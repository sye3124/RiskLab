"""Numerical checks using supplied course identities and synthetic data only."""
import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from risklab.config import Mandate, STRATEGIES, TICKERS
from risklab.portfolio import (OptimizationError, covariance_estimates, optimize,
                               portfolio_risk, strategy_weights, weights_feasible)


def example_returns(n=260):
    rng = np.random.default_rng(104)
    factor = rng.normal(size=(n, 2))
    exposures = np.array([[.8,.1], [.7,.2], [.9,.2], [-.1,.3], [.2,.3], [.1,.8], [.2,.6]])
    values = .01 * (factor @ exposures.T + rng.normal(0, .25, size=(n, 7)))
    return pd.DataFrame(values, columns=TICKERS)


class CovarianceAndRiskTests(unittest.TestCase):
    def test_constant_correlation_target_matches_ln3(self):
        data = example_returns()
        estimates = covariance_estimates(data, .35)
        sample = np.cov(data, rowvar=False, ddof=1)
        np.testing.assert_allclose(estimates['sample'], sample, rtol=0, atol=0)
        np.testing.assert_allclose(np.diag(estimates['constant_correlation']), np.diag(sample))
        vol = np.sqrt(np.diag(sample))
        corr = sample / np.outer(vol, vol)
        target_corr = estimates['constant_correlation'] / np.outer(vol, vol)
        mask = ~np.eye(7, dtype=bool)
        np.testing.assert_allclose(target_corr[mask], np.mean(corr[mask]))
        np.testing.assert_allclose(estimates['shrinkage'], .35 * estimates['constant_correlation'] + .65 * sample)
        np.testing.assert_allclose(covariance_estimates(data, 0)['shrinkage'], sample)
        np.testing.assert_allclose(covariance_estimates(data, 1)['shrinkage'], estimates['constant_correlation'])

    def test_ln5_three_asset_risk_contribution_example(self):
        # LN5 pp.10-13: volatilities 30%,20%,15%, weights 50%,20%,30%.
        vols = np.array([.3, .2, .15])
        corr = np.array([[1, .8, .5], [.8, 1, .3], [.5, .3, 1]])
        risk = portfolio_risk(np.array([.5, .2, .3]), corr * np.outer(vols, vols))
        self.assertAlmostEqual(risk['volatility'], .208698, places=6)
        np.testing.assert_allclose(risk['absolute'], [.146982, .033254, .028462], atol=6e-7)
        self.assertAlmostEqual(float(np.sum(risk['percentage'])), 1.0)
        self.assertAlmostEqual(float(np.sum(risk['absolute'])), risk['volatility'])

    def test_rejects_invalid_covariance_inputs(self):
        with self.assertRaises(ValueError):
            covariance_estimates(example_returns(), 1.1)
        with self.assertRaises(ValueError):
            covariance_estimates(example_returns(30))
        data = example_returns()
        data.iloc[:, 0] = 0
        with self.assertRaisesRegex(ValueError, 'positive'):
            covariance_estimates(data)
        cov = np.eye(7)
        cov[0, 1] = .5
        with self.assertRaisesRegex(ValueError, 'symmetric'):
            optimize(cov, 'erc')


class OptimizationTests(unittest.TestCase):
    def setUp(self):
        self.covs = covariance_estimates(example_returns())

    def test_ln5_uniform_correlation_erc_equals_inverse_volatility(self):
        # LN5 p22 identity; include a negative admissible common correlation.
        vols = np.array([.3, .25, .2, .15, .1, .18, .22])
        for correlation in (0.0, .7, -.1):
            corr = np.full((7, 7), correlation)
            np.fill_diagonal(corr, 1.0)
            cov = corr * np.outer(vols, vols)
            weights, diagnostics = optimize(cov, 'erc', return_diagnostics=True)
            expected = 1 / vols
            expected /= expected.sum()
            np.testing.assert_allclose(weights, expected, atol=3e-6)
            self.assertTrue(diagnostics['solver_success'])
            self.assertTrue(diagnostics['converged'])
            self.assertLess(diagnostics['ERC_deviation_max'], 2e-5)
            self.assertLessEqual(diagnostics['max_constraint_residual'], 2e-7)
            json.dumps(diagnostics, allow_nan=False)

    def test_erc_is_invariant_to_covariance_units(self):
        baseline = optimize(self.covs['sample'], 'erc')
        for factor in (1e-12, 1e12):
            scaled = optimize(self.covs['sample'] * factor, 'erc')
            np.testing.assert_allclose(scaled, baseline, atol=5e-7)

    def test_ln5_converges_on_low_correlation_regression(self):
        # A prior numerical scaling caused SLSQP status 8 on this ordinary
        # synthetic window even though risk residuals were small. Require
        # solver success as well as the independently checked residual.
        rng = np.random.default_rng(713)
        prices = pd.DataFrame(100 * np.cumprod(1 + rng.normal(.0002, .01, (650, 7)), axis=0),
                              index=pd.bdate_range('2018-01-01', periods=650), columns=TICKERS)
        returns = prices.pct_change().iloc[1:].loc[:'2019-03-29'].iloc[-252:]
        covs = covariance_estimates(returns)
        for key in ('sample', 'shrinkage'):
            weights, diagnostic = optimize(covs[key], 'erc', return_diagnostics=True)
            self.assertTrue(diagnostic['solver_success'])
            self.assertTrue(diagnostic['converged'])
            self.assertLess(diagnostic['ERC_deviation_max'], 2e-5)
            self.assertTrue(weights_feasible(weights))

    def test_nonuniform_unconstrained_budget_is_explicitly_unsupported(self):
        with self.assertRaisesRegex(ValueError, 'equal risk budgets only'):
            optimize(self.covs['sample'], 'erc', risk_budget=np.array([.4, .1, .1, .1, .1, .1, .1]))

    def test_failed_solver_is_rejected_even_if_weights_are_feasible(self):
        failed = SimpleNamespace(x=np.full(7, 1 / 7), fun=.1, success=False,
                                 status=9, message='Iteration limit reached', nit=1800)
        with patch('risklab.portfolio.minimize', return_value=failed):
            with self.assertRaises(OptimizationError) as caught:
                optimize(np.eye(7), 'min_variance')
        records = caught.exception.diagnostics['candidates']
        self.assertEqual(len(records), 1)
        self.assertFalse(records[0]['accepted'])
        self.assertFalse(records[0]['solver_success'])
        self.assertEqual(records[0]['solver_status'], 9)
        self.assertEqual(records[0]['iterations'], 1800)

    def test_caps_and_infeasibility(self):
        for method in ('erc', 'min_variance'):
            weights, diagnostics = optimize(self.covs['shrinkage'], method, Mandate(), return_diagnostics=True)
            self.assertTrue(weights_feasible(weights, Mandate(), tol=2e-7))
            self.assertTrue(diagnostics['converged'])
            self.assertFalse(diagnostics['global_optimality_claimed'])
            self.assertTrue(all('solver_success' in item for item in diagnostics['candidates']))
        with self.assertRaisesRegex(ValueError, 'infeasible'):
            optimize(self.covs['shrinkage'], 'erc', Mandate(max_etf_weight=.1))

    def test_planned_turnover_penalty_uses_full_l1_and_retains_feasible_holdings(self):
        previous = np.array([.1, .1, .1, .25, .2, .125, .125])
        for method in ('min_variance', 'erc'):
            base = optimize(self.covs['shrinkage'], method, Mandate(), previous_weights=previous)
            penalized, diagnostics = optimize(self.covs['shrinkage'], method, Mandate(turnover_penalty=10),
                                               previous_weights=previous, return_diagnostics=True)
            self.assertLess(np.abs(penalized - previous).sum(), np.abs(base - previous).sum())
            np.testing.assert_allclose(penalized, previous, atol=2e-6)
            actual_full_l1 = float(np.abs(penalized - previous).sum())
            self.assertAlmostEqual(diagnostics['planned_full_L1_turnover'], actual_full_l1)
            self.assertAlmostEqual(diagnostics['objective_turnover_penalty'], 10 * actual_full_l1)
            self.assertAlmostEqual(diagnostics['planned_one_way_turnover'], actual_full_l1 / 2)
            self.assertTrue(diagnostics['turnover_penalty_active'])

    def test_six_strategies_and_json_safe_diagnostics(self):
        weights, diagnostics = strategy_weights(self.covs, Mandate(), return_diagnostics=True)
        self.assertEqual(tuple(weights), STRATEGIES)
        self.assertEqual([row['strategy'] for row in diagnostics], list(STRATEGIES))
        self.assertEqual(len(diagnostics), 6)
        self.assertFalse(diagnostics[0]['mandate_applied'])
        self.assertTrue(diagnostics[2]['mandate_applied'])
        self.assertEqual(diagnostics[0]['method'], 'fixed')
        self.assertEqual(diagnostics[3]['method'], 'erc')
        json.dumps(diagnostics, allow_nan=False)


if __name__ == '__main__':
    unittest.main()
