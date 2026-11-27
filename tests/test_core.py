"""Deterministic accounting/chronology tests; all fixtures are artificial."""
import unittest
import numpy as np
import pandas as pd
from risklab import TICKERS, STRATEGIES, Mandate, PriceValidationError, backtest, clean_prices, weights_feasible


def example_prices(n=810, seed=11):
    rng = np.random.default_rng(seed)
    factors = rng.normal(0, .008, (n,3))
    exposure = np.array([[.85,.10,.10],[.77,.20,.08],[.92,.15,.10],
                         [-.08,.17,.16],[.20,.42,.10],[-.08,.05,.75],[.17,.20,.48]])
    returns = factors @ exposure.T + rng.normal(0,.002,(n,7))
    returns += np.array([.0003,.00025,.0002,.0001,.00015,.0002,.0002])
    return pd.DataFrame(100*np.cumprod(1+returns,axis=0),
                        index=pd.bdate_range('2018-01-01',periods=n),columns=TICKERS)


class CleaningTests(unittest.TestCase):
    def test_missing_required_ticker(self):
        with self.assertRaisesRegex(PriceValidationError, 'Missing required ticker'):
            clean_prices(example_prices().drop(columns=['DBC']))

    def test_interior_missing_date_is_rejected_without_aggregation(self):
        prices = example_prices()
        missing_date = prices.index[200]
        prices.loc[missing_date,'SPY'] = np.nan
        with self.assertRaises(PriceValidationError) as error:
            clean_prices(prices)
        self.assertIn(str(missing_date.date()),error.exception.report['incomplete_dates'])
        self.assertEqual(error.exception.report['dates_removed_for_missing_prices'],0)

    def test_duplicate_date_is_not_silently_overwritten(self):
        prices = example_prices()
        prices = pd.concat([prices,prices.iloc[[40]]])
        with self.assertRaisesRegex(PriceValidationError,'Duplicate dates'):
            clean_prices(prices)

    def test_leading_pre_common_history_is_documented(self):
        prices = example_prices()
        prices.loc[prices.index[:7],'DBC'] = np.nan
        clean, report = clean_prices(prices)
        self.assertEqual(clean.index[0],prices.index[7])
        self.assertEqual(report['leading_rows_trimmed'],7)
        self.assertEqual(len(report['leading_dates_trimmed']),7)

    def test_mandate_invalid_values_rejected(self):
        for kwargs in ({'transaction_cost_bps':float('nan')},{'estimation_days':60.5},
                       {'shrinkage_intensity':1.1},{'turnover_penalty':-1}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                Mandate(**kwargs)


class BacktestTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.prices = example_prices()
        cls.mandate = Mandate(estimation_days=252)
        cls.result = backtest(cls.prices,cls.mandate)

    def test_initial_cash_has_no_asset_return_before_execution(self):
        r = self.result
        expected_execution = self.prices.index[self.prices.index.get_loc(r.initial_signal)+1]
        self.assertEqual(r.wealth.index[0],expected_execution)
        initial = r.trades.loc[r.trades.trade_type.eq('initial')]
        self.assertTrue(initial['date'].eq(expected_execution).all())
        self.assertTrue(initial['signal_date'].eq(r.initial_signal).all())
        np.testing.assert_allclose(r.daily_returns.iloc[0],-.001,atol=1e-12)
        np.testing.assert_allclose(r.wealth.iloc[0],.999,atol=1e-12)

    def test_new_holdings_earn_the_next_interval(self):
        r = self.result
        date = r.daily_returns.index[1]
        asset_return = self.prices.pct_change(fill_method=None).loc[date].to_numpy()
        targets = r.target_weights.loc[r.target_weights.signal_date.eq(r.initial_signal)].set_index('strategy')
        for name in STRATEGIES:
            expected = targets.loc[name,list(TICKERS)].to_numpy(dtype=float) @ asset_return
            self.assertAlmostEqual(r.daily_returns.loc[date,name],expected,places=12)

    def test_nav_recursion_and_weight_sums(self):
        r = self.result
        np.testing.assert_allclose(r.wealth,(1+r.daily_returns).cumprod(),atol=2e-13)
        for frame in r.weights.values():
            np.testing.assert_allclose(frame.sum(axis=1),1,atol=1e-8)

    def test_rebalance_costs_use_both_trade_legs_after_market_drift(self):
        r = self.result
        name = 'ERC (shrinkage, constrained)'
        asset_returns = self.prices.pct_change(fill_method=None)
        for trade in r.trades.loc[r.trades.strategy.eq(name) & r.trades.trade_type.eq('rebalance')].itertuples():
            position = r.wealth.index.get_loc(trade.date)
            prior_date = r.wealth.index[position-1]
            prior_weights = r.weights[name].loc[prior_date].to_numpy()
            asset_return = asset_returns.loc[trade.date].to_numpy()
            gross_return = prior_weights @ asset_return
            drifted = prior_weights*(1+asset_return)/(1+gross_return)
            target = r.weights[name].loc[trade.date].to_numpy()
            full_notional = np.abs(target-drifted).sum()
            expected_cost = r.wealth.loc[prior_date,name]*(1+gross_return)*.001*full_notional
            self.assertAlmostEqual(trade.traded_notional,full_notional,places=12)
            self.assertAlmostEqual(trade.turnover,full_notional/2,places=12)
            self.assertAlmostEqual(trade.cost_nav,expected_cost,places=12)
            self.assertLess(trade.signal_date,trade.date)
            self.assertTrue(weights_feasible(target,self.mandate))

    def test_future_asset_specific_changes_do_not_change_prior_targets(self):
        r = self.result
        signals = sorted(r.target_weights.signal_date.unique())
        boundary = pd.Timestamp(signals[2])
        future = self.prices.index > boundary
        altered = self.prices.copy()
        length = int(future.sum())
        t = np.arange(length)[:,None]
        asset = np.arange(1,8)[None,:]
        multipliers = np.exp(.025*np.sin(t*.17*asset)+.002*t*asset/max(length,1))
        altered.loc[future,:] *= multipliers
        changed = backtest(altered,self.mandate)
        expected = r.target_weights.loc[r.target_weights.signal_date <= boundary,list(TICKERS)]
        actual = changed.target_weights.loc[changed.target_weights.signal_date <= boundary,list(TICKERS)]
        np.testing.assert_allclose(actual,expected,atol=1e-10)
        np.testing.assert_allclose(changed.wealth.loc[:boundary],r.wealth.loc[:boundary],atol=1e-12)
        for date in r.cov_estimates:
            if date <= boundary:
                np.testing.assert_allclose(changed.cov_estimates[date]['sample'],r.cov_estimates[date]['sample'],atol=1e-14)

    def test_solver_records_cover_each_strategy_signal(self):
        r = self.result
        d = r.solver_diagnostics
        self.assertEqual(len(d),len(r.target_weights))
        self.assertTrue(d['solver_success'].all())
        self.assertTrue(d['training_end'].eq(d['signal_date']).all())
        self.assertTrue((d['signal_date'] < d['execution_date']).all())
        self.assertTrue(d['training_observations'].eq(252).all())

    def test_realized_risk_excludes_pre_execution_interval(self):
        table = self.result.realized_risk_tracking
        self.assertFalse(table.empty)
        self.assertTrue((table.evaluation_start > table.execution_date).all())
        self.assertTrue((table.observations >= 20).all())
        self.assertIn('partial_period',table.columns)

    def test_cagr_uses_declared_calendar_span(self):
        r = self.result
        years = (r.wealth.index[-1]-r.initial_signal).total_seconds()/(365.25*86400)
        np.testing.assert_allclose(r.metrics.CAGR,(r.wealth.iloc[-1]**(1/years)-1).reindex(r.metrics.index))
        np.testing.assert_allclose(r.metrics.annualized_rebalance_turnover,r.metrics.rebalance_one_way_turnover/years)

    def test_costless_run_has_same_targets_and_higher_final_nav(self):
        costless = backtest(self.prices,Mandate(estimation_days=252,transaction_cost_bps=0))
        np.testing.assert_allclose(costless.target_weights.loc[:,list(TICKERS)],
                                   self.result.target_weights.loc[:,list(TICKERS)],atol=1e-9)
        self.assertTrue((costless.wealth.iloc[-1] >= self.result.wealth.iloc[-1]).all())


if __name__ == '__main__':
    unittest.main()
