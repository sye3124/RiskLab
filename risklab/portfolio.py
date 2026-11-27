"""Portfolio calculations using the supplied QARM II course materials.

Method references (bundled under ``reference/``):
* LN3, pp. 9-10: sample/constant-correlation covariance blend.
* LN5, p. 5: marginal, absolute and relative volatility contributions.
* LN5, p. 28: log-constraint ERC optimization, then weight rescaling.

The fixed shrinkage intensity, client caps, constrained squared-error ERC
approximation and optional planned-turnover penalty are explicit project
choices. No estimated Ledoit-Wolf intensity or external ERC formulation is
used. A convergence flag is necessary, not a proof of global optimality.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .config import ALTERNATIVES, EQUITIES, STRATEGIES, TICKERS, Mandate

_FEASIBILITY_TOL = 2e-7
_ERC_MAX_ERROR = 2e-5
_EQ_IDX = np.array([TICKERS.index(t) for t in EQUITIES])
_ALT_IDX = np.array([TICKERS.index(t) for t in ALTERNATIVES])


class OptimizationError(RuntimeError):
    """No accepted converged solution; ``diagnostics`` retains every attempt."""

    def __init__(self, message: str, diagnostics: dict[str, Any]):
        super().__init__(message)
        self.diagnostics = diagnostics


def covariance_estimates(training_returns: pd.DataFrame,
                         shrinkage_intensity: float = 0.5) -> dict[str, np.ndarray]:
    """LN3 constant-correlation target blended with the sample covariance.

    ``sample`` uses ``ddof=1``. The target keeps its variances and assigns the
    average of the sample's off-diagonal correlations to every asset pair.
    ``shrinkage = delta * target + (1-delta) * sample``. The default delta=0.5
    is a fixed project assumption, not an estimated optimal intensity.
    """
    if not np.isfinite(shrinkage_intensity) or not 0 <= shrinkage_intensity <= 1:
        raise ValueError("Shrinkage intensity must be between zero and one.")
    arr = np.asarray(training_returns, dtype=float)
    if arr.ndim != 2 or len(arr) < 60 or arr.shape[1] < 2 or not np.isfinite(arr).all():
        raise ValueError("Covariance estimation requires at least 60 complete return observations and two assets.")
    sample = np.cov(arr, rowvar=False, ddof=1)
    variances = np.diag(sample)
    if not np.isfinite(sample).all() or np.any(variances <= 0):
        raise ValueError("Every asset must have finite, strictly positive sample variance.")
    volatility = np.sqrt(variances)
    correlation = sample / np.outer(volatility, volatility)
    off_diagonal = ~np.eye(len(variances), dtype=bool)
    mean_correlation = float(correlation[off_diagonal].mean())
    target = mean_correlation * np.outer(volatility, volatility)
    np.fill_diagonal(target, variances)
    delta = float(shrinkage_intensity)
    shrinkage = delta * target + (1.0 - delta) * sample
    return {"sample": sample, "constant_correlation": target, "shrinkage": shrinkage}


def _validated_covariance(cov: np.ndarray, n: int | None = None) -> np.ndarray:
    cov = np.asarray(cov, dtype=float)
    if (cov.ndim != 2 or cov.shape[0] != cov.shape[1]
            or (n is not None and cov.shape != (n, n)) or not np.isfinite(cov).all()):
        dimension = f"{n}x{n}" if n is not None else "square"
        raise ValueError(f"Covariance must be a finite {dimension} matrix.")
    scale = float(np.max(np.abs(cov)))
    if scale <= 0 or np.any(np.diag(cov) <= 0):
        raise ValueError("Covariance must have strictly positive asset variances.")
    if not np.allclose(cov, cov.T, rtol=1e-10, atol=scale * 1e-12):
        raise ValueError("Covariance must be symmetric.")
    cov = (cov + cov.T) / 2.0
    if np.linalg.eigvalsh(cov / scale).min() < -1e-10:
        raise ValueError("Covariance must be positive semidefinite.")
    return cov


def portfolio_risk(weights: np.ndarray, cov: np.ndarray) -> dict[str, np.ndarray | float]:
    """Volatility and LN5 marginal/absolute/relative risk contributions."""
    weights = np.asarray(weights, dtype=float)
    cov = _validated_covariance(cov)
    if weights.shape != (cov.shape[0],) or not np.isfinite(weights).all():
        raise ValueError("Weights must be a finite vector matching the covariance.")
    variance = float(weights @ cov @ weights)
    if not np.isfinite(variance) or variance <= 0:
        raise ValueError("Portfolio variance must be strictly positive.")
    volatility = float(np.sqrt(variance))
    marginal = cov @ weights / volatility
    absolute = weights * marginal
    return {"volatility": volatility, "marginal": marginal,
            "absolute": absolute, "percentage": absolute / volatility}


def _weight_residual(w: np.ndarray, mandate: Mandate | None = None) -> float:
    w = np.asarray(w, dtype=float)
    if w.shape != (len(TICKERS),) or not np.isfinite(w).all():
        return float("inf")
    cap = mandate.max_etf_weight if mandate is not None else 1.0
    violations = [abs(float(w.sum()) - 1.0), float(np.max(-w)), float(np.max(w - cap)), 0.0]
    if mandate is not None:
        violations += [float(w[_EQ_IDX].sum() - mandate.max_equity_weight),
                       float(w[_ALT_IDX].sum() - mandate.max_alternatives_weight)]
    return max(violations)


def weights_feasible(w: np.ndarray, mandate: Mandate | None = None,
                     tol: float = 1e-6) -> bool:
    return _weight_residual(w, mandate) <= tol


def _restrictions(mandate: Mandate | None, size: int | None = None):
    n = len(TICKERS)
    size = n if size is None else size
    cap = mandate.max_etf_weight if mandate is not None else 1.0
    ones = np.r_[np.ones(n), np.zeros(size - n)]
    constraints = [{"type": "eq", "fun": lambda x: float(x[:n].sum() - 1.0),
                    "jac": lambda x: ones}]
    if mandate is not None:
        for indices, limit in ((_EQ_IDX, mandate.max_equity_weight),
                               (_ALT_IDX, mandate.max_alternatives_weight)):
            derivative = np.zeros(size)
            derivative[indices] = -1.0
            constraints.append({"type": "ineq",
                                "fun": lambda x, idx=indices, bound=limit: float(bound - x[idx].sum()),
                                "jac": lambda x, derivative=derivative: derivative})
    bounds = [(0.0, cap)] * n + [(0.0, 2.0)] * (size - n)
    return bounds, constraints


def _project_start(preferred: np.ndarray, mandate: Mandate | None) -> np.ndarray:
    n = len(TICKERS)
    if mandate is None:
        return np.asarray(preferred, dtype=float).copy()
    capacity = (min(3 * mandate.max_etf_weight, mandate.max_equity_weight)
                + min(2 * mandate.max_etf_weight, mandate.max_alternatives_weight)
                + 2 * mandate.max_etf_weight)
    if capacity < 1.0 - _FEASIBILITY_TOL:
        raise ValueError("Client restrictions are infeasible: asset and group caps cannot allocate 100%.")
    if weights_feasible(preferred, mandate, tol=_FEASIBILITY_TOL):
        return np.asarray(preferred, dtype=float).copy()
    bounds, constraints = _restrictions(mandate)
    initial = np.full(n, 1 / n)
    result = minimize(lambda w: .5 * float(np.sum((w - preferred) ** 2)), initial,
                      jac=lambda w: w - preferred, method="SLSQP", bounds=bounds,
                      constraints=constraints, options={"maxiter": 1000, "ftol": 1e-12})
    if not result.success or not weights_feasible(result.x, mandate, tol=_FEASIBILITY_TOL):
        raise ValueError("Could not obtain a converged feasible start; client restrictions may be infeasible. "
                         + str(result.message))
    return np.asarray(result.x, dtype=float)


def _finite_or_none(value: Any) -> float | None:
    try:
        number = float(value)
        return number if np.isfinite(number) else None
    except (ValueError, TypeError):
        return None


def _solver_record(result: Any, name: str, residual: float,
                   accepted: bool, risk_error: float | None = None) -> dict[str, Any]:
    return {
        "start": name, "solver_success": bool(getattr(result, "success", False)),
        "success": bool(getattr(result, "success", False)),
        "accepted": bool(accepted), "solver_status": int(getattr(result, "status", -1)),
        "solver_message": str(getattr(result, "message", "No solver message")),
        "iterations": int(getattr(result, "nit", 0)),
        "objective": _finite_or_none(getattr(result, "fun", None)),
        "max_constraint_residual": _finite_or_none(residual),
        "ERC_deviation_max": _finite_or_none(risk_error),
    }


def _relative_risk_objective(w: np.ndarray, cov: np.ndarray,
                             target: np.ndarray) -> tuple[float, np.ndarray]:
    cov_w = cov @ w
    variance = float(w @ cov_w)
    if variance <= 0 or not np.isfinite(variance):
        return 1e10, np.zeros_like(w)
    numerator = w * cov_w
    difference = numerator / variance - target
    # Differentiate the LN5 relative risk-contribution expression directly.
    jacobian = ((np.diag(cov_w) + w[:, None] * cov) / variance
                - np.outer(numerator, 2 * cov_w) / variance ** 2)
    return float(difference @ difference), 2 * jacobian.T @ difference


def _ln5_erc(cov: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    """Solve LN5 p28 with an interior logarithmic constraint, then rescale."""
    n = len(TICKERS)
    if not np.allclose(target, np.full(n, 1 / n), rtol=0, atol=1e-12):
        raise ValueError("The course LN5 log-constraint solver supports equal risk budgets only. "
                         "Nonuniform unconstrained budgets are not implemented.")
    cov_scale = float(np.diag(cov).mean())
    normalized_cov = cov / cov_scale
    inv_vol = 1 / np.sqrt(np.diag(normalized_cov))
    inv_vol /= inv_vol.sum()
    # LN5 permits c below -n*log(n). Choosing the product of a half-invested
    # positive starting vector is a numerical scale choice, not a new method.
    base_c = float(np.log(inv_vol / 2).sum())
    candidates: list[dict[str, Any]] = []
    accepted: list[tuple[np.ndarray, dict[str, Any], float, float, float]] = []
    for attempt, c in enumerate((base_c, base_c - n * np.log(4.0))):
        factor = float(np.exp((c - np.log(inv_vol).sum()) / n))
        start = inv_vol * factor
        # Solve in y=x/exp(c/n), so typical variables are order one and the
        # constraint is mean(log(y)) >= 0. This is the same LN5 constraint,
        # with numerical rescaling only. Tiny x made SLSQP's line search report
        # failure near the solution for otherwise ordinary low correlations.
        variable_scale = float(np.exp(c / n))
        scaled_start = start / variable_scale
        objective_scale = float(scaled_start @ normalized_cov @ scaled_start)
        if objective_scale <= 0:
            continue
        lower = min(1e-14, float(start.min()) * 1e-4)
        constraint = {"type": "ineq", "fun": lambda y: float(np.log(y).mean()),
                      "jac": lambda y: 1 / (n * y)}
        result = minimize(lambda y: .5 * float(y @ normalized_cov @ y) / objective_scale,
                          scaled_start, jac=lambda y: normalized_cov @ y / objective_scale,
                          method="SLSQP", bounds=[(lower / variable_scale, 1.0 / variable_scale)] * n,
                          constraints=[constraint], options={"maxiter": 1800, "ftol": 1e-11})
        x = np.asarray(result.x, dtype=float) * variable_scale
        valid = np.isfinite(x).all() and np.all(x > 0) and x.sum() > 0
        weights = x / x.sum() if valid else np.full(n, np.nan)
        residual = (max(0.0, c - float(np.log(x).sum()), float(np.max(lower - x)),
                        float(np.max(x - 1)), _weight_residual(weights)) if valid else float("inf"))
        risk_error = (float(np.max(np.abs(portfolio_risk(weights, normalized_cov)["percentage"] - target)))
                      if valid and float(weights @ normalized_cov @ weights) > 0 else float("inf"))
        ok = (bool(result.success) and valid and np.isfinite(result.fun)
              and residual <= _FEASIBILITY_TOL and risk_error <= _ERC_MAX_ERROR)
        record = _solver_record(result, f"LN5 log constraint scale {attempt + 1}", residual, ok, risk_error)
        record["log_constraint_c"] = float(c)
        candidates.append(record)
        if ok:
            accepted.append((weights, record, c, objective_scale, variable_scale))
            # The residual, feasibility and convergence criteria already verify
            # this unique course ERC target; a second scale is only a fallback.
            break
    if not accepted:
        raise OptimizationError("LN5 ERC failed convergence, feasibility or risk-budget tolerance checks.",
                                {"method": "erc", "formulation": "LN5 log constraint and rescale",
                                 "candidates": candidates})
    weights, chosen, c, objective_scale, variable_scale = accepted[0]
    diagnostics = {
        **chosen, "success": True, "converged": True, "method": "erc",
        "formulation": "LN5 log-constraint minimization followed by rescaling",
        "objective_units": "Half variance of scaled unnormalized variables divided by their initial quadratic value",
        "covariance_normalization": cov_scale, "log_objective_normalization": objective_scale,
        "log_variable_normalization": variable_scale,
        "log_constraint_c": float(c), "candidates": candidates,
        "risk_budget_tolerance_max": _ERC_MAX_ERROR,
    }
    return weights, diagnostics


def optimize(cov: np.ndarray, method: str, mandate: Mandate | None = None,
             risk_budget: np.ndarray | None = None, previous_weights: np.ndarray | None = None,
             return_diagnostics: bool = False):
    """Return portfolio weights, optionally together with solver diagnostics.

    Unconstrained equal-budget ERC uses LN5 p28. Capped ERC minimizes squared
    relative-risk-budget errors, an explicitly labeled project approximation.
    Minimum variance minimizes variance divided by the mean asset variance;
    this normalization leaves the zero-penalty minimizer unchanged.

    For capped strategies only, a nonzero ``mandate.turnover_penalty`` adds
    lambda * sum(abs(target - previous_weights)). Previous weights must be
    known at the signal date. This planned-turnover preference does not impose
    a hard limit on turnover at the later execution date.
    """
    n = len(TICKERS)
    cov = _validated_covariance(cov, n)
    if method not in {"erc", "min_variance"}:
        raise ValueError("Unknown optimization method.")
    target = np.full(n, 1 / n) if risk_budget is None else np.asarray(risk_budget, dtype=float)
    if (target.shape != (n,) or not np.isfinite(target).all() or (target < 0).any()
            or not np.isclose(target.sum(), 1.0, rtol=0, atol=1e-10)):
        raise ValueError("Risk budgets must be seven nonnegative values summing to one.")
    previous = None if previous_weights is None else np.asarray(previous_weights, dtype=float)
    if previous is not None and not weights_feasible(previous, tol=_FEASIBILITY_TOL):
        raise ValueError("Previous weights must be seven finite, nonnegative weights summing to one.")
    penalty = float(mandate.turnover_penalty) if mandate is not None else 0.0
    penalty_active = penalty > 0 and previous is not None
    if method == "erc" and mandate is None:
        weights, diagnostics = _ln5_erc(cov, target)
        diagnostics["objective_base"] = diagnostics["objective"]
        diagnostics["objective_turnover_penalty"] = 0.0
    else:
        normalized_cov = cov / float(np.diag(cov).mean())
        initial = _project_start(np.full(n, 1 / n), mandate)
        preferences = [("feasible equal-weight projection", initial)]
        if method == "erc":
            inv_vol = 1 / np.sqrt(np.diag(cov))
            inv_vol /= inv_vol.sum()
            preferences.append(("inverse-volatility projection", _project_start(inv_vol, mandate)))
            if previous is not None:
                preferences.append(("previous-holdings projection", _project_start(previous, mandate)))
            else:
                preferred = np.array([.12, .12, .12, .29, .18, .085, .085])
                preferences.append(("alternate deterministic projection", _project_start(preferred, mandate)))
        # De-duplicate starts; the previous package inadvertently repeated one.
        starts: list[tuple[str, np.ndarray]] = []
        for name, start in preferences:
            if not any(np.allclose(start, old, rtol=0, atol=1e-9) for _, old in starts):
                starts.append((name, start))
        size = 2 * n if penalty_active else n
        bounds, constraints = _restrictions(mandate, size)
        if penalty_active:
            # z_i >= +/- (w_i - previous_i): exact epigraph of the full L1 term.
            minus_jac = np.c_[-np.eye(n), np.eye(n)]
            plus_jac = np.c_[np.eye(n), np.eye(n)]
            constraints.extend([
                {"type": "ineq", "fun": lambda x: x[n:] - (x[:n] - previous),
                 "jac": lambda x: minus_jac},
                {"type": "ineq", "fun": lambda x: x[n:] + (x[:n] - previous),
                 "jac": lambda x: plus_jac},
            ])

        def base(w):
            if method == "min_variance":
                return float(w @ normalized_cov @ w), 2 * normalized_cov @ w
            return _relative_risk_objective(w, normalized_cov, target)

        def objective(x):
            value, _ = base(x[:n])
            return value + (penalty * float(x[n:].sum()) if penalty_active else 0.0)

        def derivative(x):
            _, grad = base(x[:n])
            return np.r_[grad, np.full(n, penalty)] if penalty_active else grad

        candidates = []
        accepted = []
        for name, w0 in starts:
            x0 = np.r_[w0, np.abs(w0 - previous)] if penalty_active else w0
            result = minimize(objective, x0, jac=derivative, method="SLSQP", bounds=bounds,
                              constraints=constraints, options={"maxiter": 1800, "ftol": 1e-12})
            w = np.asarray(result.x[:n], dtype=float)
            residual = _weight_residual(w, mandate)
            if penalty_active and np.isfinite(result.x).all():
                residual = max(residual, float(np.max(np.abs(w - previous) - result.x[n:])),
                               float(np.max(-result.x[n:])))
            risk_error = (float(np.max(np.abs(portfolio_risk(w, normalized_cov)["percentage"] - target)))
                          if np.isfinite(w).all() and float(w @ normalized_cov @ w) > 0 else float("inf"))
            ok = (bool(result.success) and np.isfinite(result.x).all()
                  and np.isfinite(result.fun) and residual <= _FEASIBILITY_TOL
                  and np.isfinite(risk_error))
            record = _solver_record(result, name, residual, ok, risk_error)
            candidates.append(record)
            if ok:
                accepted.append((w.copy(), record))
        if not accepted:
            raise OptimizationError(f"No converged feasible {method} solution passed validation.",
                                    {"method": method, "candidates": candidates})
        weights, chosen = min(accepted, key=lambda pair: pair[1]["objective"])
        value, _ = base(weights)
        penalty_value = penalty * float(np.abs(weights - previous).sum()) if penalty_active else 0.0
        formulation = ("Minimum variance under client caps" if method == "min_variance" and mandate is not None
                       else "Long-only minimum variance" if method == "min_variance"
                       else "Project approximation: relative risk-contribution squared error under caps")
        if penalty_active:
            formulation += "; plus project planned-turnover penalty"
        diagnostics = {
            **chosen, "success": True, "converged": True, "method": method,
            "formulation": formulation, "objective_base": float(value),
            "objective_turnover_penalty": float(penalty_value),
            "objective": float(value + penalty_value),
            "objective_units": ("Portfolio variance / mean asset variance + lambda * planned full L1 turnover"
                                if method == "min_variance"
                                else "Sum of squared relative risk-budget errors + lambda * planned full L1 turnover"),
            "covariance_normalization": float(np.diag(cov).mean()),
            "candidates": candidates, "risk_budget_tolerance_max": None,
        }
    risk = portfolio_risk(weights, cov)["percentage"]
    planned = None if previous is None else float(np.abs(weights - previous).sum())
    diagnostics.update({
        "ERC_deviation_L2": float(np.linalg.norm(risk - target)),
        "ERC_deviation_max": float(np.max(np.abs(risk - target))),
        "turnover_penalty": penalty, "turnover_penalty_active": bool(penalty_active),
        "planned_full_L1_turnover": planned,
        "planned_one_way_turnover": None if planned is None else planned / 2,
        "constraint_tolerance": _FEASIBILITY_TOL,
        "mandate_applied": mandate is not None,
        "global_optimality_claimed": False,
    })
    return (weights, diagnostics) if return_diagnostics else weights


def strategy_weights(covs: Mapping[str, np.ndarray], mandate: Mandate,
                     previous_weights: Mapping[str, np.ndarray] | None = None,
                     return_diagnostics: bool = False):
    """All six project portfolios, with optional per-strategy diagnostics."""
    n = len(TICKERS)
    w_6040 = np.zeros(n)
    w_6040[TICKERS.index("SPY")] = .60
    w_6040[TICKERS.index("IEF")] = .40
    weights = {STRATEGIES[0]: w_6040, STRATEGIES[1]: np.full(n, 1 / n)}
    diagnostics: list[dict[str, Any]] = []
    previous = {} if previous_weights is None else previous_weights
    for name, weight in weights.items():
        prev = previous.get(name)
        planned = None if prev is None else float(np.abs(weight - np.asarray(prev)).sum())
        risk = portfolio_risk(weight, covs["shrinkage"])["percentage"]
        diagnostics.append({
            "strategy": name, "covariance_estimator": "shrinkage", "method": "fixed",
            "formulation": "Fixed benchmark; no numerical optimization", "success": True,
            "converged": True, "solver_success": True, "solver_status": 0,
            "solver_message": "Fixed reference weights; client caps deliberately not applied",
            "iterations": 0, "objective": None, "objective_base": None,
            "objective_turnover_penalty": 0.0, "max_constraint_residual": _weight_residual(weight),
            "ERC_deviation_L2": float(np.linalg.norm(risk - 1 / n)),
            "ERC_deviation_max": float(np.max(np.abs(risk - 1 / n))),
            "turnover_penalty": 0.0, "turnover_penalty_active": False,
            "planned_full_L1_turnover": planned,
            "planned_one_way_turnover": None if planned is None else planned / 2,
            "mandate_applied": False, "global_optimality_claimed": False, "candidates": [],
        })
    specifications = [
        (STRATEGIES[2], "shrinkage", "min_variance", mandate),
        (STRATEGIES[3], "sample", "erc", None),
        (STRATEGIES[4], "shrinkage", "erc", None),
        (STRATEGIES[5], "shrinkage", "erc", mandate),
    ]
    for name, covariance_key, method, restrictions in specifications:
        weight, diagnostic = optimize(covs[covariance_key], method, restrictions,
                                      previous_weights=previous.get(name), return_diagnostics=True)
        weights[name] = weight
        diagnostics.append({"strategy": name, "covariance_estimator": covariance_key, **diagnostic})
    return (weights, diagnostics) if return_diagnostics else weights
