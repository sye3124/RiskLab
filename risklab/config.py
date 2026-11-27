"""Project choices and shared result types. See reference/ and docs/METHODS.md."""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / 'data' / 'etf_prices.csv'
START_DATE = '2005-01-01'
END_DATE = '2026-11-01'  # exclusive
ESTIMATION_WINDOWS = (504, 252)

TICKERS = ("SPY", "VGK", "EEM", "IEF", "LQD", "GLD", "DBC")
EQUITIES = ("SPY", "VGK", "EEM")
ALTERNATIVES = ("GLD", "DBC")
STRATEGIES = (
    "60/40 SPY/IEF", "Equal weight", "Minimum variance (constrained)",
    "ERC (sample)", "ERC (shrinkage)", "ERC (shrinkage, constrained)",
)

@dataclass(frozen=True)
class Mandate:
    max_etf_weight: float = 0.35
    max_equity_weight: float = 0.70
    max_alternatives_weight: float = 0.25
    transaction_cost_bps: float = 10.0
    estimation_days: int = ESTIMATION_WINDOWS[0]
    risk_free_annual: float = 0.0
    shrinkage_intensity: float = 0.5
    turnover_penalty: float = 0.0

    def __post_init__(self):
        for key in ("max_etf_weight", "max_equity_weight", "max_alternatives_weight"):
            value = getattr(self, key)
            if not np.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{key} must be between zero and one.")
        if not isinstance(self.estimation_days, (int, np.integer)) or self.estimation_days < 60:
            raise ValueError("Estimation window must be an integer of at least 60 observations.")
        if not np.isfinite(self.transaction_cost_bps) or not 0 <= self.transaction_cost_bps <= 100:
            raise ValueError("Transaction costs must be between 0 and 100 basis points per dollar traded.")
        if not np.isfinite(self.risk_free_annual) or self.risk_free_annual <= -1:
            raise ValueError("The constant annual risk-free assumption must be finite and above -100%.")
        if not np.isfinite(self.shrinkage_intensity) or not 0 <= self.shrinkage_intensity <= 1:
            raise ValueError("Shrinkage intensity must be between zero and one.")
        if not np.isfinite(self.turnover_penalty) or self.turnover_penalty < 0:
            raise ValueError("The planned-turnover penalty must be finite and nonnegative.")

@dataclass
class BacktestResult:
    daily_returns: pd.DataFrame
    wealth: pd.DataFrame
    weights: dict[str, pd.DataFrame]
    trades: pd.DataFrame
    metrics: pd.DataFrame
    initial_signal: pd.Timestamp
    cov_estimates: dict[pd.Timestamp, dict[str, np.ndarray]]
    risk_tracking: pd.DataFrame
    realized_risk_tracking: pd.DataFrame
    solver_diagnostics: pd.DataFrame = field(default_factory=pd.DataFrame)
    target_weights: pd.DataFrame = field(default_factory=pd.DataFrame)
    data_quality: dict = field(default_factory=dict)
    execution_policy: str = "Signal at quarter-end close; execute at next observed close; new holdings earn returns thereafter."
