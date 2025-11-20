"""
VAR-DCC-GARCH comparison and likelihood-ratio test (Python port).
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from scipy.stats import chi2
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller

try:
    from .utils import DCCResult, fit_dcc_model
except ImportError:  # pragma: no cover
    import sys
    from pathlib import Path

    CURRENT_DIR = Path(__file__).resolve().parent
    if str(CURRENT_DIR) not in sys.path:
        sys.path.append(str(CURRENT_DIR))
    from utils import DCCResult, fit_dcc_model


@dataclass
class VARComparisonResult:
    adf_table: pd.DataFrame
    best_lag: int
    lr_stat: float
    lr_pvalue: float
    dcc_result: DCCResult


def _adf_summary(series: pd.Series) -> pd.Series:
    statistic, p_value, _, _, crit_values, _ = adfuller(series.dropna(), autolag="AIC")
    return pd.Series(
        {
            "Test Statistic": statistic,
            "p-value": p_value,
            "Critical Value (5%)": crit_values.get("5%", float("nan")),
        }
    )


def compare_with_var(
    returns: pd.DataFrame, baseline: DCCResult, max_lags: int = 10
) -> VARComparisonResult:
    """
    Estimate VAR(p)-DCC-GARCH and perform the LR test against the baseline DCC.
    """
    adf_results = pd.DataFrame(
        {_col: _adf_summary(returns[_col]) for _col in returns.columns}
    ).T
    adf_results.index.name = "Series"

    var_model = VAR(returns.dropna())
    order_selection = var_model.select_order(maxlags=max_lags)
    best_lag = order_selection.selected_orders.get("bic")
    if best_lag is None:
        best_lag = int(order_selection.aic.idxmin())

    var_fit = var_model.fit(best_lag)
    residuals = var_fit.resid

    dcc_var = fit_dcc_model(
        residuals,
        univariate_model="garch",
        variant="dcc",
        name="VAR_DCC_GARCH",
    )

    lr_stat = max(0.0, 2.0 * (dcc_var.loglikelihood - baseline.loglikelihood))
    dof = max(best_lag, 1)
    lr_pvalue = chi2.sf(lr_stat, dof)

    return VARComparisonResult(
        adf_table=adf_results,
        best_lag=best_lag,
        lr_stat=lr_stat,
        lr_pvalue=lr_pvalue,
        dcc_result=dcc_var,
    )


if __name__ == "__main__":
    print("Invoke compare_with_var(returns_df, baseline_result) within the pipeline.")
