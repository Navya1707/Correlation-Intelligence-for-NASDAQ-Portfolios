"""
Rolling 90-day correlations between the macro-financial index and each asset.

The function ``compute_rolling_correlations`` accepts a returns DataFrame whose first column
is the macro index and the remaining columns are asset returns.
"""
from __future__ import annotations

import pandas as pd


def compute_rolling_correlations(
    returns: pd.DataFrame, window: int = 90
) -> pd.DataFrame:
    """
    Compute rolling correlations between the macro index and each asset.

    Parameters
    ----------
    returns:
        DataFrame where the first column is the macro index and the remaining
        columns are asset return series.
    window:
        Size of the symmetric rolling window in days (default: 90).
    """
    if returns.shape[1] < 2:
        raise ValueError("At least two series are required (macro index + assets).")

    macro_col = returns.columns[0]
    asset_cols = returns.columns[1:]

    effective_window = max(1, window * 2 - 1)
    rolling = pd.DataFrame(index=returns.index, columns=asset_cols, dtype=float)
    for col in asset_cols:
        rolling[col] = returns[macro_col].rolling(
            effective_window, center=True, min_periods=window
        ).corr(returns[col])

    return rolling.dropna(how="all")


if __name__ == "__main__":
    print(
        "Import compute_rolling_correlations from this module and provide a returns DataFrame."
    )
