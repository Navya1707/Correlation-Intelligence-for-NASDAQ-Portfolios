"""
Visualization helpers for the DCC pipeline.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

try:
    from .dcc_utils import DCCResult
except ImportError:  # pragma: no cover
    import sys
    from pathlib import Path

    CURRENT_DIR = Path(__file__).resolve().parent
    if str(CURRENT_DIR) not in sys.path:
        sys.path.append(str(CURRENT_DIR))
    from dcc_utils import DCCResult


def plot_rolling_correlations(rolling_corr: pd.DataFrame) -> None:
    """Stacked plot of 90-day rolling correlations."""
    ax = rolling_corr.plot(figsize=(12, 6), linewidth=2)
    ax.set_title(
        "90-Day Rolling Window Correlations between Macro Index and Assets", fontsize=16
    )
    ax.set_ylabel("Correlation")
    ax.grid(alpha=0.3)
    plt.tight_layout()


def plot_dcc_vs_rolling(dcc_result: DCCResult, rolling_corr: pd.DataFrame) -> None:
    """Compare DCC correlations to the rolling benchmark."""
    aligned_dcc, aligned_roll = dcc_result.correlations.align(rolling_corr, join="inner")
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    aligned_dcc.plot(ax=axes[0], linewidth=2.5)
    axes[0].set_title("DCC-GARCH(1,1) Correlations", fontsize=14)
    axes[0].set_ylabel("Correlation")

    aligned_roll.plot(ax=axes[1], linewidth=2.0)
    axes[1].set_title("90-Day Rolling Window Correlations", fontsize=14)
    axes[1].set_ylabel("Correlation")

    fig.suptitle(
        "Comparison of DCC-GARCH(1,1) and 90-Day Rolling Correlations", fontsize=18
    )
    plt.tight_layout()


def plot_var_vs_dcc(dcc_result: DCCResult, var_dcc_result: DCCResult) -> None:
    """Overlay VAR-DCC results against the baseline DCC."""
    aligned_var, aligned_dcc = var_dcc_result.correlations.align(
        dcc_result.correlations, join="inner"
    )
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    aligned_var.plot(ax=axes[0], linewidth=2.5)
    axes[0].set_title("VAR(1)-DCC-GARCH(1,1) Correlations", fontsize=14)

    aligned_dcc.plot(ax=axes[1], linewidth=2.5)
    axes[1].set_title("DCC-GARCH(1,1) Correlations", fontsize=14)
    axes[1].set_ylabel("Correlation")

    fig.suptitle(
        "Comparison of VAR(1)-DCC-GARCH(1,1) and DCC-GARCH(1,1)", fontsize=18
    )
    plt.tight_layout()


def plot_all(
    rolling_corr: pd.DataFrame, dcc_result: DCCResult, var_dcc_result: DCCResult
) -> None:
    """Convenience wrapper that generates the three core figures."""
    plot_rolling_correlations(rolling_corr)
    plot_dcc_vs_rolling(dcc_result, rolling_corr)
    plot_var_vs_dcc(dcc_result, var_dcc_result)


if __name__ == "__main__":
    print("Import the plotting helpers and provide the appropriate results objects.")
