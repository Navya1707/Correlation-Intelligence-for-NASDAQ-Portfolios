"""
Identification of VaR and ES breaches for the portfolio tail-risk analysis.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd
from matplotlib import pyplot as plt

from .portfolio_allocator import PortfolioSnapshot
from .tail_risk_engine import TailRiskSeries


@dataclass
class BreachLog:
    violations_95: pd.Series
    violations_99: pd.Series
    pnl: pd.Series


def identify_var_breaches(portfolio: PortfolioSnapshot, risk: TailRiskSeries) -> BreachLog:
    """
    Flag VaR violations at 95% and 99% significance levels.
    """
    common_index = risk.var_95.index.intersection(portfolio.returns.index)
    pnl = portfolio.returns.loc[common_index] * portfolio.capital

    var95 = risk.var_95.loc[common_index]
    var99 = risk.var_99.loc[common_index]

    violations_95 = (pnl < var95) & (pnl < 0)
    violations_99 = (pnl < var99) & (pnl < 0)

    return BreachLog(
        violations_95=violations_95,
        violations_99=violations_99,
        pnl=pnl,
    )


def plot_var_breaches(result: BreachLog) -> None:
    """Scatter plot highlighting VaR breaches."""
    dates = result.pnl.index
    plt.figure(figsize=(12, 4))
    plt.plot(dates, result.pnl, color="0.5", linewidth=2, label="P&L")
    plt.scatter(
        dates[result.violations_95],
        result.pnl[result.violations_95],
        color="royalblue",
        label="VaR 95% breach",
    )
    plt.scatter(
        dates[result.violations_99],
        result.pnl[result.violations_99],
        color="crimson",
        label="VaR 99% breach",
    )
    plt.title("Violations Dynamic VaR combined with Σ_DCC/GARCH of VAR-DCC-GARCH model")
    plt.ylabel("USD")
    plt.legend()
    plt.tight_layout()


def plot_es_histogram(
    portfolio: PortfolioSnapshot,
    risk: TailRiskSeries,
    comparison_returns: Optional[pd.Series] = None,
) -> None:
    """
    Plot the P&L distribution and highlight ES thresholds. Optionally overlay a comparison series
    (e.g., an alternative sample definition).
    """
    pnl = portfolio.returns * portfolio.capital
    plt.figure(figsize=(10, 5))
    plt.hist(pnl, bins=75, alpha=0.6, label="Baseline", color="0.6")

    es95 = risk.summary["ES_95_mean"]
    es99 = risk.summary["ES_99_mean"]
    plt.axvline(es95, color="royalblue", linestyle="-.", linewidth=2, label=f"ES 95% = {es95:.2f}")
    plt.axvline(es99, color="crimson", linestyle="-.", linewidth=2, label=f"ES 99% = {es99:.2f}")

    if comparison_returns is not None:
        plt.hist(
            comparison_returns * portfolio.capital,
            bins=75,
            alpha=0.35,
            label="Comparison sample",
            color="0.2",
        )

    plt.title("P&L and Expected Shortfall Comparison", fontsize=16)
    plt.xlabel("USD")
    plt.ylabel("Counts")
    plt.legend()
    plt.tight_layout()


if __name__ == "__main__":
    print("Use identify_var_breaches + plot helpers once VaR/ES series are computed.")
