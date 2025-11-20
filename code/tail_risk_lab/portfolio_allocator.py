"""
Portfolio construction utilities rewritten in a modular style.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


@dataclass
class PortfolioSnapshot:
    curve: pd.Series
    returns: pd.Series
    pnl: pd.Series
    weights: np.ndarray
    capital: float


class EqualWeightAllocator:
    """Builds an equally weighted portfolio over time."""

    def __init__(self, initial_capital: float = 10_000.0) -> None:
        self.initial_capital = float(initial_capital)

    def allocate(self, asset_return_matrix: pd.DataFrame) -> PortfolioSnapshot:
        if asset_return_matrix.empty:
            raise ValueError("No returns supplied for allocation.")

        num_assets = asset_return_matrix.shape[1]
        weights = np.full(num_assets, 1.0 / num_assets)
        blended_returns = asset_return_matrix.mul(weights, axis=1).sum(axis=1)
        equity_curve = self.initial_capital * (1 + blended_returns).cumprod()
        pnl = equity_curve.diff().fillna(0.0)

        return PortfolioSnapshot(
            curve=equity_curve,
            returns=blended_returns,
            pnl=pnl,
            weights=weights,
            capital=self.initial_capital,
        )

    @staticmethod
    def plot(snapshot: PortfolioSnapshot) -> None:
        ax = snapshot.curve.plot(figsize=(10, 4), color="dimgray", linewidth=2)
        ax.set_title("Equal-Weight Portfolio Value", fontsize=16)
        ax.set_ylabel("USD")
        ax.grid(alpha=0.3)
        plt.tight_layout()


def build_equal_weight_portfolio(
    returns: pd.DataFrame, investment: float = 10_000.0
) -> PortfolioSnapshot:
    """
    Backward-compatible wrapper returning :class:`PortfolioSnapshot`.
    """
    allocator = EqualWeightAllocator(initial_capital=investment)
    return allocator.allocate(returns)


if __name__ == "__main__":
    print("Instantiate EqualWeightAllocator and call allocate(returns_df).")
