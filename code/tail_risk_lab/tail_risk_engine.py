"""
Dynamic Value-at-Risk / Expected Shortfall engine.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.integrate import quad
from scipy.stats import norm

from .portfolio_allocator import PortfolioSnapshot


@dataclass
class TailRiskSeries:
    var_95: pd.Series
    var_99: pd.Series
    es_95: pd.Series
    es_99: pd.Series
    summary: pd.Series


class TailRiskEngine:
    """
    Consumes portfolio statistics plus VAR-DCC covariance matrices and produces VaR/ES paths.
    """

    def __init__(self, alpha_low: float = 0.05, alpha_high: float = 0.01) -> None:
        self.alpha_low = alpha_low
        self.alpha_high = alpha_high

    def evaluate(self, portfolio: PortfolioSnapshot, dcc_payload: Any) -> TailRiskSeries:
        aligned_returns = portfolio.returns.loc[dcc_payload.index]
        covariances = dcc_payload.covariances
        weights = portfolio.weights
        capital = portfolio.capital

        var_95, var_99, es_95, es_99 = [], [], [], []
        for cov_matrix, mu in zip(covariances, aligned_returns.to_numpy()):
            sigma = float(np.sqrt(weights @ cov_matrix @ weights))
            var_95.append(self._var(mu, sigma, self.alpha_low) * capital)
            var_99.append(self._var(mu, sigma, self.alpha_high) * capital)
            es_95.append(self._es(mu, sigma, self.alpha_low) * capital)
            es_99.append(self._es(mu, sigma, self.alpha_high) * capital)

        series_95 = pd.Series(var_95, index=aligned_returns.index, name="VaR_95")
        series_99 = pd.Series(var_99, index=aligned_returns.index, name="VaR_99")
        es_series_95 = pd.Series(es_95, index=aligned_returns.index, name="ES_95")
        es_series_99 = pd.Series(es_99, index=aligned_returns.index, name="ES_99")

        summary = pd.Series(
            {
                "VaR_95_mean": series_95.mean(),
                "VaR_99_mean": series_99.mean(),
                "ES_95_mean": es_series_95.mean(),
                "ES_99_mean": es_series_99.mean(),
            }
        )

        return TailRiskSeries(
            var_95=series_95,
            var_99=series_99,
            es_95=es_series_95,
            es_99=es_series_99,
            summary=summary,
        )

    @staticmethod
    def _var(mu: float, sigma: float, alpha: float) -> float:
        return -(mu + sigma * norm.ppf(1 - alpha))

    @staticmethod
    def _es(mu: float, sigma: float, alpha: float) -> float:
        integrand = lambda q: TailRiskEngine._var(mu, sigma, q)
        value, _ = quad(integrand, 0, alpha)
        return value / alpha


def compute_dynamic_var_es(
    portfolio: PortfolioSnapshot, dcc_result: Any, alpha_5: float = 0.05, alpha_1: float = 0.01
) -> TailRiskSeries:
    """
    Compatibility wrapper.
    """
    engine = TailRiskEngine(alpha_low=alpha_5, alpha_high=alpha_1)
    return engine.evaluate(portfolio, dcc_result)


if __name__ == "__main__":
    print("Instantiate TailRiskEngine(alpha_low, alpha_high) and call evaluate(portfolio, dcc).")
