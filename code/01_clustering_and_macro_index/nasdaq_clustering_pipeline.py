# =============================================================================
# NASDAQ-100 Clustering Pipeline (Python refactor)
# =============================================================================
"""
End-to-end machine-learning workflow for grouping NASDAQ-100 constituents.

Key features:
    * Modular function layout with explicit inputs/outputs.
    * Renamed intermediate variables for clarity (e.g., ``feature_table``,
      ``cluster_frame``) plus dataclasses for configuration/state.
    * Visualization helpers are separated from the data wrangling logic.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from math import sqrt
from typing import Iterable, List, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import yfinance as yf
from pandas_datareader import data as pdr
from sklearn.cluster import KMeans

px.default = "svg"
yf.pdr_override()


# =============================================================================
# Configuration containers
# =============================================================================


@dataclass(frozen=True)
class UniverseConfig:
    source_url: str = "https://en.wikipedia.org/wiki/Nasdaq-100"
    table_index: int = 4


@dataclass(frozen=True)
class DownloadConfig:
    start: datetime = datetime(2019, 1, 1)
    end: datetime = datetime(2024, 1, 1)


@dataclass
class ClusteringOutcome:
    features: pd.DataFrame
    labels: pd.Series
    centroids: np.ndarray
    inertia_series: pd.Series


# =============================================================================
# Data acquisition helpers
# =============================================================================


def fetch_nasdaq_members(config: UniverseConfig) -> List[str]:
    tables = pd.read_html(config.source_url)
    symbols = tables[config.table_index]["Symbol"].tolist()
    cleaned = [symbol.replace("\n", "").replace(".", "-").replace(" ", "") for symbol in symbols]
    return cleaned


def download_adjusted_prices(
    tickers: Sequence[str], config: DownloadConfig
) -> pd.DataFrame:
    frames = []
    for ticker in tickers:
        try:
            data = pdr.get_data_yahoo(ticker, start=config.start, end=config.end)["Adj Close"]
            frames.append(pd.DataFrame({ticker: data}))
        except Exception:
            continue
    prices = pd.concat(frames, axis=1)
    prices.sort_index(inplace=True)
    return prices


# =============================================================================
# Feature engineering + clustering
# =============================================================================


def build_feature_matrix(price_history: pd.DataFrame) -> pd.DataFrame:
    pct_changes = price_history.pct_change()
    feature_table = pd.DataFrame(
        {
            "Returns": pct_changes.mean() * 252,
            "Volatility": pct_changes.std() * sqrt(252),
        }
    ).dropna()
    return feature_table


def compute_elbow_curve(matrix: np.ndarray, k_values: Iterable[int]) -> pd.Series:
    inertia = {}
    for k in k_values:
        model = KMeans(n_clusters=k, n_init="auto", random_state=42)
        model.fit(matrix)
        inertia[k] = model.inertia_
    return pd.Series(inertia)


def cluster_universe(feature_table: pd.DataFrame, n_clusters: int = 4) -> ClusteringOutcome:
    matrix = feature_table.to_numpy()
    inertia_series = compute_elbow_curve(matrix, range(2, 20))
    kmeans = KMeans(n_clusters=n_clusters, n_init="auto", random_state=42)
    labels = pd.Series(kmeans.fit_predict(matrix), index=feature_table.index, name="Cluster")
    return ClusteringOutcome(
        features=feature_table,
        labels=labels,
        centroids=kmeans.cluster_centers_,
        inertia_series=inertia_series,
    )


def describe_clusters(outcome: ClusteringOutcome) -> pd.DataFrame:
    cluster_frame = outcome.features.join(outcome.labels)
    semantics = {
        0: "balanced profile",
        1: "high return / high risk",
        2: "lower risk",
        3: "volatility leaders",
    }
    cluster_frame["Cluster_Name"] = cluster_frame["Cluster"].map(semantics)
    return cluster_frame


def nearest_members(outcome: ClusteringOutcome) -> pd.DataFrame:
    matrix = outcome.features.to_numpy()
    centroids = outcome.centroids
    distances = np.linalg.norm(matrix[:, None, :] - centroids[None, :, :], axis=2)
    closest_indices = np.argmin(distances, axis=0)
    representatives = outcome.features.index[closest_indices]
    return pd.DataFrame({"Cluster": range(len(centroids)), "Representative": representatives})


# =============================================================================
# Visualization helpers
# =============================================================================


def plot_elbow(inertia_series: pd.Series, optimal_k: int) -> None:
    plt.figure(figsize=(10, 5))
    plt.plot(inertia_series.index, inertia_series.values, marker="o")
    plt.axvline(optimal_k, color="grey", linestyle="--", linewidth=1.5)
    plt.scatter(optimal_k, inertia_series.loc[optimal_k], color="red", s=80)
    plt.title("Elbow Method", fontsize=18, weight="bold")
    plt.xlabel("Number of Clusters (K)")
    plt.ylabel("Inertia")
    plt.grid(alpha=0.4, linestyle="--")
    plt.tight_layout()
    plt.show()


def plot_return_volatility(cluster_frame: pd.DataFrame) -> None:
    plt.figure(figsize=(14, 7))
    scatter = plt.scatter(
        cluster_frame["Returns"],
        cluster_frame["Volatility"],
        c=cluster_frame["Cluster"],
        cmap="viridis",
        s=120,
        alpha=0.6,
    )
    plt.title("Return vs Volatility", fontsize=20, weight="bold")
    plt.xlabel("Annualized Return")
    plt.ylabel("Annualized Volatility")
    plt.grid(alpha=0.5, linestyle="--")
    cbar = plt.colorbar(scatter)
    cbar.set_label("Cluster")

    highlight = ["PYPL", "CRWD", "INTU", "EA"]
    for ticker in highlight:
        if ticker not in cluster_frame.index:
            continue
        row = cluster_frame.loc[ticker]
        plt.annotate(
            ticker,
            (row["Returns"], row["Volatility"]),
            textcoords="offset points",
            xytext=(0, 20),
            ha="center",
            fontsize=12,
            bbox=dict(boxstyle="round,pad=0.3", alpha=0.7, facecolor="white"),
            arrowprops=dict(arrowstyle="-", color="black"),
        )
    plt.tight_layout()
    plt.show()


# =============================================================================
# Main entry point
# =============================================================================


def main() -> None:
    universe = fetch_nasdaq_members(UniverseConfig())
    prices = download_adjusted_prices(universe, DownloadConfig())
    features = build_feature_matrix(prices)
    outcome = cluster_universe(features, n_clusters=4)
    enriched = describe_clusters(outcome)

    print("Cluster representatives:")
    print(nearest_members(outcome))

    plot_elbow(outcome.inertia_series, optimal_k=4)
    plot_return_volatility(enriched)


if __name__ == "__main__":
    main()
