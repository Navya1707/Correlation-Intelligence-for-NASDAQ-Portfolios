"""
Model selection for DCC variants via RMSE against rolling correlations.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
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


@dataclass
class ModelSelectionResult:
    rmse_table: pd.DataFrame
    best_model: str


def _rmse(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    diff = a - b
    return np.sqrt(np.nanmean(diff * diff, axis=0))


def select_best_model(
    results: Dict[str, DCCResult], rolling_corr: pd.DataFrame
) -> ModelSelectionResult:
    """
    Compare DCC specifications using the RMSE metric.
    """
    metrics = {}
    for name, result in results.items():
        aligned_model, aligned_roll = result.correlations.align(rolling_corr, join="inner")
        if aligned_model.empty:
            raise ValueError("Rolling correlations and DCC outputs do not overlap.")
        rmse_vector = _rmse(aligned_model.to_numpy(), aligned_roll.to_numpy())
        metrics[name] = rmse_vector.sum()

    rmse_table = pd.DataFrame.from_dict(metrics, orient="index", columns=["RMSE"])
    rmse_table.sort_values("RMSE", inplace=True)
    return ModelSelectionResult(rmse_table=rmse_table, best_model=rmse_table.index[0])


if __name__ == "__main__":
    print("Use select_best_model(results, rolling_corr) after estimating the models.")
