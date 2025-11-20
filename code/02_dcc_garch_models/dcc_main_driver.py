"""
Main driver for the DCC/VAR pipeline.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import pandas as pd

try:
    from .dcc_compare_with_var import VARComparisonResult, compare_with_var
    from .dcc_compute_rolling import compute_rolling_correlations
    from .dcc_estimate_all_models import estimate_all_models
    from .dcc_generate_plots import plot_all
    from .dcc_model_selection import ModelSelectionResult, select_best_model
    from .dcc_utils import DCCResult
except ImportError:  # pragma: no cover
    import sys
    from pathlib import Path

    CURRENT_DIR = Path(__file__).resolve().parent
    if str(CURRENT_DIR) not in sys.path:
        sys.path.append(str(CURRENT_DIR))
    from dcc_compare_with_var import VARComparisonResult, compare_with_var
    from dcc_compute_rolling import compute_rolling_correlations
    from dcc_estimate_all_models import estimate_all_models
    from dcc_generate_plots import plot_all
    from dcc_model_selection import ModelSelectionResult, select_best_model
    from dcc_utils import DCCResult


@dataclass
class DCCPipelineResult:
    rolling_correlations: pd.DataFrame
    model_results: Dict[str, DCCResult]
    selection: ModelSelectionResult
    var_comparison: VARComparisonResult


def run_pipeline(
    returns: pd.DataFrame, window: int = 90, max_lags: int = 10, plot: bool = False
) -> DCCPipelineResult:
    """
    Execute the full DCC pipeline (rolling correlations → model estimation → VAR comparison).
    """
    rolling_corr = compute_rolling_correlations(returns, window=window)
    model_results = estimate_all_models(returns)
    selection = select_best_model(model_results, rolling_corr)
    best_model = model_results[selection.best_model]
    var_comparison = compare_with_var(returns, best_model, max_lags=max_lags)

    if plot:
        plot_all(rolling_corr, best_model, var_comparison.dcc_result)

    return DCCPipelineResult(
        rolling_correlations=rolling_corr,
        model_results=model_results,
        selection=selection,
        var_comparison=var_comparison,
    )


if __name__ == "__main__":
    print("Provide a returns DataFrame to run_pipeline(...) to estimate the models.")
