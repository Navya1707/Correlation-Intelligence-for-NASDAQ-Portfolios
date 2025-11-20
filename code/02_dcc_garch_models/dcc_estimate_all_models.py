"""
Estimate the suite of DCC-type models defined for the study.
"""
from __future__ import annotations

from typing import Dict

import pandas as pd

try:  # Allow running as a script without package context
    from .dcc_utils import DCCResult, fit_dcc_model
except ImportError:  # pragma: no cover - fallback for direct execution
    import sys
    from pathlib import Path

    CURRENT_DIR = Path(__file__).resolve().parent
    if str(CURRENT_DIR) not in sys.path:
        sys.path.append(str(CURRENT_DIR))
    from dcc_utils import DCCResult, fit_dcc_model


MODEL_SPECS = [
    ("DCC_GARCH", "garch", "dcc"),
    ("DCC_TARCH", "tarch", "dcc"),
    ("DCC_GJR", "gjr", "dcc"),
    ("ADCC_GARCH", "garch", "adcc"),
    ("ADCC_TARCH", "tarch", "adcc"),
    ("ADCC_GJR", "gjr", "adcc"),
]


def estimate_all_models(returns: pd.DataFrame) -> Dict[str, DCCResult]:
    """
    Estimate each model specification on the supplied returns DataFrame.
    """
    results: Dict[str, DCCResult] = {}
    for name, uni_model, variant in MODEL_SPECS:
        results[name] = fit_dcc_model(
            returns,
            univariate_model=uni_model,
            variant=variant,
            name=name,
        )
    return results


if __name__ == "__main__":
    print("Call estimate_all_models(returns_df) to estimate the DCC specifications.")
