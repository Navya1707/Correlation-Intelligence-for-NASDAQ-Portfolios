"""
Macro-financial index construction workflow.

The :class:`MacroIndexAssembler` orchestrates data ingestion, feature creation,
volatility estimation, and weighting of macro variables into a single index.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional

import numpy as np
import pandas as pd
from arch import arch_model
from matplotlib import pyplot as plt


ALIAS_BOOK: Dict[str, Iterable[str]] = {
    "date": ("date", "fulldates", "datetime"),
    "gold": ("gold", "gc=F"),
    "wti": ("wti", "cl=F", "oil"),
    "vix": ("vix", "^vix"),
    "fvx": ("fvx", "^fvx", "tnx"),
    "libor": ("libor", "ibor"),
    "ois": ("ois", "sofr"),
    "cpi": ("cpi", "cpi_us"),
    "cpi_date": ("cpi_date", "date_cpi", "date_m"),
}


@dataclass
class MacroIndexArtifacts:
    """Outputs produced by :class:`MacroIndexAssembler`."""

    returns: pd.Series
    weights: pd.Series
    vol_surface: pd.DataFrame
    garch_params: pd.DataFrame


class MacroIndexAssembler:
    """
    Builds the macro index from an Excel workbook.

    Example
    -------
    >>> builder = MacroIndexAssembler("Macro_Variables.xlsx")
    >>> artifacts = builder.build(plot=True)
    >>> print(artifacts.weights)
    """

    def __init__(
        self,
        workbook_path: Path | str,
        sheet_name: str = "Data",
        column_map: Optional[Mapping[str, str]] = None,
    ) -> None:
        self.workbook_path = Path(workbook_path)
        self.sheet_name = sheet_name
        self.column_map: Dict[str, str] = dict(column_map or {})
        self._data = self._load_workbook()
        self._columns = self._resolve_columns()

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def build(self, plot: bool = False) -> MacroIndexArtifacts:
        """
        Execute the end-to-end workflow and optionally visualize the index.
        """
        macro_inputs = self._prepare_macro_features()
        vol_surface, params = self._estimate_volatility(macro_inputs)
        weights = self._derive_weights(vol_surface)
        index_returns = self._compose_index(macro_inputs, weights)

        if plot:
            self._plot_index(index_returns)

        return MacroIndexArtifacts(
            returns=index_returns,
            weights=weights,
            vol_surface=vol_surface,
            garch_params=params,
        )

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _load_workbook(self) -> pd.DataFrame:
        if not self.workbook_path.exists():
            raise FileNotFoundError(
                f"Workbook {self.workbook_path} not found. "
                "Please supply the correct path."
            )
        return pd.read_excel(self.workbook_path, sheet_name=self.sheet_name)

    def _resolve_columns(self) -> Dict[str, str]:
        columns: Dict[str, str] = {}
        for key, aliases in ALIAS_BOOK.items():
            if key in self.column_map:
                columns[key] = self.column_map[key]
                continue
            match = next(
                (col for col in self._data.columns if col.lower() in [a.lower() for a in aliases]),
                None,
            )
            if not match:
                raise KeyError(
                    f"Column for '{key}' not found. Provide a custom mapping via column_map."
                )
            columns[key] = match
        return columns

    def _prepare_macro_features(self) -> pd.DataFrame:
        """Return a clean DataFrame of macro variables expressed as returns/differences."""
        dates = pd.to_datetime(self._data[self._columns["date"]])

        def series_from(col_key: str) -> pd.Series:
            return pd.Series(self._data[self._columns[col_key]].values, index=dates).astype(float)

        gold = series_from("gold")
        wti = series_from("wti")
        vix = series_from("vix")
        fvx = series_from("fvx")
        libor = series_from("libor")
        ois = series_from("ois")

        gold_ret = np.log(gold).diff()
        wti_ret = np.log(wti).diff()
        vix_ret = np.log(vix).diff()
        fvx_diff = fvx.diff()
        lois = (libor - ois).diff()

        cpi = self._interpolate_cpi(dates)
        cpi_ret = pd.Series(np.diff(np.log(cpi)) * 100, index=dates[1:])

        macro = (
            pd.concat(
                [
                    gold_ret.rename("GOLD"),
                    wti_ret.rename("WTI"),
                    vix_ret.rename("VIX"),
                    fvx_diff.rename("FVX"),
                    cpi_ret.rename("CPI"),
                    lois.rename("LOIS"),
                ],
                axis=1,
            )
            .dropna()
            .astype(float)
        )
        return macro

    def _interpolate_cpi(self, daily_index: pd.DatetimeIndex) -> np.ndarray:
        slice_df = self._data[[self._columns["cpi_date"], self._columns["cpi"]]].dropna()
        monthly_dates = pd.to_datetime(slice_df[self._columns["cpi_date"]].values)
        monthly_values = slice_df[self._columns["cpi"]].astype(float).to_numpy()
        daily_epoch = daily_index.view("int64").astype(float)
        monthly_epoch = monthly_dates.view("int64").astype(float)
        return np.interp(daily_epoch, monthly_epoch, monthly_values)

    def _estimate_volatility(
        self, macro_inputs: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        vol_map: Dict[str, pd.Series] = {}
        params = []
        for name, series in macro_inputs.items():
            model = arch_model(
                series.dropna(), mean="Zero", vol="GARCH", p=1, q=1, dist="normal"
            )
            fit = model.fit(disp="off")
            vol_map[name] = fit.conditional_volatility.reindex(macro_inputs.index)
            params.append(
                {
                    "Variable": name,
                    "Omega": fit.params.get("omega", np.nan),
                    "Alpha": fit.params.get("alpha[1]", np.nan),
                    "Beta": fit.params.get("beta[1]", np.nan),
                }
            )
        return pd.DataFrame(vol_map), pd.DataFrame(params).set_index("Variable")

    def _derive_weights(self, vol_surface: pd.DataFrame) -> pd.Series:
        inv_vol = 1.0 / vol_surface.mean()
        return inv_vol / inv_vol.sum()

    def _compose_index(
        self, macro_inputs: pd.DataFrame, weights: pd.Series
    ) -> pd.Series:
        return (macro_inputs * weights).sum(axis=1)

    def _plot_index(self, index_returns: pd.Series) -> None:
        fig, ax = plt.subplots(figsize=(12, 4))
        index_returns.cumsum().plot(ax=ax, color="tab:blue", linewidth=1.6)
        ax.set_title("Macro-Financial Index (Cumulative Return)", fontsize=16)
        ax.set_ylabel("Cumulative log return")
        ax.grid(alpha=0.3)
        plt.tight_layout()


def build_macro_index(
    workbook_path: Path | str,
    sheet_name: str = "Data",
    column_map: Optional[Mapping[str, str]] = None,
    plot: bool = False,
) -> MacroIndexArtifacts:
    """
    Functional wrapper around :class:`MacroIndexAssembler` for backward compatibility.
    """
    assembler = MacroIndexAssembler(
        workbook_path=workbook_path, sheet_name=sheet_name, column_map=column_map
    )
    return assembler.build(plot=plot)


if __name__ == "__main__":
    default_file = Path("Macro_Variables.xlsx")
    if default_file.exists():
        outcome = build_macro_index(default_file, plot=True)
        print("Macro index constructed. Normalized weights:")
        print(outcome.weights)
    else:
        print(
            "Macro_Variables.xlsx not found. Provide the workbook path when calling build_macro_index()."
        )
