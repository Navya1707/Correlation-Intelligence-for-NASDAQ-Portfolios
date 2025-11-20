"""
Shared utilities for estimating DCC/ADCC models in Python.

Combines univariate GARCH fits (via the ``arch`` package) with a maximum-likelihood
estimation of the Engle (2002) Dynamic Conditional Correlation specification.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from arch import arch_model
from numpy.linalg import LinAlgError
from scipy.optimize import minimize


EPS = 1e-8


@dataclass
class UnivariateFits:
    residuals: pd.DataFrame
    standardized_residuals: pd.DataFrame
    conditional_volatility: pd.DataFrame
    loglikelihood: float
    model: str


@dataclass
class DCCResult:
    name: str
    params: np.ndarray
    vcv: np.ndarray
    loglikelihood: float
    correlations: pd.DataFrame
    covariances: np.ndarray
    standardized_residuals: pd.DataFrame
    conditional_volatility: pd.DataFrame
    variant: str
    univariate_model: str
    index: pd.Index


def _build_arch_model(series: pd.Series, model_type: str):
    kwargs = {"mean": "Zero", "vol": "GARCH", "p": 1, "o": 0, "q": 1, "power": 2.0}
    if model_type.lower() == "tarch":
        kwargs.update({"o": 1, "power": 1.0})
    elif model_type.lower() == "gjr":
        kwargs.update({"o": 1})
    model = arch_model(series.dropna(), dist="normal", **kwargs)
    return model.fit(disp="off")


def fit_univariate_models(returns: pd.DataFrame, model_type: str = "garch") -> UnivariateFits:
    residuals: Dict[str, pd.Series] = {}
    std_resids: Dict[str, pd.Series] = {}
    cond_vols: Dict[str, pd.Series] = {}
    total_loglik = 0.0

    for column in returns.columns:
        fit = _build_arch_model(returns[column].astype(float), model_type)
        residuals[column] = fit.resid.reindex(returns.index)
        std_resids[column] = fit.std_resid.reindex(returns.index)
        cond_vols[column] = fit.conditional_volatility.reindex(returns.index)
        total_loglik += float(fit.loglikelihood)

    return UnivariateFits(
        residuals=pd.DataFrame(residuals),
        standardized_residuals=pd.DataFrame(std_resids),
        conditional_volatility=pd.DataFrame(cond_vols),
        loglikelihood=total_loglik,
        model=model_type,
    )


def _is_valid_parameters(params: np.ndarray, variant: str) -> bool:
    if np.any(params <= 0):
        return False
    if variant == "adcc":
        a, b, g = params
        return (a + b + 0.5 * g) < 1 - 1e-5
    a, b = params
    return (a + b) < 1 - 1e-5


def _dcc_loglik(eps: np.ndarray, params: np.ndarray, variant: str) -> Tuple[float, np.ndarray]:
    if variant == "adcc":
        a, b, g = params
    else:
        a, b = params
        g = 0.0

    if not _is_valid_parameters(params, variant):
        return -np.inf, np.empty(0)

    t_obs, n_assets = eps.shape
    q_bar = np.cov(eps.T)
    qt = q_bar.copy()
    loglik = 0.0
    r_collection = np.zeros((t_obs, n_assets, n_assets))

    for t in range(t_obs):
        e_prev = eps[t - 1] if t > 0 else eps[0]
        qt = (1 - a - b - g / 2.0) * q_bar + a * np.outer(e_prev, e_prev) + b * qt
        if variant == "adcc":
            eta_prev = np.minimum(e_prev, 0.0)
            qt += g * np.outer(eta_prev, eta_prev)

        qt = (qt + qt.T) / 2.0
        diag = np.sqrt(np.clip(np.diag(qt), 1e-12, None))
        d_inv = np.diag(1.0 / diag)
        rt = d_inv @ qt @ d_inv
        rt = (rt + rt.T) / 2.0
        det_r = np.linalg.det(rt)
        if not np.isfinite(det_r) or det_r <= 0:
            return -np.inf, np.empty(0)
        inv_r = np.linalg.inv(rt)
        e_t = eps[t]
        loglik += -0.5 * (np.log(det_r) + e_t @ inv_r @ e_t - e_t @ e_t)
        r_collection[t] = rt

    return loglik, r_collection


def _numerical_hessian(func, params: np.ndarray, step: float = 1e-5) -> np.ndarray:
    n_params = len(params)
    hessian = np.zeros((n_params, n_params))
    f0 = func(params)
    for i in range(n_params):
        e_i = np.zeros(n_params)
        e_i[i] = step
        f_plus = func(params + e_i)
        f_minus = func(params - e_i)
        hessian[i, i] = (f_plus - 2 * f0 + f_minus) / (step**2)
        for j in range(i + 1, n_params):
            e_j = np.zeros(n_params)
            e_j[j] = step
            f_pp = func(params + e_i + e_j)
            f_pm = func(params + e_i - e_j)
            f_mp = func(params - e_i + e_j)
            f_mm = func(params - e_i - e_j)
            value = (f_pp - f_pm - f_mp + f_mm) / (4 * step**2)
            hessian[i, j] = value
            hessian[j, i] = value
    return hessian


def fit_dcc_model(
    returns: pd.DataFrame,
    univariate_model: str = "garch",
    variant: str = "dcc",
    name: Optional[str] = None,
) -> DCCResult:
    """
    Estimate a (A)symmetric DCC specification for the provided returns.
    """
    uni = fit_univariate_models(returns, model_type=univariate_model)
    valid_mask = (
        ~uni.standardized_residuals.isna().any(axis=1)
        & ~uni.conditional_volatility.isna().any(axis=1)
    )
    std_resids = uni.standardized_residuals.loc[valid_mask]
    cond_vol = uni.conditional_volatility.loc[valid_mask]

    if std_resids.empty:
        raise ValueError("Not enough valid observations to estimate the DCC model.")

    eps = std_resids.to_numpy()

    def neg_loglik(theta: np.ndarray) -> float:
        loglik, _ = _dcc_loglik(eps, theta, variant)
        return np.inf if not np.isfinite(loglik) else -loglik

    if variant == "adcc":
        x0 = np.array([0.03, 0.90, 0.05])
        bounds = [(1e-4, 0.999), (1e-4, 0.999), (1e-4, 0.999)]
        constraints = [{"type": "ineq", "fun": lambda x: 0.999 - (x[0] + x[1] + 0.5 * x[2])}]
    else:
        x0 = np.array([0.03, 0.94])
        bounds = [(1e-4, 0.999), (1e-4, 0.999)]
        constraints = [{"type": "ineq", "fun": lambda x: 0.999 - (x[0] + x[1])}]

    opt = minimize(
        neg_loglik,
        x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 500, "ftol": 1e-9},
    )
    if not opt.success:
        raise RuntimeError(f"DCC optimization failed: {opt.message}")

    loglik_corr, r_collection = _dcc_loglik(eps, opt.x, variant)
    if not np.isfinite(loglik_corr):
        raise RuntimeError("DCC estimation returned a non-finite likelihood.")

    def objective(theta: np.ndarray) -> float:
        return neg_loglik(theta)

    try:
        hessian = _numerical_hessian(objective, opt.x)
        cov_params = np.linalg.inv(hessian + np.eye(hessian.shape[0]) * 1e-6)
    except (LinAlgError, ValueError):
        cov_params = np.eye(len(opt.x)) * np.nan

    vols = cond_vol.to_numpy()
    covariances = np.zeros_like(r_collection)
    for t in range(r_collection.shape[0]):
        diag = np.diag(vols[t])
        covariances[t] = diag @ r_collection[t] @ diag

    columns = list(returns.columns)
    macro_vs_assets = pd.DataFrame(
        r_collection[:, 0, 1:],
        index=std_resids.index,
        columns=columns[1:],
    )

    return DCCResult(
        name=name or f"{variant.upper()}_{univariate_model.upper()}",
        params=opt.x,
        vcv=cov_params,
        loglikelihood=float(uni.loglikelihood + loglik_corr),
        correlations=macro_vs_assets,
        covariances=covariances,
        standardized_residuals=std_resids,
        conditional_volatility=cond_vol,
        variant=variant,
        univariate_model=univariate_model,
        index=std_resids.index,
    )
