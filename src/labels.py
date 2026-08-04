"""Labeling: triple-barrier and forward-return targets."""

from __future__ import annotations

import numpy as np
import polars as pl
from numba import njit


@njit
def _triple_barrier_nb(
    mid: np.ndarray, sigma: np.ndarray, horizon: int, k: float
) -> np.ndarray:
    """For each t, scan mid[t+1 : t+horizon] for first barrier touch."""
    n = mid.shape[0]
    out = np.zeros(n, dtype=np.int8)
    # Need a full horizon of future bars; last `horizon` rows stay 0.
    for t in range(n - horizon):
        m = mid[t]
        s = sigma[t]
        if np.isnan(m) or np.isnan(s):
            continue
        upper = m + k * s * m
        lower = m - k * s * m
        for j in range(t + 1, t + horizon + 1):
            mj = mid[j]
            if np.isnan(mj):
                continue
            if mj >= upper:
                out[t] = 1
                break
            if mj <= lower:
                out[t] = -1
                break
    return out


def triple_barrier(
    mid: pl.Series, sigma: pl.Series, horizon: int, k: float
) -> pl.Series:
    """Return integer labels in {-1, 0, +1}, same length as mid."""
    if len(mid) != len(sigma):
        raise ValueError("mid and sigma must have the same length")
    labels = _triple_barrier_nb(
        mid.to_numpy().astype(np.float64),
        sigma.to_numpy().astype(np.float64),
        int(horizon),
        float(k),
    )
    return pl.Series("label_tb", labels)


def forward_return_sign(
    mid: pl.Series, horizon: int, threshold_bps: float
) -> pl.Series:
    """+1/-1/0 from H-bar forward log return vs ±threshold_bps."""
    thr = float(threshold_bps) / 1e4  # 1 bp = 1e-4
    m = mid.to_numpy().astype(np.float64)
    n = m.shape[0]
    out = np.zeros(n, dtype=np.int8)
    h = int(horizon)
    for t in range(n - h):
        m0 = m[t]
        m1 = m[t + h]
        if np.isnan(m0) or np.isnan(m1) or m0 <= 0.0:
            continue
        r = np.log(m1 / m0)
        if r > thr:
            out[t] = 1
        elif r < -thr:
            out[t] = -1
    return pl.Series("label_fr", out)


from src.config import LABEL_HORIZON_BARS, LABEL_K_SIGMA


def build_labels(df: pl.DataFrame) -> pl.DataFrame:
    """Append label_tb (triple-barrier) and label_fr (forward-return)."""
    mid = df["mid"]
    sigma = df["vol_50"]
    label_tb = triple_barrier(mid, sigma, LABEL_HORIZON_BARS, LABEL_K_SIGMA)
    label_fr = forward_return_sign(mid, LABEL_HORIZON_BARS, threshold_bps=1.0)
    return df.with_columns(label_tb, label_fr)