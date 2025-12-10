"""Loaders for external benchmark datasets (IHDP, Criteo uplift)."""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd


def _select_first(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
    for name in candidates:
        if name in df.columns:
            return name
    return None


def _infer_feature_columns(df: pd.DataFrame, excluded: List[str]) -> List[str]:
    excluded_set = set(excluded)
    return [c for c in df.columns if c not in excluded_set]


def load_ihdp_replicate(path: str) -> Dict[str, np.ndarray]:
    """
    Load a single IHDP-style simulated replicate CSV.

    The loader is lenient to column naming conventions that appear in
    common IHDP releases. It looks for treatment labels such as ``A`` or
    ``treatment``, potential outcome columns (``y0``, ``y1``), and observed
    outcomes (``y`` or ``yf``). Remaining columns are treated as covariates.
    """

    df = pd.read_csv(path)

    a_col = _select_first(df, ["A", "a", "treatment", "treat"])
    if a_col is None:
        raise ValueError("Treatment column not found in IHDP file.")

    y_obs_col = _select_first(df, ["Y", "y", "yf", "outcome"])
    y0_col = _select_first(df, ["Y0", "y0"])
    y1_col = _select_first(df, ["Y1", "y1"])

    exclude = [a_col]
    if y_obs_col:
        exclude.append(y_obs_col)
    if y0_col:
        exclude.append(y0_col)
    if y1_col:
        exclude.append(y1_col)

    feature_cols = _infer_feature_columns(df, exclude)
    if not feature_cols:
        raise ValueError("No feature columns detected in IHDP file.")

    X = df[feature_cols].to_numpy()
    A = df[a_col].to_numpy().astype(int)
    Y = df[y_obs_col].to_numpy() if y_obs_col else (A * df[y1_col] + (1 - A) * df[y0_col]).to_numpy()

    results: Dict[str, np.ndarray] = {"X": X, "A": A, "Y": Y}
    if y0_col and y1_col:
        Y0 = df[y0_col].to_numpy()
        Y1 = df[y1_col].to_numpy()
        results.update({"Y0": Y0, "Y1": Y1, "tau_true": float(np.mean(Y1 - Y0))})

    return results


def load_criteo_uplift(path: str) -> Dict[str, np.ndarray]:
    """
    Load Criteo uplift-style datasets.

    Expected columns (case-insensitive):
    - treatment indicator: ``treatment`` / ``A`` / ``a``
    - binary outcome: ``conversion`` / ``y`` / ``label`` / ``outcome``
    Remaining columns are treated as covariates ``X``.
    """

    df = pd.read_csv(path)

    a_col = _select_first(df, ["treatment", "A", "a", "exposure"])
    if a_col is None:
        raise ValueError("Treatment column not found in Criteo file.")

    y_col = _select_first(df, ["conversion", "y", "label", "outcome", "target"])
    if y_col is None:
        raise ValueError("Outcome column not found in Criteo file.")

    feature_cols = _infer_feature_columns(df, [a_col, y_col])
    if not feature_cols:
        raise ValueError("No feature columns detected in Criteo file.")

    X = df[feature_cols].to_numpy()
    A = df[a_col].to_numpy().astype(int)
    Y = df[y_col].to_numpy()

    return {"X": X, "A": A, "Y": Y}


__all__ = ["load_ihdp_replicate", "load_criteo_uplift"]
