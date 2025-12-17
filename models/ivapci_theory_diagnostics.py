"""Theory/representation diagnostics for IVAPCI-style estimators.

This module exists mainly to support benchmark harnesses that try to import
`models.ivapci_theory_diagnostics` to compute post-fit diagnostics.

The functions are written to be *robust* to estimator implementations:
- If the estimator already exposes `training_diagnostics` (dict), we reuse it.
- If the estimator has `_post_fit_quality_diagnostics(...)`, we can invoke it.
- Otherwise we fall back to simple correlation-based checks.

All outputs are returned as a dict of Python scalars (float/int/str).
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple
import math

import numpy as np


def _as_1d(x: Any) -> np.ndarray:
    arr = np.asarray(x)
    if arr.ndim == 2 and arr.shape[1] == 1:
        arr = arr[:, 0]
    return arr.reshape(-1)


def _safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        if isinstance(x, (float, int, np.floating, np.integer)):
            if isinstance(x, bool):
                return float(int(x))
            return float(x)
        if isinstance(x, np.ndarray) and x.size == 1:
            return float(x.reshape(-1)[0])
        return float(x)
    except Exception:
        return None


def _safe_corr(a: np.ndarray, b: np.ndarray) -> Optional[float]:
    try:
        a = _as_1d(a)
        b = _as_1d(b)
        if a.size != b.size or a.size < 3:
            return None
        sa = np.std(a)
        sb = np.std(b)
        if sa <= 0 or sb <= 0:
            return 0.0
        c = float(np.corrcoef(a, b)[0, 1])
        if math.isnan(c):
            return 0.0
        return c
    except Exception:
        return None


def _regress_residual(y: np.ndarray, X: np.ndarray) -> np.ndarray:
    # Simple OLS residual: y - X (X^+ y)
    y = _as_1d(y)
    X = np.asarray(X)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    X = np.column_stack([X, np.ones(X.shape[0])])  # intercept
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    yhat = X @ coef
    return y - yhat


def compute_ivapci_theory_diagnostics(
    estimator: Any,
    V_all: np.ndarray,
    A: np.ndarray,
    Y: np.ndarray,
    *,
    max_rows: int = 5000,
    random_state: int = 0,
) -> Dict[str, Any]:
    """Compute a standardized set of diagnostics (best-effort).

    Returns a dict that can be merged into a benchmark diagnostics row.
    """
    out: Dict[str, Any] = {}

    # Reuse existing estimator diagnostics if present.
    td = getattr(estimator, "training_diagnostics", None)
    if isinstance(td, dict):
        for k, v in td.items():
            if isinstance(v, (str, bool)):
                out[k] = v
            else:
                fv = _safe_float(v)
                if fv is not None:
                    out[k] = fv

    # Basic data diagnostics
    try:
        V = np.asarray(V_all)
        if V.ndim == 2 and V.shape[0] > 0:
            stds = np.std(V, axis=0)
            out.setdefault("min_std", float(np.min(stds)))
            out.setdefault("num_low_std", int(np.sum(stds < 1e-3)))
    except Exception:
        pass

    # Try to run the estimator's own post-fit checks if available.
    try:
        if hasattr(estimator, "_post_fit_quality_diagnostics"):
            # NOTE: this method is expected to update estimator.training_diagnostics.
            estimator._post_fit_quality_diagnostics(V_all=V_all, A=A, Y=Y)
            td2 = getattr(estimator, "training_diagnostics", None)
            if isinstance(td2, dict):
                for k, v in td2.items():
                    if isinstance(v, (str, bool)):
                        out[k] = v
                    else:
                        fv = _safe_float(v)
                        if fv is not None:
                            out[k] = fv
    except Exception as e:
        out["post_fit_diag_error"] = f"{type(e).__name__}: {e}"

    # If key IV diagnostics still missing, compute lightweight versions.
    if "iv_relevance_abs_corr" not in out or "iv_exclusion_abs_corr_resid" not in out:
        try:
            V = np.asarray(V_all)
            if V.ndim == 2 and V.shape[0] > 0:
                # Heuristic: assume Z is the last block, with dimensionality if estimator exposes it.
                z_dim = None
                for attr in ("z_dim", "Z_dim", "dim_z"):
                    if hasattr(estimator, "cfg") and hasattr(estimator.cfg, attr):
                        z_dim = int(getattr(estimator.cfg, attr))
                        break
                    if hasattr(estimator, attr):
                        try:
                            z_dim = int(getattr(estimator, attr))
                            break
                        except Exception:
                            pass
                if z_dim is None:
                    z_dim = min(4, V.shape[1])  # fallback
                Z = V[:, -z_dim:]
                z_mean = np.mean(Z, axis=1)
                a1 = _as_1d(A)
                y1 = _as_1d(Y)
                out.setdefault("iv_relevance_abs_corr", float(abs(_safe_corr(z_mean, a1) or 0.0)))
                # exclusion: corr(Z, residual of Y after regressing on A and V (excluding Z))
                X = np.column_stack([a1, V[:, : max(0, V.shape[1] - z_dim)]])
                y_res = _regress_residual(y1, X)
                out.setdefault("iv_exclusion_abs_corr_resid", float(abs(_safe_corr(z_mean, y_res) or 0.0)))
        except Exception:
            pass

    return out


# Common aliases used by different harnesses
compute_theory_diagnostics = compute_ivapci_theory_diagnostics
compute_diagnostics = compute_ivapci_theory_diagnostics
run = compute_ivapci_theory_diagnostics
