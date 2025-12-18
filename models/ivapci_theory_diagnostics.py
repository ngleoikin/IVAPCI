"""Theory-aligned diagnostics for IVAPCI encoders (Theorem 1/2/3 proxies).

This module provides best-effort, *model-agnostic* diagnostics that can be run
after an IVAPCI-style estimator is fitted.

Design goals
------------
- Robust to estimator implementations:
  * Accept either (_v_mean/_v_std) or (_x_mean/_x_std) preprocessing stats.
  * Work even if the estimator does not expose a block-splitting helper.
  * Avoid dtype/device mismatches when calling torch modules.

The estimator is expected to expose (best effort):
- get_latent(X_all: np.ndarray) -> np.ndarray
- enc_x / enc_w / enc_z / enc_n : torch.nn.Module
- config with x_dim, w_dim, z_dim (or recoverable from encoders)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import r2_score, roc_auc_score


@dataclass
class TheoremDiagnosticsConfig:
    """Configuration for theory diagnostics.

    Attributes
    ----------
    max_n_for_mi:
        Maximum number of samples used when estimating mutual information proxies.
    random_state:
        Seed for subsampling during MI estimation.
    min_group_n:
        Minimum group size when computing conditional MI by treatment group.
    """

    max_n_for_mi: int = 5000
    random_state: int = 42
    min_group_n: int = 30


class TheoremComplianceDiagnostics:
    """Best-effort diagnostics approximating three theory pillars.

    Usage
    -----
        diag = TheoremComplianceDiagnostics(fitted_estimator)
        out = diag.run_all_diagnostics(X_all, A, Y)

    Notes
    -----
    - These are *proxies*, not exact mutual informations.
    - For stability, torch inputs are cast to the encoder parameter dtype.
    """

    def __init__(self, estimator, config: Optional[TheoremDiagnosticsConfig] = None) -> None:
        if not getattr(estimator, "_is_fit", False):
            raise RuntimeError("Estimator must be fit before running diagnostics.")
        self.est = estimator
        self.cfg = config or TheoremDiagnosticsConfig()
        self.device = getattr(estimator, "device", torch.device("cpu"))

    # ---------- public entry ----------

    def run_all_diagnostics(
        self,
        X_all: np.ndarray,
        A: np.ndarray,
        Y: np.ndarray,
        n_recon_features: int = 5,
    ) -> Dict[str, Any]:
        """Compute proxies for the three theorem-inspired diagnostics."""
        X_all = np.asarray(X_all, dtype=np.float64)
        A = np.asarray(A).reshape(-1)
        Y = np.asarray(Y).reshape(-1)

        # causal latent (concatenated blocks)
        U_c = self.est.get_latent(X_all)

        # standardized proxy blocks (safe + dtype compatible)
        X_std = self._standardize_inputs_safe(X_all)
        tx, tw, tz, tn = self._encode_blocks(X_std)

        diagnostics: Dict[str, Any] = {}
        diagnostics.update(self.theorem1_diagnosis(X_all, U_c, n_features=n_recon_features))
        diagnostics.update(self.theorem2_diagnosis(U_c, A, Y))
        diagnostics.update(self.theorem3_diagnosis(tx, tw, tz, tn, A, Y))
        return diagnostics

    # ---------- preprocessing & encoding helpers ----------

    def _standardize_inputs_safe(self, X: np.ndarray) -> np.ndarray:
        """Standardize with estimator-provided stats, avoiding divide-by-zero."""
        mean = getattr(self.est, "_v_mean", None)
        std = getattr(self.est, "_v_std", None)
        if mean is None or std is None:
            mean = getattr(self.est, "_x_mean", None)
            std = getattr(self.est, "_x_std", None)
        if mean is None or std is None:
            # fall back: per-feature standardization
            mean = X.mean(axis=0, keepdims=True)
            std = X.std(axis=0, keepdims=True)

        mean = np.asarray(mean, dtype=np.float64).reshape(1, -1)
        std = np.asarray(std, dtype=np.float64).reshape(1, -1)
        std = np.where(std > 0, std, 1.0)
        X_std = (X - mean) / std

        # Important: torch encoders are typically float32; cast here to avoid dtype mismatch.
        return X_std.astype(np.float32, copy=False)

    def _infer_block_dims(self, X_std: np.ndarray) -> Tuple[int, int, int]:
        """Infer x_dim/w_dim/z_dim.

        Prefer estimator.config.{x_dim,w_dim,z_dim}, else fall back to attributes on estimator.
        """
        cfg = getattr(self.est, "config", None)
        if cfg is not None and all(hasattr(cfg, k) for k in ("x_dim", "w_dim", "z_dim")):
            return int(cfg.x_dim), int(cfg.w_dim), int(cfg.z_dim)

        # Best-effort: try estimator attributes
        for keys in (("x_dim", "w_dim", "z_dim"), ("d_x", "d_w", "d_z")):
            if all(hasattr(self.est, k) for k in keys):
                return int(getattr(self.est, keys[0])), int(getattr(self.est, keys[1])), int(getattr(self.est, keys[2]))

        # Last resort: split evenly (should rarely happen)
        d = X_std.shape[1]
        x = d // 3
        w = d // 3
        z = d - x - w
        return x, w, z

    def _module_dtype(self, module: torch.nn.Module) -> torch.dtype:
        """Get parameter dtype for a torch module (default float32)."""
        try:
            p = next(module.parameters())
            return p.dtype
        except StopIteration:
            return torch.float32

    def _encode_blocks(self, X_std: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        x_dim, w_dim, z_dim = self._infer_block_dims(X_std)
        x_part = X_std[:, :x_dim]
        w_part = X_std[:, x_dim : x_dim + w_dim]
        z_part = X_std[:, x_dim + w_dim : x_dim + w_dim + z_dim]

        # eval mode
        for m in ("enc_x", "enc_w", "enc_z", "enc_n"):
            mod = getattr(self.est, m, None)
            if mod is not None and hasattr(mod, "eval"):
                mod.eval()

        with torch.no_grad():
            # force encoder inputs to float32 to avoid dtype mismatch with float32 weights
            tx = self.est.enc_x(torch.as_tensor(x_part, device=self.device, dtype=torch.float32))
            tw = self.est.enc_w(torch.as_tensor(w_part, device=self.device, dtype=torch.float32))
            tz = self.est.enc_z(torch.as_tensor(z_part, device=self.device, dtype=torch.float32))
            tn = self.est.enc_n(torch.as_tensor(X_std, device=self.device, dtype=torch.float32))

        return tx.detach().cpu().numpy(), tw.detach().cpu().numpy(), tz.detach().cpu().numpy(), tn.detach().cpu().numpy()

    # ---------- theorem 1': info loss / reconstruction ----------

    def theorem1_diagnosis(self, X_all: np.ndarray, U_c: np.ndarray, n_features: int = 5) -> Dict[str, Any]:
        """Proxy for representation information loss via linear reconstruction.

        Returns:
            - theorem1_avg_reconstruction_mse
            - theorem1_avg_reconstruction_r2
            - theorem1_info_loss_proxy   (MSE / Var)
        """
        n, d = X_all.shape
        n_features = int(min(d, max(1, n_features)))
        X_sub = X_all[:, :n_features]

        mses, r2s = [], []
        for j in range(X_sub.shape[1]):
            reg = Ridge(alpha=1e-6)
            reg.fit(U_c, X_sub[:, j])
            pred = reg.predict(U_c)
            mses.append(float(np.mean((pred - X_sub[:, j]) ** 2)))
            if np.var(X_sub[:, j]) > 0:
                r2s.append(float(r2_score(X_sub[:, j], pred)))
            else:
                r2s.append(0.0)

        avg_mse = float(np.mean(mses)) if mses else 0.0
        avg_r2 = float(np.mean(r2s)) if r2s else 0.0
        total_var = float(np.var(X_sub))
        info_loss_proxy = avg_mse / (total_var + 1e-8) if total_var > 0 else avg_mse

        return {
            "theorem1_avg_reconstruction_mse": avg_mse,
            "theorem1_avg_reconstruction_r2": avg_r2,
            "theorem1_info_loss_proxy": float(info_loss_proxy),
        }

    # ---------- theorem 2': separability / bottleneck ----------

    def _mi_u_target(self, U: np.ndarray, target: np.ndarray) -> float:
        """Estimate I(U;target) proxy and aggregate over U dimensions."""
        target = np.asarray(target).reshape(-1)
        # choose MI estimator
        uniq = np.unique(target)
        if uniq.size == 2 and np.all(np.isin(uniq, [0, 1])):
            mi = mutual_info_classif(U, target, discrete_features=False, random_state=self.cfg.random_state)
        else:
            mi = mutual_info_regression(U, target, random_state=self.cfg.random_state)
        mi = np.asarray(mi, dtype=float)
        # aggregate across U dims (mean is scale-stable)
        return float(np.nanmean(mi)) if mi.size else 0.0

    def theorem2_diagnosis(self, U_c: np.ndarray, A: np.ndarray, Y: np.ndarray) -> Dict[str, Any]:
        """Estimate proxies for I(U_c;A) and I(U_c;Y|A)."""
        n = U_c.shape[0]
        rng = np.random.RandomState(self.cfg.random_state)
        idx = np.arange(n)
        if n > self.cfg.max_n_for_mi:
            idx = rng.choice(n, size=self.cfg.max_n_for_mi, replace=False)
        # subsample slice (guard against NameError)
        U_sub, A_sub, Y_sub = U_c[idx], A[idx], Y[idx]

        mi_ua = self._mi_u_target(U_sub, A_sub)

        # conditional MI proxy: weighted by treatment group frequency
        mi_uy_given_a = 0.0
        mass = 0.0
        for a_val in np.unique(A_sub):
            mask = A_sub == a_val
            if int(mask.sum()) < self.cfg.min_group_n:
                continue
            w = float(mask.mean())
            mi_group = self._mi_u_target(U_sub[mask], Y_sub[mask])
            mi_uy_given_a += w * mi_group
            mass += w
        if mass > 0:
            mi_uy_given_a /= mass

        bottleneck_ratio = mi_ua / (mi_uy_given_a + 1e-8)

        return {
            "theorem2_I_Uc_A_proxy": float(mi_ua),
            "theorem2_I_Uc_Y_given_A_proxy": float(mi_uy_given_a),
            "theorem2_bottleneck_ratio": float(bottleneck_ratio),
        }

    # ---------- theorem 3': hierarchy / exclusivity ----------

    def theorem3_diagnosis(
        self,
        tx: np.ndarray,
        tw: np.ndarray,
        tz: np.ndarray,
        tn: np.ndarray,
        A: np.ndarray,
        Y: np.ndarray,
    ) -> Dict[str, Any]:
        """Check (proxy) hierarchy/exclusivity and cross-block entanglement.

        Returns:
            - theorem3_Tw_vs_A_auc
            - theorem3_Tw_vs_A_gamma   (0..1, higher means more treatment information in Tw)
            - theorem3_Tz_vs_Y_R2_linear
            - theorem3_avg_cross_block_corr
        """
        out: Dict[str, Any] = {}

        # Tw -> A (AUC preferred to accuracy; robust to imbalance)
        auc_w = 0.5
        gamma_w = 0.0
        if len(np.unique(A)) == 2 and tw.shape[0] == A.shape[0]:
            try:
                clf = LogisticRegression(max_iter=5000, n_jobs=None)
                clf.fit(tw, A)
                p = clf.predict_proba(tw)[:, 1]
                auc_w = float(roc_auc_score(A, p))
                gamma_w = float(max(0.0, min(1.0, 2.0 * abs(auc_w - 0.5))))
            except Exception:
                auc_w = 0.5
                gamma_w = 0.0

        # Tz -> Y (linear R2 proxy)
        r2_z = 0.0
        try:
            reg = LinearRegression()
            reg.fit(tz, Y)
            y_hat = reg.predict(tz)
            r2_z = float(r2_score(Y, y_hat)) if np.var(Y) > 0 else 0.0
        except Exception:
            r2_z = 0.0

        # Partial R^2 for Tz after controlling for A, Tx, Tw
        partial_r2 = 0.0
        try:
            base_X = np.hstack([A.reshape(-1, 1), tx, tw])
            full_X = np.hstack([base_X, tz])
            base_model = LinearRegression().fit(base_X, Y)
            full_model = LinearRegression().fit(full_X, Y)
            base_resid = Y - base_model.predict(base_X)
            full_resid = Y - full_model.predict(full_X)
            ssr_base = float(np.sum(base_resid ** 2))
            ssr_full = float(np.sum(full_resid ** 2))
            if ssr_base > 0:
                partial_r2 = max(0.0, 1.0 - ssr_full / ssr_base)
        except Exception:
            partial_r2 = 0.0

        # Cross-block correlations (average abs corr between blocks)
        def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
            if a.size == 0 or b.size == 0:
                return 0.0
            a = a - a.mean(axis=0, keepdims=True)
            b = b - b.mean(axis=0, keepdims=True)
            sa = a.std(axis=0, keepdims=True) + 1e-8
            sb = b.std(axis=0, keepdims=True) + 1e-8
            a = a / sa
            b = b / sb
            c = (a.T @ b) / float(a.shape[0])
            return float(np.nanmean(np.abs(c)))

        avg_cross = float(np.mean([
            _safe_corr(tx, tw),
            _safe_corr(tx, tz),
            _safe_corr(tw, tz),
        ]))

        out.update(
            {
                "theorem3_Tw_vs_A_auc": float(auc_w),
                "theorem3_Tw_vs_A_gamma": float(gamma_w),
                "theorem3_Tz_vs_Y_R2_linear": float(r2_z),
                "theorem3_Tz_vs_Y_partial_R2": float(partial_r2),
                "theorem3_avg_cross_block_corr": float(avg_cross),
            }
        )
        return out


__all__ = ["TheoremDiagnosticsConfig", "TheoremComplianceDiagnostics"]
