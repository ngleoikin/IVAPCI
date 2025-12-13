"""Theory-aligned diagnostics for IVAPCI encoders (Theorem 1/2/3 proxies)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import torch
from sklearn.feature_selection import mutual_info_regression
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import r2_score


@dataclass
class TheoremDiagnosticsConfig:
    """Configuration for theory diagnostics.

    Attributes:
        max_n_for_mi: Maximum number of samples used when estimating mutual information
            proxies to control runtime.
        random_state: Seed for subsampling during MI estimation.
    """

    max_n_for_mi: int = 5000
    random_state: int = 42


class TheoremComplianceDiagnostics:
    """Best-effort diagnostics approximating the three theory pillars.

    Usage:
        diag = TheoremComplianceDiagnostics(fitted_estimator)
        result = diag.run_all_diagnostics(X_all, A, Y)

    The estimator is expected to expose:
      * get_latent(X) -> U_c (concatenated causal blocks)
      * _split_blocks_np(X_std)
      * enc_x/enc_w/enc_z/enc_n modules
      * _x_mean/_x_std for preprocessing
    """

    def __init__(
        self, estimator, config: Optional[TheoremDiagnosticsConfig] = None
    ) -> None:
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

        X_all = np.asarray(X_all, dtype=float)
        A = np.asarray(A).reshape(-1)
        Y = np.asarray(Y).reshape(-1)

        # causal latent (concatenated blocks)
        U_c = self.est.get_latent(X_all)

        # standardized proxy blocks
        X_std = (X_all - self.est._x_mean) / self.est._x_std
        tx, tw, tz, tn = self._encode_blocks(X_std)

        diagnostics: Dict[str, Any] = {}
        diagnostics.update(
            self.theorem1_diagnosis(X_all, U_c, n_features=n_recon_features)
        )
        diagnostics.update(self.theorem2_diagnosis(U_c, A, Y))
        diagnostics.update(self.theorem3_diagnosis(tx, tw, tz, tn, A, Y))
        return diagnostics

    # ---------- encoding helpers ----------

    def _encode_blocks(self, X_std: np.ndarray):
        cfg = self.est.config
        x_part = X_std[:, : cfg.x_dim]
        w_part = X_std[:, cfg.x_dim : cfg.x_dim + cfg.w_dim]
        z_part = X_std[:, cfg.x_dim + cfg.w_dim :]

        self.est.enc_x.eval()
        self.est.enc_w.eval()
        self.est.enc_z.eval()
        self.est.enc_n.eval()

        with torch.no_grad():
            tx = self.est.enc_x(torch.from_numpy(x_part).to(self.device))
            tw = self.est.enc_w(torch.from_numpy(w_part).to(self.device))
            tz = self.est.enc_z(torch.from_numpy(z_part).to(self.device))
            tn = self.est.enc_n(torch.from_numpy(X_std).to(self.device))

        return tx.cpu().numpy(), tw.cpu().numpy(), tz.cpu().numpy(), tn.cpu().numpy()

    # ---------- theorem 1': info loss / reconstruction ----------

    def theorem1_diagnosis(
        self, X_all: np.ndarray, U_c: np.ndarray, n_features: int = 5
    ) -> Dict[str, Any]:
        """Proxy for representation information loss via linear recon MSE."""

        n, d = X_all.shape
        n_features = min(d, max(1, n_features))
        X_sub = X_all[:, :n_features]

        recons = []
        for j in range(X_sub.shape[1]):
            reg = LinearRegression()
            reg.fit(U_c, X_sub[:, j])
            pred = reg.predict(U_c)
            recons.append(float(np.mean((pred - X_sub[:, j]) ** 2)))

        avg_recon_mse = float(np.mean(recons))
        total_var = float(np.var(X_sub))
        info_loss_proxy = (
            avg_recon_mse / (total_var + 1e-8) if total_var > 0 else avg_recon_mse
        )

        return {
            "theorem1_avg_reconstruction_mse": avg_recon_mse,
            "theorem1_info_loss_proxy": info_loss_proxy,
        }

    # ---------- theorem 2': separability / bottleneck ----------

    def theorem2_diagnosis(
        self, U_c: np.ndarray, A: np.ndarray, Y: np.ndarray
    ) -> Dict[str, Any]:
        """Estimate proxies for I(U_c;A) and I(U_c;Y|A)."""

        n = U_c.shape[0]
        rng = np.random.RandomState(self.cfg.random_state)
        idx = np.arange(n)
        if n > self.cfg.max_n_for_mi:
            idx = rng.choice(n, size=self.cfg.max_n_for_mi, replace=False)
        U_sub, A_sub, Y_sub = U_c[idx], A[idx], Y[idx]

        mi_ua = float(mutual_info_regression(U_sub, A_sub)[0])

        mi_uy_given_a = 0.0
        mass = 0.0
        for a_val in [0, 1]:
            mask = A_sub == a_val
            if mask.sum() < 20:
                continue
            weight = float(mask.mean())
            mi_group = float(mutual_info_regression(U_sub[mask], Y_sub[mask])[0])
            mi_uy_given_a += weight * mi_group
            mass += weight
        if mass > 0:
            mi_uy_given_a /= mass

        bottleneck_ratio = mi_ua / (mi_uy_given_a + 1e-8)

        return {
            "theorem2_I_Uc_A_proxy": mi_ua,
            "theorem2_I_Uc_Y_given_A_proxy": mi_uy_given_a,
            "theorem2_bottleneck_ratio": bottleneck_ratio,
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
        """Check treatment/outcome exclusivity and cross-block correlations."""

        out: Dict[str, Any] = {}

        if len(np.unique(A)) == 2:
            clf_w = LogisticRegression(max_iter=2000)
            clf_w.fit(tw, A)
            acc_w = float(clf_w.score(tw, A))
        else:
            acc_w = 0.5
        gamma_w = abs(acc_w - 0.5) * 2.0

        reg_z = LinearRegression()
        reg_z.fit(tz, Y)
        y_pred_from_z = reg_z.predict(tz)
        r2_z = float(r2_score(Y, y_pred_from_z)) if np.var(Y) > 0 else 0.0

        H = np.hstack([tx, tw, tz])
        H_center = H - H.mean(axis=0, keepdims=True)
        corr = np.corrcoef(H_center.T)
        d_tx, d_tw, d_tz = tx.shape[1], tw.shape[1], tz.shape[1]

        cross_corrs = []
        for i in range(d_tx):
            for j in range(d_tx, d_tx + d_tw):
                cross_corrs.append(abs(corr[i, j]))
        for i in range(d_tx):
            for j in range(d_tx + d_tw, d_tx + d_tw + d_tz):
                cross_corrs.append(abs(corr[i, j]))
        for i in range(d_tx, d_tx + d_tw):
            for j in range(d_tx + d_tw, d_tx + d_tw + d_tz):
                cross_corrs.append(abs(corr[i, j]))

        avg_cross = float(np.mean(cross_corrs)) if cross_corrs else 0.0

        out.update(
            {
                "theorem3_Tw_vs_A_accuracy": acc_w,
                "theorem3_Tw_vs_A_gamma": gamma_w,
                "theorem3_Tz_vs_Y_R2_linear": r2_z,
                "theorem3_avg_cross_block_corr": avg_cross,
                "theorem3_hierarchy_score": (1.0 - gamma_w)
                * (1.0 - max(0.0, r2_z)),
            }
        )
        return out


__all__ = ["TheoremDiagnosticsConfig", "TheoremComplianceDiagnostics"]

