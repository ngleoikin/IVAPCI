"""Theory-aligned diagnostics for IVAPCI encoders.

This module is intentionally self-contained so benchmark harnesses can safely do:
    from models.ivapci_theory_diagnostics import TheoremComplianceDiagnostics

It provides:
- "theorem*" keys: best-effort proxies for Theorem 1/2/3 style conditions
- "rep_*" keys: pragmatic quality metrics used by the benchmark/analysis

Design goals:
- No dependency on your training code beyond a fitted estimator that exposes:
    * get_latent(X_all) -> U_c (np.ndarray)
    * config.x_dim, config.w_dim
    * _x_mean, _x_std (numpy arrays) for standardization
    * enc_x/enc_w/enc_z/enc_n (torch modules), optional
- Robust to edge cases: constant targets, tiny groups, zero-variance columns, NaNs
- Avoids "training-set accuracy" pitfalls: uses a holdout split by default
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit, ShuffleSplit


@dataclass
class TheoremDiagnosticsConfig:
    """Configuration for theory diagnostics.

    Attributes:
        max_n_for_mi: Subsample cap for mutual information estimates to control runtime.
        random_state: Seed used for subsampling and splits.
        holdout_frac: Fraction of data used as holdout for predictive diagnostics.
        min_group_n: Minimum samples required per treatment group for conditional MI.
        mi_n_neighbors: kNN neighbors for sklearn MI estimators (tradeoff bias/variance).
    """

    max_n_for_mi: int = 5000
    random_state: int = 42
    holdout_frac: float = 0.3
    min_group_n: int = 30
    mi_n_neighbors: int = 3


class TheoremComplianceDiagnostics:
    """Best-effort diagnostics approximating the three theory pillars.

    Typical usage inside estimator.fit():
        try:
            from models.ivapci_theory_diagnostics import TheoremComplianceDiagnostics
            diag = TheoremComplianceDiagnostics(self)
            self.training_diagnostics.update(diag.run_all_diagnostics(V_all, A, Y))
        except Exception as e:
            self.training_diagnostics["diagnostics_error"] = str(e)
    """

    def __init__(
        self,
        estimator,
        config: Optional[TheoremDiagnosticsConfig] = None,
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
        """Compute proxies for theorem-style compliance and practical rep quality."""
        X_all = np.asarray(X_all, dtype=float)
        A = np.asarray(A).reshape(-1)
        Y = np.asarray(Y).reshape(-1)

        if X_all.ndim != 2:
            raise ValueError("X_all must be 2D array [n, d].")
        n = X_all.shape[0]
        if A.shape[0] != n or Y.shape[0] != n:
            raise ValueError("A and Y must have same length as X_all.")

        # ---------- latent U_c ----------
        U_c = self._get_latent_safe(X_all)

        # ---------- block representations (tx, tw, tz, tn) ----------
        tx, tw, tz, tn = self._encode_blocks_safe(X_all)

        diagnostics: Dict[str, Any] = {}

        # Theorem proxies
        diagnostics.update(self.theorem1_diagnosis(X_all, U_c, n_features=n_recon_features))
        diagnostics.update(self.theorem2_diagnosis(U_c, A, Y))
        diagnostics.update(self.theorem3_diagnosis(tx, tw, tz, tn, A, Y))

        # Practical rep quality (names used in benchmark analysis)
        diagnostics.update(self.rep_quality_diagnosis(tx, tw, tz, A, Y))

        return diagnostics

    # ---------- helpers ----------

    def _get_latent_safe(self, X_all: np.ndarray) -> np.ndarray:
        """Get causal latent U_c from estimator, with sanity checks."""
        U_c = self.est.get_latent(X_all)
        U_c = np.asarray(U_c, dtype=float)
        if U_c.ndim != 2 or U_c.shape[0] != X_all.shape[0]:
            raise ValueError("estimator.get_latent must return 2D array [n, d_latent].")
        U_c = np.nan_to_num(U_c, nan=0.0, posinf=0.0, neginf=0.0)
        return U_c

    def _standardize_inputs_safe(self, X_all: np.ndarray) -> np.ndarray:
        """Standardize using estimator stats if present; otherwise return centered inputs."""
        if hasattr(self.est, "_x_mean") and hasattr(self.est, "_x_std"):
            mean = np.asarray(self.est._x_mean, dtype=float)
            std = np.asarray(self.est._x_std, dtype=float)
            std_safe = np.where(std > 1e-12, std, 1.0)
            X_std = (X_all - mean) / std_safe
        else:
            X_std = X_all - X_all.mean(axis=0, keepdims=True)
        X_std = np.nan_to_num(X_std, nan=0.0, posinf=0.0, neginf=0.0)
        return X_std

    def _encode_blocks_safe(self, X_all: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Encode blocks using encoders if available; else fall back to splitting U_c."""
        X_std = self._standardize_inputs_safe(X_all)

        # If encoder modules exist, use them (preferred: matches training-time blocks)
        has_enc = all(hasattr(self.est, k) for k in ["enc_x", "enc_w", "enc_z", "enc_n"]) and hasattr(self.est, "config")
        if has_enc:
            cfg = self.est.config
            x_dim = int(getattr(cfg, "x_dim"))
            w_dim = int(getattr(cfg, "w_dim"))
            x_part = X_std[:, :x_dim]
            w_part = X_std[:, x_dim : x_dim + w_dim]
            z_part = X_std[:, x_dim + w_dim :]

            # ensure eval mode
            self.est.enc_x.eval()
            self.est.enc_w.eval()
            self.est.enc_z.eval()
            self.est.enc_n.eval()

            with torch.no_grad():
                tx = self.est.enc_x(torch.from_numpy(x_part).to(self.device)).cpu().numpy()
                tw = self.est.enc_w(torch.from_numpy(w_part).to(self.device)).cpu().numpy()
                tz = self.est.enc_z(torch.from_numpy(z_part).to(self.device)).cpu().numpy()
                tn = self.est.enc_n(torch.from_numpy(X_std).to(self.device)).cpu().numpy()

            return (np.nan_to_num(tx), np.nan_to_num(tw), np.nan_to_num(tz), np.nan_to_num(tn))

        # Fallback: if estimator exposes _split_latent or _split_blocks_np, use it
        U_c = self._get_latent_safe(X_all)
        if hasattr(self.est, "_split_latent"):
            blocks = self.est._split_latent(U_c)
            if isinstance(blocks, (list, tuple)) and len(blocks) == 4:
                tx, tw, tz, tn = blocks
                return (np.asarray(tx), np.asarray(tw), np.asarray(tz), np.asarray(tn))

        if hasattr(self.est, "_split_blocks_np"):
            blocks = self.est._split_blocks_np(X_std)
            if isinstance(blocks, (list, tuple)) and len(blocks) >= 3:
                # best-effort: expects [X,W,Z,(N)]
                x_part, w_part, z_part = blocks[:3]
                # encode by identity (not ideal but gives something)
                tx = np.asarray(x_part)
                tw = np.asarray(w_part)
                tz = np.asarray(z_part)
                tn = np.zeros((X_all.shape[0], 1), dtype=float)
                return (tx, tw, tz, tn)

        # last resort: treat whole latent as tx and zeros for others
        tx = U_c
        tw = np.zeros((X_all.shape[0], 1), dtype=float)
        tz = np.zeros((X_all.shape[0], 1), dtype=float)
        tn = np.zeros((X_all.shape[0], 1), dtype=float)
        return tx, tw, tz, tn

    def _holdout_split(self, A: np.ndarray, n: int) -> Tuple[np.ndarray, np.ndarray]:
        """Return train/test indices. Stratify if binary treatment."""
        rng = self.cfg.random_state
        frac = self.cfg.holdout_frac
        if n < 50:
            # tiny: use all for both (will degrade to in-sample, but avoids errors)
            idx = np.arange(n)
            return idx, idx

        if len(np.unique(A)) == 2 and np.all(np.isin(A, [0, 1])):
            sss = StratifiedShuffleSplit(n_splits=1, test_size=frac, random_state=rng)
            tr, te = next(sss.split(np.zeros(n), A))
            return tr, te

        ss = ShuffleSplit(n_splits=1, test_size=frac, random_state=rng)
        tr, te = next(ss.split(np.zeros(n)))
        return tr, te

    # ---------- theorem 1': info loss / reconstruction ----------

    def theorem1_diagnosis(
        self,
        X_all: np.ndarray,
        U_c: np.ndarray,
        n_features: int = 5,
    ) -> Dict[str, Any]:
        """Proxy for representation information loss via linear recon MSE."""
        n, d = X_all.shape
        n_features = int(min(d, max(1, n_features)))
        X_sub = X_all[:, :n_features]

        # recon each feature from U_c (linear)
        recons = []
        for j in range(X_sub.shape[1]):
            y = X_sub[:, j]
            if np.allclose(np.var(y), 0.0):
                recons.append(0.0)
                continue
            reg = Ridge(alpha=1.0)
            reg.fit(U_c, y)
            pred = reg.predict(U_c)
            recons.append(float(np.mean((pred - y) ** 2)))

        avg_recon_mse = float(np.mean(recons))
        total_var = float(np.var(X_sub))
        info_loss_proxy = avg_recon_mse / (total_var + 1e-8) if total_var > 0 else avg_recon_mse

        # add a low-var input alert signal (useful for your low-var-proxy scenario)
        x_std = self._standardize_inputs_safe(X_all)
        min_std = float(np.min(np.std(x_std, axis=0)))
        frac_near_zero_std = float(np.mean(np.std(x_std, axis=0) < 1e-6))

        return {
            "theorem1_avg_reconstruction_mse": avg_recon_mse,
            "theorem1_info_loss_proxy": info_loss_proxy,
            "theorem1_min_std_after_standardize": min_std,
            "theorem1_frac_near_zero_std": frac_near_zero_std,
        }

    # ---------- theorem 2': separability / bottleneck ----------

    def theorem2_diagnosis(
        self,
        U_c: np.ndarray,
        A: np.ndarray,
        Y: np.ndarray,
    ) -> Dict[str, Any]:
        """Estimate proxies for I(U_c;A) and I(U_c;Y|A).

        Notes:
        - For binary A we use mutual_info_classif (more appropriate than regression MI).
        - For continuous Y we use mutual_info_regression.
        - We aggregate MI across latent dimensions by mean.
        """
        n = U_c.shape[0]
        rng = np.random.RandomState(self.cfg.random_state)
        idx = np.arange(n)
        if n > self.cfg.max_n_for_mi:
            idx = rng.choice(n, size=self.cfg.max_n_for_mi, replace=False)

        U_sub, A_sub, Y_sub = U_c[idx], A[idx], Y[idx]

        # I(U;A): average across dims
        mi_ua_vec = mutual_info_classif(
            U_sub,
            A_sub.astype(int) if len(np.unique(A_sub)) == 2 else A_sub,
            discrete_features=False,
            n_neighbors=self.cfg.mi_n_neighbors,
            random_state=self.cfg.random_state,
        )
        mi_ua = float(np.mean(mi_ua_vec)) if np.size(mi_ua_vec) else 0.0

        # I(U;Y|A): weighted avg MI within A groups (for binary A)
        mi_uy_given_a = 0.0
        mass = 0.0
        if len(np.unique(A_sub)) == 2 and np.all(np.isin(A_sub, [0, 1])):
            for a_val in [0, 1]:
                mask = A_sub == a_val
                if mask.sum() < self.cfg.min_group_n or np.allclose(np.var(Y_sub[mask]), 0.0):
                    continue
                weight = float(mask.mean())
                mi_vec = mutual_info_regression(
                    U_sub[mask],
                    Y_sub[mask],
                    n_neighbors=self.cfg.mi_n_neighbors,
                    random_state=self.cfg.random_state,
                )
                mi_group = float(np.mean(mi_vec)) if np.size(mi_vec) else 0.0
                mi_uy_given_a += weight * mi_group
                mass += weight
            if mass > 0:
                mi_uy_given_a /= mass
        else:
            # non-binary A: just compute unconditional proxy
            if not np.allclose(np.var(Y_sub), 0.0):
                mi_vec = mutual_info_regression(
                    U_sub,
                    Y_sub,
                    n_neighbors=self.cfg.mi_n_neighbors,
                    random_state=self.cfg.random_state,
                )
                mi_uy_given_a = float(np.mean(mi_vec)) if np.size(mi_vec) else 0.0

        bottleneck_ratio = mi_ua / (mi_uy_given_a + 1e-8)

        return {
            "theorem2_I_Uc_A_proxy": mi_ua,
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
        """Check exclusivity and cross-block dependence proxies.

        Improvements over the original version you pasted:
        - Uses holdout ROC-AUC / R^2 rather than in-sample accuracy/R^2.
        - Handles constant targets and zero-variance features gracefully.
        """
        out: Dict[str, Any] = {}
        n = len(A)
        tr, te = self._holdout_split(A, n)

        # --- W should not predict A (use ROC-AUC; closer to 0.5 is better) ---
        auc_w = np.nan
        if len(np.unique(A)) == 2 and np.all(np.isin(A, [0, 1])) and tw.shape[1] > 0:
            try:
                clf = LogisticRegression(max_iter=3000, solver="lbfgs")
                clf.fit(tw[tr], A[tr])
                p = clf.predict_proba(tw[te])[:, 1]
                auc_w = float(roc_auc_score(A[te], p))
            except Exception:
                auc_w = np.nan
        gamma_w = float(abs((auc_w if np.isfinite(auc_w) else 0.5) - 0.5) * 2.0)  # 0 is best

        # --- Z should not predict Y (exclusion leakage proxy) ---
        r2_z = np.nan
        if tz.shape[1] > 0 and not np.allclose(np.var(Y[te]), 0.0):
            try:
                reg = Ridge(alpha=1.0)
                reg.fit(tz[tr], Y[tr])
                pred = reg.predict(tz[te])
                r2_z = float(r2_score(Y[te], pred))
            except Exception:
                r2_z = np.nan

        # --- cross-block linear dependence (avg abs corr) ---
        H = np.hstack([tx, tw, tz]).astype(float)
        H = H - H.mean(axis=0, keepdims=True)
        std = H.std(axis=0, keepdims=True)
        std = np.where(std > 1e-12, std, 1.0)
        Hn = H / std

        corr = np.corrcoef(Hn.T)
        corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)

        d_tx, d_tw, d_tz = tx.shape[1], tw.shape[1], tz.shape[1]
        cross_corrs = []
        # tx - tw
        for i in range(d_tx):
            for j in range(d_tx, d_tx + d_tw):
                cross_corrs.append(abs(corr[i, j]))
        # tx - tz
        for i in range(d_tx):
            for j in range(d_tx + d_tw, d_tx + d_tw + d_tz):
                cross_corrs.append(abs(corr[i, j]))
        # tw - tz
        for i in range(d_tx, d_tx + d_tw):
            for j in range(d_tx + d_tw, d_tx + d_tw + d_tz):
                cross_corrs.append(abs(corr[i, j]))
        avg_cross = float(np.mean(cross_corrs)) if cross_corrs else 0.0

        out.update(
            {
                "theorem3_Tw_vs_A_auc": float(auc_w) if np.isfinite(auc_w) else np.nan,
                "theorem3_Tw_vs_A_gamma": gamma_w,
                "theorem3_Tz_vs_Y_R2_holdout": float(r2_z) if np.isfinite(r2_z) else np.nan,
                "theorem3_avg_cross_block_corr": avg_cross,
                # heuristic combined score: higher is better
                "theorem3_hierarchy_score": float((1.0 - gamma_w) * (1.0 - max(0.0, (r2_z if np.isfinite(r2_z) else 0.0)))),
            }
        )
        return out

    # ---------- practical rep quality (benchmark-friendly names) ----------

    def rep_quality_diagnosis(
        self,
        tx: np.ndarray,
        tw: np.ndarray,
        tz: np.ndarray,
        A: np.ndarray,
        Y: np.ndarray,
    ) -> Dict[str, Any]:
        """Diagnostics used for model selection / debugging.

        Keys produced (when applicable):
          - rep_auc_z_to_a: ROC-AUC of tz -> A (higher means stronger IV signal)
          - rep_auc_w_to_a: ROC-AUC of tw -> A (should be ~0.5 if W independent of A)
          - rep_r2_xw_a_to_y: holdout R^2 of [tx,tw,A] -> Y (outcome predictability)
          - rep_exclusion_leakage_r2: holdout R^2 of [tz,A] -> Y (Z leakage; should be low)
        """
        n = len(A)
        tr, te = self._holdout_split(A, n)
        out: Dict[str, Any] = {}

        # tz -> A (strength proxy)
        if len(np.unique(A)) == 2 and np.all(np.isin(A, [0, 1])) and tz.shape[1] > 0:
            try:
                clf = LogisticRegression(max_iter=3000, solver="lbfgs")
                clf.fit(tz[tr], A[tr])
                p = clf.predict_proba(tz[te])[:, 1]
                out["rep_auc_z_to_a"] = float(roc_auc_score(A[te], p))
            except Exception:
                out["rep_auc_z_to_a"] = np.nan
        else:
            out["rep_auc_z_to_a"] = np.nan

        # tw -> A (should be ~0.5)
        if len(np.unique(A)) == 2 and np.all(np.isin(A, [0, 1])) and tw.shape[1] > 0:
            try:
                clf = LogisticRegression(max_iter=3000, solver="lbfgs")
                clf.fit(tw[tr], A[tr])
                p = clf.predict_proba(tw[te])[:, 1]
                out["rep_auc_w_to_a"] = float(roc_auc_score(A[te], p))
            except Exception:
                out["rep_auc_w_to_a"] = np.nan
        else:
            out["rep_auc_w_to_a"] = np.nan

        # [tx, tw, A] -> Y
        X_xw_a = np.hstack([tx, tw, A.reshape(-1, 1)]).astype(float)
        if X_xw_a.shape[1] > 0 and not np.allclose(np.var(Y[te]), 0.0):
            try:
                reg = Ridge(alpha=1.0)
                reg.fit(X_xw_a[tr], Y[tr])
                pred = reg.predict(X_xw_a[te])
                out["rep_r2_xw_a_to_y"] = float(r2_score(Y[te], pred))
            except Exception:
                out["rep_r2_xw_a_to_y"] = np.nan
        else:
            out["rep_r2_xw_a_to_y"] = np.nan

        # [tz, A] -> Y (exclusion leakage proxy)
        X_z_a = np.hstack([tz, A.reshape(-1, 1)]).astype(float)
        if X_z_a.shape[1] > 0 and not np.allclose(np.var(Y[te]), 0.0):
            try:
                reg = Ridge(alpha=1.0)
                reg.fit(X_z_a[tr], Y[tr])
                pred = reg.predict(X_z_a[te])
                out["rep_exclusion_leakage_r2"] = float(r2_score(Y[te], pred))
            except Exception:
                out["rep_exclusion_leakage_r2"] = np.nan
        else:
            out["rep_exclusion_leakage_r2"] = np.nan

        return out


__all__ = ["TheoremDiagnosticsConfig", "TheoremComplianceDiagnostics"]
