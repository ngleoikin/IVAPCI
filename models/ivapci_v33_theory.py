"""IVAPCI v3.3 (TheoryComplete): hierarchical encoder + theorem-aware extras."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler

from . import BaseCausalEstimator


# -------------------------
# Theorem 1: preprocessing
# -------------------------

def _info_aware_standardize(
    train: np.ndarray,
    min_std: float = 1e-2,
    low_var_min_std: float = 1e-4,
    clip_value: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Standardize while *clamping* (not flattening) near-constant dimensions.

    Near-deterministic proxy channels keep their small variance (plus a tiny
    floor) instead of being inflated to the global ``min_std``. This protects
    low-variance informative proxies in "low-var" scenarios from being washed
    out by preprocessing.

    Returns:
        train_std, mean, std_clamped, protected_mask
    """
    mean = train.mean(axis=0, keepdims=True)
    std = train.std(axis=0, keepdims=True)
    protected = (std < min_std).squeeze(0)

    # Use a softer floor for extremely low-variance channels so their signal is
    # preserved (Theorem-1 guidance for low-var proxies).
    std_clamped = np.where(
        std < min_std,
        np.maximum(std, low_var_min_std),
        std,
    )
    standardized = (train - mean) / std_clamped
    if clip_value is not None:
        standardized = np.clip(standardized, -clip_value, clip_value)
    return standardized, mean, std_clamped, protected


def _apply_standardize(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (x - mean) / std


class InformationLossMonitor:
    """Reconstruction MSE / variance as a lightweight ΔI proxy (Theorem 1)."""

    @staticmethod
    def estimate(model: "IVAPCIv33TheoryHierEstimator", V_all: np.ndarray) -> Dict[str, float]:
        V_std = _apply_standardize(V_all.astype(np.float32), model._v_mean, model._v_std)
        with torch.no_grad():
            V_t = torch.from_numpy(V_std).to(model.device)
            tx, tw, tz, tn = model._encode_blocks(V_t)
            recon = model.decoder(torch.cat([tx, tw, tz, tn], dim=1)).cpu().numpy()
        recon_mse = float(np.mean((recon - V_std) ** 2))
        var = float(np.var(V_std) + 1e-8)
        return {"recon_mse": recon_mse, "info_loss_proxy": recon_mse / var}


# -------------------------
# small NN building blocks
# -------------------------

def _mlp(input_dim: int, hidden: Sequence[int], out_dim: int, dropout: float = 0.0) -> nn.Sequential:
    layers: list[nn.Module] = []
    last = input_dim
    for h in hidden:
        layers.append(nn.Linear(last, h))
        layers.append(nn.ReLU())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        last = h
    layers.append(nn.Linear(last, out_dim))
    return nn.Sequential(*layers)


class _GroupEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden: Sequence[int], out_dim: int, dropout: float = 0.0):
        super().__init__()
        if hidden:
            self.body = _mlp(input_dim, hidden, hidden[-1], dropout=dropout)
            self.out = nn.Linear(hidden[-1], out_dim)
        else:
            self.body = nn.Identity()
            self.out = nn.Linear(input_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.body(x)
        return self.out(h)


class _Decoder(nn.Module):
    def __init__(self, input_dim: int, hidden: Sequence[int], out_dim: int):
        super().__init__()
        self.net = _mlp(input_dim, hidden, out_dim)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class _AClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden: Sequence[int]):
        super().__init__()
        self.net = _mlp(input_dim, hidden, 1)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.net(t).squeeze(-1)


class _YRegressor(nn.Module):
    def __init__(self, input_dim: int, hidden: Sequence[int]):
        super().__init__()
        self.net = _mlp(input_dim + 1, hidden, 1)

    def forward(self, t: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([t, a.unsqueeze(-1)], dim=1)).squeeze(-1)


class _YAdversary(nn.Module):
    def __init__(self, input_dim: int, hidden: Sequence[int]):
        super().__init__()
        self.net = _mlp(input_dim, hidden, 1)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.net(t).squeeze(-1)


# -------------------------
# Orthogonality penalties
# -------------------------

def _center(x: torch.Tensor) -> torch.Tensor:
    return x - x.mean(dim=0, keepdim=True)


def _offdiag_corr_penalty(blocks: list[torch.Tensor], eps: float = 1e-8) -> torch.Tensor:
    pen = torch.zeros((), device=blocks[0].device)
    centered = [_center(b) for b in blocks]
    norms = [torch.sqrt(torch.mean(b ** 2, dim=0, keepdim=True) + eps) for b in centered]
    for i in range(len(blocks)):
        for j in range(i + 1, len(blocks)):
            bi = centered[i] / norms[i]
            bj = centered[j] / norms[j]
            corr = (bi.T @ bj) / bi.shape[0]
            pen = pen + torch.mean(corr ** 2)
    return pen


def _conditional_orthogonal_penalty(blocks: list[torch.Tensor], cond: torch.Tensor, ridge: float = 1e-3) -> torch.Tensor:
    pen = torch.zeros((), device=blocks[0].device)
    C = _center(cond)
    CtC = C.T @ C / C.shape[0]
    I = torch.eye(CtC.shape[0], device=C.device)
    Ct = C.T / C.shape[0]
    resid = []
    for B in blocks:
        Bc = _center(B)
        Beta = torch.linalg.solve(CtC + ridge * I, Ct @ Bc)
        resid.append(Bc - C @ Beta)
    return _offdiag_corr_penalty(resid)


def _padic_ultrametric_loss(Uc: torch.Tensor, num_triplets: int = 128) -> torch.Tensor:
    if Uc.shape[0] < 3:
        return torch.zeros((), device=Uc.device)
    n = Uc.shape[0]
    t = min(num_triplets, max(1, n // 2))
    idx = torch.randint(0, n, (t, 3), device=Uc.device)
    u_i = Uc[idx[:, 0]]
    u_j = Uc[idx[:, 1]]
    u_k = Uc[idx[:, 2]]
    d_ij = torch.norm(u_i - u_j, dim=1)
    d_jk = torch.norm(u_j - u_k, dim=1)
    d_ik = torch.norm(u_i - u_k, dim=1)
    viol = torch.relu(d_ik - torch.maximum(d_ij, d_jk))
    return torch.mean(viol ** 2)


# -------------------------
# Config (Theorem 4–5 guided)
# -------------------------

@dataclass
class IVAPCIV33TheoryConfig:
    x_dim: int = 0
    w_dim: int = 0
    z_dim: int = 0

    latent_x_dim: int = 4
    latent_w_dim: int = 4
    latent_z_dim: int = 2
    latent_n_dim: int = 4

    enc_x_hidden: Sequence[int] = (128, 64)
    enc_w_hidden: Sequence[int] = (256, 128, 64)
    enc_z_hidden: Sequence[int] = (64, 32)
    enc_n_hidden: Sequence[int] = (128, 64)
    dropout_z: float = 0.2

    dec_hidden: Sequence[int] = (128, 128)

    a_hidden: Sequence[int] = (64, 32)
    y_hidden: Sequence[int] = (128, 64)
    adv_a_hidden: Sequence[int] = (64,)
    adv_y_hidden: Sequence[int] = (64,)

    lambda_recon: float = 1.0
    lambda_a: float = 0.1
    lambda_y: float = 0.5
    lambda_ortho: float = 0.01
    lambda_cond_ortho: float = 1e-3
    lambda_consistency: float = 0.05
    ridge_alpha: float = 1e-2
    standardize_nuisance: bool = True
    gamma_adv_w: float = 0.1
    gamma_adv_z: float = 0.1
    gamma_adv_n: float = 0.1
    gamma_padic: float = 1e-3

    min_std: float = 1e-2
    low_var_min_std: float = 1e-3
    std_clip: Optional[float] = 10.0
    use_noise_in_latent: bool = True

    lr_main: float = 1e-3
    lr_adv: float = 1e-3
    batch_size: int = 128
    epochs_pretrain: int = 50
    epochs_main: int = 150
    val_frac: float = 0.1
    early_stopping_patience: int = 15
    early_stopping_min_delta: float = 0.0

    n_splits_dr: int = 2
    clip_prop: float = 5e-3
    clip_prop_adaptive_max: float = 1e-2
    clip_prop_radr: float = 1e-2
    ipw_cap: float = 20.0
    ipw_cap_high: float = 100.0
    ipw_cap_radr: Optional[float] = None
    adaptive_ipw: bool = True

    seed: int = 42
    device: str = "cpu"

    cond_ortho_warmup_epochs: int = 10

    n_samples_hint: Optional[int] = None
    adaptive: bool = True

    def apply_theorem45_defaults(self) -> "IVAPCIV33TheoryConfig":
        n = self.n_samples_hint
        if (not self.adaptive) or (n is None) or (n <= 0):
            return self
        scale = float(n) ** 0.4
        base = max(6, int(2 * np.log1p(n)))
        self.latent_x_dim = max(2, base // 3)
        self.latent_w_dim = max(2, base // 3)
        self.latent_z_dim = max(2, base // 6)
        self.latent_n_dim = max(2, base // 3)

        self.lambda_ortho = 0.005 * np.sqrt(np.log1p(n))
        self.lambda_consistency = 0.02 * np.sqrt(np.log1p(n))

        decay = 1.0 / np.sqrt(np.log1p(n))
        self.gamma_adv_w = 0.2 * decay
        self.gamma_adv_z = 0.2 * decay
        self.gamma_adv_n = 0.2 * decay

        self.gamma_padic = 0.001 * np.sqrt(np.log1p(n))
        self.min_std = max(1e-4, 1e-2 / np.sqrt(np.log1p(n)))
        self.low_var_min_std = min(self.low_var_min_std, self.min_std * 0.1)

        self.epochs_pretrain = int(min(80, 30 + 10 * np.log1p(n / 500)))
        self.epochs_main = int(min(250, 80 + 20 * np.log1p(n / 500)))
        return self


# -------------------------
# Estimator (no RADR)
# -------------------------

class IVAPCIv33TheoryHierEstimator(BaseCausalEstimator):
    """Hierarchical encoder with theorem-1-5 components (DR stage)."""

    def __init__(self, config: Optional[IVAPCIV33TheoryConfig] = None):
        self.config = config or IVAPCIV33TheoryConfig()
        if self.config.n_samples_hint is not None:
            self.config.apply_theorem45_defaults()
        self.device = torch.device(self.config.device)
        torch.manual_seed(self.config.seed)
        np.random.seed(self.config.seed)
        self._is_fit = False
        self._protected_mask: Optional[np.ndarray] = None
        self.training_diagnostics: Dict[str, float] = {}
        self.info_monitor = InformationLossMonitor()
        self._block_slices: Optional[Tuple[slice, slice, slice]] = None

    @staticmethod
    def _toggle_requires_grad(modules: Iterable[nn.Module], flag: bool) -> None:
        for mod in modules:
            for p in mod.parameters():
                p.requires_grad = flag

    # --- model wiring ---
    def _build(self, d_all: int) -> None:
        cfg = self.config
        if not (cfg.x_dim and cfg.w_dim and cfg.z_dim):
            raise ValueError("IVAPCI v3.3 requires x_dim, w_dim, z_dim to split X|W|Z blocks")
        if cfg.x_dim + cfg.w_dim + cfg.z_dim != d_all:
            raise ValueError("x_dim + w_dim + z_dim must equal input dimension")

        self._block_slices = (
            slice(0, cfg.x_dim),
            slice(cfg.x_dim, cfg.x_dim + cfg.w_dim),
            slice(cfg.x_dim + cfg.w_dim, cfg.x_dim + cfg.w_dim + cfg.z_dim),
        )

        self.enc_x = _GroupEncoder(cfg.x_dim, cfg.enc_x_hidden, cfg.latent_x_dim).to(self.device)
        self.enc_w = _GroupEncoder(cfg.w_dim, cfg.enc_w_hidden, cfg.latent_w_dim).to(self.device)
        self.enc_z = _GroupEncoder(cfg.z_dim, cfg.enc_z_hidden, cfg.latent_z_dim, dropout=cfg.dropout_z).to(
            self.device
        )
        self.enc_n = _GroupEncoder(d_all, cfg.enc_n_hidden, cfg.latent_n_dim).to(self.device)

        total_lat = cfg.latent_x_dim + cfg.latent_w_dim + cfg.latent_z_dim + cfg.latent_n_dim
        self.decoder = _Decoder(total_lat, cfg.dec_hidden, d_all).to(self.device)

        self.a_head = _AClassifier(cfg.latent_x_dim + cfg.latent_z_dim, cfg.a_hidden).to(self.device)
        self.y_head = _YRegressor(cfg.latent_x_dim + cfg.latent_w_dim, cfg.y_hidden).to(self.device)

        self.a_from_z = _AClassifier(cfg.latent_z_dim, (32,)).to(self.device)
        self.y_from_w = _YRegressor(cfg.latent_w_dim, (64, 32)).to(self.device)

        self.adv_w = _AClassifier(cfg.latent_w_dim, cfg.adv_a_hidden).to(self.device)
        self.adv_n = _AClassifier(cfg.latent_n_dim, cfg.adv_a_hidden).to(self.device)
        self.adv_z = _YAdversary(cfg.latent_z_dim, cfg.adv_y_hidden).to(self.device)

        main_params = (
            list(self.enc_x.parameters())
            + list(self.enc_w.parameters())
            + list(self.enc_z.parameters())
            + list(self.enc_n.parameters())
            + list(self.decoder.parameters())
            + list(self.a_head.parameters())
            + list(self.y_head.parameters())
            + list(self.a_from_z.parameters())
            + list(self.y_from_w.parameters())
        )
        self.main_opt = torch.optim.Adam(main_params, lr=cfg.lr_main)
        self.main_sched = ReduceLROnPlateau(self.main_opt, mode="min", factor=0.5, patience=10)

        adv_params = list(self.adv_w.parameters()) + list(self.adv_z.parameters()) + list(self.adv_n.parameters())
        self.adv_opt = torch.optim.Adam(adv_params, lr=cfg.lr_adv)

    def _split_blocks_tensor(
        self, V_t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        assert self._block_slices is not None, "Block slices must be initialized in _build"
        return (
            V_t[:, self._block_slices[0]],
            V_t[:, self._block_slices[1]],
            V_t[:, self._block_slices[2]],
        )

    def _encode_blocks(
        self, V_t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x_part, w_part, z_part = self._split_blocks_tensor(V_t)
        tx = self.enc_x(x_part)
        tw = self.enc_w(w_part)
        tz = self.enc_z(z_part)
        tn = self.enc_n(V_t)
        return tx, tw, tz, tn

    # --- training ---
    def fit(self, V_all: np.ndarray, A: np.ndarray, Y: np.ndarray) -> None:
        V_all = np.asarray(V_all, dtype=np.float32)
        A = np.asarray(A, dtype=np.float32).reshape(-1)
        Y = np.asarray(Y, dtype=np.float32).reshape(-1)
        n, d_all = V_all.shape
        cfg = self.config

        if cfg.n_samples_hint is None:
            cfg.n_samples_hint = int(n)
            if cfg.adaptive:
                cfg.apply_theorem45_defaults()

        tr_idx, va_idx = train_test_split(
            np.arange(n),
            test_size=cfg.val_frac,
            random_state=cfg.seed,
            stratify=A if np.unique(A).size > 1 else None,
        )
        V_tr, V_va = V_all[tr_idx], V_all[va_idx]
        A_tr, A_va = A[tr_idx], A[va_idx]
        Y_tr, Y_va = Y[tr_idx], Y[va_idx]

        V_tr_std, self._v_mean, self._v_std, protected = _info_aware_standardize(
            V_tr,
            min_std=cfg.min_std,
            low_var_min_std=cfg.low_var_min_std,
            clip_value=cfg.std_clip,
        )
        self._protected_mask = protected
        V_va_std = _apply_standardize(V_va, self._v_mean, self._v_std)
        if cfg.std_clip is not None:
            V_tr_std = np.clip(V_tr_std, -cfg.std_clip, cfg.std_clip)
            V_va_std = np.clip(V_va_std, -cfg.std_clip, cfg.std_clip)

        Y_tr_std, self._y_mean, self._y_std, _ = _info_aware_standardize(
            Y_tr.reshape(-1, 1), min_std=1e-6, low_var_min_std=1e-6, clip_value=cfg.std_clip
        )
        Y_tr_std = Y_tr_std.squeeze(1)
        Y_va_std = _apply_standardize(Y_va.reshape(-1, 1), self._y_mean, self._y_std).squeeze(1)

        self._build(d_all)

        bce = nn.BCEWithLogitsLoss()
        mse = nn.MSELoss()

        tr_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(
                torch.from_numpy(V_tr_std), torch.from_numpy(A_tr), torch.from_numpy(Y_tr_std)
            ),
            batch_size=cfg.batch_size,
            shuffle=True,
        )
        va_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(
                torch.from_numpy(V_va_std), torch.from_numpy(A_va), torch.from_numpy(Y_va_std)
            ),
            batch_size=cfg.batch_size,
            shuffle=False,
        )

        # Stage 0: reconstruction
        for _ in range(cfg.epochs_pretrain):
            self.enc_x.train(); self.enc_w.train(); self.enc_z.train(); self.enc_n.train(); self.decoder.train()
            for vb, _, _ in tr_loader:
                vb = vb.to(self.device)
                tx, tw, tz, tn = self._encode_blocks(vb)
                recon = self.decoder(torch.cat([tx, tw, tz, tn], dim=1))
                loss = mse(recon, vb)
                self.main_opt.zero_grad(); loss.backward(); self.main_opt.step()

        best_val = float("inf")
        patience = 0
        best_state = None

        # Stage 1: full training
        for epoch in range(cfg.epochs_main):
            self.enc_x.train(); self.enc_w.train(); self.enc_z.train(); self.enc_n.train()
            self.decoder.train(); self.a_head.train(); self.y_head.train()
            self.a_from_z.train(); self.y_from_w.train()
            self.adv_w.train(); self.adv_z.train(); self.adv_n.train()

            for vb, ab, yb in tr_loader:
                vb = vb.to(self.device); ab = ab.to(self.device); yb = yb.to(self.device)

                # adversaries update
                with torch.no_grad():
                    tx_d, tw_d, tz_d, tn_d = self._encode_blocks(vb)
                adv_loss = bce(self.adv_w(tw_d), ab) + bce(self.adv_n(tn_d), ab) + mse(self.adv_z(tz_d), yb)
                self.adv_opt.zero_grad(); adv_loss.backward(); self.adv_opt.step()

                # main update
                self._toggle_requires_grad([self.adv_w, self.adv_n, self.adv_z], False)
                tx, tw, tz, tn = self._encode_blocks(vb)
                recon = self.decoder(torch.cat([tx, tw, tz, tn], dim=1))
                logits_a = self.a_head(torch.cat([tx, tz], dim=1))
                y_pred = self.y_head(torch.cat([tx, tw], dim=1), ab)

                cons_a = bce(self.a_from_z(tz), ab)
                cons_y = mse(self.y_from_w(tw, ab), yb)
                consistency = cons_a + cons_y

                ortho = _offdiag_corr_penalty([tx, tw, tz, tn])
                warmup = cfg.cond_ortho_warmup_epochs
                cond_weight = 0.0
                if cfg.lambda_cond_ortho > 0 and epoch >= warmup:
                    ramp = min(1.0, (epoch - warmup + 1) / max(1, warmup))
                    cond_weight = float(cfg.lambda_cond_ortho * ramp)
                cond_ortho = torch.zeros((), device=self.device)
                if cond_weight > 0:
                    cond_ortho = _conditional_orthogonal_penalty([tw, tz, tn], torch.cat([tx, tz], dim=1))

                adv_w_logits = self.adv_w(tw)
                adv_n_logits = self.adv_n(tn)
                adv_z_pred = self.adv_z(tz)

                loss_main = (
                    cfg.lambda_recon * mse(recon, vb)
                    + cfg.lambda_a * bce(logits_a, ab)
                    + cfg.lambda_y * mse(y_pred, yb)
                    + cfg.lambda_consistency * consistency
                    + cfg.lambda_ortho * ortho
                    + cond_weight * cond_ortho
                    + cfg.gamma_padic * _padic_ultrametric_loss(torch.cat([tx, tz], dim=1))
                    - cfg.gamma_adv_w * bce(adv_w_logits, ab)
                    - cfg.gamma_adv_n * bce(adv_n_logits, ab)
                    - cfg.gamma_adv_z * mse(adv_z_pred, yb)
                )

                self.main_opt.zero_grad(); loss_main.backward(); self.main_opt.step()
                self._toggle_requires_grad([self.adv_w, self.adv_n, self.adv_z], True)

            # validation
            self.enc_x.eval(); self.enc_w.eval(); self.enc_z.eval(); self.enc_n.eval()
            self.decoder.eval(); self.a_head.eval(); self.y_head.eval()
            vals = []
            with torch.no_grad():
                for vb, ab, yb in va_loader:
                    vb = vb.to(self.device); ab = ab.to(self.device); yb = yb.to(self.device)
                    tx, tw, tz, tn = self._encode_blocks(vb)
                    recon = self.decoder(torch.cat([tx, tw, tz, tn], dim=1))
                    logits_a = self.a_head(torch.cat([tx, tz], dim=1))
                    y_pred = self.y_head(torch.cat([tx, tw], dim=1), ab)
                    loss_val = (
                        cfg.lambda_recon * mse(recon, vb)
                        + cfg.lambda_a * bce(logits_a, ab)
                        + cfg.lambda_y * mse(y_pred, yb)
                    )
                    vals.append(float(loss_val.item()))
            mean_val = float(np.mean(vals)) if vals else float("inf")
            self.main_sched.step(mean_val)

            if mean_val + cfg.early_stopping_min_delta < best_val:
                best_val = mean_val
                patience = 0
                best_state = {k: v.state_dict() for k, v in {
                    "enc_x": self.enc_x, "enc_w": self.enc_w, "enc_z": self.enc_z, "enc_n": self.enc_n,
                    "decoder": self.decoder, "a_head": self.a_head, "y_head": self.y_head,
                    "a_from_z": self.a_from_z, "y_from_w": self.y_from_w,
                    "adv_w": self.adv_w, "adv_z": self.adv_z, "adv_n": self.adv_n,
                }.items()}
            else:
                patience += 1
                if patience >= cfg.early_stopping_patience:
                    break

        if best_state is not None:
            self.enc_x.load_state_dict(best_state["enc_x"])
            self.enc_w.load_state_dict(best_state["enc_w"])
            self.enc_z.load_state_dict(best_state["enc_z"])
            self.enc_n.load_state_dict(best_state["enc_n"])
            self.decoder.load_state_dict(best_state["decoder"])
            self.a_head.load_state_dict(best_state["a_head"])
            self.y_head.load_state_dict(best_state["y_head"])
            self.a_from_z.load_state_dict(best_state["a_from_z"])
            self.y_from_w.load_state_dict(best_state["y_from_w"])
            self.adv_w.load_state_dict(best_state["adv_w"])
            self.adv_z.load_state_dict(best_state["adv_z"])
            self.adv_n.load_state_dict(best_state["adv_n"])

        self._is_fit = True

        # Diagnostics
        self.training_diagnostics = self.info_monitor.estimate(self, V_all)
        self.training_diagnostics["protected_features"] = (
            int(np.sum(self._protected_mask)) if self._protected_mask is not None else 0
        )
        self.training_diagnostics["min_std"] = float(cfg.min_std)

        # Adversary diagnostics (use raw-scale targets for comparability)
        V_std = _apply_standardize(V_all.astype(np.float32), self._v_mean, self._v_std)
        V_t = torch.from_numpy(V_std).to(self.device)

        self.enc_x.eval(); self.enc_w.eval(); self.enc_z.eval(); self.enc_n.eval()
        self.adv_w.eval(); self.adv_n.eval(); self.adv_z.eval()
        with torch.no_grad():
            tx, tw, tz, tn = self._encode_blocks(V_t)
            adv_w_probs = torch.sigmoid(self.adv_w(tw)).cpu().numpy()
            adv_n_probs = torch.sigmoid(self.adv_n(tn)).cpu().numpy()
            adv_z_pred = self.adv_z(tz).cpu().numpy()

        adv_w_acc = float(((adv_w_probs > 0.5) == (A == 1)).mean()) if adv_w_probs.size else np.nan
        adv_n_acc = float(((adv_n_probs > 0.5) == (A == 1)).mean()) if adv_n_probs.size else np.nan

        # adv_z is trained on standardized Y; convert back to raw scale for diagnostics
        if hasattr(self, "_y_std") and hasattr(self, "_y_mean"):
            y_scale = float(self._y_std.squeeze())
            y_mean = float(self._y_mean.squeeze())
            adv_z_pred_raw = adv_z_pred * y_scale + y_mean
        else:
            adv_z_pred_raw = adv_z_pred
        adv_z_r2 = float(r2_score(Y, adv_z_pred_raw)) if np.var(Y) > 0 else np.nan

        self.training_diagnostics.update(
            {
                "adv_w_acc": adv_w_acc,
                "adv_n_acc": adv_n_acc,
                "adv_z_r2": adv_z_r2,
            }
        )
    # --- latent + DR ---
    def get_latent(self, V_all: np.ndarray) -> np.ndarray:
        if not self._is_fit:
            raise RuntimeError("Estimator must be fit before get_latent.")
        V_all = np.asarray(V_all, dtype=np.float32)
        V_std = _apply_standardize(V_all, self._v_mean, self._v_std)
        V_t = torch.from_numpy(V_std).to(self.device)
        self.enc_x.eval(); self.enc_w.eval(); self.enc_z.eval(); self.enc_n.eval()
        with torch.no_grad():
            tx, tw, tz, tn = self._encode_blocks(V_t)
        if self.config.use_noise_in_latent:
            return torch.cat([tx, tw, tz, tn], dim=1).cpu().numpy()
        return torch.cat([tx, tw, tz], dim=1).cpu().numpy()

    def _dr_ate(self, U: np.ndarray, A: np.ndarray, Y: np.ndarray) -> float:
        cfg = self.config
        U = np.asarray(U)
        A = np.asarray(A).reshape(-1)
        Y = np.asarray(Y).reshape(-1)
        stats = {k: [] for k in [
            "e_min", "e_max", "e_q01", "e_q05", "e_q95", "e_q99",
            "clip_used", "cap_used", "frac_e_clipped",
            "ipw_abs_max_raw", "ipw_abs_max_capped", "frac_ipw_capped",
        ]}

        def _quantiles(x: np.ndarray) -> Dict[str, float]:
            if x.size == 0:
                return {"min": np.nan, "max": np.nan, "q01": np.nan, "q05": np.nan, "q95": np.nan, "q99": np.nan}
            return {
                "min": float(np.min(x)),
                "max": float(np.max(x)),
                "q01": float(np.quantile(x, 0.01)),
                "q05": float(np.quantile(x, 0.05)),
                "q95": float(np.quantile(x, 0.95)),
                "q99": float(np.quantile(x, 0.99)),
            }

        kf = KFold(n_splits=cfg.n_splits_dr, shuffle=True, random_state=cfg.seed)
        psi = np.zeros_like(Y, dtype=float)
        for tr, te in kf.split(U):
            U_tr, U_te = U[tr], U[te]
            A_tr, A_te = A[tr], A[te]
            Y_tr, Y_te = Y[tr], Y[te]

            if cfg.standardize_nuisance:
                scaler_u = StandardScaler().fit(U_tr)
                U_tr_s = scaler_u.transform(U_tr)
                U_te_s = scaler_u.transform(U_te)
            else:
                U_tr_s, U_te_s = U_tr, U_te
            if np.unique(A_tr).size < 2:
                e_raw = np.full_like(A_te, float(np.mean(A_tr)) if len(A_tr) else 0.5, dtype=float)
            else:
                prop = LogisticRegression(max_iter=2000, solver="lbfgs")
                prop.fit(U_tr_s, A_tr)
                e_raw = prop.predict_proba(U_te_s)[:, 1]
            clip_use = float(cfg.clip_prop)
            cap_use = float(cfg.ipw_cap) if cfg.ipw_cap is not None else np.nan
            e_hat = np.clip(e_raw, clip_use, 1 - clip_use)

            qs = _quantiles(e_raw)
            for k, v in qs.items():
                stats[f"e_{k}"].append(v)
            stats["clip_used"].append(clip_use)
            stats["cap_used"].append(cap_use)
            stats["frac_e_clipped"].append(float(np.mean((e_raw < clip_use) | (e_raw > 1 - clip_use))))

            # Arm-specific outcome models to accommodate heterogeneous effects
            if np.any(A_tr == 1):
                m1_model: Ridge | DummyRegressor = Ridge(alpha=cfg.ridge_alpha)
                m1_model.fit(U_tr_s[A_tr == 1], Y_tr[A_tr == 1])
            else:
                m1_model = DummyRegressor(strategy="constant", constant=float(np.mean(Y_tr)))
                m1_model.fit(np.zeros((1, U_tr.shape[1])), [0.0])

            if np.any(A_tr == 0):
                m0_model: Ridge | DummyRegressor = Ridge(alpha=cfg.ridge_alpha)
                m0_model.fit(U_tr_s[A_tr == 0], Y_tr[A_tr == 0])
            else:
                m0_model = DummyRegressor(strategy="constant", constant=float(np.mean(Y_tr)))
                m0_model.fit(np.zeros((1, U_tr.shape[1])), [0.0])

            m1 = m1_model.predict(U_te_s)
            m0 = m0_model.predict(U_te_s)
            m_hat = np.where(A_te == 1, m1, m0)

            eps_raw = 1e-6
            w_raw = (A_te - np.clip(e_raw, eps_raw, 1 - eps_raw)) / (
                np.clip(e_raw, eps_raw, 1 - eps_raw)
                * (1 - np.clip(e_raw, eps_raw, 1 - eps_raw))
            )
            stats["ipw_abs_max_raw"].append(float(np.max(np.abs(w_raw))) if w_raw.size else np.nan)

            w = (A_te - e_hat) / (e_hat * (1 - e_hat))
            if cfg.ipw_cap and cfg.ipw_cap > 0:
                w = np.clip(w, -float(cfg.ipw_cap), float(cfg.ipw_cap))
                stats["ipw_abs_max_capped"].append(float(np.max(np.abs(w))) if w.size else np.nan)
                # use absolute raw weights to count how many observations needed clipping
                stats["frac_ipw_capped"].append(
                    float(np.mean(np.abs(w_raw) >= float(cfg.ipw_cap))) if w_raw.size else np.nan
                )
            else:
                stats["ipw_abs_max_capped"].append(np.nan)
                stats["frac_ipw_capped"].append(np.nan)

            psi[te] = m1 - m0 + w * (Y_te - m_hat)
        if stats and hasattr(self, "training_diagnostics"):
            agg = {k: float(np.nanmean(v)) for k, v in stats.items() if v}
            self.training_diagnostics.update({f"dr_{k}": v for k, v in agg.items()})
        return float(np.mean(psi))

    def estimate_ate(self, V_all: np.ndarray, A: np.ndarray, Y: np.ndarray) -> float:
        U = self.get_latent(V_all)
        return self._dr_ate(U, A, Y)

    def get_training_diagnostics(self) -> Dict[str, float]:
        return dict(self.training_diagnostics)


class IVAPCIv33TheoryHierRADREstimator(IVAPCIv33TheoryHierEstimator):
    """Same encoder, plus RADR-style calibration with fold-specific nuisances."""

    # ---- helpers for RADR features ----
    def _destandardize_y(self, y_std: np.ndarray) -> np.ndarray:
        scale = float(self._y_std.squeeze()) if hasattr(self, "_y_std") else 1.0
        mean = float(self._y_mean.squeeze()) if hasattr(self, "_y_mean") else 0.0
        return y_std * scale + mean

    def _head_features(
        self, U: np.ndarray, A: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Propensity/outcome head predictions for RADR nuisance models."""

        tx, tw, tz, tn = self._split_latent(U)
        A = np.asarray(A, dtype=np.float32).reshape(-1)

        tx_t = torch.from_numpy(tx.astype(np.float32)).to(self.device)
        tw_t = torch.from_numpy(tw.astype(np.float32)).to(self.device)
        tz_t = torch.from_numpy(tz.astype(np.float32)).to(self.device)
        A_t = torch.from_numpy(A).to(self.device)

        self.a_head.eval()
        self.y_head.eval()
        with torch.no_grad():
            s_logits = self.a_head(torch.cat([tx_t, tz_t], dim=1)).cpu().numpy()
            t_obs_std = self.y_head(torch.cat([tx_t, tw_t], dim=1), A_t).cpu().numpy()
            ones = torch.ones_like(A_t)
            zeros = torch.zeros_like(A_t)
            t1_std = self.y_head(torch.cat([tx_t, tw_t], dim=1), ones).cpu().numpy()
            t0_std = self.y_head(torch.cat([tx_t, tw_t], dim=1), zeros).cpu().numpy()

        t_obs = self._destandardize_y(t_obs_std)
        t1 = self._destandardize_y(t1_std)
        t0 = self._destandardize_y(t0_std)
        return s_logits, t_obs, t1, t0

    def _split_latent(self, U: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
        cfg = self.config
        dx, dw, dz = cfg.latent_x_dim, cfg.latent_w_dim, cfg.latent_z_dim
        tx = U[:, :dx]
        tw = U[:, dx : dx + dw]
        tz = U[:, dx + dw : dx + dw + dz]
        tn: Optional[np.ndarray] = None
        if cfg.use_noise_in_latent and U.shape[1] > dx + dw + dz:
            tn = U[:, dx + dw + dz :]
        return tx, tw, tz, tn

    def _dr_ate(self, U: np.ndarray, A: np.ndarray, Y: np.ndarray) -> float:
        cfg = self.config
        U = np.asarray(U)
        A = np.asarray(A).reshape(-1)
        Y = np.asarray(Y).reshape(-1)
        stats: Dict[str, list] = {
            k: []
            for k in [
                "e_min",
                "e_max",
                "e_q01",
                "e_q05",
                "e_q95",
                "e_q99",
                "clip_used",
                "cap_used",
                "overlap_score",
                "frac_e_clipped",
                "ipw_abs_max_raw",
                "ipw_abs_max_postclip_precap",
                "ipw_abs_max_capped",
                "frac_ipw_capped",
                "frac_ipw_capped_raw",
            ]
        }

        def _quantiles(x: np.ndarray) -> Dict[str, float]:
            if x.size == 0:
                return {"min": np.nan, "max": np.nan, "q01": np.nan, "q05": np.nan, "q95": np.nan, "q99": np.nan}
            return {
                "min": float(np.min(x)),
                "max": float(np.max(x)),
                "q01": float(np.quantile(x, 0.01)),
                "q05": float(np.quantile(x, 0.05)),
                "q95": float(np.quantile(x, 0.95)),
                "q99": float(np.quantile(x, 0.99)),
            }

        kf = KFold(n_splits=cfg.n_splits_dr, shuffle=True, random_state=cfg.seed)
        psi = np.zeros_like(Y, dtype=float)
        clip = float(max(cfg.clip_prop_radr, cfg.clip_prop))
        ipw_cap = cfg.ipw_cap_radr if getattr(cfg, "ipw_cap_radr", None) is not None else cfg.ipw_cap

        for tr, te in kf.split(U):
            A_tr, A_te = A[tr], A[te]
            Y_tr, Y_te = Y[tr], Y[te]

            s_tr, t_obs_tr, t1_tr, t0_tr = self._head_features(U[tr], A_tr)
            s_te, t_obs_te, t1_te, t0_te = self._head_features(U[te], A_te)

            tx_tr, tw_tr, tz_tr, tn_tr = self._split_latent(U[tr])
            tx_te, tw_te, tz_te, tn_te = self._split_latent(U[te])

            s_tr_clipped = np.clip(s_tr, -10, 10)
            s_te_clipped = np.clip(s_te, -10, 10)
            prop_feats_tr = [s_tr_clipped.reshape(-1, 1), tx_tr, tz_tr]
            prop_feats_te = [s_te_clipped.reshape(-1, 1), tx_te, tz_te]
            if tn_tr is not None:
                prop_feats_tr.append(tn_tr)
                prop_feats_te.append(tn_te)
            Xp_tr = np.column_stack(prop_feats_tr)
            Xp_te = np.column_stack(prop_feats_te)
            if cfg.standardize_nuisance:
                scaler_prop = StandardScaler().fit(Xp_tr)
                Xp_tr = scaler_prop.transform(Xp_tr)
                Xp_te = scaler_prop.transform(Xp_te)
            if np.unique(A_tr).size < 2:
                e_raw = np.full_like(A_te, float(np.mean(A_tr)) if len(A_tr) else 0.5, dtype=float)
            else:
                prop = LogisticRegression(max_iter=2000, solver="lbfgs")
                prop.fit(Xp_tr, A_tr)
                e_raw = prop.predict_proba(Xp_te)[:, 1]

            qs = _quantiles(e_raw)
            clip_use = clip
            cap_use = ipw_cap
            e_hat = np.clip(e_raw, clip_use, 1 - clip_use)

            stats["clip_used"].append(float(clip_use))
            stats["cap_used"].append(float(cap_use) if cap_use is not None else np.nan)
            eps = max(clip_use, 0.05)
            overlap_score = float(np.mean((e_raw > eps) & (e_raw < 1 - eps))) if e_raw.size else np.nan
            stats["overlap_score"].append(overlap_score)
            stats["frac_e_clipped"].append(float(np.mean((e_raw < clip_use) | (e_raw > 1 - clip_use))))
            for k, v in qs.items():
                stats[f"e_{k}"].append(v)

            def _outcome_features(a_vec, tx_block, tw_block, t_val, tn_block):
                feats = [
                    a_vec.reshape(-1, 1),
                    tx_block,
                    tw_block,
                    t_val.reshape(-1, 1),
                    (a_vec * t_val).reshape(-1, 1),
                    (a_vec[:, None] * tx_block),
                    (a_vec[:, None] * tw_block),
                ]
                if tn_block is not None:
                    feats.append(tn_block)
                    feats.append(a_vec[:, None] * tn_block)
                return np.column_stack(feats)

            Xo_tr = _outcome_features(A_tr, tx_tr, tw_tr, t_obs_tr, tn_tr)
            Xo_te = _outcome_features(A_te, tx_te, tw_te, t_obs_te, tn_te)
            if cfg.standardize_nuisance:
                scaler_out = StandardScaler().fit(Xo_tr)
                Xo_tr = scaler_out.transform(Xo_tr)
                Xo_te = scaler_out.transform(Xo_te)

            if len(A_tr) == 0:
                out_model: Ridge | DummyRegressor = DummyRegressor(strategy="mean")
                out_model.fit(np.zeros((1, Xo_tr.shape[1])), [0.0])
            else:
                out_model = Ridge(alpha=cfg.ridge_alpha)
                out_model.fit(Xo_tr, Y_tr)

            X1 = _outcome_features(np.ones_like(A_te), tx_te, tw_te, t1_te, tn_te)
            X0 = _outcome_features(np.zeros_like(A_te), tx_te, tw_te, t0_te, tn_te)
            if cfg.standardize_nuisance:
                X1 = scaler_out.transform(X1)
                X0 = scaler_out.transform(X0)

            m_hat = out_model.predict(Xo_te)
            m1 = out_model.predict(X1)
            m0 = out_model.predict(X0)

            eps_raw = 1e-6
            e_safe = np.clip(e_raw, eps_raw, 1 - eps_raw)
            w_raw = (A_te - e_safe) / (e_safe * (1 - e_safe))

            stats["ipw_abs_max_raw"].append(float(np.max(np.abs(w_raw))) if w_raw.size else np.nan)
            # post-clip (propensity) but pre-cap weights
            w = (A_te - e_hat) / (e_hat * (1 - e_hat))
            stats["ipw_abs_max_postclip_precap"].append(float(np.max(np.abs(w))) if w.size else np.nan)

            w_capped = np.clip(w, -float(ipw_cap), float(ipw_cap)) if ipw_cap and ipw_cap > 0 else w
            stats["ipw_abs_max_capped"].append(float(np.max(np.abs(w_capped))) if w_capped.size else np.nan)
            if ipw_cap and ipw_cap > 0:
                stats["frac_ipw_capped"].append(
                    float(np.mean(np.abs(w) >= float(ipw_cap))) if w.size else np.nan
                )
                stats["frac_ipw_capped_raw"].append(
                    float(np.mean(np.abs(w_raw) >= float(ipw_cap))) if w_raw.size else np.nan
                )
            else:
                stats["frac_ipw_capped"].append(np.nan)
                stats["frac_ipw_capped_raw"].append(np.nan)

            psi[te] = m1 - m0 + w_capped * (Y_te - m_hat)

        if not hasattr(self, "training_diagnostics"):
            self.training_diagnostics = {}
        if stats:
            agg = {k: float(np.nanmean(v)) for k, v in stats.items() if v}
            self.training_diagnostics.update({f"radr_{k}": v for k, v in agg.items()})

        return float(np.mean(psi))


__all__ = [
    "IVAPCIV33TheoryConfig",
    "IVAPCIv33TheoryHierEstimator",
    "IVAPCIv33TheoryHierRADREstimator",
]
