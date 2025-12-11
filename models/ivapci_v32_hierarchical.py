"""IVAPCI v3.2 hierarchical encoder with group-specific heads and layered adversaries."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import KFold, train_test_split

from . import BaseCausalEstimator


# ----------------- basic helpers -----------------


class _GroupEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden: Sequence[int],
        latent_dim: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        layers: list[nn.Module] = []
        last = input_dim
        for h in hidden:
            layers.append(nn.Linear(last, h))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            last = h
        layers.append(nn.Linear(last, latent_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _ProxyDecoder(nn.Module):
    def __init__(self, latent_dim: int, output_dim: int, hidden: Sequence[int]):
        super().__init__()
        layers: list[nn.Module] = []
        last = latent_dim
        for h in hidden:
            layers.extend([nn.Linear(last, h), nn.ReLU()])
            last = h
        layers.append(nn.Linear(last, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class _AHead(nn.Module):
    def __init__(self, latent_dim: int, hidden: Sequence[int]):
        super().__init__()
        layers: list[nn.Module] = []
        last = latent_dim
        for h in hidden:
            layers.extend([nn.Linear(last, h), nn.ReLU()])
            last = h
        layers.append(nn.Linear(last, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z).squeeze(-1)


class _YHead(nn.Module):
    def __init__(self, latent_dim: int, hidden: Sequence[int]):
        super().__init__()
        layers: list[nn.Module] = []
        last = latent_dim + 1  # concat with A
        for h in hidden:
            layers.extend([nn.Linear(last, h), nn.ReLU()])
            last = h
        layers.append(nn.Linear(last, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([z, a.unsqueeze(-1)], dim=1)).squeeze(-1)


class _Adversary(nn.Module):
    """Simple MLP adversary (scalar logit / prediction)."""

    def __init__(self, latent_dim: int, hidden: Sequence[int]):
        super().__init__()
        layers: list[nn.Module] = []
        last = latent_dim
        for h in hidden:
            layers.extend([nn.Linear(last, h), nn.ReLU()])
            last = h
        layers.append(nn.Linear(last, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z).squeeze(-1)


class _ScalarRegHead(nn.Module):
    """Simple scalar regression head (used for consistency)."""

    def __init__(self, latent_dim: int, hidden: Sequence[int]):
        super().__init__()
        layers: list[nn.Module] = []
        last = latent_dim
        for h in hidden:
            layers.extend([nn.Linear(last, h), nn.ReLU()])
            last = h
        layers.append(nn.Linear(last, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z).squeeze(-1)


def _standardize(train: np.ndarray, min_std: float = 1e-2):
    mean = train.mean(axis=0, keepdims=True)
    std = train.std(axis=0, keepdims=True)
    std = np.maximum(std, min_std)
    return (train - mean) / std, mean, std


def _apply_standardize(x: np.ndarray, mean: np.ndarray, std: np.ndarray):
    return (x - mean) / std


def _padic_ultrametric_loss(Uc: torch.Tensor, num_triplets: int = 128) -> torch.Tensor:
    """Ultrametric-style regularizer on U_c using random triplets."""
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
    return torch.mean(viol**2)


def _conditional_orthogonal_penalty(
    blocks: list[torch.Tensor], U: torch.Tensor
) -> torch.Tensor:
    """Approximate I(T_i; T_j | U) ≈ 0 using residual cross-covariance.

    For each block pair (T_i, T_j), regress them on U via least squares,
    take residuals, and penalize their empirical cross-covariance.
    """

    if len(blocks) < 2:
        return torch.zeros((), device=U.device)

    penalty = torch.zeros((), device=U.device)
    for i in range(len(blocks)):
        for j in range(i + 1, len(blocks)):
            Ti = blocks[i]
            Tj = blocks[j]

            sol_i = torch.linalg.lstsq(U, Ti).solution
            sol_j = torch.linalg.lstsq(U, Tj).solution

            resid_i = Ti - U @ sol_i
            resid_j = Tj - U @ sol_j

            ri = resid_i - resid_i.mean(dim=0, keepdim=True)
            rj = resid_j - resid_j.mean(dim=0, keepdim=True)
            cov = ri.T @ rj / ri.shape[0]
            penalty = penalty + (cov**2).mean()

    return penalty


# ----------------- config -----------------


@dataclass
class IVAPCIV32HierConfig:
    """Configuration for hierarchical IVAPCI v3.2 encoders.

    x_dim, w_dim, z_dim specify the column partition of X_all = [X | W | Z].
    """

    x_dim: int
    w_dim: int
    z_dim: int

    # latent sizes
    latent_x_dim: int = 4
    latent_w_dim: int = 4
    latent_z_dim: int = 4
    latent_n_dim: int = 4

    # encoder architectures
    enc_x_hidden: Sequence[int] = (64, 32)      # balanced
    enc_w_hidden: Sequence[int] = (128, 64)    # outcome-oriented, deeper
    enc_z_hidden: Sequence[int] = (32,)        # smaller
    enc_z_dropout: float = 0.2                 # dropout on treatment block
    noise_hidden: Sequence[int] = (64, 32)     # noise encoder over all proxies

    # decoder + heads
    decoder_hidden: Sequence[int] = (64, 64)
    a_hidden: Sequence[int] = (32,)
    y_hidden: Sequence[int] = (64, 32)

    # adversaries
    adv_w_hidden: Sequence[int] = (32,)
    adv_z_hidden: Sequence[int] = (32,)
    adv_n_hidden: Sequence[int] = (64,)

    # consistency heads (T_W -> Y, T_Z -> A)
    consistency_hidden: Sequence[int] = (32,)
    lambda_consistency: float = 0.05

    # loss weights
    lambda_recon: float = 1.0
    lambda_a: float = 0.2
    lambda_y: float = 0.2
    lambda_ortho: float = 0.01
    gamma_adv_w: float = 0.1
    gamma_adv_z: float = 0.1
    gamma_adv_n: float = 0.1
    gamma_padic: float = 1e-3

    # optimization
    lr_main: float = 1e-3
    lr_adv: float = 1e-3
    batch_size: int = 128
    epochs_pretrain: int = 30
    epochs_main: int = 120
    val_frac: float = 0.1
    early_stopping_patience: int = 15
    early_stopping_min_delta: float = 0.0

    # DR / DML
    n_splits_dr: int = 2

    # misc
    seed: int = 42
    device: str = "cpu"


# ----------------- base hierarchical encoder + DR -----------------


class IVAPCIv32HierEncoderEstimator(BaseCausalEstimator):
    """Hierarchical IVAPCI encoder (X/W/Z blocks + noise) + simple DR on U_c."""

    def __init__(self, config: Optional[IVAPCIV32HierConfig] = None):
        if config is None:
            raise ValueError("IVAPCIV32HierConfig with x_dim/w_dim/z_dim must be provided.")
        self.config = config
        self.device = torch.device(self.config.device)
        torch.manual_seed(self.config.seed)
        np.random.seed(self.config.seed)
        self._is_fit = False
        # training/theory diagnostics cache
        self.training_diagnostics: dict = {}

    # -------------- training --------------

    def fit(self, X_all: np.ndarray, A: np.ndarray, Y: np.ndarray) -> None:
        X_all = np.asarray(X_all, dtype=np.float32)
        A = np.asarray(A, dtype=np.float32).reshape(-1)
        Y = np.asarray(Y, dtype=np.float32).reshape(-1)

        n, d_all = X_all.shape
        cfg = self.config
        assert cfg.x_dim + cfg.w_dim + cfg.z_dim == d_all, "X_all 列数必须等于 x_dim + w_dim + z_dim"

        # split train / val
        train_idx, val_idx = train_test_split(
            np.arange(n),
            test_size=cfg.val_frac,
            random_state=cfg.seed,
            stratify=A if np.unique(A).size > 1 else None,
        )
        X_train, X_val = X_all[train_idx], X_all[val_idx]
        A_train, A_val = A[train_idx], A[val_idx]
        Y_train, Y_val = Y[train_idx], Y[val_idx]

        # standardize X, Y
        X_train_std, self._x_mean, self._x_std = _standardize(X_train)
        Y_train_std, self._y_mean, self._y_std = _standardize(Y_train.reshape(-1, 1))
        Y_train_std = Y_train_std.squeeze(1)
        X_val_std = _apply_standardize(X_val, self._x_mean, self._x_std)
        Y_val_std = _apply_standardize(Y_val.reshape(-1, 1), self._y_mean, self._y_std).squeeze(1)

        def split_blocks(x: torch.Tensor):
            x_part = x[:, :cfg.x_dim]
            w_part = x[:, cfg.x_dim : cfg.x_dim + cfg.w_dim]
            z_part = x[:, cfg.x_dim + cfg.w_dim :]
            return x_part, w_part, z_part

        d_x, d_w, d_z = cfg.x_dim, cfg.w_dim, cfg.z_dim
        total_c_dim = cfg.latent_x_dim + cfg.latent_w_dim + cfg.latent_z_dim
        total_latent = total_c_dim + cfg.latent_n_dim

        # build networks
        self.enc_x = _GroupEncoder(d_x, cfg.enc_x_hidden, cfg.latent_x_dim, dropout=0.0).to(self.device)
        self.enc_w = _GroupEncoder(d_w, cfg.enc_w_hidden, cfg.latent_w_dim, dropout=0.0).to(self.device)
        self.enc_z = _GroupEncoder(d_z, cfg.enc_z_hidden, cfg.latent_z_dim, dropout=cfg.enc_z_dropout).to(
            self.device
        )
        self.enc_n = _GroupEncoder(d_all, cfg.noise_hidden, cfg.latent_n_dim, dropout=0.0).to(self.device)

        self.proxy_decoder = _ProxyDecoder(total_latent, d_all, cfg.decoder_hidden).to(self.device)
        self.a_head = _AHead(total_c_dim, cfg.a_hidden).to(self.device)
        self.y_head = _YHead(total_c_dim, cfg.y_hidden).to(self.device)
        self.cons_y_from_w = _ScalarRegHead(cfg.latent_w_dim, cfg.consistency_hidden).to(self.device)
        self.cons_a_from_z = _AHead(cfg.latent_z_dim, cfg.consistency_hidden).to(self.device)

        self.adv_w = _Adversary(cfg.latent_w_dim, cfg.adv_w_hidden).to(self.device)
        self.adv_z = _Adversary(cfg.latent_z_dim, cfg.adv_z_hidden).to(self.device)
        self.adv_n = _Adversary(cfg.latent_n_dim, cfg.adv_n_hidden).to(self.device)

        main_params = (
            list(self.enc_x.parameters())
            + list(self.enc_w.parameters())
            + list(self.enc_z.parameters())
            + list(self.enc_n.parameters())
            + list(self.proxy_decoder.parameters())
            + list(self.a_head.parameters())
            + list(self.y_head.parameters())
            + list(self.cons_y_from_w.parameters())
            + list(self.cons_a_from_z.parameters())
        )
        self.main_opt = torch.optim.Adam(main_params, lr=cfg.lr_main)
        self.adv_opt_w = torch.optim.Adam(self.adv_w.parameters(), lr=cfg.lr_adv)
        self.adv_opt_z = torch.optim.Adam(self.adv_z.parameters(), lr=cfg.lr_adv)
        self.adv_opt_n = torch.optim.Adam(self.adv_n.parameters(), lr=cfg.lr_adv)

        bce_loss = nn.BCEWithLogitsLoss()
        mse_loss = nn.MSELoss()

        train_dataset = torch.utils.data.TensorDataset(
            torch.from_numpy(X_train_std), torch.from_numpy(A_train), torch.from_numpy(Y_train_std)
        )
        val_dataset = torch.utils.data.TensorDataset(
            torch.from_numpy(X_val_std), torch.from_numpy(A_val), torch.from_numpy(Y_val_std)
        )
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False)

        def forward_blocks(
            xb: torch.Tensor, ab: torch.Tensor, yb: torch.Tensor
        ) -> tuple[torch.Tensor, ...]:
            x_part, w_part, z_part = split_blocks(xb)
            tx = self.enc_x(x_part)
            tw = self.enc_w(w_part)
            tz = self.enc_z(z_part)
            tn = self.enc_n(xb)

            Uc = torch.cat([tx, tw, tz], dim=1)
            z_all = torch.cat([Uc, tn], dim=1)

            x_recon = self.proxy_decoder(z_all)
            logits_a = self.a_head(Uc)
            y_pred = self.y_head(Uc, ab)

            adv_w_logits = self.adv_w(tw)
            adv_z_pred = self.adv_z(tz)
            adv_n_logits = self.adv_n(tn)

            ortho = _conditional_orthogonal_penalty([tx, tw, tz], Uc)
            padic = _padic_ultrametric_loss(Uc)

            return (
                x_recon,
                logits_a,
                y_pred,
                adv_w_logits,
                adv_z_pred,
                adv_n_logits,
                ortho,
                padic,
                tx,
                tw,
                tz,
                tn,
                Uc,
            )

        # ---------- Stage 0: reconstruction + orthogonality + p-adic ----------

        for _ in range(cfg.epochs_pretrain):
            self.enc_x.train()
            self.enc_w.train()
            self.enc_z.train()
            self.enc_n.train()
            self.proxy_decoder.train()

            for xb, ab, yb in train_loader:
                xb = xb.to(self.device)
                ab = ab.to(self.device)
                yb = yb.to(self.device)

                x_recon, _, _, _, _, _, ortho, padic, *_ = forward_blocks(xb, ab, yb)
                loss = (
                    cfg.lambda_recon * mse_loss(x_recon, xb)
                    + cfg.lambda_ortho * ortho
                    + cfg.gamma_padic * padic
                )

                self.main_opt.zero_grad()
                loss.backward()
                self.main_opt.step()

        # ---------- Stage 1: full training with layered adversaries ----------

        best_val = float("inf")
        patience = 0
        best_state: dict[str, dict] | None = None

        for _epoch in range(cfg.epochs_main):
            self.enc_x.train()
            self.enc_w.train()
            self.enc_z.train()
            self.enc_n.train()
            self.proxy_decoder.train()
            self.a_head.train()
            self.y_head.train()
            self.cons_y_from_w.train()
            self.cons_a_from_z.train()
            self.adv_w.train()
            self.adv_z.train()
            self.adv_n.train()

            for xb, ab, yb in train_loader:
                xb = xb.to(self.device)
                ab = ab.to(self.device)
                yb = yb.to(self.device)

                # --- update adversaries (encoders detached) ---
                with torch.no_grad():
                    x_part, w_part, z_part = split_blocks(xb)
                    tw = self.enc_w(w_part)
                    tz = self.enc_z(z_part)
                    tn = self.enc_n(xb)

                # adv_w(T_W) vs A
                self.adv_opt_w.zero_grad()
                adv_w_logits_det = self.adv_w(tw.detach())
                loss_adv_w = bce_loss(adv_w_logits_det, ab)
                loss_adv_w.backward()
                self.adv_opt_w.step()

                # adv_z(T_Z) vs Y
                self.adv_opt_z.zero_grad()
                adv_z_pred_det = self.adv_z(tz.detach())
                loss_adv_z = mse_loss(adv_z_pred_det, yb)
                loss_adv_z.backward()
                self.adv_opt_z.step()

                # adv_n(T_noise) vs A
                self.adv_opt_n.zero_grad()
                adv_n_logits_det = self.adv_n(tn.detach())
                loss_adv_n = bce_loss(adv_n_logits_det, ab)
                loss_adv_n.backward()
                self.adv_opt_n.step()

                # --- main update (encoders + decoder + heads) ---
                (
                    x_recon,
                    logits_a,
                    y_pred,
                    adv_w_logits,
                    adv_z_pred,
                    adv_n_logits,
                    ortho,
                    padic,
                    tx,
                    tw,
                    tz,
                    _tn,
                    _Uc,
                ) = forward_blocks(xb, ab, yb)

                y_from_w = self.cons_y_from_w(tw)
                a_from_z_logits = self.cons_a_from_z(tz)
                consistency_loss = mse_loss(y_from_w, yb) + bce_loss(a_from_z_logits, ab)

                loss_main = (
                    cfg.lambda_recon * mse_loss(x_recon, xb)
                    + cfg.lambda_a * bce_loss(logits_a, ab)
                    + cfg.lambda_y * mse_loss(y_pred, yb)
                    + cfg.lambda_ortho * ortho
                    + cfg.gamma_padic * padic
                    - cfg.gamma_adv_w * bce_loss(adv_w_logits, ab)
                    - cfg.gamma_adv_z * mse_loss(adv_z_pred, yb)
                    - cfg.gamma_adv_n * bce_loss(adv_n_logits, ab)
                    + cfg.lambda_consistency * consistency_loss
                )

                self.main_opt.zero_grad()
                loss_main.backward()
                self.main_opt.step()

            # ---------- validation (without adversaries) ----------
            self.enc_x.eval()
            self.enc_w.eval()
            self.enc_z.eval()
            self.enc_n.eval()
            self.proxy_decoder.eval()
            self.a_head.eval()
            self.y_head.eval()
            self.cons_y_from_w.eval()
            self.cons_a_from_z.eval()

            val_losses: list[float] = []
            with torch.no_grad():
                for xb, ab, yb in val_loader:
                    xb = xb.to(self.device)
                    ab = ab.to(self.device)
                    yb = yb.to(self.device)

                    (
                        x_recon,
                        logits_a,
                        y_pred,
                        _advw,
                        _advz,
                        _advn,
                        ortho,
                        padic,
                        *_rest,
                    ) = forward_blocks(xb, ab, yb)
                    loss_val = (
                        cfg.lambda_recon * mse_loss(x_recon, xb)
                        + cfg.lambda_a * bce_loss(logits_a, ab)
                        + cfg.lambda_y * mse_loss(y_pred, yb)
                        + cfg.lambda_ortho * ortho
                        + cfg.gamma_padic * padic
                    )
                    val_losses.append(float(loss_val.item()))
            mean_val = float(np.mean(val_losses)) if val_losses else float("inf")

            if mean_val + cfg.early_stopping_min_delta < best_val:
                best_val = mean_val
                patience = 0
                best_state = {
                    "enc_x": self.enc_x.state_dict(),
                    "enc_w": self.enc_w.state_dict(),
                    "enc_z": self.enc_z.state_dict(),
                    "enc_n": self.enc_n.state_dict(),
                    "decoder": self.proxy_decoder.state_dict(),
                    "a_head": self.a_head.state_dict(),
                    "y_head": self.y_head.state_dict(),
                    "cons_y_from_w": self.cons_y_from_w.state_dict(),
                    "cons_a_from_z": self.cons_a_from_z.state_dict(),
                    "adv_w": self.adv_w.state_dict(),
                    "adv_z": self.adv_z.state_dict(),
                    "adv_n": self.adv_n.state_dict(),
                }
            else:
                patience += 1
                if patience >= cfg.early_stopping_patience:
                    break

        if best_state is None:
            best_state = {
                "enc_x": self.enc_x.state_dict(),
                "enc_w": self.enc_w.state_dict(),
                "enc_z": self.enc_z.state_dict(),
                "enc_n": self.enc_n.state_dict(),
                "decoder": self.proxy_decoder.state_dict(),
                "a_head": self.a_head.state_dict(),
                "y_head": self.y_head.state_dict(),
                "cons_y_from_w": self.cons_y_from_w.state_dict(),
                "cons_a_from_z": self.cons_a_from_z.state_dict(),
                "adv_w": self.adv_w.state_dict(),
                "adv_z": self.adv_z.state_dict(),
                "adv_n": self.adv_n.state_dict(),
            }

        self.enc_x.load_state_dict(best_state["enc_x"])
        self.enc_w.load_state_dict(best_state["enc_w"])
        self.enc_z.load_state_dict(best_state["enc_z"])
        self.enc_n.load_state_dict(best_state["enc_n"])
        self.proxy_decoder.load_state_dict(best_state["decoder"])
        self.a_head.load_state_dict(best_state["a_head"])
        self.y_head.load_state_dict(best_state["y_head"])
        self.cons_y_from_w.load_state_dict(best_state["cons_y_from_w"])
        self.cons_a_from_z.load_state_dict(best_state["cons_a_from_z"])
        self.adv_w.load_state_dict(best_state["adv_w"])
        self.adv_z.load_state_dict(best_state["adv_z"])
        self.adv_n.load_state_dict(best_state["adv_n"])

        self._is_fit = True

        # post-training diagnostics (best effort; non-fatal)
        try:
            from .ivapci_theory_diagnostics import TheoremComplianceDiagnostics

            diag = TheoremComplianceDiagnostics(self)
            self.training_diagnostics = diag.run_all_diagnostics(
                X_all, A, Y, n_recon_features=min(10, X_all.shape[1])
            )
        except Exception as exc:  # pragma: no cover - diagnostics best-effort
            self.training_diagnostics = {"diagnostics_error": str(exc)}

    # -------------- latent + DR --------------

    def _split_blocks_np(self, X_std: np.ndarray):
        cfg = self.config
        x_part = X_std[:, : cfg.x_dim]
        w_part = X_std[:, cfg.x_dim : cfg.x_dim + cfg.w_dim]
        z_part = X_std[:, cfg.x_dim + cfg.w_dim :]
        return x_part, w_part, z_part

    def get_latent(self, X_all: np.ndarray) -> np.ndarray:
        """Return U_c = [T_X | T_W | T_Z]."""
        if not self._is_fit:
            raise RuntimeError("Estimator must be fit before get_latent.")

        X_all = np.asarray(X_all, dtype=np.float32)
        X_std = _apply_standardize(X_all, self._x_mean, self._x_std)
        x_part, w_part, z_part = self._split_blocks_np(X_std)

        self.enc_x.eval()
        self.enc_w.eval()
        self.enc_z.eval()

        with torch.no_grad():
            tx = self.enc_x(torch.from_numpy(x_part).to(self.device))
            tw = self.enc_w(torch.from_numpy(w_part).to(self.device))
            tz = self.enc_z(torch.from_numpy(z_part).to(self.device))
            Uc = torch.cat([tx, tw, tz], dim=1)

        return Uc.cpu().numpy()

    def _dr_ate(self, U: np.ndarray, A: np.ndarray, Y: np.ndarray) -> float:
        """Standard DR on U (no RADR calibration)."""
        cfg = self.config
        U = np.asarray(U)
        A = np.asarray(A).reshape(-1)
        Y = np.asarray(Y).reshape(-1)

        kf = KFold(n_splits=cfg.n_splits_dr, shuffle=True, random_state=cfg.seed)
        psi = np.zeros_like(Y, dtype=float)

        for train_idx, test_idx in kf.split(U):
            U_tr, U_te = U[train_idx], U[test_idx]
            A_tr, A_te = A[train_idx], A[test_idx]
            Y_tr, Y_te = Y[train_idx], Y[test_idx]

            if np.unique(A_tr).size < 2:
                clf_full = LogisticRegression(max_iter=2000, solver="lbfgs")
                clf_full.fit(U, A)
                e_hat = clf_full.predict_proba(U_te)[:, 1].clip(1e-3, 1 - 1e-3)

                reg_full = LinearRegression().fit(np.column_stack([A, U]), Y)
                m1 = reg_full.predict(np.column_stack([np.ones_like(A_te), U_te]))
                m0 = reg_full.predict(np.column_stack([np.zeros_like(A_te), U_te]))
                psi[test_idx] = (
                    m1
                    - m0
                    + A_te * (Y_te - m1) / e_hat
                    - (1 - A_te) * (Y_te - m0) / (1 - e_hat)
                )
                continue

            prop = LogisticRegression(max_iter=2000, solver="lbfgs")
            prop.fit(U_tr, A_tr)
            e_hat = prop.predict_proba(U_te)[:, 1].clip(1e-3, 1 - 1e-3)

            reg = LinearRegression().fit(np.column_stack([A_tr, U_tr]), Y_tr)
            m1 = reg.predict(np.column_stack([np.ones_like(A_te), U_te]))
            m0 = reg.predict(np.column_stack([np.zeros_like(A_te), U_te]))
            m_hat = reg.predict(np.column_stack([A_te, U_te]))

            psi[test_idx] = (
                m1
                - m0
                + A_te * (Y_te - m_hat) / e_hat
                - (1 - A_te) * (Y_te - m_hat) / (1 - e_hat)
            )

        return float(np.mean(psi))

    def estimate_ate(self, X_all: np.ndarray, A: np.ndarray, Y: np.ndarray) -> float:
        if not self._is_fit:
            raise RuntimeError("Estimator must be fit before estimating ATE.")
        U_c = self.get_latent(X_all)
        return self._dr_ate(U_c, A, Y)

    def get_training_diagnostics(self) -> dict:
        """Return theory/training diagnostics captured after fit()."""
        return getattr(self, "training_diagnostics", {})


# ----------------- RADR variant on top of hierarchical encoder -----------------


class IVAPCIv32HierRADREstimator(IVAPCIv32HierEncoderEstimator):
    """Hierarchical encoder + RADR calibration (same spirit as v3.1 RADR)."""

    def _head_features(self, U: np.ndarray, A: np.ndarray):
        U_t = torch.from_numpy(U.astype(np.float32)).to(self.device)
        A_t = torch.from_numpy(A.astype(np.float32)).to(self.device)

        self.a_head.eval()
        self.y_head.eval()

        with torch.no_grad():
            s_logits = self.a_head(U_t).cpu().numpy()
            t_obs_std = self.y_head(U_t, A_t).cpu().numpy()
            ones = torch.ones_like(A_t)
            zeros = torch.zeros_like(A_t)
            t1_std = self.y_head(U_t, ones).cpu().numpy()
            t0_std = self.y_head(U_t, zeros).cpu().numpy()

        y_std = float(self._y_std.squeeze()) if hasattr(self, "_y_std") else 1.0
        y_mean = float(self._y_mean.squeeze()) if hasattr(self, "_y_mean") else 0.0

        def _destandardize(arr: np.ndarray) -> np.ndarray:
            return arr * y_std + y_mean

        return (
            s_logits,
            _destandardize(t_obs_std),
            _destandardize(t1_std),
            _destandardize(t0_std),
        )

    def _dr_ate(self, U: np.ndarray, A: np.ndarray, Y: np.ndarray) -> float:
        """RADR-style calibrated DR using s(U), t(A,U) from heads."""
        cfg = self.config
        U = np.asarray(U)
        A = np.asarray(A).reshape(-1)
        Y = np.asarray(Y).reshape(-1)

        s_logits, t_obs, t1_all, t0_all = self._head_features(U, A)

        kf = KFold(n_splits=cfg.n_splits_dr, shuffle=True, random_state=cfg.seed)
        psi = np.zeros_like(Y, dtype=float)

        for train_idx, test_idx in kf.split(U):
            U_train, U_test = U[train_idx], U[test_idx]
            A_train, A_test = A[train_idx], A[test_idx]
            Y_train, Y_test = Y[train_idx], Y[test_idx]
            s_train, s_test = s_logits[train_idx], s_logits[test_idx]
            t_train, t_test = t_obs[train_idx], t_obs[test_idx]
            t1_test, t0_test = t1_all[test_idx], t0_all[test_idx]

            # Propensity calibration p(A=1 | s(U), U)
            if np.unique(A_train).size < 2:
                e_hat = np.full_like(
                    A_test,
                    fill_value=float(np.mean(A_train)) if len(A_train) else 0.5,
                    dtype=float,
                )
            else:
                prop = LogisticRegression(max_iter=2000, solver="lbfgs")
                X_prop_train = np.column_stack([s_train, U_train])
                X_prop_test = np.column_stack([s_test, U_test])
                prop.fit(X_prop_train, A_train)
                e_hat = prop.predict_proba(X_prop_test)[:, 1]
            e_hat = np.clip(e_hat, 1e-3, 1 - 1e-3)

            # Outcome calibration E[Y | A,U,t(A,U)]
            X_out_train = np.column_stack([A_train, U_train, t_train, A_train * t_train])
            if len(A_train) == 0:
                out_model: LinearRegression | DummyRegressor
                out_model = DummyRegressor(
                    strategy="constant",
                    constant=float(np.mean(Y_train)) if len(Y_train) else 0.0,
                ).fit(np.zeros((1, X_out_train.shape[1])), [0.0])
            else:
                out_model = LinearRegression()
                out_model.fit(X_out_train, Y_train)

            X_out_test = np.column_stack([A_test, U_test, t_test, A_test * t_test])
            m_hat = out_model.predict(X_out_test)

            X1 = np.column_stack([np.ones_like(A_test), U_test, t1_test, t1_test])
            X0 = np.column_stack(
                [np.zeros_like(A_test), U_test, t0_test, np.zeros_like(t0_test)]
            )
            m1_hat = out_model.predict(X1)
            m0_hat = out_model.predict(X0)

            psi[test_idx] = m1_hat - m0_hat + (A_test - e_hat) / (e_hat * (1 - e_hat)) * (
                Y_test - m_hat
            )

        return float(np.mean(psi))


__all__ = [
    "IVAPCIV32HierConfig",
    "IVAPCIv32HierEncoderEstimator",
    "IVAPCIv32HierRADREstimator",
]
