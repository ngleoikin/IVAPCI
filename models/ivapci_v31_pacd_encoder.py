"""IVAPCI v3.1: proxy-only encoder with causal/noise split, adversary, and p-adic regularizer."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import KFold, train_test_split

from . import BaseCausalEstimator


def _standardize(train: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = train.mean(axis=0, keepdims=True)
    std = train.std(axis=0, keepdims=True)
    std[std == 0] = 1.0
    return (train - mean) / std, mean, std


def _apply_standardize(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (x - mean) / std


class _SharedEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden: Sequence[int]):
        super().__init__()
        layers: Iterable[nn.Module] = []
        last = input_dim
        for h in hidden:
            layers.extend([nn.Linear(last, h), nn.ReLU()])
            last = h
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _LatentHead(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int):
        super().__init__()
        self.mu = nn.Linear(input_dim, latent_dim)
        self.logvar = nn.Linear(input_dim, latent_dim)

    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.mu(h), self.logvar(h)


class _ProxyDecoder(nn.Module):
    def __init__(self, latent_dim: int, output_dim: int, hidden: Sequence[int]):
        super().__init__()
        layers: Iterable[nn.Module] = []
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
        layers: Iterable[nn.Module] = []
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
        layers: Iterable[nn.Module] = []
        last = latent_dim + 1
        for h in hidden:
            layers.extend([nn.Linear(last, h), nn.ReLU()])
            last = h
        layers.append(nn.Linear(last, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([z, a.unsqueeze(-1)], dim=1)).squeeze(-1)


class _Adversary(nn.Module):
    def __init__(self, latent_dim: int, hidden: Sequence[int]):
        super().__init__()
        layers: Iterable[nn.Module] = []
        last = latent_dim
        for h in hidden:
            layers.extend([nn.Linear(last, h), nn.ReLU()])
            last = h
        layers.append(nn.Linear(last, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z).squeeze(-1)


def _kl_loss(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())


def _reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std


def _padic_ultrametric_loss(Uc: torch.Tensor, num_triplets: int = 128) -> torch.Tensor:
    if Uc.shape[0] < 3:
        return torch.tensor(0.0, device=Uc.device)
    n = Uc.shape[0]
    idx = torch.randint(0, n, (num_triplets, 3), device=Uc.device)
    u_i = Uc[idx[:, 0]]
    u_j = Uc[idx[:, 1]]
    u_k = Uc[idx[:, 2]]
    d_ij = torch.norm(u_i - u_j, dim=1)
    d_jk = torch.norm(u_j - u_k, dim=1)
    d_ik = torch.norm(u_i - u_k, dim=1)
    viol = torch.relu(d_ik - torch.maximum(d_ij, d_jk))
    return torch.mean(viol ** 2)


@dataclass
class IVAPCIV31Config:
    latent_c_dim: int = 4
    latent_n_dim: int = 4
    encoder_hidden: Sequence[int] = (128, 64)
    decoder_hidden: Sequence[int] = (64, 64)
    a_hidden: Sequence[int] = (32,)
    y_hidden: Sequence[int] = (64, 32)
    adv_hidden: Sequence[int] = (64,)

    beta_c: float = 1.0
    beta_n: float = 1.0
    lambda_a: float = 0.2
    lambda_y: float = 0.2
    gamma_adv: float = 1.0
    gamma_padic: float = 0.02

    lr_main: float = 1e-3
    lr_adv: float = 1e-3
    batch_size: int = 128
    epochs_pretrain: int = 50
    epochs_main: int = 150
    val_frac: float = 0.1
    early_stopping_patience: int = 15
    early_stopping_min_delta: float = 0.0

    n_splits_dr: int = 2
    seed: int = 42
    device: str = "cpu"


class IVAPCIv31PACDEncoderEstimator(BaseCausalEstimator):
    """IVAPCI v3.1 with causal/noise split, adversary, and p-adic geometry regularization."""

    def __init__(self, config: Optional[IVAPCIV31Config] = None):
        self.config = config or IVAPCIV31Config()
        self.device = torch.device(self.config.device)
        torch.manual_seed(self.config.seed)
        np.random.seed(self.config.seed)
        self._is_fit = False

    # -------------------- public API --------------------
    def fit(self, X_all: np.ndarray, A: np.ndarray, Y: np.ndarray) -> None:
        X_all = np.asarray(X_all, dtype=np.float32)
        A = np.asarray(A, dtype=np.float32).reshape(-1)
        Y = np.asarray(Y, dtype=np.float32).reshape(-1)
        n = X_all.shape[0]
        cfg = self.config

        train_idx, val_idx = train_test_split(
            np.arange(n),
            test_size=cfg.val_frac,
            random_state=cfg.seed,
            stratify=A if len(np.unique(A)) > 1 else None,
        )

        X_train, X_val = X_all[train_idx], X_all[val_idx]
        A_train, A_val = A[train_idx], A[val_idx]
        Y_train, Y_val = Y[train_idx], Y[val_idx]

        X_train_std, self._x_mean, self._x_std = _standardize(X_train)
        Y_train_std, self._y_mean, self._y_std = _standardize(Y_train.reshape(-1, 1))
        Y_train_std = Y_train_std.squeeze(1)
        X_val_std = _apply_standardize(X_val, self._x_mean, self._x_std)
        Y_val_std = _apply_standardize(Y_val.reshape(-1, 1), self._y_mean, self._y_std).squeeze(1)

        d_all = X_train_std.shape[1]
        total_latent = cfg.latent_c_dim + cfg.latent_n_dim
        self.shared = _SharedEncoder(d_all, cfg.encoder_hidden).to(self.device)
        self.c_head = _LatentHead(cfg.encoder_hidden[-1] if cfg.encoder_hidden else d_all, cfg.latent_c_dim).to(
            self.device
        )
        self.n_head = _LatentHead(cfg.encoder_hidden[-1] if cfg.encoder_hidden else d_all, cfg.latent_n_dim).to(
            self.device
        )
        self.proxy_decoder = _ProxyDecoder(total_latent, d_all, cfg.decoder_hidden).to(self.device)
        self.a_head = _AHead(cfg.latent_c_dim, cfg.a_hidden).to(self.device)
        self.y_head = _YHead(cfg.latent_c_dim, cfg.y_hidden).to(self.device)
        self.adv_head = _Adversary(cfg.latent_n_dim, cfg.adv_hidden).to(self.device)

        main_params = (
            list(self.shared.parameters())
            + list(self.c_head.parameters())
            + list(self.n_head.parameters())
            + list(self.proxy_decoder.parameters())
            + list(self.a_head.parameters())
            + list(self.y_head.parameters())
        )
        self.main_opt = torch.optim.Adam(main_params, lr=cfg.lr_main)
        self.adv_opt = torch.optim.Adam(self.adv_head.parameters(), lr=cfg.lr_adv)
        bce_loss = nn.BCEWithLogitsLoss()
        mse_loss = nn.MSELoss()

        train_dataset = torch.utils.data.TensorDataset(
            torch.from_numpy(X_train_std),
            torch.from_numpy(A_train),
            torch.from_numpy(Y_train_std),
        )
        val_dataset = torch.utils.data.TensorDataset(
            torch.from_numpy(X_val_std),
            torch.from_numpy(A_val),
            torch.from_numpy(Y_val_std),
        )
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False)

        def forward_batch(xb: torch.Tensor, ab: torch.Tensor, yb: torch.Tensor):
            h = self.shared(xb)
            mu_c, logvar_c = self.c_head(h)
            mu_n, logvar_n = self.n_head(h)
            z_c = _reparameterize(mu_c, logvar_c)
            z_n = _reparameterize(mu_n, logvar_n)
            z = torch.cat([z_c, z_n], dim=1)

            x_recon = self.proxy_decoder(z)
            logits_a = self.a_head(z_c)
            y_pred = self.y_head(z_c, ab)
            adv_logits = self.adv_head(z_n)

            recon_loss = mse_loss(x_recon, xb)
            kl_c = _kl_loss(mu_c, logvar_c)
            kl_n = _kl_loss(mu_n, logvar_n)
            a_loss = bce_loss(logits_a, ab)
            y_loss = mse_loss(y_pred, yb)
            adv_loss = bce_loss(adv_logits, ab)
            padic_loss = _padic_ultrametric_loss(z_c)
            return recon_loss, kl_c, kl_n, a_loss, y_loss, adv_loss, padic_loss

        def run_pretrain_epoch() -> float:
            self.shared.train(); self.c_head.train(); self.n_head.train(); self.proxy_decoder.train()
            total = 0.0
            count = 0
            for xb, _, _ in train_loader:
                xb = xb.to(self.device)
                recon, kl_c, kl_n, _, _, _, _ = forward_batch(xb, torch.zeros_like(xb[:, 0]), torch.zeros_like(xb[:, 0]))
                loss = recon + cfg.beta_c * kl_c + cfg.beta_n * kl_n
                self.main_opt.zero_grad()
                loss.backward()
                self.main_opt.step()
                total += float(loss.item()) * xb.size(0)
                count += xb.size(0)
            return total / max(count, 1)

        def compute_main_loss(xb: torch.Tensor, ab: torch.Tensor, yb: torch.Tensor):
            recon, kl_c, kl_n, a_loss, y_loss, adv_loss, padic_loss = forward_batch(xb, ab, yb)
            loss = (
                recon
                + cfg.beta_c * kl_c
                + cfg.beta_n * kl_n
                + cfg.lambda_a * a_loss
                + cfg.lambda_y * y_loss
                + cfg.gamma_adv * (-adv_loss)
                + cfg.gamma_padic * padic_loss
            )
            return loss, adv_loss

        # Stage 0: proxy-only pretrain
        for _ in range(cfg.epochs_pretrain):
            run_pretrain_epoch()

        best_val = float("inf")
        patience = 0
        best_state = None

        # Stage 1: main training with adversary + padic
        for _ in range(cfg.epochs_main):
            self.shared.train(); self.c_head.train(); self.n_head.train(); self.proxy_decoder.train(); self.a_head.train(); self.y_head.train(); self.adv_head.train()
            for xb, ab, yb in train_loader:
                xb = xb.to(self.device)
                ab = ab.to(self.device)
                yb = yb.to(self.device)

                # adversary update (detach encoder)
                with torch.no_grad():
                    h = self.shared(xb)
                    mu_c, logvar_c = self.c_head(h)
                    mu_n, logvar_n = self.n_head(h)
                    z_n = _reparameterize(mu_n, logvar_n)
                adv_logits = self.adv_head(z_n.detach())
                adv_loss = bce_loss(adv_logits, ab)
                self.adv_opt.zero_grad()
                adv_loss.backward()
                self.adv_opt.step()

                # main update
                loss, adv_loss_main = compute_main_loss(xb, ab, yb)
                self.main_opt.zero_grad()
                loss.backward()
                self.main_opt.step()

            # validation
            self.shared.eval(); self.c_head.eval(); self.n_head.eval(); self.proxy_decoder.eval(); self.a_head.eval(); self.y_head.eval(); self.adv_head.eval()
            val_losses = []
            with torch.no_grad():
                for xb, ab, yb in val_loader:
                    xb = xb.to(self.device)
                    ab = ab.to(self.device)
                    yb = yb.to(self.device)
                    val_loss, _ = compute_main_loss(xb, ab, yb)
                    val_losses.append(float(val_loss.item()))
            mean_val = float(np.mean(val_losses)) if val_losses else float("inf")

            if mean_val + cfg.early_stopping_min_delta < best_val:
                best_val = mean_val
                patience = 0
                best_state = {
                    "shared": self.shared.state_dict(),
                    "c_head": self.c_head.state_dict(),
                    "n_head": self.n_head.state_dict(),
                    "decoder": self.proxy_decoder.state_dict(),
                    "a_head": self.a_head.state_dict(),
                    "y_head": self.y_head.state_dict(),
                    "adv_head": self.adv_head.state_dict(),
                }
            else:
                patience += 1
                if patience >= cfg.early_stopping_patience:
                    break

        if best_state is None:
            best_state = {
                "shared": self.shared.state_dict(),
                "c_head": self.c_head.state_dict(),
                "n_head": self.n_head.state_dict(),
                "decoder": self.proxy_decoder.state_dict(),
                "a_head": self.a_head.state_dict(),
                "y_head": self.y_head.state_dict(),
                "adv_head": self.adv_head.state_dict(),
            }

        self.shared.load_state_dict(best_state["shared"])
        self.c_head.load_state_dict(best_state["c_head"])
        self.n_head.load_state_dict(best_state["n_head"])
        self.proxy_decoder.load_state_dict(best_state["decoder"])
        self.a_head.load_state_dict(best_state["a_head"])
        self.y_head.load_state_dict(best_state["y_head"])
        self.adv_head.load_state_dict(best_state["adv_head"])

        self._is_fit = True

    def get_latent(self, X_all: np.ndarray) -> np.ndarray:
        if not self._is_fit:
            raise RuntimeError("Estimator must be fit before get_latent.")
        X_all = np.asarray(X_all, dtype=np.float32)
        X_std = _apply_standardize(X_all, self._x_mean, self._x_std)
        X_t = torch.from_numpy(X_std).to(self.device)
        self.shared.eval(); self.c_head.eval()
        with torch.no_grad():
            h = self.shared(X_t)
            mu_c, _ = self.c_head(h)
        return mu_c.cpu().numpy()

    def estimate_ate(self, X_all: np.ndarray, A: np.ndarray, Y: np.ndarray) -> float:
        if not self._is_fit:
            raise RuntimeError("Estimator must be fit before estimating ATE.")
        U_c = self.get_latent(X_all)
        return self._dr_ate_glm(U_c, A, Y)

    # -------------------- DR / DML --------------------
    def _dr_ate_glm(self, U: np.ndarray, A: np.ndarray, Y: np.ndarray) -> float:
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
                # fallback to single-model DR using full data
                clf_full = LogisticRegression(max_iter=2000, solver="lbfgs")
                clf_full.fit(U, A)
                e_hat = clf_full.predict_proba(U_te)[:, 1].clip(1e-3, 1 - 1e-3)
                reg_full = LinearRegression().fit(np.column_stack([A, U]), Y)
                m1 = reg_full.predict(np.column_stack([np.ones_like(A_te), U_te]))
                m0 = reg_full.predict(np.column_stack([np.zeros_like(A_te), U_te]))
                psi[test_idx] = m1 - m0 + A_te * (Y_te - m1) / e_hat - (1 - A_te) * (Y_te - m0) / (1 - e_hat)
                continue

            prop = LogisticRegression(max_iter=2000, solver="lbfgs")
            prop.fit(U_tr, A_tr)
            e_hat = prop.predict_proba(U_te)[:, 1].clip(1e-3, 1 - 1e-3)

            reg = LinearRegression().fit(np.column_stack([A_tr, U_tr]), Y_tr)
            m_hat = reg.predict(np.column_stack([A_te, U_te]))
            m1 = reg.predict(np.column_stack([np.ones_like(A_te), U_te]))
            m0 = reg.predict(np.column_stack([np.zeros_like(A_te), U_te]))

            psi[test_idx] = (
                m1
                - m0
                + A_te * (Y_te - m1) / e_hat
                - (1 - A_te) * (Y_te - m0) / (1 - e_hat)
            )

        return float(np.mean(psi))


__all__ = ["IVAPCIV31Config", "IVAPCIv31PACDEncoderEstimator"]
