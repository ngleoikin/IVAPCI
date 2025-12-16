"""IVAPCI-Gold: proxy-only VAE representation + cross-fitted GLM-DR ATE."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import KFold, train_test_split

from . import BaseCausalEstimator


# -------------------- 标准化工具 -------------------- #

def _standardize(train: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = train.mean(axis=0, keepdims=True)
    std = train.std(axis=0, keepdims=True)
    std[std == 0] = 1.0
    return (train - mean) / std, mean, std


def _apply_standardize(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (x - mean) / std


# -------------------- VAE 子网络 -------------------- #

class _Encoder(nn.Module):
    """Encoder: X_all -> (mu, logvar) of U."""

    def __init__(self, input_dim: int, latent_dim: int, hidden: Sequence[int]):
        super().__init__()
        layers: Iterable[nn.Module] = []
        last = input_dim
        for h in hidden:
            layers.extend([nn.Linear(last, h), nn.ReLU()])
            last = h
        self.net = nn.Sequential(*layers)
        self.mu = nn.Linear(last, latent_dim)
        self.logvar = nn.Linear(last, latent_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.net(x)
        return self.mu(h), self.logvar(h)


class _ProxyDecoder(nn.Module):
    """Decoder: U -> X_all."""

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


# -------------------- 配置 -------------------- #

@dataclass
class IVAPCIGoldConfig:
    # 表示学习相关
    latent_dim: int = 4
    encoder_hidden: Sequence[int] = (128, 64)
    decoder_hidden: Sequence[int] = (64, 64)
    beta: float = 1.0
    lr_vae: float = 1e-3
    batch_size_vae: int = 128
    epochs_vae: int = 300
    val_frac_repr: float = 0.1
    early_stopping_patience_vae: int = 30
    early_stopping_min_delta_vae: float = 0.0

    # DR / DML 相关
    n_splits_dr: int = 2
    seed: int = 42
    device: str = "cpu"


# -------------------- Gold Estimator -------------------- #

class IVAPCIGoldEstimator(BaseCausalEstimator):
    """
    Gold 版 IVAPCI:
      - Stage 1: 纯 proxy-VAE 学 U_hat(X_all)，完全不看 A/Y。
      - Stage 2: Encoder 冻结，用 U_hat 做 cross-fitted GLM-DR 估计 ATE。
    """

    def __init__(self, config: Optional[IVAPCIGoldConfig] = None):
        self.config = config or IVAPCIGoldConfig()
        self.device = torch.device(self.config.device)
        torch.manual_seed(self.config.seed)
        np.random.seed(self.config.seed)
        self._is_fit = False

    # -------------------- public API -------------------- #
    def fit(self, X_all: np.ndarray, A: np.ndarray, Y: np.ndarray) -> None:
        X_all = np.asarray(X_all, dtype=np.float32)
        n, d_all = X_all.shape
        cfg = self.config

        idx_all = np.arange(n)
        train_idx, val_idx = train_test_split(
            idx_all,
            test_size=cfg.val_frac_repr,
            random_state=cfg.seed,
            shuffle=True,
        )

        X_train = X_all[train_idx]
        X_val = X_all[val_idx]

        X_train_std, self._x_mean, self._x_std = _standardize(X_train)
        X_val_std = _apply_standardize(X_val, self._x_mean, self._x_std)

        self.encoder = _Encoder(d_all, cfg.latent_dim, cfg.encoder_hidden).to(self.device)
        self.proxy_decoder = _ProxyDecoder(cfg.latent_dim, d_all, cfg.decoder_hidden).to(self.device)

        optimizer = torch.optim.Adam(
            list(self.encoder.parameters()) + list(self.proxy_decoder.parameters()),
            lr=cfg.lr_vae,
        )
        mse_loss = nn.MSELoss(reduction="mean")

        train_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(torch.from_numpy(X_train_std)),
            batch_size=cfg.batch_size_vae,
            shuffle=True,
        )
        val_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(torch.from_numpy(X_val_std)),
            batch_size=cfg.batch_size_vae,
            shuffle=False,
        )

        best_val = float("inf")
        patience = 0
        best_state = None

        def _batch_vae_loss(xb: torch.Tensor) -> torch.Tensor:
            mu, logvar = self.encoder(xb)
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = mu + eps * std
            x_recon = self.proxy_decoder(z)
            recon = mse_loss(x_recon, xb)
            kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            return recon + cfg.beta * kl

        for _ in range(cfg.epochs_vae):
            self.encoder.train(); self.proxy_decoder.train()
            for (xb,) in train_loader:
                xb = xb.to(self.device)
                loss = _batch_vae_loss(xb)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            self.encoder.eval(); self.proxy_decoder.eval()
            val_losses = []
            with torch.no_grad():
                for (xb,) in val_loader:
                    xb = xb.to(self.device)
                    val_losses.append(float(_batch_vae_loss(xb).item()))
            mean_val = float(np.mean(val_losses)) if val_losses else float("inf")

            if mean_val + cfg.early_stopping_min_delta_vae < best_val:
                best_val = mean_val
                patience = 0
                best_state = {
                    "encoder": self.encoder.state_dict(),
                    "decoder": self.proxy_decoder.state_dict(),
                }
            else:
                patience += 1
                if patience >= cfg.early_stopping_patience_vae:
                    break

        if best_state is None:
            best_state = {
                "encoder": self.encoder.state_dict(),
                "decoder": self.proxy_decoder.state_dict(),
            }

        self.encoder.load_state_dict(best_state["encoder"])
        self.proxy_decoder.load_state_dict(best_state["decoder"])
        self._is_fit = True

    def get_latent(self, X_all: np.ndarray) -> np.ndarray:
        if not self._is_fit:
            raise RuntimeError("IVAPCIGoldEstimator must be fit before get_latent")
        X_all = np.asarray(X_all, dtype=np.float32)
        X_std = _apply_standardize(X_all, self._x_mean, self._x_std)
        X_t = torch.from_numpy(X_std).to(self.device)
        self.encoder.eval()
        with torch.no_grad():
            mu, _ = self.encoder(X_t)
        return mu.cpu().numpy()

    def estimate_ate(self, X_all: np.ndarray, A: np.ndarray, Y: np.ndarray) -> float:
        if not self._is_fit:
            raise RuntimeError("IVAPCIGoldEstimator must be fit before estimate_ate")
        U_hat = self.get_latent(X_all)
        return self._dr_ate_glm(U_hat, A, Y)

    # -------------------- DR / DML 辅助 -------------------- #
    def _dr_ate_glm(self, U: np.ndarray, A: np.ndarray, Y: np.ndarray) -> float:
        cfg = self.config
        U = np.asarray(U, dtype=np.float32)
        A = np.asarray(A).reshape(-1)
        Y = np.asarray(Y).reshape(-1)

        n = U.shape[0]
        kf = KFold(n_splits=cfg.n_splits_dr, shuffle=True, random_state=cfg.seed)
        psi = np.zeros(n, dtype=float)

        for train_idx, test_idx in kf.split(U):
            U_tr, U_te = U[train_idx], U[test_idx]
            A_tr, A_te = A[train_idx], A[test_idx]
            Y_tr, Y_te = Y[train_idx], Y[test_idx]

            unique_classes = np.unique(A_tr)
            if len(unique_classes) < 2:
                p = float(np.clip(np.mean(A_tr), 1e-3, 1 - 1e-3))
                e_hat = np.full_like(A_te, fill_value=p, dtype=float)
            else:
                clf = LogisticRegression(max_iter=2000, solver="lbfgs")
                clf.fit(U_tr, A_tr)
                e_hat = clf.predict_proba(U_te)[:, 1].clip(1e-3, 1 - 1e-3)

            def _fit_or_dummy(mask: np.ndarray) -> LinearRegression | DummyRegressor:
                sub_u = U_tr[mask]
                sub_y = Y_tr[mask]
                if len(sub_y) == 0:
                    return DummyRegressor(strategy="constant", constant=float(np.mean(Y_tr) if len(Y_tr) else 0.0)).fit(
                        U_tr, Y_tr
                    )
                return LinearRegression().fit(sub_u, sub_y)

            reg0 = _fit_or_dummy(A_tr == 0)
            reg1 = _fit_or_dummy(A_tr == 1)

            m0 = reg0.predict(U_te)
            m1 = reg1.predict(U_te)

            psi[test_idx] = m1 - m0 + A_te * (Y_te - m1) / e_hat - (1 - A_te) * (Y_te - m0) / (1 - e_hat)

        return float(psi.mean())


__all__ = ["IVAPCIGoldConfig", "IVAPCIGoldEstimator"]
