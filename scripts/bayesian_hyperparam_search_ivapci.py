#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
bayesian_hyperparam_search_ivapci.py — IVAPCI v3.3 贝叶斯（Optuna）超参数优化，可直接运行版

为什么用贝叶斯优化：
- 你的训练一次成本高，且参数多为连续/半连续；网格搜索会“指数爆炸”
- Optuna(TPE) 对混合空间（连续 + 类别）更稳，且支持并行与剪枝（pruning）

本脚本做了什么：
- 复用你工程里的“场景生成器”，按 scenario_name + n + seed 生成 (V,A,Y,tau_true,x_dim,w_dim,z_dim)
- 自动探测 estimator 模块（优先 v27 -> v26 -> ivapci_v33_theory）
- Optuna 负责采样参数；每个 trial 评估多个 scenario × seed × repeat，产出：
    mean_rmse, mean_w_auc, mean_z_leak, mean_overlap_ess_min, mean_train_time
- 默认单目标（加权 composite），支持 --multiobjective 输出帕累托（RMSE, |W_AUC-0.5|, Z_leak）

依赖：
    pip install optuna pandas numpy scikit-learn

可选可视化：
    pip install optuna-dashboard plotly kaleido
    optuna-dashboard sqlite:///YOUR_OUTDIR/optuna_study.db

使用示例：
    # 先跑 quick，验证 W 泄漏能否压下来（建议 30~60 trials）
    python bayesian_hyperparam_search_ivapci.py --mode quick --scenarios EASY-linear-weak,MODERATE-nonlinear --n-trials 40

    # balanced（更全面）
    python bayesian_hyperparam_search_ivapci.py --mode balanced --scenarios all --n-trials 80 --n-jobs 4

    # focused_w（专攻 W 泄漏）
    python bayesian_hyperparam_search_ivapci.py --mode focused_w --scenarios MODERATE-nonlinear --n-trials 60

注意：
- 你的训练是“有闭环控制器”的，目标函数会有噪声；TPE 通常比 GP 更稳。
- Python 的 hash() 不是稳定的，本脚本用 md5 做稳定 seed 组合（便于复现/对比）。
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import optuna
import pandas as pd


# -------------------- 稳定工具 --------------------

def stable_hash_int(s: str, mod: int = 2**31 - 1) -> int:
    h = hashlib.md5(s.encode("utf-8")).hexdigest()
    return int(h[:8], 16) % mod


def parse_hidden_spec(v: Any) -> Any:
    if isinstance(v, str) and "-" in v and all(p.isdigit() for p in v.split("-")):
        return tuple(int(x) for x in v.split("-"))
    return v


def safe_float(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def median_abs_deviation(arr: np.ndarray) -> float:
    arr = np.asarray(arr, dtype=float)
    if arr.size == 0:
        return float("nan")
    med = np.median(arr)
    return float(np.median(np.abs(arr - med)))


def check_convergence_plateau(
    study: optuna.Study,
    *,
    window: int = 25,
    rel_impr_thr: float = 0.003,
    min_trials: int = 30,
    objective_index: int = 0,
) -> bool:
    """Detect a plateau when recent improvements fall below a threshold.

    Compares the best value before the last ``window`` trials with the best value
    *within* the last window. If the relative improvement is smaller than
    ``rel_impr_thr`` the search is considered converged and can be stopped early.
    """

    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if len(completed) < max(min_trials, window + 5):
        return False

    values: list[float] = []
    for t in completed:
        # ``values`` exists for both single/multi objective; fall back to ``value``
        val: Optional[float]
        if t.values is not None and len(t.values) > objective_index:
            val = t.values[objective_index]
        else:
            val = t.value
        if val is None or not np.isfinite(val):
            continue
        values.append(float(val))

    if len(values) < max(min_trials, window + 5):
        return False

    best_before = float(np.min(values[:-window]))
    best_recent = float(np.min(values[-window:]))
    denom = max(abs(best_before), 1e-12)
    rel_impr = (best_before - best_recent) / denom
    return rel_impr < rel_impr_thr


# -------------------- 自动探测：estimator module --------------------

def try_import_first(candidates: Sequence[str]) -> Tuple[str, Any]:
    last_err = None
    for m in candidates:
        try:
            return m, importlib.import_module(m)
        except Exception as e:
            last_err = e
    raise ImportError(f"Could not import estimator module candidates: {candidates}. Last error: {last_err}")


def resolve_estimator_module(user_spec: Optional[str]) -> Any:
    """Resolve estimator module, handling repo-local imports by default.

    When running from the repository root, the ``models`` package is not on
    ``sys.path`` by default. We prepend the repo root so that imports such as
    ``models.ivapci_v33_theory`` work without additional flags. Users can still
    override via ``--estimator-module``.
    """

    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    candidates = []
    if user_spec:
        candidates.append(user_spec)
    candidates += [
        "models.ivapci_v33_theory_v27",
        "models.ivapci_v33_theory_v26",
        "models.ivapci_v33_theory",
        "ivapci_v33_theory_v27",
        "ivapci_v33_theory_v26",
        "ivapci_v33_theory",
    ]
    mod_name, mod = try_import_first(candidates)
    print(f"[Info] Using estimator module: {mod_name}")
    return mod


# -------------------- 自动探测：scenario generator --------------------

def resolve_scenario_generator(module_name: Optional[str], fn_name: Optional[str]) -> Tuple[str, str, Callable[..., Any]]:
    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    module_candidates = []
    if module_name:
        module_candidates.append(module_name)
    module_candidates += [
        "simulation_configs",
        "scripts.simulation_configs",
        "simulation_scenarios",
        "scripts.simulation_scenarios",
        "simulators",
        "simulators.simulators",
        "scripts.simulators",
        "scripts.simulators.simulators",
        "ivapci.simulation_configs",
        "ivapci.simulation_scenarios",
    ]
    fn_candidates = []
    if fn_name:
        fn_candidates.append(fn_name)
    fn_candidates += [
        "generate_scenario",
        "make_scenario",
        "sample_scenario",
        "generate",
        "simulate_scenario",
        "simulate",
    ]

    last_err = None
    for mn in module_candidates:
        try:
            m = importlib.import_module(mn)
        except Exception as e:
            last_err = e
            continue
        for fn in fn_candidates:
            f = getattr(m, fn, None)
            if callable(f):
                print(f"[Info] Using scenario generator: {mn}:{fn}")
                return mn, fn, f

    raise ImportError(
        "Could not resolve scenario generator. "
        "Pass --scenario-module <module> and --scenario-fn <function>. "
        f"Last import error: {last_err}"
    )


def list_available_scenarios(gen_module_name: str) -> Optional[List[str]]:
    try:
        m = importlib.import_module(gen_module_name)
    except Exception:
        return None

    if hasattr(m, "get_available_scenarios") and callable(getattr(m, "get_available_scenarios")):
        try:
            return [str(x) for x in list(getattr(m, "get_available_scenarios")())]
        except Exception:
            pass

    if hasattr(m, "list_scenarios") and callable(getattr(m, "list_scenarios")):
        try:
            vals = getattr(m, "list_scenarios")()
            if isinstance(vals, dict):
                return [str(k) for k in vals.keys()]
            return [str(x) for x in list(vals)]
        except Exception:
            pass

    for attr in ["AVAILABLE_SCENARIOS", "SCENARIOS", "SCENARIO_REGISTRY"]:
        if hasattr(m, attr):
            obj = getattr(m, attr)
            if isinstance(obj, dict):
                return [str(k) for k in obj.keys()]
            if isinstance(obj, (list, tuple)):
                return [str(x) for x in obj]
    return None


def normalize_scenarios_arg(s: str, gen_module_name: str) -> List[str]:
    s = s.strip()
    if s.lower() == "all":
        avail = list_available_scenarios(gen_module_name)
        if not avail:
            raise ValueError(f"scenarios=all but cannot list scenarios from {gen_module_name}")
        return avail
    if "," in s:
        parts = [p.strip() for p in s.split(",") if p.strip()]
    else:
        parts = [p.strip() for p in s.split() if p.strip()]
    if not parts:
        raise ValueError("No scenarios provided.")
    return parts


# -------------------- 解析场景输出 --------------------

def extract_data_and_dims(out: Any) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, int, int, int]:
    """
    支持：
      - dict: {V,A,Y,tau_true|tau,x_dim,w_dim,z_dim}
      - dict: {X,W,Z,A,Y,tau_true|tau}
      - tuple: (V,A,Y,tau_true|tau,meta)  meta含 x_dim/w_dim/z_dim 或 X/W/Z
      - tuple: (X,W,Z,A,Y,tau_true|tau)

    兼容 simulators.simulators 的返回（使用 ``tau`` 键且 meta 不含 x_dim/w_dim/z_dim）。
    """

    def _get_tau(payload: dict) -> float:
        if "tau_true" in payload:
            return float(payload["tau_true"])
        if "tau" in payload:
            return float(payload["tau"])
        raise ValueError("Scenario output missing tau/tau_true")

    if isinstance(out, dict):
        # Already concatenated V
        if {"V", "A", "Y"}.issubset(out.keys()) and ({"tau_true"}.issubset(out.keys()) or {"tau"}.issubset(out.keys())):
            V = np.asarray(out["V"])
            A = np.asarray(out["A"]).reshape(-1)
            Y = np.asarray(out["Y"]).reshape(-1)
            tau = _get_tau(out)
            if {"x_dim", "w_dim", "z_dim"}.issubset(out.keys()):
                return V, A, Y, tau, int(out["x_dim"]), int(out["w_dim"]), int(out["z_dim"])
            meta = out.get("meta", None)
            if isinstance(meta, dict) and {"x_dim", "w_dim", "z_dim"}.issubset(meta.keys()):
                return V, A, Y, tau, int(meta["x_dim"]), int(meta["w_dim"]), int(meta["z_dim"])
            raise ValueError("Scenario dict has V/A/Y but missing x_dim/w_dim/z_dim metadata.")

        # Separate X/W/Z
        if {"X", "W", "Z", "A", "Y"}.issubset(out.keys()) and ({"tau_true"}.issubset(out.keys()) or {"tau"}.issubset(out.keys())):
            X = np.asarray(out["X"])
            W = np.asarray(out["W"])
            Z = np.asarray(out["Z"])
            V = np.concatenate([X, W, Z], axis=1)
            A = np.asarray(out["A"]).reshape(-1)
            Y = np.asarray(out["Y"]).reshape(-1)
            tau = _get_tau(out)
            return V, A, Y, tau, X.shape[1], W.shape[1], Z.shape[1]
        raise ValueError(f"Scenario dict keys not recognized: {sorted(out.keys())[:20]}")

    if isinstance(out, (tuple, list)):
        if len(out) == 6:
            X, W, Z, A, Y, tau = out
            X = np.asarray(X)
            W = np.asarray(W)
            Z = np.asarray(Z)
            V = np.concatenate([X, W, Z], axis=1)
            return V, np.asarray(A).reshape(-1), np.asarray(Y).reshape(-1), float(tau), X.shape[1], W.shape[1], Z.shape[1]
        if len(out) == 5:
            V, A, Y, tau, meta = out
            V = np.asarray(V)
            A = np.asarray(A).reshape(-1)
            Y = np.asarray(Y).reshape(-1)
            tau = float(tau)
            if isinstance(meta, dict) and {"x_dim", "w_dim", "z_dim"}.issubset(meta.keys()):
                return V, A, Y, tau, int(meta["x_dim"]), int(meta["w_dim"]), int(meta["z_dim"])
            if isinstance(meta, dict) and {"X", "W", "Z"}.issubset(meta.keys()):
                X = np.asarray(meta["X"])
                W = np.asarray(meta["W"])
                Z = np.asarray(meta["Z"])
                return V, A, Y, tau, X.shape[1], W.shape[1], Z.shape[1]
            raise ValueError("Scenario tuple has meta but meta lacks x_dim/w_dim/z_dim (or X/W/Z).")

    raise ValueError(f"Unsupported scenario output type: {type(out)}")


# -------------------- Optuna search space（按 mode） --------------------

def define_params(trial: optuna.trial.Trial, mode: str) -> Dict[str, Any]:
    # 说明：这里刻意用“更稳”的范围（避免把 TW 压塌）。
    if mode == "quick":
        return {
            "gamma_adv_w": trial.suggest_float("gamma_adv_w", 0.18, 0.35),
            "lambda_hsic": trial.suggest_float("lambda_hsic", 0.03, 0.15, log=True),
            "lambda_hsic_w_a": trial.suggest_float("lambda_hsic_w_a", 0.02, 0.10),

            "gamma_adv_z": 0.18,
            "dropout_z": 0.30,
            "enc_z_hidden": "32-16",

            "ctrl_w_auc_target": trial.suggest_float("ctrl_w_auc_target", 0.52, 0.55),
            "ctrl_kp_w": trial.suggest_float("ctrl_kp_w", 1.8, 3.0),
            "ctrl_ki_w": trial.suggest_float("ctrl_ki_w", 0.45, 0.75),
        }

    if mode == "balanced":
        return {
            # W
            "gamma_adv_w": trial.suggest_float("gamma_adv_w", 0.15, 0.40),
            "gamma_adv_w_cond": trial.suggest_float("gamma_adv_w_cond", 0.14, 0.30),
            "lambda_hsic": trial.suggest_float("lambda_hsic", 0.02, 0.20, log=True),
            "lambda_hsic_w_a": trial.suggest_float("lambda_hsic_w_a", 0.02, 0.12),

            # Z
            "gamma_adv_z": trial.suggest_float("gamma_adv_z", 0.12, 0.28),
            "gamma_adv_z_cond": trial.suggest_float("gamma_adv_z_cond", 0.10, 0.22),
            "dropout_z": trial.suggest_float("dropout_z", 0.20, 0.40),
            "enc_z_hidden": trial.suggest_categorical("enc_z_hidden", ["64-32", "48-24", "32-16"]),

            # orthogonal
            "lambda_cond_ortho": trial.suggest_float("lambda_cond_ortho", 0.004, 0.03, log=True),
            "cond_ortho_warmup_epochs": trial.suggest_int("cond_ortho_warmup_epochs", 3, 10),

            # controller
            "ctrl_w_auc_target": trial.suggest_float("ctrl_w_auc_target", 0.52, 0.55),
            "ctrl_kp_w": trial.suggest_float("ctrl_kp_w", 1.8, 3.2),
            "ctrl_ki_w": trial.suggest_float("ctrl_ki_w", 0.45, 0.80),
            "ctrl_z_r2_target": trial.suggest_float("ctrl_z_r2_target", 0.08, 0.14),
        }

    if mode == "focused_w":
        return {
            # W (dense)
            "gamma_adv_w": trial.suggest_float("gamma_adv_w", 0.18, 0.36),
            "gamma_adv_w_cond": trial.suggest_float("gamma_adv_w_cond", 0.15, 0.30),
            "lambda_hsic": trial.suggest_float("lambda_hsic", 0.04, 0.18, log=True),
            "lambda_hsic_w_a": trial.suggest_float("lambda_hsic_w_a", 0.02, 0.14),

            # controller W
            "ctrl_w_auc_target": trial.suggest_float("ctrl_w_auc_target", 0.51, 0.55),
            "ctrl_kp_w": trial.suggest_float("ctrl_kp_w", 2.0, 3.5),
            "ctrl_ki_w": trial.suggest_float("ctrl_ki_w", 0.45, 0.85),

            # Z fixed
            "gamma_adv_z": 0.18,
            "dropout_z": 0.30,
            "enc_z_hidden": "32-16",
        }

    raise ValueError(f"Unknown mode: {mode}")


# -------------------- 单次 trial 执行 --------------------

def make_seed(base_seed: int, trial_num: int, scenario: str, repeat: int) -> int:
    return (
        base_seed * 1_000_000
        + trial_num * 10_000
        + stable_hash_int(scenario) % 10_000
        + repeat
    ) % (2**31 - 1)


def _ess_ratio_from_diag(diag: Dict[str, Any], n: int) -> float:
    ess = safe_float(diag.get("overlap_ess_min", np.nan))
    if not np.isfinite(ess):
        ess = safe_float(diag.get("dr_ess_min", np.nan))
        if np.isfinite(ess) and n > 0 and ess > 1.0:
            ess = ess / float(n)
    if not np.isfinite(ess):
        ess = 0.0
    return float(ess)


def _y_scale(Y: np.ndarray, tau_true: float, mode: str) -> float:
    y = np.asarray(Y, dtype=float)
    if mode == "y_mad":
        scale = median_abs_deviation(y)
    elif mode == "tau_abs":
        scale = abs(float(tau_true))
    else:
        scale = float(np.std(y))
    if not np.isfinite(scale) or scale <= 0:
        scale = float(np.std(y) + 1e-8)
    # 防止极低方差场景下的缩放噪声被放大
    scale = max(scale, 0.5)
    return float(scale + 1e-8)


def run_one_fit(
    est_mod: Any,
    estimator_kind: str,
    cfg_overrides: Dict[str, Any],
    scenario_fn: Callable[..., Any],
    scenario_name: str,
    n: int,
    base_seed: int,
    seed: int,
    rep: int,
    epochs_main: int,
    pretrain_epochs: Optional[int],
    *,
    w_auc_hinge: float,
    z_r2_hinge: float,
    ess_ratio_target: float,
    ate_scale: str,
) -> Dict[str, Any]:
    """Simulate, fit, and return diagnostics plus per-run objective components."""

    Cfg = getattr(est_mod, "IVAPCIV33TheoryConfig", None)
    if Cfg is None:
        raise AttributeError("Estimator module missing IVAPCIV33TheoryConfig")

    if estimator_kind == "hier":
        Est = getattr(est_mod, "IVAPCIv33TheoryHierEstimator", None)
    else:
        Est = getattr(est_mod, "IVAPCIv33TheoryHierRADREstimator", None)
    if Est is None:
        raise AttributeError(f"Estimator module missing estimator for kind={estimator_kind}")

    out = scenario_fn(scenario_name, n=n, seed=seed)
    V, A, Y, tau_true, x_dim, w_dim, z_dim = extract_data_and_dims(out)

    cfg = Cfg()
    cfg.x_dim, cfg.w_dim, cfg.z_dim = int(x_dim), int(w_dim), int(z_dim)
    cfg.epochs_main = int(epochs_main)
    if pretrain_epochs is not None:
        cfg.epochs_pretrain = int(pretrain_epochs)

    for k, v in cfg_overrides.items():
        setattr(cfg, k, parse_hidden_spec(v))

    t0 = time.time()
    est = Est(config=cfg)
    est.fit(V, A, Y)
    train_time = time.time() - t0

    ate_hat = safe_float(est.estimate_ate(V, A, Y))
    tau_true_f = float(tau_true)
    y_scale_val = _y_scale(Y, tau_true_f, ate_scale)
    abs_err = abs(ate_hat - tau_true_f)
    rel_abs_err = abs_err / (y_scale_val + 1e-8)

    try:
        diag = est.get_training_diagnostics()
    except Exception:
        diag = {}

    w_auc_raw = safe_float(diag.get("rep_auc_w_to_a", np.nan))
    if not np.isfinite(w_auc_raw):
        w_auc_raw = 0.5
    w_auc_eff = max(float(w_auc_raw), 1.0 - float(w_auc_raw))
    w_leak = max(0.0, w_auc_eff - 0.5)

    z_r2 = safe_float(
        diag.get(
            "rep_exclusion_leakage_r2_cond",
            diag.get("rep_exclusion_leakage_r2", np.nan),
        )
    )
    if not np.isfinite(z_r2):
        z_r2 = 0.0
    z_leak = max(0.0, float(z_r2))

    ess_ratio = _ess_ratio_from_diag(diag, n)

    w_violation = max(0.0, float(w_auc_eff) - float(w_auc_hinge))
    z_violation = max(0.0, float(z_r2) - float(z_r2_hinge))
    ess_violation = max(0.0, float(ess_ratio_target) - float(ess_ratio))

    # Sanity checks: violations must be zero when metrics are within hinge/target.
    if z_r2 <= float(z_r2_hinge) + 1e-12 and z_violation > 1e-9:
        raise RuntimeError(
            f"z_violation inconsistent with hinge: z_r2={z_r2}, hinge={z_r2_hinge}, viol={z_violation}, "
            f"scenario={scenario_name}, seed={seed}, base_seed={base_seed}, rep={rep}"
        )
    if ess_ratio >= float(ess_ratio_target) - 1e-12 and ess_violation > 1e-9:
        raise RuntimeError(
            f"ess_violation inconsistent with target: ess_ratio={ess_ratio}, target={ess_ratio_target}, "
            f"viol={ess_violation}, scenario={scenario_name}, seed={seed}, base_seed={base_seed}, rep={rep}"
        )

    return {
        "status": "success",
        "scenario": scenario_name,
        "seed": int(seed),
        "base_seed": int(base_seed),
        "rep": int(rep),
        "ate_hat": float(ate_hat),
        "tau_true": tau_true_f,
        "y_scale": y_scale_val,
        "abs_err": float(abs_err),
        "rel_abs_err": float(rel_abs_err),
        # legacy alias kept for downstream code
        "ate_err": float(rel_abs_err),
        "w_auc": float(w_auc_raw),
        "w_auc_eff": float(w_auc_eff),
        "w_leak": float(w_leak),
        "z_r2": float(z_r2),
        "z_leak": float(z_leak),
        "ess_ratio": float(ess_ratio),
        "w_violation": float(w_violation),
        "z_violation": float(z_violation),
        "ess_violation": float(ess_violation),
        "train_time": float(train_time),
        "iv_first_stage_f": safe_float(diag.get("iv_first_stage_f", np.nan)),
        "weak_iv_flag": safe_float(diag.get("weak_iv_flag", np.nan)),
    }


def _agg(values: List[float], method: str) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    if method == "mean":
        return float(np.nanmean(arr))
    return float(np.nanmedian(arr))


def aggregate_runs(
    runs: List[Dict[str, Any]], *, scenarios: List[str], agg_repeats: str
) -> Tuple[Dict[str, float], List[Dict[str, Any]]]:
    """Aggregate metrics with repeat-level robustness → seeds → scenarios.

    Repeat-level aggregation uses ``agg_repeats`` (median/mean) within each base seed
    so ``n_repeats`` truly stabilizes noisy training. Scenario-level aggregation uses
    a weighted mean across seeds (weight = log1p(n_seeds)) and across scenarios.
    """

    scenario_stats: List[Dict[str, Any]] = []
    for sc in scenarios:
        sc_rows = [r for r in runs if r.get("scenario") == sc and r.get("status") == "success"]
        if not sc_rows:
            continue
        base_seeds = sorted(set(int(r.get("base_seed", -1)) for r in sc_rows))
        seed_stats: List[Dict[str, float]] = []
        for bs in base_seeds:
            seed_rows = [r for r in sc_rows if int(r.get("base_seed", -1)) == bs]
            seed_stats.append({
                "rel_abs_err": _agg([r.get("rel_abs_err", np.nan) for r in seed_rows], agg_repeats),
                "ate_err": _agg([r.get("ate_err", np.nan) for r in seed_rows], agg_repeats),
                "w_violation": _agg([r.get("w_violation", np.nan) for r in seed_rows], agg_repeats),
                "z_violation": _agg([r.get("z_violation", np.nan) for r in seed_rows], agg_repeats),
                "ess_violation": _agg([r.get("ess_violation", np.nan) for r in seed_rows], agg_repeats),
                "w_auc_eff": _agg([r.get("w_auc_eff", np.nan) for r in seed_rows], agg_repeats),
                "w_leak": _agg([r.get("w_leak", np.nan) for r in seed_rows], agg_repeats),
                "w_auc": _agg([r.get("w_auc", np.nan) for r in seed_rows], agg_repeats),
                "z_r2": _agg([r.get("z_r2", np.nan) for r in seed_rows], agg_repeats),
                "z_leak": _agg([r.get("z_leak", np.nan) for r in seed_rows], agg_repeats),
                "ess_ratio": _agg([r.get("ess_ratio", np.nan) for r in seed_rows], agg_repeats),
                "train_time": _agg([r.get("train_time", np.nan) for r in seed_rows], agg_repeats),
            })

        weight = float(np.log1p(len(seed_stats))) if seed_stats else 1.0
        scenario_stats.append({
            "scenario": sc,
            "weight": weight,
            "rel_abs_err": float(np.nanmean([s["rel_abs_err"] for s in seed_stats])),
            "ate_err": float(np.nanmean([s["ate_err"] for s in seed_stats])),
            "w_violation": float(np.nanmean([s["w_violation"] for s in seed_stats])),
            "z_violation": float(np.nanmean([s["z_violation"] for s in seed_stats])),
            "ess_violation": float(np.nanmean([s["ess_violation"] for s in seed_stats])),
            "w_auc_eff": float(np.nanmean([s["w_auc_eff"] for s in seed_stats])),
            "w_leak": float(np.nanmean([s["w_leak"] for s in seed_stats])),
            "w_auc": float(np.nanmean([s["w_auc"] for s in seed_stats])),
            "z_r2": float(np.nanmean([s["z_r2"] for s in seed_stats])),
            "z_leak": float(np.nanmean([s["z_leak"] for s in seed_stats])),
            "ess_ratio": float(np.nanmean([s["ess_ratio"] for s in seed_stats])),
            "train_time": float(np.nanmean([s["train_time"] for s in seed_stats])),
            "n_seeds": len(seed_stats),
        })

    if not scenario_stats:
        return {
            "rel_abs_err": float("inf"),
            "ate_err": float("inf"),
            "w_violation": float("inf"),
            "z_violation": float("inf"),
            "ess_violation": float("inf"),
            "w_auc_eff": float("nan"),
            "w_leak": float("nan"),
            "w_auc": float("nan"),
            "z_r2": float("nan"),
            "z_leak": float("nan"),
            "ess_ratio": float("nan"),
            "train_time": float("inf"),
        }, []

    w_sum = float(sum(s.get("weight", 1.0) for s in scenario_stats))
    def _wmean(key: str) -> float:
        return float(
            sum((s.get("weight", 1.0) * s.get(key, np.nan)) for s in scenario_stats) / max(w_sum, 1e-8)
        )

    global_stats = {
        "rel_abs_err": _wmean("rel_abs_err"),
        "ate_err": _wmean("ate_err"),
        "w_violation": _wmean("w_violation"),
        "z_violation": _wmean("z_violation"),
        "ess_violation": _wmean("ess_violation"),
        "w_auc_eff": _wmean("w_auc_eff"),
        "w_leak": _wmean("w_leak"),
        "w_auc": _wmean("w_auc"),
        "z_r2": _wmean("z_r2"),
        "z_leak": _wmean("z_leak"),
        "ess_ratio": _wmean("ess_ratio"),
        "train_time": _wmean("train_time"),
    }
    return global_stats, scenario_stats


def validate_best_config(
    *,
    est_mod: Any,
    estimator_kind: str,
    scenario_fn: Callable[..., Any],
    scenarios: List[str],
    params: Dict[str, Any],
    args: argparse.Namespace,
    outdir: Path,
) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    for sc in scenarios:
        for i in range(args.val_repeats):
            base_seed = int(args.val_seed_offset + i)
            seed = make_seed(base_seed, trial_num=-1, scenario=sc, repeat=0)
            try:
                r = run_one_fit(
                    est_mod=est_mod,
                    estimator_kind=estimator_kind,
                    cfg_overrides=params,
                    scenario_fn=scenario_fn,
                    scenario_name=sc,
                    n=args.n,
                    base_seed=base_seed,
                    seed=seed,
                    rep=0,
                    epochs_main=args.epochs,
                    pretrain_epochs=args.pretrain_epochs,
                    w_auc_hinge=args.w_auc_hinge,
                    z_r2_hinge=args.z_r2_hinge,
                    ess_ratio_target=args.ess_ratio_target,
                    ate_scale=args.ate_scale,
                )
            except Exception as e:  # noqa: PIE786 - best-effort validation
                rows.append(
                    {"scenario": sc, "base_seed": base_seed, "rep": 0, "status": "failed", "error": str(e)[:200]}
                )
                continue
            rows.append(r)

    df_val = pd.DataFrame(rows)
    df_path = outdir / "validation_runs.csv"
    df_val.to_csv(df_path, index=False)

    ok = df_val[df_val.get("status", "success") == "success"]
    summary: Dict[str, Any]
    if ok.empty:
        summary = {
            "n_success": 0,
            "n_total": len(df_val),
            "message": "No successful validation runs",
        }
    else:
        summary = {
            "n_success": int(len(ok)),
            "n_total": int(len(df_val)),
            "mean_auc_eff": float(ok["w_auc_eff"].mean()),
            "mean_z_r2": float(ok["z_r2"].mean()),
            "mean_ess_ratio": float(ok["ess_ratio"].mean()),
            "median_rel_abs_err": float(ok["rel_abs_err"].median()),
            "scenarios": scenarios,
        }

    (outdir / "validation_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def composite_score_from_scenarios(
    scenario_stats: List[Dict[str, Any]], *, w_auc_hinge: float, z_r2_hinge: float, ess_ratio_target: float
) -> float:
    """Log-sum objective that penalizes any violated condition.

    L_s = log1p(ate) + log1p(2*w_term) + log1p(2*z_term) + log1p(ess_term)
    Global = Σ_s w_s * L_s / Σ_s w_s, with w_s = log1p(n_seeds).
    """

    if not scenario_stats:
        return float("inf")

    weighted = []
    for s in scenario_stats:
        weight = float(s.get("weight", 1.0))
        w_auc_eff = float(s.get("w_auc_eff", np.nan))
        z_r2 = float(s.get("z_r2", np.nan))
        ess_ratio = float(s.get("ess_ratio", np.nan))
        ate = float(s.get("rel_abs_err", np.nan))

        if not np.isfinite(ate):
            ate = float("inf")

        w_term = float(s.get("w_violation", np.nan))
        if not np.isfinite(w_term):
            w_term = max(0.0, w_auc_eff - w_auc_hinge) if np.isfinite(w_auc_eff) else 0.5

        z_term = float(s.get("z_violation", np.nan))
        if not np.isfinite(z_term):
            z_term = max(0.0, z_r2 - z_r2_hinge) if np.isfinite(z_r2) else 0.5

        ess_term = float(s.get("ess_violation", np.nan))
        if not np.isfinite(ess_term):
            ess_term = max(0.0, ess_ratio_target - ess_ratio) if np.isfinite(ess_ratio) else ess_ratio_target

        loss_s = (
            np.log1p(ate)
            + np.log1p(2.0 * w_term)
            + np.log1p(2.0 * z_term)
            + np.log1p(ess_term)
        )
        weighted.append((weight, loss_s))

    w_sum = float(sum(w for w, _ in weighted))
    if w_sum <= 0:
        return float("inf")
    return float(sum(w * l for w, l in weighted) / w_sum)


# -------------------- 主程序 --------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="IVAPCI v3.3 Bayesian hyperparameter optimization (Optuna)")
    parser.add_argument("--mode", choices=["quick", "balanced", "focused_w"], default="quick")
    parser.add_argument("--scenarios", default="EASY-linear-weak,MODERATE-nonlinear", help="comma/space separated or 'all'")
    parser.add_argument("--n", type=int, default=500)
    parser.add_argument("--seeds", type=str, default="0,1")
    parser.add_argument("--n-repeats", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--pretrain-epochs", type=int, default=None)

    parser.add_argument("--n-trials", type=int, default=50)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--timeout-hours", type=float, default=None)

    parser.add_argument("--estimator", choices=["hier", "hier_radr"], default="hier")
    parser.add_argument("--estimator-module", type=str, default=None)
    parser.add_argument("--scenario-module", type=str, default=None)
    parser.add_argument("--scenario-fn", type=str, default=None)

    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--study-name", type=str, default=None)
    parser.add_argument("--multiobjective", action="store_true", help="(deprecated) same as --objective-mode pareto")
    parser.add_argument("--objective-mode", choices=["single", "pareto"], default="pareto")
    parser.add_argument("--sampler", choices=["tpe", "nsga2"], default="tpe")
    parser.add_argument("--pruner", choices=["median", "hyperband", "none"], default="median")
    parser.add_argument("--agg", choices=["median", "mean"], default="median", help="repeat-level aggregation")
    parser.add_argument("--w-auc-hinge", "--w-auc-thr", dest="w_auc_hinge", type=float, default=0.58, help="W→A AUC hinge threshold")
    parser.add_argument("--w-auc-feasible", type=float, default=0.56, help="W→A feasibility upper bound")
    parser.add_argument("--z-r2-hinge", "--z-r2-thr", dest="z_r2_hinge", type=float, default=0.08, help="Z→Y leakage hinge R2")
    parser.add_argument("--z-r2-feasible", type=float, default=0.05, help="Z→Y feasibility R2 upper bound")
    parser.add_argument("--ess-ratio-target", type=float, default=0.35, help="Minimum ESS ratio target (hinge)")
    parser.add_argument("--ess-ratio-feasible", type=float, default=0.40, help="ESS feasibility floor")
    parser.add_argument("--ate-scale", choices=["y_std", "y_mad", "tau_abs"], default="y_std", help="normalize ATE error")
    parser.add_argument("--plateau-window", type=int, default=25, help="Plateau detection window (trials)")
    parser.add_argument("--plateau-rel-impr", type=float, default=0.003, help="Relative improvement threshold for plateau stop")
    parser.add_argument("--plateau-min-trials", type=int, default=30, help="Minimum completed trials before plateau check")
    parser.add_argument("--validate-best", action="store_true", help="Run independent-seed validation on recommended params")
    parser.add_argument("--val-repeats", type=int, default=10, help="Validation runs per scenario (distinct seeds)")
    parser.add_argument("--val-seed-offset", type=int, default=10_000, help="Offset applied to validation seeds")
    parser.add_argument("--val-scenarios", type=str, default=None, help="Optional holdout scenarios for validation (comma/space)")
    parser.add_argument("--quiet", action="store_true")

    args = parser.parse_args()

    if args.multiobjective:
        args.objective_mode = "pareto"

    est_mod = resolve_estimator_module(args.estimator_module)
    gen_mod_name, gen_fn_name, scenario_fn = resolve_scenario_generator(args.scenario_module, args.scenario_fn)
    scenarios = normalize_scenarios_arg(args.scenarios, gen_mod_name)

    seeds = [int(x) for x in args.seeds.replace(",", " ").split() if x.strip()]
    val_scenarios = normalize_scenarios_arg(args.val_scenarios, gen_mod_name) if args.val_scenarios else list(scenarios)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = Path(args.output_dir or f"optuna_search_{args.mode}_{ts}")
    outdir.mkdir(parents=True, exist_ok=True)

    study_name = args.study_name or f"ivapci_{args.mode}_{ts}"
    storage = f"sqlite:///{outdir}/optuna_study.db"

    objective_mode = args.objective_mode
    if objective_mode == "pareto" and args.sampler == "tpe":
        # NSGA-II is more appropriate for Pareto fronts.
        args.sampler = "nsga2"

    # sampler
    if args.sampler == "tpe":
        sampler: optuna.samplers.BaseSampler = optuna.samplers.TPESampler(
            n_startup_trials=max(10, args.n_trials // 5),
            multivariate=True,
            seed=42,
        )
    else:
        sampler = optuna.samplers.NSGAIISampler(seed=42)

    # pruner
    if objective_mode == "pareto":
        pruner: optuna.pruners.BasePruner = optuna.pruners.NopPruner()
    elif args.pruner == "median":
        pruner = optuna.pruners.MedianPruner(n_startup_trials=8, n_warmup_steps=0)
    elif args.pruner == "hyperband":
        pruner = optuna.pruners.HyperbandPruner()
    else:
        pruner = optuna.pruners.NopPruner()

    directions = ["minimize"] * 4 if objective_mode == "pareto" else ["minimize"]

    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        load_if_exists=True,
        directions=directions,
        sampler=sampler,
        pruner=pruner,
    )

    def objective(trial: optuna.trial.Trial):
        params = define_params(trial, args.mode)
        all_rows: List[Dict[str, Any]] = []
        step = 0

        try:
            for sc in scenarios:
                for base_seed in seeds:
                    for rep in range(args.n_repeats):
                        seed = make_seed(base_seed, trial.number, sc, rep)
                        r = run_one_fit(
                            est_mod=est_mod,
                            estimator_kind=args.estimator,
                            cfg_overrides=params,
                            scenario_fn=scenario_fn,
                            scenario_name=sc,
                            n=args.n,
                            base_seed=base_seed,
                            seed=seed,
                            rep=rep,
                            epochs_main=args.epochs,
                            pretrain_epochs=args.pretrain_epochs,
                            w_auc_hinge=args.w_auc_hinge,
                            z_r2_hinge=args.z_r2_hinge,
                            ess_ratio_target=args.ess_ratio_target,
                            ate_scale=args.ate_scale,
                        )
                        all_rows.append(r)

                        if objective_mode == "single" and not isinstance(pruner, optuna.pruners.NopPruner):
                            _, sc_stats = aggregate_runs(all_rows, scenarios=scenarios, agg_repeats=args.agg)
                            score = composite_score_from_scenarios(
                                sc_stats,
                                w_auc_hinge=args.w_auc_hinge,
                                z_r2_hinge=args.z_r2_hinge,
                                ess_ratio_target=args.ess_ratio_target,
                            )
                            trial.report(score, step=step)
                            step += 1
                            if trial.should_prune():
                                raise optuna.TrialPruned()

        except optuna.TrialPruned:
            raise
        except Exception as e:  # noqa: PIE786 - best-effort diagnostics
            trial.set_user_attr("error", repr(e)[:300])
            trial.set_user_attr("fail_stage", "simulate/fit/estimate")
            if objective_mode == "pareto":
                return (float("inf"), float("inf"), float("inf"), float("inf"))
            return float("inf")

        m, sc_stats = aggregate_runs(all_rows, scenarios=scenarios, agg_repeats=args.agg)

        trial.set_user_attr("mean_ate_err", m["ate_err"])
        trial.set_user_attr("mean_w_auc", m["w_auc"])
        trial.set_user_attr("mean_w_auc_eff", m.get("w_auc_eff", np.nan))
        trial.set_user_attr("mean_w_leak", m.get("w_leak", np.nan))
        trial.set_user_attr("mean_z_r2", m["z_r2"])
        trial.set_user_attr("mean_z_leak", m.get("z_leak", np.nan))
        trial.set_user_attr("mean_ess_ratio", m["ess_ratio"])
        trial.set_user_attr("mean_train_time", m["train_time"])
        trial.set_user_attr("scenario_stats", json.dumps(sc_stats))

        if objective_mode == "pareto":
            return (
                m["ate_err"],
                m["w_violation"],
                m["z_violation"],
                m["ess_violation"],
            )

        return composite_score_from_scenarios(
            sc_stats,
            w_auc_hinge=args.w_auc_hinge,
            z_r2_hinge=args.z_r2_hinge,
            ess_ratio_target=args.ess_ratio_target,
        )

    timeout = args.timeout_hours * 3600 if args.timeout_hours else None

    if not args.quiet:
        print("\n" + "=" * 90)
        print("Optuna Bayesian Search")
        print("=" * 90)
        print(f"Output dir:     {outdir}")
        print(f"Storage:        {storage}")
        print(f"Study:          {study_name}")
        print(f"Mode:           {args.mode}")
        print(f"Objective:      {objective_mode}")
        print(f"Trials:         {args.n_trials}")
        print(f"Jobs:           {args.n_jobs}")
        print(f"Scenarios:      {scenarios}")
        print(f"n:              {args.n}")
        print(f"seeds:          {seeds}  repeats={args.n_repeats}")
        print(f"epochs:         {args.epochs}  pretrain={args.pretrain_epochs}")
        print(
            "thresholds: "
            f"W_auc<=hinge {args.w_auc_hinge} / feasible {args.w_auc_feasible}, "
            f"Z_r2<=hinge {args.z_r2_hinge} / feasible {args.z_r2_feasible}, "
            f"ESS>=hinge {args.ess_ratio_target} / feasible {args.ess_ratio_feasible}, "
            f"ate_scale={args.ate_scale}"
        )
        if args.plateau_window > 0:
            print(
                f"plateau: window={args.plateau_window}, rel_impr_thr={args.plateau_rel_impr}, min_trials={args.plateau_min_trials}"
            )
        if args.validate_best:
            print(
                f"validation: repeats={args.val_repeats}, seed_offset={args.val_seed_offset}, scenarios={val_scenarios}"
            )
        print("=" * 90 + "\n")

    callbacks: List[Callable[[optuna.Study, optuna.trial.FrozenTrial], None]] = []
    if args.plateau_window > 0:
        def _plateau_cb(study: optuna.Study, _trial: optuna.trial.FrozenTrial) -> None:
            if check_convergence_plateau(
                study,
                window=args.plateau_window,
                rel_impr_thr=args.plateau_rel_impr,
                min_trials=args.plateau_min_trials,
                objective_index=0,
            ):
                if not args.quiet:
                    print("[Stop] Plateau detected. Early stopping.")
                study.stop()

        callbacks.append(_plateau_cb)

    study.optimize(
        objective,
        n_trials=args.n_trials,
        n_jobs=args.n_jobs,
        timeout=timeout,
        show_progress_bar=(not args.quiet),
        callbacks=callbacks,
    )

    attr_cols = ("number", "values", "params", "user_attrs", "state") if objective_mode == "pareto" else (
        "number",
        "value",
        "params",
        "user_attrs",
        "state",
    )
    df = study.trials_dataframe(attrs=attr_cols)
    df.to_csv(outdir / "trials.csv", index=False)

    best_params_for_validation: Optional[Dict[str, Any]] = None
    if objective_mode == "pareto":
        pareto = study.best_trials
        pareto_out = []
        for t in pareto:
            pareto_out.append(
                {
                    "trial": t.number,
                    "values": list(t.values),
                    "params": t.params,
                    "user_attrs": dict(t.user_attrs),
                }
            )
        (outdir / "pareto_best_trials.json").write_text(json.dumps(pareto_out, indent=2))
        pareto_sorted = sorted(pareto_out, key=lambda x: x["values"][0])
        (outdir / "pareto_top10_by_ate_err.json").write_text(json.dumps(pareto_sorted[:10], indent=2))

        def _metric_from_trial(tr: optuna.trial.FrozenTrial) -> Dict[str, float]:
            ua = tr.user_attrs
            return {
                "ate_err": safe_float(ua.get("mean_ate_err", np.nan)),
                "w_auc_eff": safe_float(ua.get("mean_w_auc_eff", ua.get("mean_w_auc", np.nan))),
                "z_r2": safe_float(ua.get("mean_z_r2", np.nan)),
                "ess_ratio": safe_float(ua.get("mean_ess_ratio", np.nan)),
            }

        feasible: List[Dict[str, Any]] = []
        completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        for t in completed:
            met = _metric_from_trial(t)
            if (
                np.isfinite(list(met.values())).all()
                and met["w_auc_eff"] <= args.w_auc_feasible
                and met["z_r2"] <= args.z_r2_feasible
                and met["ess_ratio"] >= args.ess_ratio_feasible
            ):
                feasible.append({"trial": t.number, "params": t.params, "metrics": met, "user_attrs": dict(t.user_attrs)})

        recommended: Dict[str, Any]
        if feasible:
            recommended = min(feasible, key=lambda x: x["metrics"].get("ate_err", float("inf")))
            (outdir / "pareto_feasible.json").write_text(json.dumps(feasible, indent=2))
            best_params_for_validation = recommended.get("params")
        else:
            fallback: List[Dict[str, Any]] = []
            for t in completed:
                met = _metric_from_trial(t)
                if not np.isfinite(list(met.values())).all():
                    continue
                fallback.append(
                    {
                        "trial": t.number,
                        "params": t.params,
                        "user_attrs": dict(t.user_attrs),
                        "metrics": met,
                        "violation": max(0.0, met["w_auc_eff"] - args.w_auc_feasible)
                        + max(0.0, met["z_r2"] - args.z_r2_feasible)
                        + max(0.0, args.ess_ratio_feasible - met["ess_ratio"]),
                    }
                )
            recommended = min(fallback, key=lambda x: x.get("violation", float("inf"))) if fallback else {}
            if recommended:
                best_params_for_validation = recommended.get("params")

        if not best_params_for_validation and pareto_out:
            best_params_for_validation = pareto_out[0].get("params")

        if recommended:
            (outdir / "recommended_params.json").write_text(json.dumps(recommended, indent=2))
    else:
        best = {
            "best_value": float(study.best_value),
            "best_trial": int(study.best_trial.number),
            "best_params": dict(study.best_params),
            "user_attrs": dict(study.best_trial.user_attrs),
            "n_trials": len(study.trials),
        }
        (outdir / "best_params.json").write_text(json.dumps(best, indent=2))
        best_params_for_validation = dict(study.best_params)

    val_summary: Optional[Dict[str, Any]] = None
    if args.validate_best and best_params_for_validation:
        try:
            val_summary = validate_best_config(
                est_mod=est_mod,
                estimator_kind=args.estimator,
                scenario_fn=scenario_fn,
                scenarios=val_scenarios,
                params=best_params_for_validation,
                args=args,
                outdir=outdir,
            )
            if not args.quiet:
                print("Validation summary:", val_summary)
        except Exception as e:  # noqa: PIE786 - best-effort validation
            (outdir / "validation_error.txt").write_text(str(e))

    meta = {
        "mode": args.mode,
        "multiobjective": bool(objective_mode == "pareto"),
        "n_trials": args.n_trials,
        "n_jobs": args.n_jobs,
        "scenarios": scenarios,
        "n": args.n,
        "seeds": seeds,
        "n_repeats": args.n_repeats,
        "epochs": args.epochs,
        "pretrain_epochs": args.pretrain_epochs,
        "estimator": args.estimator,
        "estimator_module": getattr(est_mod, "__name__", str(est_mod)),
        "scenario_module": gen_mod_name,
        "scenario_fn": gen_fn_name,
        "storage": storage,
        "study_name": study_name,
        "objective_mode": objective_mode,
        "agg": args.agg,
        "w_auc_hinge": args.w_auc_hinge,
        "w_auc_feasible": args.w_auc_feasible,
        "z_r2_hinge": args.z_r2_hinge,
        "z_r2_feasible": args.z_r2_feasible,
        "ess_ratio_target": args.ess_ratio_target,
        "ess_ratio_feasible": args.ess_ratio_feasible,
        "ate_scale": args.ate_scale,
        "plateau_window": args.plateau_window,
        "plateau_rel_impr": args.plateau_rel_impr,
        "plateau_min_trials": args.plateau_min_trials,
        "validate_best": args.validate_best,
        "val_repeats": args.val_repeats,
        "val_seed_offset": args.val_seed_offset,
        "val_scenarios": val_scenarios,
        "validation_summary": val_summary,
        "timestamp": datetime.now().isoformat(),
    }
    (outdir / "run_meta.json").write_text(json.dumps(meta, indent=2))

    if not args.quiet:
        print("\n✓ Done.")
        print(f"✓ Saved: {outdir}")
        print(f"✓ Trials CSV: {outdir / 'trials.csv'}")
        print(f"✓ Study DB: {outdir / 'optuna_study.db'}")
        if objective_mode == "pareto":
            print(f"✓ Pareto JSON: {outdir / 'pareto_best_trials.json'}")
            print(f"✓ Recommended: {outdir / 'recommended_params.json'}")
        else:
            print(f"✓ Best params: {outdir / 'best_params.json'}")


if __name__ == "__main__":
    main()
