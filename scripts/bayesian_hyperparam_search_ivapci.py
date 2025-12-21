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
      - dict: {V,A,Y,tau_true,x_dim,w_dim,z_dim}
      - dict: {X,W,Z,A,Y,tau_true}
      - tuple: (V,A,Y,tau_true,meta)  meta含 x_dim/w_dim/z_dim 或 X/W/Z
      - tuple: (X,W,Z,A,Y,tau_true)
    """
    if isinstance(out, dict):
        if {"V", "A", "Y", "tau_true"}.issubset(out.keys()):
            V = np.asarray(out["V"])
            A = np.asarray(out["A"]).reshape(-1)
            Y = np.asarray(out["Y"]).reshape(-1)
            tau = float(out["tau_true"])
            if {"x_dim", "w_dim", "z_dim"}.issubset(out.keys()):
                return V, A, Y, tau, int(out["x_dim"]), int(out["w_dim"]), int(out["z_dim"])
            meta = out.get("meta", None)
            if isinstance(meta, dict) and {"x_dim", "w_dim", "z_dim"}.issubset(meta.keys()):
                return V, A, Y, tau, int(meta["x_dim"]), int(meta["w_dim"]), int(meta["z_dim"])
            raise ValueError("Scenario dict has V/A/Y/tau_true but missing x_dim/w_dim/z_dim.")
        if {"X", "W", "Z", "A", "Y", "tau_true"}.issubset(out.keys()):
            X = np.asarray(out["X"])
            W = np.asarray(out["W"])
            Z = np.asarray(out["Z"])
            V = np.concatenate([X, W, Z], axis=1)
            A = np.asarray(out["A"]).reshape(-1)
            Y = np.asarray(out["Y"]).reshape(-1)
            tau = float(out["tau_true"])
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


def run_one_fit(
    est_mod: Any,
    estimator_kind: str,
    cfg_overrides: Dict[str, Any],
    scenario_fn: Callable[..., Any],
    scenario_name: str,
    n: int,
    seed: int,
    epochs_main: int,
    pretrain_epochs: Optional[int],
) -> Dict[str, Any]:
    # resolve classes
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

    # apply overrides
    for k, v in cfg_overrides.items():
        setattr(cfg, k, parse_hidden_spec(v))

    t0 = time.time()
    est = Est(config=cfg)
    est.fit(V, A, Y)
    train_time = time.time() - t0

    ate_hat = safe_float(est.estimate_ate(V, A, Y))
    err = ate_hat - float(tau_true)

    try:
        diag = est.get_training_diagnostics()
    except Exception:
        diag = {}

    return {
        "status": "success",
        "scenario": scenario_name,
        "seed": int(seed),
        "true_ate": float(tau_true),
        "ate_est": float(ate_hat),
        "sq_err": float(err * err),
        "train_time": float(train_time),
        "rep_auc_w_to_a": safe_float(diag.get("rep_auc_w_to_a", np.nan)),
        "rep_exclusion_leakage_r2": safe_float(diag.get("rep_exclusion_leakage_r2", np.nan)),
        "overlap_ess_min": safe_float(diag.get("overlap_ess_min", np.nan)),
        "iv_first_stage_f": safe_float(diag.get("iv_first_stage_f", np.nan)),
        "weak_iv_flag": safe_float(diag.get("weak_iv_flag", np.nan)),
    }


def aggregate_metrics(rows: List[Dict[str, Any]]) -> Dict[str, float]:
    ok = [r for r in rows if r.get("status") == "success"]
    if not ok:
        return {
            "mean_rmse": float("inf"),
            "mean_w_auc": float("nan"),
            "mean_z_leak": float("nan"),
            "mean_overlap_ess_min": float("nan"),
            "mean_train_time": float("inf"),
        }
    rmse = float(np.sqrt(np.mean([r["sq_err"] for r in ok])))
    w_auc = float(np.nanmean([r["rep_auc_w_to_a"] for r in ok]))
    z_leak = float(np.nanmean([r["rep_exclusion_leakage_r2"] for r in ok]))
    ess = float(np.nanmean([r["overlap_ess_min"] for r in ok]))
    tt = float(np.mean([r["train_time"] for r in ok]))
    return {
        "mean_rmse": rmse,
        "mean_w_auc": w_auc,
        "mean_z_leak": z_leak,
        "mean_overlap_ess_min": ess,
        "mean_train_time": tt,
    }


def composite_score(m: Dict[str, float]) -> float:
    """
    单目标：越小越好。
    这里不做全局 min-max（Optuna 运行中拿不到全局信息），用“可解释的罚项”：
      score = RMSE
            + 0.6 * |W_AUC-0.5|
            + 1.2 * max(0, Z_leak-0.10)
            + 0.2 * max(0, 0.20-ESSmin)   # ESS 太低时惩罚
    你可以按经验调整权重/阈值。
    """
    rmse = m["mean_rmse"]
    w_dev = abs(m["mean_w_auc"] - 0.5) if np.isfinite(m["mean_w_auc"]) else 0.25
    z = m["mean_z_leak"]
    ess = m["mean_overlap_ess_min"]

    z_pen = max(0.0, (z - 0.10)) if np.isfinite(z) else 0.2
    ess_pen = max(0.0, (0.20 - ess)) if np.isfinite(ess) else 0.2
    return float(rmse + 0.6 * w_dev + 1.2 * z_pen + 0.2 * ess_pen)


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
    parser.add_argument("--multiobjective", action="store_true", help="optimize (rmse, |w_auc-0.5|, z_leak) pareto front")
    parser.add_argument("--sampler", choices=["tpe", "nsga2"], default="tpe")
    parser.add_argument("--pruner", choices=["median", "hyperband", "none"], default="median")
    parser.add_argument("--quiet", action="store_true")

    args = parser.parse_args()

    est_mod = resolve_estimator_module(args.estimator_module)
    gen_mod_name, gen_fn_name, scenario_fn = resolve_scenario_generator(args.scenario_module, args.scenario_fn)
    scenarios = normalize_scenarios_arg(args.scenarios, gen_mod_name)

    seeds = [int(x) for x in args.seeds.replace(",", " ").split() if x.strip()]

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = Path(args.output_dir or f"optuna_search_{args.mode}_{ts}")
    outdir.mkdir(parents=True, exist_ok=True)

    study_name = args.study_name or f"ivapci_{args.mode}_{ts}"
    storage = f"sqlite:///{outdir}/optuna_study.db"

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
    if args.pruner == "median":
        pruner: optuna.pruners.BasePruner = optuna.pruners.MedianPruner(n_startup_trials=8, n_warmup_steps=0)
    elif args.pruner == "hyperband":
        pruner = optuna.pruners.HyperbandPruner()
    else:
        pruner = optuna.pruners.NopPruner()

    directions = ["minimize", "minimize", "minimize"] if args.multiobjective else ["minimize"]

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

        for sc in scenarios:
            for base_seed in seeds:
                for rep in range(args.n_repeats):
                    seed = make_seed(base_seed, trial.number, sc, rep)
                    try:
                        r = run_one_fit(
                            est_mod=est_mod,
                            estimator_kind=args.estimator,
                            cfg_overrides=params,
                            scenario_fn=scenario_fn,
                            scenario_name=sc,
                            n=args.n,
                            seed=seed,
                            epochs_main=args.epochs,
                            pretrain_epochs=args.pretrain_epochs,
                        )
                    except Exception:
                        return float("inf") if not args.multiobjective else (float("inf"), 1.0, 1.0)

                    all_rows.append(r)

                    if not args.multiobjective:
                        m = aggregate_metrics(all_rows)
                        score = composite_score(m)
                        trial.report(score, step=step)
                        step += 1
                        if trial.should_prune():
                            raise optuna.TrialPruned()

        m = aggregate_metrics(all_rows)

        trial.set_user_attr("mean_rmse", m["mean_rmse"])
        trial.set_user_attr("mean_w_auc", m["mean_w_auc"])
        trial.set_user_attr("mean_z_leak", m["mean_z_leak"])
        trial.set_user_attr("mean_overlap_ess_min", m["mean_overlap_ess_min"])
        trial.set_user_attr("mean_train_time", m["mean_train_time"])

        if args.multiobjective:
            w_dev = abs(m["mean_w_auc"] - 0.5) if np.isfinite(m["mean_w_auc"]) else 0.25
            z = m["mean_z_leak"] if np.isfinite(m["mean_z_leak"]) else 1.0
            return (m["mean_rmse"], w_dev, z)

        return composite_score(m)

    timeout = args.timeout_hours * 3600 if args.timeout_hours else None

    if not args.quiet:
        print("\n" + "=" * 90)
        print("Optuna Bayesian Search")
        print("=" * 90)
        print(f"Output dir:     {outdir}")
        print(f"Storage:        {storage}")
        print(f"Study:          {study_name}")
        print(f"Mode:           {args.mode}")
        print(f"Multiobjective: {args.multiobjective}")
        print(f"Trials:         {args.n_trials}")
        print(f"Jobs:           {args.n_jobs}")
        print(f"Scenarios:      {scenarios}")
        print(f"n:              {args.n}")
        print(f"seeds:          {seeds}  repeats={args.n_repeats}")
        print(f"epochs:         {args.epochs}  pretrain={args.pretrain_epochs}")
        print("=" * 90 + "\n")

    study.optimize(
        objective,
        n_trials=args.n_trials,
        n_jobs=args.n_jobs,
        timeout=timeout,
        show_progress_bar=(not args.quiet),
    )

    df = study.trials_dataframe(attrs=("number", "value", "params", "user_attrs", "state"))
    df.to_csv(outdir / "trials.csv", index=False)

    if args.multiobjective:
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
        (outdir / "pareto_top10_by_rmse.json").write_text(json.dumps(pareto_sorted[:10], indent=2))
    else:
        best = {
            "best_value": float(study.best_value),
            "best_trial": int(study.best_trial.number),
            "best_params": dict(study.best_params),
            "user_attrs": dict(study.best_trial.user_attrs),
            "n_trials": len(study.trials),
        }
        (outdir / "best_params.json").write_text(json.dumps(best, indent=2))

    meta = {
        "mode": args.mode,
        "multiobjective": bool(args.multiobjective),
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
        "timestamp": datetime.now().isoformat(),
    }
    (outdir / "run_meta.json").write_text(json.dumps(meta, indent=2))

    if not args.quiet:
        print("\n✓ Done.")
        print(f"✓ Saved: {outdir}")
        print(f"✓ Trials CSV: {outdir / 'trials.csv'}")
        print(f"✓ Study DB: {outdir / 'optuna_study.db'}")
        if args.multiobjective:
            print(f"✓ Pareto JSON: {outdir / 'pareto_best_trials.json'}")
        else:
            print(f"✓ Best params: {outdir / 'best_params.json'}")


if __name__ == "__main__":
    main()
