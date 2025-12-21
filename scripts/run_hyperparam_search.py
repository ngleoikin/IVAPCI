#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_hyperparam_search.py — IVAPCI v3.3 超参数搜索（可直接运行版）

目标：
- 在你现有的 simulation 场景生成器上，批量测试一组 IVAPCI(v3.3) 超参数
- 自动汇总 ATE 误差（RMSE） + 泄漏诊断（W→A AUC、Z 泄漏 R²、overlap ESS）+ 训练耗时
- 输出 results.csv / best_config.json / top5_configs.json / checkpoint.json 等

用法示例：
    # Quick（推荐先跑）
    python run_hyperparam_search.py --mode quick --scenarios EASY-linear-weak,MODERATE-nonlinear

    # Balanced（更全面）
    python run_hyperparam_search.py --mode balanced --scenarios all --n-repeats 5 --epochs 120

    # Focused W（专攻 W 泄漏）
    python run_hyperparam_search.py --mode focused_w --scenarios MODERATE-nonlinear --n-repeats 3

你只需要保证：
1) 你的代码库里有“场景生成函数”，能按 scenario_name + seed + n 生成数据
2) 你的 IVAPCI v3.3 estimator 模块可 import（本脚本会自动探测 v27/v26/主模块）

场景生成函数要求（推荐二选一）：
A) 返回 dict:
   {
     "V": np.ndarray, "A": np.ndarray, "Y": np.ndarray, "tau_true": float,
     "x_dim": int, "w_dim": int, "z_dim": int
   }
B) 返回 tuple:
   (V, A, Y, tau_true, meta)
   其中 meta 至少包含 x_dim/w_dim/z_dim，或者包含 X/W/Z 三个数组。

如果你的生成函数名字/模块不同，用参数指定：
    --scenario-module simulation_configs
    --scenario-fn generate_scenario

说明：
- 本脚本不会调用 scripts/run_simulation_benchmark.py，而是直接 in-process 训练 estimator，
  这样可以对 config 做细粒度覆盖，不需要改你的 benchmark CLI。
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterGrid


# ==================== 搜索空间（可按需修改） ====================

SEARCH_SPACES: Dict[str, Dict[str, List[Any]]] = {
    "quick": {
        # 🔴 W 独立性（核心问题）
        "gamma_adv_w": [0.20, 0.25, 0.30],
        "lambda_hsic": [0.05, 0.08],
        "lambda_hsic_w_a": [0.03, 0.05],

        # 🟡 Z 排他性（固定推荐值）
        "gamma_adv_z": [0.18],
        "dropout_z": [0.30],
        "enc_z_hidden": ["32-16"],

        # 🟢 控制器（固定较稳的目标；0.51 太狠，容易长期饱和）
        "ctrl_w_auc_target": [0.54],
        "ctrl_kp_w": [2.2],
        "ctrl_ki_w": [0.55],
    },
    "balanced": {
        # W 独立性
        "gamma_adv_w": [0.20, 0.25, 0.30],
        "gamma_adv_w_cond": [0.16, 0.20, 0.22],
        "lambda_hsic": [0.04, 0.06, 0.08],
        "lambda_hsic_w_a": [0.03, 0.05, 0.07],

        # Z 排他性
        "gamma_adv_z": [0.15, 0.18, 0.22],
        "gamma_adv_z_cond": [0.12, 0.16],
        "dropout_z": [0.25, 0.30, 0.35],
        "enc_z_hidden": ["64-32", "32-16"],

        # 条件正交（略早、略强）
        "lambda_cond_ortho": [0.008, 0.012, 0.015],
        "cond_ortho_warmup_epochs": [5, 8],

        # 控制器
        "ctrl_w_auc_target": [0.53, 0.54],
        "ctrl_kp_w": [2.0, 2.2],
        "ctrl_ki_w": [0.50, 0.55],
        "ctrl_z_r2_target": [0.10, 0.12],
    },
    "focused_w": {
        # 密集搜索 W
        "gamma_adv_w": [0.20, 0.23, 0.26, 0.29],
        "gamma_adv_w_cond": [0.16, 0.18, 0.20, 0.22],
        "lambda_hsic": [0.05, 0.07, 0.09, 0.11],
        "lambda_hsic_w_a": [0.03, 0.05, 0.07, 0.09],

        # 控制器 W
        "ctrl_w_auc_target": [0.52, 0.53, 0.54],
        "ctrl_kp_w": [2.1, 2.3, 2.5],
        "ctrl_ki_w": [0.50, 0.55, 0.60],

        # Z 固定（避免扩大搜索维度）
        "gamma_adv_z": [0.18],
        "dropout_z": [0.30],
        "enc_z_hidden": ["32-16"],
    },
}


# ==================== 工具函数 ====================


def _stable_hash_int(s: str, mod: int = 2**31 - 1) -> int:
    h = hashlib.md5(s.encode("utf-8")).hexdigest()
    return int(h[:8], 16) % mod


def _parse_hidden_spec(spec: Any) -> Any:
    if isinstance(spec, str) and "-" in spec and all(p.isdigit() for p in spec.split("-")):
        return tuple(int(x) for x in spec.split("-"))
    return spec


def _try_import_first(candidates: Sequence[str]) -> Tuple[str, Any]:
    last_err = None
    for m in candidates:
        try:
            return m, importlib.import_module(m)
        except Exception as e:
            last_err = e
    raise ImportError(f"Could not import any estimator module candidates: {candidates}. Last error: {last_err}")


def _resolve_estimator_module(user_spec: Optional[str]) -> Any:
    candidates = []
    if user_spec:
        candidates.append(user_spec)
    candidates += [
        "ivapci_v33_theory_v27",
        "ivapci_v33_theory_v26",
        "ivapci_v33_theory",
    ]
    mod_name, mod = _try_import_first(candidates)
    print(f"[Info] Using estimator module: {mod_name}")
    return mod


def _resolve_scenario_generator(module_name: Optional[str], fn_name: Optional[str]) -> Tuple[str, str, Callable[..., Any]]:
    module_candidates = []
    if module_name:
        module_candidates.append(module_name)
    module_candidates += [
        "simulation_configs",
        "scripts.simulation_configs",
        "simulation_scenarios",
        "scripts.simulation_scenarios",
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


def _list_available_scenarios(gen_module_name: str) -> Optional[List[str]]:
    try:
        m = importlib.import_module(gen_module_name)
    except Exception:
        return None

    if hasattr(m, "get_available_scenarios") and callable(getattr(m, "get_available_scenarios")):
        try:
            vals = list(getattr(m, "get_available_scenarios")())
            return [str(v) for v in vals]
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


def _normalize_scenarios_arg(s: str, gen_module_name: Optional[str]) -> List[str]:
    s = s.strip()
    if s.lower() == "all":
        if not gen_module_name:
            raise ValueError("scenarios=all requires a resolvable scenario module.")
        avail = _list_available_scenarios(gen_module_name)
        if not avail:
            raise ValueError(
                f"Could not list scenarios from module {gen_module_name}. "
                "Please pass explicit scenario names, e.g. --scenarios EASY-linear-weak,MODERATE-nonlinear"
            )
        return avail
    if "," in s:
        parts = [p.strip() for p in s.split(",") if p.strip()]
    else:
        parts = [p.strip() for p in s.split() if p.strip()]
    if not parts:
        raise ValueError("No scenarios provided.")
    return parts


def _extract_data_and_dims(out: Any) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, int, int, int]:
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
            raise ValueError("Scenario generator returned dict with V/A/Y/tau_true but missing x_dim/w_dim/z_dim.")
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
            X = np.asarray(X); W = np.asarray(W); Z = np.asarray(Z)
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
                X = np.asarray(meta["X"]); W = np.asarray(meta["W"]); Z = np.asarray(meta["Z"])
                return V, A, Y, tau, X.shape[1], W.shape[1], Z.shape[1]
            raise ValueError("Scenario tuple has meta but meta lacks x_dim/w_dim/z_dim (or X/W/Z).")
    raise ValueError(f"Unsupported scenario output type: {type(out)}")


def _safe_float(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def _minmax_norm(arr: np.ndarray) -> np.ndarray:
    a = np.asarray(arr, dtype=float)
    mn = np.nanmin(a)
    mx = np.nanmax(a)
    if not np.isfinite(mn) or not np.isfinite(mx) or abs(mx - mn) < 1e-12:
        return np.zeros_like(a)
    return (a - mn) / (mx - mn)


# ==================== 搜索运行器 ====================


class HyperparameterSearchRunner:
    def __init__(
        self,
        mode: str,
        scenarios: List[str],
        n: int,
        base_seeds: List[int],
        n_repeats: int,
        epochs_main: int,
        epochs_pretrain: Optional[int],
        estimator_module: Any,
        scenario_fn: Callable[..., Any],
        estimator_kind: str = "hier",
        output_dir: str = "",
        verbose: bool = True,
        checkpoint_every: int = 5,
    ):
        if mode not in SEARCH_SPACES:
            raise ValueError(f"Unknown mode '{mode}'. Choose from {list(SEARCH_SPACES.keys())}")
        if estimator_kind not in ["hier", "hier_radr"]:
            raise ValueError("estimator_kind must be one of: hier, hier_radr")

        self.mode = mode
        self.scenarios = scenarios
        self.n = int(n)
        self.base_seeds = [int(s) for s in base_seeds]
        self.n_repeats = int(n_repeats)
        self.epochs_main = int(epochs_main)
        self.epochs_pretrain = None if epochs_pretrain is None else int(epochs_pretrain)
        self.estimator_module = estimator_module
        self.scenario_fn = scenario_fn
        self.estimator_kind = estimator_kind
        self.verbose = verbose
        self.checkpoint_every = int(checkpoint_every)

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.param_grid = list(ParameterGrid(SEARCH_SPACES[mode]))
        self.results_runs: List[Dict[str, Any]] = []
        self.results_scenario: List[Dict[str, Any]] = []
        self.results_config: List[Dict[str, Any]] = []
        self.start_time = time.time()

        self._resume_from_checkpoint()

        if self.verbose:
            self._print_header()

    def _print_header(self):
        total = len(self.param_grid) * len(self.scenarios) * (len(self.base_seeds) * self.n_repeats)
        print("\n" + "=" * 90)
        print("IVAPCI v3.3 Hyperparam Search")
        print("=" * 90)
        print(f"Mode:              {self.mode}")
        print(f"Estimator:         {self.estimator_kind}")
        print(f"Scenarios:         {len(self.scenarios)} -> {self.scenarios}")
        print(f"n:                 {self.n}")
        print(f"Base seeds:        {self.base_seeds}")
        print(f"Repeats/seed:      {self.n_repeats}")
        print(f"Configs:           {len(self.param_grid)}")
        print(f"Total runs:        {total}")
        print(f"epochs_main:       {self.epochs_main}")
        if self.epochs_pretrain is not None:
            print(f"epochs_pretrain:   {self.epochs_pretrain}")
        print(f"Output dir:        {self.output_dir}")
        print("=" * 90 + "\n")

    def _resume_from_checkpoint(self):
        ck = self.output_dir / "checkpoint.json"
        if not ck.exists():
            return
        try:
            payload = json.loads(ck.read_text())
            self.results_runs = payload.get("results_runs", [])
            self.results_scenario = payload.get("results_scenario", [])
            self.results_config = payload.get("results_config", [])
            if self.verbose:
                print(f"[Info] Resumed from checkpoint: {ck} (runs={len(self.results_runs)})")
        except Exception:
            pass

    def _save_checkpoint(self):
        ck = self.output_dir / "checkpoint.json"
        payload = {
            "mode": self.mode,
            "scenarios": self.scenarios,
            "n": self.n,
            "base_seeds": self.base_seeds,
            "n_repeats": self.n_repeats,
            "epochs_main": self.epochs_main,
            "epochs_pretrain": self.epochs_pretrain,
            "estimator_kind": self.estimator_kind,
            "timestamp": datetime.now().isoformat(),
            "elapsed_sec": time.time() - self.start_time,
            "results_runs": self.results_runs,
            "results_scenario": self.results_scenario,
            "results_config": self.results_config,
        }
        ck.write_text(json.dumps(payload, indent=2))

    def _get_config_and_estimator_classes(self):
        Cfg = getattr(self.estimator_module, "IVAPCIV33TheoryConfig", None)
        if Cfg is None:
            raise AttributeError("Estimator module missing IVAPCIV33TheoryConfig")
        if self.estimator_kind == "hier":
            Est = getattr(self.estimator_module, "IVAPCIv33TheoryHierEstimator", None)
        else:
            Est = getattr(self.estimator_module, "IVAPCIv33TheoryHierRADREstimator", None)
        if Est is None:
            raise AttributeError(f"Estimator module missing estimator class for kind={self.estimator_kind}")
        return Cfg, Est

    def _apply_params(self, base_cfg: Any, params: Dict[str, Any]) -> Any:
        import copy

        cfg = copy.deepcopy(base_cfg)
        for k, v in params.items():
            setattr(cfg, k, _parse_hidden_spec(v))

        cfg.epochs_main = self.epochs_main
        if self.epochs_pretrain is not None:
            cfg.epochs_pretrain = self.epochs_pretrain
        cfg.early_stopping_patience = min(
            getattr(cfg, "early_stopping_patience", 18), max(8, self.epochs_main // 8)
        )
        return cfg

    def _seed_for(self, config_id: int, scenario: str, base_seed: int, repeat: int) -> int:
        return (
            base_seed * 1_000_000
            + config_id * 10_000
            + _stable_hash_int(scenario) % 10_000
            + repeat
        ) % (2**31 - 1)

    def _run_one(self, scenario_name: str, cfg: Any, Est: Any, seed: int) -> Dict[str, Any]:
        t0 = time.time()
        out = self.scenario_fn(scenario_name, n=self.n, seed=seed)
        V, A, Y, tau_true, x_dim, w_dim, z_dim = _extract_data_and_dims(out)
        cfg.x_dim, cfg.w_dim, cfg.z_dim = int(x_dim), int(w_dim), int(z_dim)

        est = Est(config=cfg)
        est.fit(V, A, Y)
        train_time = time.time() - t0

        ate_hat = _safe_float(est.estimate_ate(V, A, Y))
        err = ate_hat - float(tau_true)
        try:
            diag = est.get_training_diagnostics()
        except Exception:
            diag = {}

        return {
            "status": "success",
            "scenario": scenario_name,
            "seed": int(seed),
            "ate_est": float(ate_hat),
            "true_ate": float(tau_true),
            "err": float(err),
            "abs_err": float(abs(err)),
            "sq_err": float(err * err),
            "train_time": float(train_time),
            "rep_auc_w_to_a": _safe_float(diag.get("rep_auc_w_to_a", np.nan)),
            "rep_exclusion_leakage_r2": _safe_float(diag.get("rep_exclusion_leakage_r2", np.nan)),
            "overlap_ess_min": _safe_float(diag.get("overlap_ess_min", np.nan)),
            "iv_first_stage_f": _safe_float(diag.get("iv_first_stage_f", np.nan)),
            "weak_iv_flag": _safe_float(diag.get("weak_iv_flag", np.nan)),
        }

    def run(self) -> pd.DataFrame:
        Cfg, Est = self._get_config_and_estimator_classes()

        done_cfg_ids = set()
        for row in self.results_config:
            try:
                done_cfg_ids.add(int(row["config_id"]))
            except Exception:
                pass

        total_cfg = len(self.param_grid)

        for cfg_idx, params in enumerate(self.param_grid, start=1):
            if cfg_idx in done_cfg_ids:
                continue

            if self.verbose:
                print("\n" + "-" * 90)
                print(f"[Config {cfg_idx}/{total_cfg}] params:")
                for k, v in params.items():
                    print(f"  {k}: {v}")
                print("-" * 90)

            base_cfg = Cfg()
            cfg = self._apply_params(base_cfg, params)

            per_scenario_rows = []
            for sc in self.scenarios:
                run_rows = []
                for base_seed in self.base_seeds:
                    for rep in range(self.n_repeats):
                        seed = self._seed_for(cfg_idx, sc, base_seed, rep)
                        try:
                            r = self._run_one(sc, cfg, Est, seed)
                        except Exception as e:
                            r = {
                                "status": "failed",
                                "scenario": sc,
                                "seed": int(seed),
                                "error": str(e)[:220],
                            }
                        r["config_id"] = int(cfg_idx)
                        r.update(params)
                        self.results_runs.append(r)
                        run_rows.append(r)

                        if self.verbose and r.get("status") == "success":
                            print(
                                f"  {sc} seed={base_seed} rep={rep+1}: "
                                f"RMSE={np.sqrt(r['sq_err']):.4f}  "
                                f"W_AUC={r['rep_auc_w_to_a']:.3f}  "
                                f"Z_leak={r['rep_exclusion_leakage_r2']:.3f}  "
                                f"ESSmin={r['overlap_ess_min']:.3f}"
                            )
                        elif self.verbose:
                            print(f"  {sc} seed={base_seed} rep={rep+1}: FAILED {r.get('error','')}")

                ok = [x for x in run_rows if x.get("status") == "success"]
                if ok:
                    sc_row = {
                        "config_id": int(cfg_idx),
                        "scenario": sc,
                        "n_success": len(ok),
                        "n_total": len(run_rows),
                        "mean_rmse": float(np.sqrt(np.mean([x["sq_err"] for x in ok]))),
                        "mean_abs_err": float(np.mean([x["abs_err"] for x in ok])),
                        "mean_w_auc": float(np.nanmean([x["rep_auc_w_to_a"] for x in ok])),
                        "mean_z_leak": float(np.nanmean([x["rep_exclusion_leakage_r2"] for x in ok])),
                        "mean_overlap_ess_min": float(np.nanmean([x["overlap_ess_min"] for x in ok])),
                        "mean_iv_f": float(np.nanmean([x["iv_first_stage_f"] for x in ok])),
                        "mean_weak_iv_flag": float(np.nanmean([x["weak_iv_flag"] for x in ok])),
                        "mean_train_time": float(np.mean([x["train_time"] for x in ok])),
                    }
                    sc_row.update(params)
                    per_scenario_rows.append(sc_row)
                    self.results_scenario.append(sc_row)

            if per_scenario_rows:
                df_sc = pd.DataFrame(per_scenario_rows)
                cfg_row = {
                    "config_id": int(cfg_idx),
                    "mode": self.mode,
                    "estimator": self.estimator_kind,
                    "mean_rmse": float(df_sc["mean_rmse"].mean()),
                    "mean_w_auc": float(df_sc["mean_w_auc"].mean()),
                    "mean_z_leak": float(df_sc["mean_z_leak"].mean()),
                    "mean_overlap_ess_min": float(df_sc["mean_overlap_ess_min"].mean()),
                    "mean_iv_f": float(df_sc["mean_iv_f"].mean()),
                    "mean_weak_iv_flag": float(df_sc["mean_weak_iv_flag"].mean()),
                    "mean_train_time": float(df_sc["mean_train_time"].mean()),
                    "n_scenarios": int(len(df_sc)),
                }
                cfg_row.update(params)
                self.results_config.append(cfg_row)

            if cfg_idx % self.checkpoint_every == 0:
                self._save_checkpoint()

        self._save_final()
        return pd.DataFrame(self.results_config)

    def _save_final(self):
        df_cfg = pd.DataFrame(self.results_config)
        df_sc = pd.DataFrame(self.results_scenario) if self.results_scenario else pd.DataFrame()
        df_runs = pd.DataFrame(self.results_runs) if self.results_runs else pd.DataFrame()

        if not df_cfg.empty:
            w_dev = np.abs(df_cfg["mean_w_auc"].values - 0.5)
            rmse_n = _minmax_norm(df_cfg["mean_rmse"].values)
            wdev_n = _minmax_norm(w_dev)
            z_n = _minmax_norm(df_cfg["mean_z_leak"].values)
            t_n = _minmax_norm(df_cfg["mean_train_time"].values)

            df_cfg["w_auc_dev"] = w_dev
            df_cfg["score"] = 0.40 * rmse_n + 0.30 * wdev_n + 0.20 * z_n + 0.10 * t_n
            df_cfg = df_cfg.sort_values("score", ascending=True)

        (self.output_dir / "results.csv").write_text(df_cfg.to_csv(index=False))
        if not df_sc.empty:
            (self.output_dir / "results_by_scenario.csv").write_text(df_sc.to_csv(index=False))
        if not df_runs.empty:
            (self.output_dir / "results_runs.csv").write_text(df_runs.to_csv(index=False))

        full = {
            "mode": self.mode,
            "estimator": self.estimator_kind,
            "scenarios": self.scenarios,
            "n": self.n,
            "base_seeds": self.base_seeds,
            "n_repeats": self.n_repeats,
            "epochs_main": self.epochs_main,
            "epochs_pretrain": self.epochs_pretrain,
            "search_space": SEARCH_SPACES[self.mode],
            "results_config": self.results_config,
            "results_scenario": self.results_scenario,
            "n_runs": len(self.results_runs),
            "elapsed_sec": time.time() - self.start_time,
        }
        (self.output_dir / "results_full.json").write_text(json.dumps(full, indent=2))

        if not df_cfg.empty:
            best = df_cfg.iloc[0].to_dict()
            (self.output_dir / "best_config.json").write_text(json.dumps(best, indent=2))
            top5 = df_cfg.head(5).to_dict(orient="records")
            (self.output_dir / "top5_configs.json").write_text(json.dumps(top5, indent=2))

        self._save_checkpoint()

        if self.verbose:
            print("\n" + "=" * 90)
            print("Search complete.")
            print(f"Elapsed: {(time.time() - self.start_time)/3600:.2f} hours")
            print(f"Saved to: {self.output_dir}")
            if not df_cfg.empty:
                print("Best config:")
                cols = [
                    "config_id",
                    "score",
                    "mean_rmse",
                    "mean_w_auc",
                    "mean_z_leak",
                    "mean_overlap_ess_min",
                    "mean_train_time",
                ]
                print(df_cfg[cols].head(1).to_string(index=False))
            print("=" * 90 + "\n")


# ==================== CLI ====================


def main():
    p = argparse.ArgumentParser(description="IVAPCI v3.3 hyperparameter search (in-process).")
    p.add_argument("--mode", choices=["quick", "balanced", "focused_w"], default="quick")
    p.add_argument("--scenarios", default="EASY-linear-weak,MODERATE-nonlinear", help="comma/space separated or 'all'")
    p.add_argument("--n", type=int, default=500)
    p.add_argument("--seeds", type=str, default="0,1", help="base seeds, e.g. '0,1,2' or '0 1'")
    p.add_argument("--n-repeats", type=int, default=3)
    p.add_argument("--epochs", type=int, default=100, help="epochs_main for search")
    p.add_argument("--pretrain-epochs", type=int, default=None, help="override epochs_pretrain (optional)")
    p.add_argument("--estimator", choices=["hier", "hier_radr"], default="hier")
    p.add_argument("--estimator-module", type=str, default=None, help="override estimator module import path")
    p.add_argument("--scenario-module", type=str, default=None, help="module containing scenario generator")
    p.add_argument("--scenario-fn", type=str, default=None, help="scenario function name")
    p.add_argument("--output-dir", type=str, default=None, help="output directory")
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--checkpoint-every", type=int, default=5)

    args = p.parse_args()

    seeds_str = args.seeds.replace(",", " ")
    base_seeds = [int(x) for x in seeds_str.split() if x.strip()]

    est_mod = _resolve_estimator_module(args.estimator_module)

    gen_mod_name, gen_fn_name, gen_fn = _resolve_scenario_generator(args.scenario_module, args.scenario_fn)
    scenarios = _normalize_scenarios_arg(args.scenarios, gen_mod_name)

    if args.output_dir is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"grid_search_{args.mode}_{ts}"
    else:
        output_dir = args.output_dir

    runner = HyperparameterSearchRunner(
        mode=args.mode,
        scenarios=scenarios,
        n=args.n,
        base_seeds=base_seeds,
        n_repeats=args.n_repeats,
        epochs_main=args.epochs,
        epochs_pretrain=args.pretrain_epochs,
        estimator_module=est_mod,
        scenario_fn=gen_fn,
        estimator_kind=args.estimator,
        output_dir=output_dir,
        verbose=(not args.quiet),
        checkpoint_every=args.checkpoint_every,
    )
    runner.run()


if __name__ == "__main__":
    main()
