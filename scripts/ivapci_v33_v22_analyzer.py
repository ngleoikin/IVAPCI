"""IVAPCI v3.3 (version 22) result analyzer.

Reads benchmark/diagnostic CSVs and prints summary + quality/overlap analyses.
Defaults match the filenames mentioned in user feedback; override via CLI flags.
"""

from __future__ import annotations

import argparse
import warnings
from typing import Dict, List

import numpy as np
import pandas as pd
import seaborn as sns


warnings.filterwarnings("ignore")
sns.set_style("whitegrid")


class IVAPCIv22Analyzer:
    """Analyze IVAPCI v3.3 version-22 experiment results."""

    def __init__(
        self,
        benchmark_file: str = "simulation_benchmark_results 22.csv",
        diagnostics_file: str = "simulation_diagnostics_results 22.csv",
        summary_file: str = "simulation_benchmark_summary 22.csv",
    ):
        print("📂 加载数据...")
        self.df_bench = pd.read_csv(benchmark_file)
        self.df_diag = pd.read_csv(diagnostics_file)
        self.df_summary = pd.read_csv(summary_file)

        print(f"   ✓ Benchmark: {len(self.df_bench)} 行")
        print(f"   ✓ Diagnostics: {len(self.df_diag)} 行")
        print(f"   ✓ Summary: {len(self.df_summary)} 行")

        self._compute_quality_scores()

    def _get(self, df: pd.DataFrame, key: str, default: float = np.nan) -> pd.Series:
        return df[key] if key in df.columns else pd.Series([default] * len(df))

    def _compute_quality_scores(self) -> None:
        """Compute a simple 0–4 quality score per row."""

        def score_row(row: pd.Series) -> int:
            score = 0
            if row.get("rep_auc_z_to_a", 0) > 0.7:
                score += 1
            w_auc = row.get("rep_auc_w_to_a", 0.5)
            if 0.45 < w_auc < 0.55:
                score += 1
            if row.get("rep_exclusion_leakage_r2", 1.0) < 0.1:
                score += 1
            if row.get("rep_r2_xw_a_to_y", 0) > 0.3:
                score += 1
            return score

        self.df_bench["quality_score"] = self.df_bench.apply(score_row, axis=1)
        self.df_diag["quality_score"] = self.df_diag.apply(score_row, axis=1)

    # ---------------- executive summary ----------------
    def executive_summary(self) -> None:
        print("\n" + "=" * 80)
        print(" " * 20 + "EXECUTIVE SUMMARY")
        print("=" * 80)

        print("\n【整体性能】")
        print(f"  平均绝对误差: {self.df_bench['abs_err'].mean():.4f}")
        print(f"  RMSE: {np.sqrt(self.df_bench['sq_err'].mean()):.4f}")
        print(f"  平均运行时间: {self.df_bench['runtime_sec'].mean():.2f}秒")

        if "method" in self.df_summary.columns:
            print("\n【方法对比】")
            for method in self.df_summary["method"].unique():
                df_m = self.df_summary[self.df_summary["method"] == method]
                print(f"  {method}:")
                print(f"    RMSE: {df_m['rmse'].mean():.4f}")
                print(f"    运行时间: {df_m['mean_runtime'].mean():.2f}秒")

        print("\n【质量评分分布】(0=最差, 4=最好；按方法汇总)")
        for method, df_m in self.df_bench.groupby("method"):
            qual_dist = df_m["quality_score"].value_counts().sort_index()
            bars = []
            for score, count in qual_dist.items():
                pct = count / len(df_m) * 100
                bars.append(f"{int(score)}分:{count:3d}({pct:5.1f}%)")
            bars_str = " | ".join(bars)
            print(f"  {method}: {bars_str}")

        iv_rel = self._get(self.df_bench, "iv_relevance_abs_corr")
        excl = self._get(self.df_bench, "iv_exclusion_abs_corr_resid")
        overlap = self._get(self.df_bench, "dr_overlap_score")

        print("\n【关键问题检测】")
        total = len(self.df_bench)
        weak_iv = (iv_rel < 0.15).sum()
        excl_viol = (excl > 0.2).sum()
        poor_overlap = (overlap < 0.7).sum()
        print(f"  ⚠️  弱IV: {weak_iv}/{total} ({weak_iv/total*100:.1f}%)")
        print(f"  ⚠️  排他性违反: {excl_viol}/{total} ({excl_viol/total*100:.1f}%)")
        print(f"  ⚠️  差重叠: {poor_overlap}/{total} ({poor_overlap/total*100:.1f}%)")
        print("=" * 80)

    # ---------------- identifiability ----------------
    def identifiability_analysis(self) -> None:
        print("\n" + "=" * 80)
        print("1️⃣  可识别性分析")
        print("=" * 80)

        iv_rel = self._get(self.df_bench, "iv_relevance_abs_corr")
        print("\n【IV相关性统计】")
        print(f"  均值: {iv_rel.mean():.4f}")
        print(f"  中位数: {iv_rel.median():.4f}")
        print(f"  标准差: {iv_rel.std():.4f}")
        print(f"  范围: [{iv_rel.min():.4f}, {iv_rel.max():.4f}]")

        strong = (iv_rel > 0.3).sum()
        moderate = ((iv_rel >= 0.15) & (iv_rel <= 0.3)).sum()
        weak = (iv_rel < 0.15).sum()

        print("\n  强度分类:")
        print(f"    强 (>0.3):     {strong:3d} ({strong/len(iv_rel)*100:5.1f}%)")
        print(f"    中等 (0.15-0.3): {moderate:3d} ({moderate/len(iv_rel)*100:5.1f}%)")
        print(f"    弱 (<0.15):    {weak:3d} ({weak/len(iv_rel)*100:5.1f}%) ⚠️")

        mask = iv_rel.notna()
        corr_iv_err = np.corrcoef(
            iv_rel[mask],
            self.df_bench.loc[mask, "abs_err"],
        )[0, 1]
        print(f"\n  📊 IV强度与误差的相关性: {corr_iv_err:.4f}")
        if corr_iv_err < -0.2:
            print("     ✓ 强IV显著减少误差")
        elif corr_iv_err < -0.1:
            print("     → IV强度有助于减少误差")
        else:
            print("     ⚠️ IV强度与误差关系不明显")

        iv_exc = self._get(self.df_bench, "iv_exclusion_abs_corr_resid")
        print("\n【排他性约束检查】")
        print(f"  均值: {iv_exc.mean():.4f}")
        print(f"  中位数: {iv_exc.median():.4f}")

        good_excl = (iv_exc < 0.1).sum()
        quest_excl = ((iv_exc >= 0.1) & (iv_exc < 0.2)).sum()
        viol_excl = (iv_exc >= 0.2).sum()

        print("\n  违反程度:")
        print(f"    良好 (<0.1):   {good_excl:3d} ({good_excl/len(iv_exc)*100:5.1f}%) ✓")
        print(f"    可疑 (0.1-0.2): {quest_excl:3d} ({quest_excl/len(iv_exc)*100:5.1f}%)")
        print(f"    违反 (>0.2):   {viol_excl:3d} ({viol_excl/len(iv_exc)*100:5.1f}%) ⚠️")

    # ---------------- representation ----------------
    def representation_quality_analysis(self) -> None:
        print("\n" + "=" * 80)
        print("2️⃣  表征质量分析")
        print("=" * 80)

        if "method" not in self.df_bench.columns:
            # Fallback to overall statistics when method column is absent
            groups = [("ALL", self.df_bench)]
        else:
            groups = list(self.df_bench.groupby("method"))

        for method, df_m in groups:
            print(f"\n--- 方法: {method} ---")

            z_auc = self._get(df_m, "rep_auc_z_to_a")
            print("【Z→A预测（IV强度）】")
            print(f"  均值: {z_auc.mean():.4f}")
            print(f"  中位数: {z_auc.median():.4f}")
            strong_z = (z_auc > 0.7).sum()
            mod_z = ((z_auc >= 0.6) & (z_auc <= 0.7)).sum()
            weak_z = (z_auc < 0.6).sum()
            print(f"  强 (>0.7):   {strong_z:3d} ({strong_z/len(z_auc)*100:5.1f}%) ✓")
            print(f"  中等 (0.6-0.7): {mod_z:3d} ({mod_z/len(z_auc)*100:5.1f}%)")
            print(f"  弱 (<0.6):   {weak_z:3d} ({weak_z/len(z_auc)*100:5.1f}%) ⚠️")

            w_auc = self._get(df_m, "rep_auc_w_to_a", 0.5)
            print("【W→A预测（应该~0.5）】")
            print(f"  均值: {w_auc.mean():.4f}")
            print(f"  中位数: {w_auc.median():.4f}")
            indep_w = ((w_auc > 0.45) & (w_auc < 0.55)).sum()
            dep_w = ((w_auc <= 0.45) | (w_auc >= 0.55)).sum()
            print(f"  独立 (0.45-0.55): {indep_w:3d} ({indep_w/len(w_auc)*100:5.1f}%) ✓")
            print(f"  依赖 (其他):    {dep_w:3d} ({dep_w/len(w_auc)*100:5.1f}%) ⚠️")

            leak = self._get(df_m, "rep_exclusion_leakage_r2", 0.0)
            print("【排他性泄露（Z→Y）】")
            print(f"  均值: {leak.mean():.4f}")
            print(f"  中位数: {leak.median():.4f}")
            good_leak = (leak < 0.1).sum()
            mod_leak = ((leak >= 0.1) & (leak < 0.2)).sum()
            bad_leak = (leak >= 0.2).sum()
            print(f"  低 (<0.1):    {good_leak:3d} ({good_leak/len(leak)*100:5.1f}%) ✓")
            print(f"  中等 (0.1-0.2): {mod_leak:3d} ({mod_leak/len(leak)*100:5.1f}%)")
            print(f"  高 (>0.2):    {bad_leak:3d} ({bad_leak/len(leak)*100:5.1f}%) ⚠️")

            r2_y = self._get(df_m, "rep_r2_xw_a_to_y", 0.0)
            print("【结果预测（[X,W,A]→Y）】")
            print(f"  均值: {r2_y.mean():.4f}")
            print(f"  中位数: {r2_y.median():.4f}")
            good_r2 = (r2_y > 0.3).sum()
            poor_r2 = (r2_y <= 0.3).sum()
            print(f"  好 (>0.3): {good_r2:3d} ({good_r2/len(r2_y)*100:5.1f}%) ✓")
            print(f"  差 (≤0.3): {poor_r2:3d} ({poor_r2/len(r2_y)*100:5.1f}%) ⚠️")

    # ---------------- propensity / overlap ----------------
    def propensity_overlap_analysis(self) -> None:
        print("\n" + "=" * 80)
        print("3️⃣  倾向性得分与重叠性分析")
        print("=" * 80)

        print("\n【倾向性得分范围】")
        e_min = self._get(self.df_bench, "dr_e_min")
        e_max = self._get(self.df_bench, "dr_e_max")
        print(f"  最小值均值: {e_min.mean():.4f}")
        print(f"  最大值均值: {e_max.mean():.4f}")

        extreme_low = (e_min < 0.01).sum()
        extreme_high = (e_max > 0.99).sum()
        if extreme_low > 0:
            print(f"\n  ⚠️ {extreme_low}个运行有极低倾向性得分 (<0.01)")
        if extreme_high > 0:
            print(f"  ⚠️ {extreme_high}个运行有极高倾向性得分 (>0.99)")

        print("\n【截断统计】")
        clip_used = self._get(self.df_bench, "dr_clip_used").mean()
        frac_clipped = self._get(self.df_bench, "dr_frac_e_clipped").mean()
        print(f"  使用的截断阈值: {clip_used:.4f}")
        print(f"  被截断的平均比例: {frac_clipped*100:.1f}%")
        if frac_clipped > 0.25:
            print("  ⚠️ 警告：超过25%的观测被截断，可能过于激进")
        elif frac_clipped > 0.15:
            print("  ⚠️ 注意：截断比例较高")
        else:
            print("  ✓ 截断比例合理")

        print("\n【重叠性得分】")
        overlap = self._get(self.df_bench, "dr_overlap_score")
        print(f"  均值: {overlap.mean():.4f}")
        print(f"  中位数: {overlap.median():.4f}")
        print(f"  范围: [{overlap.min():.4f}, {overlap.max():.4f}]")
        good_overlap = (overlap > 0.7).sum()
        mod_overlap = ((overlap >= 0.5) & (overlap <= 0.7)).sum()
        poor_overlap = (overlap < 0.5).sum()
        print(f"\n  好 (>0.7):   {good_overlap:3d} ({good_overlap/len(overlap)*100:5.1f}%) ✓")
        print(f"  中等 (0.5-0.7): {mod_overlap:3d} ({mod_overlap/len(overlap)*100:5.1f}%)")
        print(f"  差 (<0.5):   {poor_overlap:3d} ({poor_overlap/len(overlap)*100:5.1f}%) ⚠️")

        if "dr_ess_raw" in self.df_bench.columns:
            ess = self.df_bench["dr_ess_raw"]
            n_total = len(self.df_bench)
            ess_ratio = ess / n_total
            print("\n【有效样本量(ESS)】")
            print(f"  ESS均值: {ess.mean():.1f}")
            print(f"  ESS/n比例: {ess_ratio.mean():.1%}")
            if ess_ratio.mean() < 0.8:
                print("  ⚠️ ESS比例偏低，权重方差较大")

    # ---------------- weights ----------------
    def weight_analysis(self) -> None:
        print("\n" + "=" * 80)
        print("4️⃣  IPW权重分析")
        print("=" * 80)

        max_raw = self._get(self.df_bench, "dr_ipw_abs_max_raw")
        print("\n【原始权重（截断前）】")
        print(f"  最大绝对值均值: {max_raw.mean():.2f}")
        print(f"  最大绝对值中位数: {max_raw.median():.2f}")
        print(f"  最大绝对值最大值: {max_raw.max():.2f}")
        extreme = (max_raw > 100).sum()
        if extreme > 0:
            print(f"\n  ⚠️ {extreme}个运行有极端权重 (>100)")

        print("\n【截断后权重】")
        max_capped = self._get(self.df_bench, "dr_ipw_abs_max_capped")
        print(f"  最大绝对值均值: {max_capped.mean():.2f}")
        print(f"  最大绝对值中位数: {max_capped.median():.2f}")

        frac_capped = self._get(self.df_bench, "dr_frac_ipw_capped")
        print(f"\n【权重截断比例】")
        print(f"  均值: {frac_capped.mean()*100:.1f}%")
        print(f"  中位数: {frac_capped.median()*100:.1f}%")
        high_cap = (frac_capped > 0.2).sum()
        if high_cap > 0:
            print(f"  ⚠️ {high_cap}个运行超过20%的权重被截断")

        cap_used = self._get(self.df_bench, "dr_cap_used").mean()
        print(f"\n【权重上限设置】")
        print(f"  平均使用的上限: {cap_used:.1f}")

    # ---------------- adversarial ----------------
    def adversarial_training_analysis(self) -> None:
        print("\n" + "=" * 80)
        print("5️⃣  对抗训练效果分析")
        print("=" * 80)

        adv_w = self._get(self.df_bench, "adv_w_acc", 0.5)
        print("\n【W对抗器（应该~0.5）】")
        print(f"  均值: {adv_w.mean():.4f}")
        print(f"  中位数: {adv_w.median():.4f}")
        print(f"  标准差: {adv_w.std():.4f}")
        good_w = ((adv_w > 0.45) & (adv_w < 0.55)).sum()
        print(f"  独立性好 (0.45-0.55): {good_w}/{len(adv_w)} ({good_w/len(adv_w)*100:.1f}%)")
        if abs(adv_w.mean() - 0.5) > 0.05:
            print("  ⚠️ W对抗器偏离0.5较多，W可能未完全独立于A")
        else:
            print("  ✓ W对抗器表现良好，W基本独立于A")

        adv_n = self._get(self.df_bench, "adv_n_acc", 0.5)
        print("\n【N对抗器（噪声，应该~0.5）】")
        print(f"  均值: {adv_n.mean():.4f}")
        print(f"  标准差: {adv_n.std():.4f}")

        adv_z = self._get(self.df_bench, "adv_z_r2", 0.0)
        print("\n【Z对抗器（排他性，应该~0）】")
        print(f"  均值: {adv_z.mean():.4f}")
        print(f"  中位数: {adv_z.median():.4f}")
        good_z = (adv_z < 0.1).sum()
        print(f"  排他性好 (R²<0.1): {good_z}/{len(adv_z)} ({good_z/len(adv_z)*100:.1f}%)")
        if adv_z.mean() > 0.1:
            print("  ⚠️ Z对Y有较强预测能力，可能违反排他性")
        else:
            print("  ✓ Z对抗器表现良好，Z基本不预测Y")

    # ---------------- scenario comparison ----------------
    def scenario_comparison(self) -> None:
        print("\n" + "=" * 80)
        print("6️⃣  场景对比分析")
        print("=" * 80)

        if "scenario" not in self.df_bench.columns:
            print("  ⚠️ 数据中没有场景信息")
            return

        scenarios = self.df_bench["scenario"].unique()
        print(f"\n共有 {len(scenarios)} 个场景")
        print("\n场景性能对比：")
        print("  场景名称".ljust(30) + "  RMSE   误差   IV强度  重叠")
        print("  " + "-" * 60)

        for scenario in sorted(scenarios):
            df_s = self.df_bench[self.df_bench["scenario"] == scenario]
            rmse = np.sqrt(df_s["sq_err"].mean())
            mae = df_s["abs_err"].mean()
            iv_rel = self._get(df_s, "iv_relevance_abs_corr").mean()
            overlap = self._get(df_s, "dr_overlap_score").mean()
            print(f"  {scenario[:28].ljust(30)} {rmse:6.3f} {mae:6.3f} {iv_rel:6.3f} {overlap:6.3f}")

        print("\n【问题场景识别】")
        stats: List[Dict[str, float]] = []
        for scenario in scenarios:
            df_s = self.df_bench[self.df_bench["scenario"] == scenario]
            stats.append(
                {
                    "scenario": scenario,
                    "rmse": np.sqrt(df_s["sq_err"].mean()),
                    "iv_rel": self._get(df_s, "iv_relevance_abs_corr").mean(),
                    "overlap": self._get(df_s, "dr_overlap_score").mean(),
                }
            )
        df_stats = pd.DataFrame(stats)

        worst = df_stats.nlargest(3, "rmse")
        print("\n  RMSE最高的场景:")
        for _, row in worst.iterrows():
            print(f"    {row['scenario']}: RMSE={row['rmse']:.4f}")

        weak_iv_scenarios = df_stats[df_stats["iv_rel"] < 0.15]
        if len(weak_iv_scenarios) > 0:
            print("\n  弱IV场景:")
            for _, row in weak_iv_scenarios.iterrows():
                print(f"    {row['scenario']}: IV={row['iv_rel']:.4f}")

        poor_overlap_scenarios = df_stats[df_stats["overlap"] < 0.6]
        if len(poor_overlap_scenarios) > 0:
            print("\n  差重叠场景:")
            for _, row in poor_overlap_scenarios.iterrows():
                print(f"    {row['scenario']}: overlap={row['overlap']:.4f}")

    # ---------------- recommendations ----------------
    def generate_recommendations(self) -> None:
        print("\n" + "=" * 80)
        print("💡 改进建议")
        print("=" * 80)

        recommendations: List[Dict[str, str]] = []
        iv_rel_mean = self._get(self.df_bench, "iv_relevance_abs_corr").mean()
        if iv_rel_mean < 0.15:
            recommendations.append(
                {
                    "priority": "🔴 高",
                    "issue": "整体IV强度偏弱",
                    "recommendation": (
                        f"平均IV相关性={iv_rel_mean:.4f} < 0.15\n"
                        "  → 添加F统计量检验\n"
                        "  → 在弱IV场景自动警告用户\n"
                        "  → 考虑更强的instruments或TSLS方法"
                    ),
                }
            )

        frac_clipped = self._get(self.df_bench, "dr_frac_e_clipped").mean()
        if frac_clipped > 0.2:
            recommendations.append(
                {
                    "priority": "🔴 高",
                    "issue": "倾向性得分截断过度",
                    "recommendation": (
                        f"平均截断比例={frac_clipped*100:.1f}% > 20%\n"
                        "  → 提高clip_prop或实现数据驱动的最优截断\n"
                        "  → 考虑使用重叠权重"
                    ),
                }
            )

        w_auc_mean = self._get(self.df_bench, "rep_auc_w_to_a", 0.5).mean()
        if abs(w_auc_mean - 0.5) > 0.05:
            recommendations.append(
                {
                    "priority": "🟡 中",
                    "issue": "W未充分独立于A",
                    "recommendation": (
                        f"W→A的AUC={w_auc_mean:.4f}，偏离0.5\n"
                        "  → 启用或加强HSIC惩罚（lambda_hsic）\n"
                        "  → 增加gamma_adv_w对抗强度\n"
                        "  → 检查条件正交惩罚实现"
                    ),
                }
            )

        leak_mean = self._get(self.df_bench, "rep_exclusion_leakage_r2", 0.0).mean()
        if leak_mean > 0.15:
            recommendations.append(
                {
                    "priority": "🟡 中",
                    "issue": "Z→Y存在泄露",
                    "recommendation": (
                        f"排他性泄露R²={leak_mean:.4f} > 0.15\n"
                        "  → 增加gamma_adv_z对抗强度\n"
                        "  → 添加 Sargan-Hansen 检验\n"
                        "  → 考虑移除可疑的 instruments"
                    ),
                }
            )

        max_w_mean = self._get(self.df_bench, "dr_ipw_abs_max_raw").mean()
        if max_w_mean > 50:
            recommendations.append(
                {
                    "priority": "🟢 低",
                    "issue": "IPW权重过大",
                    "recommendation": (
                        f"平均最大权重={max_w_mean:.1f} > 50\n"
                        "  → 可考虑降低 ipw_cap 或使用更平滑的截断\n"
                        "  → 极端情况考虑匹配方法"
                    ),
                }
            )

        if recommendations:
            print("")
            for i, rec in enumerate(recommendations, 1):
                print(f"{i}. {rec['priority']} {rec['issue']}")
                print(f"   {rec['recommendation']}\n")
        else:
            print("\n✓ 未检测到重大问题，算法整体表现良好！\n")

    # ---------------- full pipeline ----------------
    def full_analysis(self) -> None:
        self.executive_summary()
        self.identifiability_analysis()
        self.representation_quality_analysis()
        self.propensity_overlap_analysis()
        self.weight_analysis()
        self.adversarial_training_analysis()
        self.scenario_comparison()
        self.generate_recommendations()
        print("\n" + "=" * 80)
        print(" " * 25 + "分析完成")
        print("=" * 80)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze IVAPCI v3.3 (v22) results.")
    parser.add_argument("--benchmark-file", type=str, default="simulation_benchmark_results 22.csv")
    parser.add_argument("--diagnostics-file", type=str, default="simulation_diagnostics_results 22.csv")
    parser.add_argument("--summary-file", type=str, default="simulation_benchmark_summary 22.csv")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    analyzer = IVAPCIv22Analyzer(
        benchmark_file=args.benchmark_file,
        diagnostics_file=args.diagnostics_file,
        summary_file=args.summary_file,
    )
    analyzer.full_analysis()
