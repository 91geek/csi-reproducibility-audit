"""
F-37 DANN 统计检验
==================

对比 DANN (f37_dann.json) vs baseline (f32 lenet_attn / f33 F-21 multiRx)
做配对 t 检验 / Wilcoxon / Cohen's d / 方差分解

产出：B13 finding
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from scipy import stats

BASE = Path("F:/python_workspace/wifi识别/wifi-crossenv-lab/src/bvp_test")


def load(name: str) -> Dict:
    with open(BASE / name, encoding="utf-8") as f:
        return json.load(f)


def paired_test(a: np.ndarray, b: np.ndarray, label_a: str, label_b: str):
    """配对检验：a vs b"""
    d = a - b
    n = len(d)
    mean_a, mean_b = a.mean(), b.mean()
    diff = mean_a - mean_b

    # 配对 t 检验
    t_stat, p_t = stats.ttest_rel(a, b)
    # Wilcoxon 符号秩
    try:
        w_stat, p_w = stats.wilcoxon(a, b)
    except Exception:
        w_stat, p_w = np.nan, np.nan
    # Cohen's d (paired)
    sd_d = d.std(ddof=1)
    cohen_d = diff / sd_d if sd_d > 0 else 0.0
    # 95% CI of diff
    se = sd_d / np.sqrt(n)
    t_crit = stats.t.ppf(0.975, n - 1)
    ci = (diff - t_crit * se, diff + t_crit * se)

    return {
        "n_pairs": n,
        "mean_a": float(mean_a),
        "mean_b": float(mean_b),
        "diff": float(diff),
        "diff_pp": float(diff * 100),
        "std_a": float(a.std(ddof=1)),
        "std_b": float(b.std(ddof=1)),
        "std_diff": float(sd_d),
        "t_stat": float(t_stat),
        "p_ttest": float(p_t),
        "wilcoxon_p": float(p_w),
        "cohen_d": float(cohen_d),
        "ci95": [float(ci[0]), float(ci[1])],
        "ci95_pp": [float(ci[0] * 100), float(ci[1] * 100)],
        "wins": int((d > 0).sum()),
        "losses": int((d < 0).sum()),
        "ties": int((d == 0).sum()),
    }


def main():
    f37 = load("f37_dann.json")
    f32 = load("f32_multiseed.json")
    f33 = load("f33_rx_vs_ms.json")

    # 提取 (fold, seed) -> acc
    dann: Dict[Tuple[int, int], float] = {}
    for k, v in f37.items():
        dann[(v["fold"], v["seed"])] = v["acc"]

    base_f32: Dict[Tuple[int, int], float] = {}
    for k, v in f32.items():
        # 精确匹配 conv 版（lenet_attn_d{fold}_s{seed}），排除 lenet_attn_mha
        if k.startswith("lenet_attn_d"):
            base_f32[(v["fold"], v["seed"])] = v["acc"]

    base_f33: Dict[Tuple[int, int], float] = {}
    for k, v in f33.items():
        if k.startswith("F-21_multiRx"):
            base_f33[(v["fold"], v["seed"])] = v["acc"]

    print("=" * 70)
    print("F-37 DANN 统计检验")
    print("=" * 70)
    print(f"DANN   : {len(dann)} runs, folds={sorted(set(k[0] for k in dann))}")
    print(f"F32 base: {len(base_f32)} runs")
    print(f"F33 base: {len(base_f33)} runs")

    def match(dann_d, base_d):
        keys = sorted(set(dann_d) & set(base_d))
        a = np.array([dann_d[k] for k in keys])
        b = np.array([base_d[k] for k in keys])
        return keys, a, b

    results = {}

    # --- 1. DANN vs F-32 baseline (lenet_attn) ---
    keys1, a1, b1 = match(dann, base_f32)
    r1 = paired_test(a1, b1, "DANN", "F-32 lenet_attn")
    results["dann_vs_f32"] = r1
    print(f"\n[1] DANN vs F-32 baseline (n={r1['n_pairs']} pairs)")
    print(f"    DANN  mean = {r1['mean_a']:.4f} ± {r1['std_a']:.4f}")
    print(f"    Base  mean = {r1['mean_b']:.4f} ± {r1['std_b']:.4f}")
    print(f"    diff  = {r1['diff_pp']:+.2f} pp  95%CI [{r1['ci95_pp'][0]:+.2f}, {r1['ci95_pp'][1]:+.2f}]")
    print(f"    t={r1['t_stat']:.3f}  p_ttest={r1['p_ttest']:.4f}  Wilcoxon p={r1['wilcoxon_p']:.4f}")
    print(f"    Cohen's d = {r1['cohen_d']:.3f}   wins/losses = {r1['wins']}/{r1['losses']}")

    # --- 2. DANN vs F-33 baseline (F-21 multiRx) ---
    keys2, a2, b2 = match(dann, base_f33)
    r2 = paired_test(a2, b2, "DANN", "F-21 multiRx")
    results["dann_vs_f33"] = r2
    print(f"\n[2] DANN vs F-21 multiRx (n={r2['n_pairs']} pairs)")
    print(f"    DANN  mean = {r2['mean_a']:.4f} ± {r2['std_a']:.4f}")
    print(f"    Base  mean = {r2['mean_b']:.4f} ± {r2['std_b']:.4f}")
    print(f"    diff  = {r2['diff_pp']:+.2f} pp  95%CI [{r2['ci95_pp'][0]:+.2f}, {r2['ci95_pp'][1]:+.2f}]")
    print(f"    t={r2['t_stat']:.3f}  p_ttest={r2['p_ttest']:.4f}  Wilcoxon p={r2['wilcoxon_p']:.4f}")
    print(f"    Cohen's d = {r2['cohen_d']:.3f}   wins/losses = {r2['wins']}/{r2['losses']}")

    # --- 3. 每 fold 明细 ---
    print(f"\n[3] 每 fold 明细 (DANN vs F-32 baseline)")
    print(f"    {'fold':<6}{'DANN':>10}{'Base':>10}{'diff_pp':>10}")
    for fold in sorted(set(k[0] for k in keys1)):
        d_arr = np.array([dann[k] for k in keys1 if k[0] == fold])
        b_arr = np.array([base_f32[k] for k in keys1 if k[0] == fold])
        print(f"    d{fold:<5}{d_arr.mean():>10.4f}{b_arr.mean():>10.4f}{(d_arr.mean()-b_arr.mean())*100:>+10.2f}")

    # --- 4. 方差分解（验证 B8） ---
    print(f"\n[4] 方差分解（B8 验证）")
    # DANN 内部：seed 间方差 vs fold 间方差
    dann_by_fold = {}
    for (fold, seed), acc in dann.items():
        dann_by_fold.setdefault(fold, []).append(acc)
    within_seed_std = np.mean([np.std(v, ddof=1) for v in dann_by_fold.values()])
    fold_means = [np.mean(v) for v in dann_by_fold.values()]
    across_fold_std = np.std(fold_means, ddof=1)
    print(f"    DANN  seed 内标准差 (within-fold) = {within_seed_std:.4f} ({within_seed_std*100:.2f} pp)")
    print(f"    DANN  fold 间标准差 (across-fold) = {across_fold_std:.4f} ({across_fold_std*100:.2f} pp)")

    base_by_fold = {}
    for (fold, seed), acc in base_f32.items():
        base_by_fold.setdefault(fold, []).append(acc)
    b_within = np.mean([np.std(v, ddof=1) for v in base_by_fold.values()])
    print(f"    Base  seed 内标准差 (within-fold) = {b_within:.4f} ({b_within*100:.2f} pp)")
    print(f"    → 架构间差异 (DANN vs Base) = {abs(r1['diff'])*100:.2f} pp")
    print(f"    → seed 内方差 (DANN)         = {within_seed_std*100:.2f} pp")
    print(f"    → 方差 / 效应 比 = {within_seed_std / max(abs(r1['diff']), 1e-9):.2f}x")

    # --- 5. 结论 ---
    print(f"\n[5] 结论")
    sig = "显著" if r1["p_ttest"] < 0.05 else "不显著"
    print(f"    DANN 效果 {sig} (p={r1['p_ttest']:.4f})")
    if r1["cohen_d"] > 0.5:
        print(f"    效应量中等以上 (d={r1['cohen_d']:.2f})")
    elif r1["cohen_d"] > 0.2:
        print(f"    效应量小 (d={r1['cohen_d']:.2f})")
    else:
        print(f"    效应量可忽略 (d={r1['cohen_d']:.2f})")
    if r1["diff"] < 0:
        print(f"    → DANN 反而比 baseline 差 {abs(r1['diff_pp']):.2f} pp（负向结果，证伪 finding）")

    # 保存
    out = {"dann_vs_f32": r1, "dann_vs_f33": r2,
           "variance": {"dann_within_seed_std": float(within_seed_std),
                        "dann_across_fold_std": float(across_fold_std),
                        "base_within_seed_std": float(b_within)}}
    out_path = BASE / "f37_stats.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存 -> {out_path}")


if __name__ == "__main__":
    main()
