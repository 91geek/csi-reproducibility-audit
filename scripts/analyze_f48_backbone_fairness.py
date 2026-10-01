#!/usr/bin/env python3
# F-48 分析脚本：CBAM+ResNet18 vs LeNetCSI_Attn 配对 backbone 对比
#
# 输入：
#   - f32_multiseed.json (前 12 条 = baseline LeNetCSI_Attn on F=[1,3,5,7] × S=[0,1,2])
#   - f48_cbam_resnet18.json (CBAM+ResNet18 on same folds/seeds)
#
# 输出：
#   - 配对 Δ, 95% CI, paired t-test p, Cohen's d
#   - JSON + 文字报告 + 一张对比图 (matplotlib)

import json
from pathlib import Path
import numpy as np
from scipy import stats

BVP_DIR = Path("F:/python_workspace/wifi识别/wifi-crossenv-lab/src/bvp_test")
OUT_DIR = Path("F:/python_workspace/wifi识别/wifi-crossenv-lab/scripts")
OUT_DIR.mkdir(exist_ok=True)

# Load baseline (LeNetCSI_Attn) — first 12 entries from f32_multiseed.json
with open(BVP_DIR / "f32_multiseed.json") as f:
    f32 = json.load(f)
base_keys = sorted([k for k in f32 if k.startswith("lenet_attn")])[:12]
assert len(base_keys) == 12, f"Expected 12 baseline runs, got {len(base_keys)}"

# Load CBAM+ResNet18 results
with open(BVP_DIR / "f48_cbam_resnet18.json") as f:
    f48 = json.load(f)

# Pair by (fold, seed)
def extract(d, prefix):
    out = {}
    for k, v in d.items():
        if k.startswith(prefix):
            out[(v["fold"], v["seed"])] = v["acc"] * 100
    return out

base_dict = extract(f32, "lenet_attn")
cbam_dict = extract(f48, "cbam_resnet18")

# Find common (fold, seed) pairs — 12 expected
common = sorted(set(base_dict) & set(cbam_dict))
n_paired = len(common)
print(f"Paired runs: {n_paired}")

base_acc = np.array([base_dict[k] for k in common])
cbam_acc = np.array([cbam_dict[k] for k in common])

# Paired Δ = CBAM - LeNet
delta = cbam_acc - base_acc
mean_delta = delta.mean()
std_delta = delta.std(ddof=1)
sem_delta = std_delta / np.sqrt(n_paired)
t_stat, p_val = stats.ttest_rel(cbam_acc, base_acc)
# Cohen's d for paired samples
cohen_d = mean_delta / std_delta
# 95% CI for mean_delta
t_crit = stats.t.ppf(0.975, df=n_paired - 1)
ci_lo = mean_delta - t_crit * sem_delta
ci_hi = mean_delta + t_crit * sem_delta

# Report
print(f"\n{'='*60}")
print(f"Backbone Fairness Audit: CBAM+ResNet18 vs LeNetCSI_Attn")
print(f"{'='*60}")
print(f"Folds×Seeds: {n_paired} paired runs")
print(f"LeNetCSI_Attn: mean = {base_acc.mean():.2f}pp, std = {base_acc.std(ddof=1):.2f}pp")
print(f"CBAM+ResNet18: mean = {cbam_acc.mean():.2f}pp, std = {cbam_acc.std(ddof=1):.2f}pp")
print(f"\nPaired Δ (CBAM - LeNet):")
print(f"  mean Δ = {mean_delta:+.2f}pp")
print(f"  95% CI = [{ci_lo:+.2f}pp, {ci_hi:+.2f}pp]")
print(f"  t({n_paired-1}) = {t_stat:.3f}, p = {p_val:.4f}")
print(f"  Cohen's d = {cohen_d:.3f}")
print(f"  wins/total = {int((delta > 0).sum())}/{n_paired}")
print(f"{'='*60}")

# Save results
results = {
    "n_paired": int(n_paired),
    "baseline": "LeNetCSI_Attn (~170K params)",
    "treatment": "CBAM+ResNet18 (~11.4M params, 67x larger)",
    "baseline_mean_acc_pp": float(base_acc.mean()),
    "baseline_std_pp": float(base_acc.std(ddof=1)),
    "treatment_mean_acc_pp": float(cbam_acc.mean()),
    "treatment_std_pp": float(cbam_acc.std(ddof=1)),
    "mean_delta_pp": float(mean_delta),
    "ci95_lo_pp": float(ci_lo),
    "ci95_hi_pp": float(ci_hi),
    "t_stat": float(t_stat),
    "p_value": float(p_val),
    "cohen_d": float(cohen_d),
    "wins": int((delta > 0).sum()),
    "losses": int((delta < 0).sum()),
    "ties": int((delta == 0).sum()),
    "paired": [
        {"fold": int(k[0]), "seed": int(k[1]),
         "lenet_acc_pp": float(base_dict[k]),
         "cbam_acc_pp": float(cbam_dict[k]),
         "delta_pp": float(cbam_dict[k] - base_dict[k])}
        for k in common
    ],
}

out_json = OUT_DIR / "f48_backbone_fairness.json"
with open(out_json, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
print(f"\nResults saved to {out_json}")

# Try to make a plot
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # Panel 1: per-pair boxplot + connecting lines
    ax = axes[0]
    positions = [1, 2]
    bp = ax.boxplot([base_acc, cbam_acc], positions=positions, widths=0.55,
                    patch_artist=True, showmeans=True, meanline=True,
                    boxprops=dict(facecolor="#f0f0f0"))
    for patch, color in zip(bp["boxes"], ["#7ba9d6", "#e6915c"]):
        patch.set_facecolor(color); patch.set_alpha(0.6)
    # Paired connecting lines
    for i in range(n_paired):
        ax.plot([1, 2], [base_acc[i], cbam_acc[i]],
                color="gray", alpha=0.4, lw=0.8, zorder=1)
    ax.scatter(np.ones(n_paired), base_acc, color="#1f4e79", alpha=0.7, zorder=3, s=30)
    ax.scatter(np.ones(n_paired) * 2, cbam_acc, color="#a8421c", alpha=0.7, zorder=3, s=30)
    ax.set_xticks(positions)
    ax.set_xticklabels(["LeNetCSI_Attn\n(~170K params)", "CBAM+ResNet18\n(~11.4M params)"])
    ax.set_ylabel("Cross-domain accuracy (pp)")
    ax.set_title(f"Backbone Fairness Audit (n={n_paired} paired runs)")
    ax.grid(axis="y", alpha=0.3)

    # Stats annotation
    sig_str = "***" if p_val < 0.001 else ("**" if p_val < 0.01 else ("*" if p_val < 0.05 else "n.s."))
    ax.text(0.5, 0.97,
            f"Δ = {mean_delta:+.2f}pp  [{ci_lo:+.2f}, {ci_hi:+.2f}]\n"
            f"d = {cohen_d:.2f}  p = {p_val:.4f}  {sig_str}\n"
            f"Wins: {int((delta > 0).sum())}/{n_paired}",
            transform=ax.transAxes, ha="center", va="top",
            fontsize=10, family="monospace",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.6))

    # Panel 2: per-(fold, seed) bar chart
    ax = axes[1]
    x = np.arange(n_paired)
    width_bar = 0.35
    ax.bar(x - width_bar / 2, base_acc, width_bar, label="LeNetCSI_Attn", color="#7ba9d6")
    ax.bar(x + width_bar / 2, cbam_acc, width_bar, label="CBAM+ResNet18", color="#e6915c")
    ax.set_xticks(x)
    ax.set_xticklabels([f"d{k[0]}_s{k[1]}" for k in common], rotation=45, fontsize=8)
    ax.set_ylabel("Cross-domain accuracy (pp)")
    ax.set_title("Per-pair (fold, seed) detail")
    ax.legend(loc="upper left")
    ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Figure: CBAM+ResNet18 (Liu 2025) vs LeNetCSI_Attn (ours) — same 12 paired runs",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    out_png = BVP_DIR / "f48_backbone_fairness.png"
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"Figure saved to {out_png}")
except Exception as e:
    print(f"Plot failed: {e}")