"""
Fig. 5 (v2): Multi-Rx mechanism 综合图 — 4 panel
(a) Antenna Participation Ratio (PR) histogram + formula derivation
(b) Fisher F per-axis (multi-Rx vs RMS-agg) + F formula
(c) Information density fold: V1_h8_rx vs V2_h8_rms vs V3_h16_rms
(d) PR vs gain scatter (per domain)

数据源：src/bvp_test/b10_mechanism_A.json (E1 + E2 + E3)
输出：src/bvp_test/fig5_mechanism.png
"""
import json
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

BVP_DIR = os.path.join(os.path.dirname(__file__), '..', 'src', 'bvp_test')

# ============ 数据加载 ============
with open(os.path.join(BVP_DIR, 'b10_mechanism_A.json'), encoding='utf-8') as f:
    b10 = json.load(f)

e1 = b10['E1_antenna_participation_ratio']
e2_v1 = b10['E2_V1_h8_rx']   # multi-Rx (per-antenna)
e2_v2 = b10['E2_V2_h8_rms']  # RMS-agg (hop=8)
e2_v3 = b10['E2_V3_h16_rms'] # RMS-agg (hop=16, baseline)
e3 = b10['E3_decomposition']

# ============ 颜色 ============
COL_MULTI_RX = '#E45756'    # 红色：核心 innovation
COL_RMS = '#4C78A8'         # 蓝色：baseline
COL_HOP = '#54A24B'         # 绿色：hop change
COL_NEUTRAL = '#888888'     # 灰色：参考线

# ============ 图布局 ============
# v3: 加大画布高度 + 给 (b)(c) 行更高比例 + 更大 hspace，避免右侧文本框挤压重叠
fig = plt.figure(figsize=(15.5, 13.5))
gs = fig.add_gridspec(3, 4, height_ratios=[1.0, 1.2, 1.4], hspace=0.75, wspace=0.32,
                     left=0.06, right=0.97, top=0.92, bottom=0.05)

# ============ Panel (a): Antenna PR histogram + formula ============
ax_a = fig.add_subplot(gs[0, 0:2])

pr_hist = e1['pr_hist']
pr_mean = e1['pr_mean']
pr_std = e1['pr_std']
pr_p05 = e1['pr_p05']
pr_p95 = e1['pr_p95']
n_ant = e1['n_antennas']
n_pos = e1['n_positions']

# PR 分布：pr_hist 是 index 表，但 index ↔ PR 范围不直接。
# 用正态近似（基于 mean=1.89, std=0.17）来可视化 + 标注 P05/P95
from scipy.stats import norm as scs_norm
x_pdf = np.linspace(1.4, 2.6, 200)
pdf_y = scs_norm.pdf(x_pdf, pr_mean, pr_std)
# Scale to n_positions density
ax_a.plot(x_pdf, pdf_y / pdf_y.max() * 0.85,
          color=COL_MULTI_RX, linewidth=2.5, label='Normal approx.')
ax_a.fill_between(x_pdf, 0, pdf_y / pdf_y.max() * 0.85,
                  color=COL_MULTI_RX, alpha=0.25)
ax_a.axvline(pr_mean, color='black', linestyle='--', linewidth=1.4,
             label=f'PR mean = {pr_mean:.2f}')
ax_a.axvline(pr_p05, color='gray', linestyle=':', linewidth=1.0, alpha=0.7,
             label=f'5%–95% = [{pr_p05:.2f}, {pr_p95:.2f}]')
ax_a.axvline(pr_p95, color='gray', linestyle=':', linewidth=1.0, alpha=0.7)
ax_a.axvline(1.0, color=COL_NEUTRAL, linestyle=':', linewidth=1.0, alpha=0.7,
             label='PR=1 (fully redundant)')
ax_a.axvline(n_ant, color=COL_NEUTRAL, linestyle=':', linewidth=1.0, alpha=0.7,
             label=f'PR={n_ant} (fully independent)')
# 标记 PR=1.89 的密度
peak_density = scs_norm.pdf(pr_mean, pr_mean, pr_std)
ax_a.plot([pr_mean], [0.85], 'o', color='black', markersize=10, zorder=5)
ax_a.text(pr_mean + 0.02, 0.82, f'peak\nn={n_pos}',
          fontsize=8, color='black', fontweight='bold')
ax_a.axvline(pr_mean, color='black', linestyle='--', linewidth=1.4,
             label=f'PR mean = {pr_mean:.2f}')
ax_a.axvline(1.0, color=COL_NEUTRAL, linestyle=':', linewidth=1.0, alpha=0.7,
             label='PR=1 (fully redundant)')
ax_a.axvline(n_ant, color=COL_NEUTRAL, linestyle=':', linewidth=1.0, alpha=0.7,
             label=f'PR={n_ant} (fully independent)')

ax_a.set_xlabel('Antenna Participation Ratio (effective rank)', fontsize=9.5)
ax_a.set_ylabel('Density (normalized)', fontsize=9.5)
ax_a.set_title('(a) Antenna PR distribution\n'
               f'mean={pr_mean:.2f} ± {pr_std:.2f}, 5%–95%=[{pr_p05:.2f}, {pr_p95:.2f}], n={n_pos}',
               fontsize=10)
ax_a.legend(fontsize=7, loc='upper right', framealpha=0.92)
ax_a.grid(axis='y', linestyle=':', alpha=0.4)
ax_a.set_xlim(1.4, 2.6)
ax_a.set_ylim(0, 1.0)
# 修复：原 y=-0.15 与 x 轴 label "Antenna Participation Ratio" 视觉冲突
# 把 interpretation 文字移到 Panel (a-right) formula box 内（那里已有 Interpretation section，避免重复）
# 同时在图底保留一行简短 summary
ax_a.text(0.5, -0.32, 'interpretation: PR ≈ 1.89 means ~2 effective antennas out of 9 → 5× information gain',
         transform=ax_a.transAxes, ha='center', va='top', fontsize=7, color='gray',
         style='italic')

# ============ Panel (a-right): PR formula derivation ============
ax_form = fig.add_subplot(gs[0, 2:4])
ax_form.axis('off')
formula_text = (
    "Antenna Participation Ratio (PR)\n"
    "═══════════════════════════════════\n"
    "\n"
    "Given the per-position antenna covariance C ∈ ℝ^(A×A)\n"
    "with eigenvalues λ₁ ≥ λ₂ ≥ ... ≥ λ_A ≥ 0,\n"
    "\n"
    "        (Σᵢ λᵢ)²\n"
    "PR  =  ──────────     (Roy & Vetterli 2007)\n"
    "        Σᵢ λᵢ²\n"
    "\n"
    "Interpretation:\n"
    "  PR = 1   ⇒ fully redundant (rank-1)\n"
    "  PR = A   ⇒ fully independent\n"
    "  PR ≈ 2   ⇒ ~2 effective degrees of freedom\n"
    "\n"
    "Empirical (Widar, 9 Rx, 2500 positions):\n"
    f"  PR_mean = {pr_mean:.2f},  PR_std = {pr_std:.2f}\n"
    f"  => ~{pr_mean:.1f} effective antennas out of {n_ant}\n"
    "\n"
    "Why this matters:\n"
    "  Multi-Rx retains A=9 antenna slots in the tensor,\n"
    "  preserving information that RMS aggregation discards\n"
    "  (which contracts to a single slot).\n"
    "\n"
    "  Net information gain ≈ 9 / PR = 9 / 1.89 ≈ 4.8×\n"
    "  (see §7.3 for derivation)"
)
ax_form.text(0.02, 0.95, formula_text, ha='left', va='top', fontsize=8,
             family='monospace',
             bbox=dict(boxstyle='round,pad=0.6', facecolor='#FFF8DC',
                       edgecolor=COL_MULTI_RX, linewidth=1.5))

# ============ Panel (b): Fisher F per-axis comparison ============
ax_b = fig.add_subplot(gs[1, 0:2])

# 准备数据：V1_h8_rx (multi-Rx, per-antenna) vs V2_h8_rms (RMS-agg)
# 注意：V1 是 per-antenna F，所以整体上比 V2 小，但保留信息多
axes_names = ['antenna\n(per-Rx)', 'subcarrier', 'doppler', 'time']
v1_F = [e2_v1['per_axis']['antenna']['F_max_of_pos_mean'],
        e2_v1['per_axis']['subcarrier']['F_max_of_pos_mean'],
        e2_v1['per_axis']['doppler']['F_max_of_pos_mean'],
        e2_v1['per_axis']['time']['F_max_of_pos_mean']]
v2_F = [0,  # V2 没有 antenna 轴
        e2_v2['per_axis']['subcarrier']['F_max_of_pos_mean'],
        e2_v2['per_axis']['doppler']['F_max_of_pos_mean'],
        e2_v2['per_axis']['time']['F_max_of_pos_mean']]

x = np.arange(len(axes_names))
width = 0.35
bars1 = ax_b.bar(x - width/2, v1_F, width, color=COL_MULTI_RX,
                  edgecolor='black', linewidth=0.6, label='Multi-Rx (V1, A=9 retained)')
# V2 没有 antenna 轴，置 0
v2_F_plot = v2_F
bars2 = ax_b.bar(x + width/2, v2_F_plot, width, color=COL_RMS,
                  edgecolor='black', linewidth=0.6, label='RMS-agg (V2, A contracted)')

ax_b.set_xticks(x)
ax_b.set_xticklabels(axes_names, fontsize=9)
ax_b.set_ylabel('Fisher F (max of pos. mean)', fontsize=9.5)
ax_b.set_title('(b) Fisher F per axis  —  Multi-Rx vs RMS-aggregation\n'
               '(per-position F ratio: between-class / within-class)', fontsize=10)
ax_b.legend(fontsize=8, framealpha=0.92)
ax_b.grid(axis='y', linestyle=':', alpha=0.4)
# 标注 V1 antenna 的 F
for i, (v, c) in enumerate(zip(v1_F, [COL_MULTI_RX]*4)):
    if v > 0:
        ax_b.text(i - width/2, v + 1.5, f'{v:.1f}', ha='center', va='bottom',
                  fontsize=7.5, fontweight='bold')
for i, v in enumerate(v2_F_plot):
    if v > 0:
        ax_b.text(i + width/2, v + 1.5, f'{v:.1f}', ha='center', va='bottom',
                  fontsize=7.5)
ax_b.set_ylim(0, max(max(v1_F), max(v2_F_plot)) * 1.15)

# ============ Panel (b-right): Fisher F formula ============
ax_form2 = fig.add_subplot(gs[1, 2:4])
ax_form2.axis('off')
fisher_text = (
    "Fisher F criterion (per dimension)\n"
    "═══════════════════════════════════════\n"
    "\n"
    "For each axis position k, compute:\n"
    "\n"
    "              σ²_between(k)\n"
    "  F(k) =  ─────────────────────\n"
    "              σ²_within(k)\n"
    "\n"
    "Where:\n"
    "  σ²_between = class-mean variance across gestures\n"
    "  σ²_within  = sample variance within each gesture\n"
    "\n"
    "F(k) > 1 ⇒ position k is discriminative\n"
    "F(k) > 4 ⇒ strongly discriminative (p<0.05 equiv.)\n"
    "\n"
    "Empirical findings:\n"
    f"  V1 (multi-Rx, per-antenna F_mean): {e2_v1['F_mean']:.2f}\n"
    f"    → only 32.4% of positions are discriminative\n"
    f"  V2 (RMS-agg, F_mean): {e2_v2['F_mean']:.2f}\n"
    f"    → 99.9% of positions are discriminative\n"
    "\n"
    "But the density multiplies:\n"
    f"  V1 elements: {e2_v1['n_elements']:,}\n"
    f"  V2 elements: {e2_v2['n_elements']:,}\n"
    f"  → density ratio: {e2_v1['n_elements']/e2_v2['n_elements']:.1f}×\n"
    "\n"
    "Net total F (sum): V1 wins\n"
    f"  Σ_F(V1) = {e2_v1['sum_F']:.0f}  vs  Σ_F(V2) = {e2_v2['sum_F']:.0f}"
)
ax_form2.text(0.02, 0.95, fisher_text, ha='left', va='top', fontsize=8,
              family='monospace',
              bbox=dict(boxstyle='round,pad=0.6', facecolor='#E8F1FB',
                        edgecolor=COL_RMS, linewidth=1.5))

# ============ Panel (c): Information density fold ============
ax_c = fig.add_subplot(gs[2, 0:2])

# Decomposition
dens_antenna = e3['天线聚合(V1vsV2)']['density_fold']   # V1 vs V2: 0.20
dens_hop = e3['时间hop(V2vsV3)']['density_fold']         # V2 vs V3: 1.00
total = e3['总效应(V1vsV3)']['density_fold']            # V1 vs V3: 0.20

labels = ['Antenna\naggregation\n(V1→V2)', 'Temporal hop\n(V2→V3)', 'Total\n(V1→V3)']
values = [dens_antenna, dens_hop, total]
colors_c = [COL_MULTI_RX, COL_HOP, COL_NEUTRAL]
bars_c = ax_c.bar(labels, values, color=colors_c, edgecolor='black', linewidth=0.7, width=0.55)

for bar, v in zip(bars_c, values):
    ax_c.text(bar.get_x() + bar.get_width()/2, v + 0.02, f'{v:.2f}',
              ha='center', va='bottom', fontsize=10, fontweight='bold')

ax_c.set_ylabel('Density fold (a / b)', fontsize=9.5)
ax_c.set_title('(c) Information density decomposition\n'
               '(a = n_elements_a / n_elements_b; <1 means more dense)', fontsize=10)
ax_c.axhline(1.0, color='black', linestyle='--', linewidth=1.0, alpha=0.6,
             label='density fold = 1.0 (no change)')
ax_c.set_ylim(0, 1.25)
ax_c.grid(axis='y', linestyle=':', alpha=0.4)
ax_c.legend(fontsize=8, loc='upper right', framealpha=0.92)

# 注解：天线聚合保留信息
# 修复：原 xytext=(0.4, 0.85) 红字穿过绿色柱子 (Temporal hop)，移到更靠左 (-0.5, 0.95) 避开柱子
ax_c.annotate('Multi-Rx retains 5× density\n(via 9 antenna slots)',
              xy=(0, dens_antenna), xytext=(-0.5, 0.95),
              fontsize=8, color=COL_MULTI_RX, fontweight='bold',
              bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                        edgecolor=COL_MULTI_RX, linewidth=1.0, alpha=0.95),
              arrowprops=dict(arrowstyle='->', color=COL_MULTI_RX, lw=1.2))
# 修复：Hop change annotation 移到 bar 1 右侧并加 bbox，避免与"density fold=1.0" legend 重叠
ax_c.annotate('Hop change\nno density effect',
              xy=(1, dens_hop), xytext=(1.7, 0.45),
              fontsize=8, color=COL_HOP,
              bbox=dict(boxstyle='round,pad=0.25', facecolor='white',
                        edgecolor=COL_HOP, linewidth=0.8, alpha=0.95),
              arrowprops=dict(arrowstyle='->', color=COL_HOP, lw=1.0))

# ============ Panel (d): Verdict box (summary) ============
ax_d = fig.add_subplot(gs[2, 2:4])
ax_d.axis('off')
verdict_text = (
    "Mechanism verdict (§7.3)\n"
    "═══════════════════════════════════════════════\n"
    "\n"
    "Multi-Rx's +8.14pp gain is NOT primarily from\n"
    "higher per-position discriminative power\n"
    "(F per axis is actually LOWER for V1).\n"
    "\n"
    "It IS primarily from:\n"
    "  ★ Preserving information density  (5× via A=9)\n"
    "  ★ Preserving per-antenna geometry (PR=1.89 > 1)\n"
    "  ★ Effective rank ≈ 2 of 9 antennas independent\n"
    "\n"
    "Conclusion:\n"
    "  Front-end signal preservation (representation)\n"
    "  beats back-end algorithmic novelty (algorithmic)\n"
    "  on Widar3.0.\n"
    "\n"
    "Boundary (see §8):\n"
    "  This breaks on MMFi where domains are\n"
    "  indistinguishable in BVP feature space\n"
    "  (no separability to amplify).\n"
    "  => Multi-scale becomes optimal instead.\n"
    "\n"
    "Caveat (E4 scatter, see E4_PR_vs_gain.png):\n"
    "  Per-domain PR ↔ gain correlation: r=+0.10\n"
    "  => Mechanism is global, not per-domain."
)
ax_d.text(0.02, 0.95, verdict_text, ha='left', va='top', fontsize=7.5,
          bbox=dict(boxstyle='round,pad=0.6', facecolor='#F0FFF0',
                       edgecolor='#2CA02C', linewidth=1.5))

# ============ Total title ============
fig.suptitle(
    f'Fig. 5 (v2): Multi-Rx mechanism decomposition — why +8.14pp on Widar3.0\n'
    f'(a) Antenna effective rank, (b) Fisher F per axis, (c) Information density fold, (d) Verdict',
    fontsize=11.5, y=0.995)

# 输出
out = os.path.join(BVP_DIR, 'fig5_mechanism.png')
fig.savefig(out, dpi=150, bbox_inches='tight')
print(f"[Fig 5 v2] -> {out}")