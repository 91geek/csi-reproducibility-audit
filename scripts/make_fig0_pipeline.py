"""
Figure 0 v4: Methodology Pipeline + MDE formula + 嵌入 Violin Plot
==================================================================
v3 基础上加入：
  - §5 LODO Protocol 区域右下角嵌入 "Acc distribution violin plot"
  - 展示 4 方法 × 2 数据集 = 8 个 acc 分布
  - 关键可视化：Widar 上 Multi-Rx (红) 最高，MMFi 上 Multi-scale (蓝) 最高 → 反转
  - 让 reviewer 一眼看到 §8 dataset-dependent 现象

风格与论文其他 Fig (1-6) 一致（matplotlib / DejaVu Sans / 学术配色）。
"""
import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 8.5,
    'axes.labelsize': 9,
    'axes.titlesize': 10,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 7.5,
    'figure.dpi': 130,
    'savefig.dpi': 200,
})

# 配色
COL_INPUT = '#5B7C99'
COL_BVP = '#7E6B8F'
COL_REPR_OURS = '#D62728'
COL_REPR_OTH = '#AEC7E8'
COL_LODO = '#2CA02C'
COL_BACKBONE = '#FF7F0E'
COL_EVAL = '#9467BD'
COL_STAT = '#8C564B'
COL_TEST = '#FFD580'
COL_MDE = '#17BECF'

# 方法配色 (violin plot 用)
COL_MR = '#D62728'      # Multi-Rx (红)
COL_MS = '#1F77B4'      # Multi-scale (蓝)
COL_RMS = '#888888'     # RMS-agg (灰)
COL_SR = '#BBBBBB'      # Single-Rx (浅灰)

ROOT = r'F:\python_workspace\wifi识别\wifi-crossenv-lab'
OUT_DIR = os.path.join(ROOT, 'src', 'bvp_test')


def box(ax, x, y, w, h, text, color, text_color='white', fontsize=9, weight='normal', zorder=2):
    p = FancyBboxPatch((x, y), w, h,
                       boxstyle="round,pad=0.02,rounding_size=0.04",
                       linewidth=1.2, edgecolor='black', facecolor=color, zorder=zorder)
    ax.add_patch(p)
    ax.text(x + w/2, y + h/2, text, ha='center', va='center',
            color=text_color, fontsize=fontsize, weight=weight, zorder=zorder+1)


def arrow(ax, x1, y1, x2, y2, color='black', lw=1.0, style='-', zorder=1):
    a = FancyArrowPatch((x1, y1), (x2, y2),
                        arrowstyle='-|>', mutation_scale=11,
                        color=color, lw=lw, linestyle=style, zorder=zorder)
    ax.add_patch(a)


def make_fig0():
    # v4 加高画布 8.5 → 10.0 以容纳 violin plot
    fig, ax = plt.subplots(figsize=(15, 10.0))
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 10.0)
    ax.set_aspect('equal')
    ax.axis('off')

    # ============== 顶部标题 ==============
    ax.text(7.5, 9.70,
            'Figure 0: Reproducibility Audit Pipeline — from raw CSI to paired multi-seed statistical inference',
            ha='center', va='center', fontsize=11, weight='bold')
    ax.text(7.5, 9.42,
            'Three core contributions: (★) Multi-Rx representation, (§3.3) LODO protocol, (§3.5) Paired × 5-seed × MDE protocol',
            ha='center', va='center', fontsize=8.5, color='#444', style='italic')

    # ============== 阶段标签 ==============
    ax.text(1.5, 8.95, '1. Input', fontsize=9.5, weight='bold', color=COL_INPUT, ha='center')
    ax.text(4.2, 8.95, '2. Preprocessing', fontsize=9.5, weight='bold', color=COL_BVP, ha='center')
    ax.text(7.0, 8.95, '3. Representation  (★ = our key axis)',
            fontsize=9.5, weight='bold', color=COL_REPR_OURS, ha='center')
    ax.text(10.0, 8.95, '4. Backbone', fontsize=9.5, weight='bold', color=COL_BACKBONE, ha='center')

    # ============== Row 1: DATA → BVP → REPRESENTATION → Backbone ==============
    box(ax, 0.3, 7.95, 2.4, 0.85, 'Raw CSI Tensor\n$H \\in \\mathbb{C}^{T \\times A \\times S}$',
        COL_INPUT, fontsize=8.5)
    ax.text(1.5, 7.77, 'Widar: (200, 9, 30) | MMFi: (128, 10, 114)',
            ha='center', va='top', fontsize=7.3, style='italic', color='#444')

    box(ax, 3.1, 7.95, 2.3, 0.85, 'BVP Preprocessing\nSTFT + MTI filter',
        COL_BVP, fontsize=8.5)
    ax.text(4.25, 7.77, 'per-Rx Doppler maps\n$\\to B \\in \\mathbb{R}^{D \\times S \\times T\'}$',
            ha='center', va='top', fontsize=7.3, style='italic', color='#444')

    box(ax, 5.95, 8.35, 2.6, 0.55, '★ Multi-Rx (ours)  |  keep A dim',
        COL_REPR_OURS, fontsize=8.5, weight='bold')
    ax.text(7.25, 8.28, 'Widar: (N, 9, 30, 33, 25) | MMFi: (N, 10, 114, 33, 29)',
            ha='center', va='top', fontsize=7, style='italic', color='#444')

    box(ax, 5.95, 7.70, 2.6, 0.45, 'Multi-scale (baseline)  |  hop ∈ {8, 16, 32}',
        COL_REPR_OTH, fontsize=8, text_color='#222')
    ax.text(7.25, 7.63, '(N, 30, 33, 25)',
            ha='center', va='top', fontsize=7, style='italic', color='#444')

    box(ax, 5.95, 7.05, 2.6, 0.45, 'RMS-agg (vanilla)  |  aggregate 9→1',
        COL_REPR_OTH, fontsize=8, text_color='#222')
    ax.text(7.25, 6.98, '(N, 30, 33, 25)',
            ha='center', va='top', fontsize=7, style='italic', color='#444')

    box(ax, 8.95, 7.95, 2.2, 0.85, 'Backbone\nLeNetCSI_Attn',
        COL_BACKBONE, fontsize=8.5)
    ax.text(10.05, 7.77, '3×(Conv+BN+ReLU) + Attn\n+ 2×FC, ~0.13M params',
            ha='center', va='top', fontsize=7.3, style='italic', color='#444')

    arrow(ax, 2.7, 8.38, 3.1, 8.38, color='black', lw=1.5)
    arrow(ax, 5.4, 8.38, 5.95, 8.60, color=COL_REPR_OURS, lw=1.5)
    arrow(ax, 5.4, 8.38, 5.95, 7.92, color='black', lw=1.0)
    arrow(ax, 5.4, 8.38, 5.95, 7.27, color='black', lw=1.0)
    arrow(ax, 8.55, 8.60, 8.95, 8.55, color=COL_REPR_OURS, lw=1.5)
    arrow(ax, 8.55, 7.92, 8.95, 8.42, color='black', lw=1.0)
    arrow(ax, 8.55, 7.27, 8.95, 8.28, color='black', lw=1.0)

    # 顶部箭头标签 (简化版，只保留 ★ ours / baseline / vanilla 三个关键标签)
    ax.text(5.60, 8.62, '★ ours', fontsize=6.5, ha='center', va='center',
            color=COL_REPR_OURS, weight='bold')
    ax.text(5.60, 7.85, 'baseline', fontsize=6.5, ha='center', va='center',
            color='#444', style='italic')
    ax.text(5.60, 7.20, 'vanilla', fontsize=6.5, ha='center', va='center',
            color='#444', style='italic')
    ax.text(8.70, 8.30, '★→', fontsize=7, ha='center', va='center',
            color=COL_REPR_OURS, weight='bold')

    # ============== LODO Protocol ==============
    ax.text(0.3, 6.35, '5. LODO Protocol', fontsize=10, weight='bold', color=COL_LODO)
    ax.text(0.3, 6.15, 'Leave-One-Domain-Out', fontsize=8, color='#444', style='italic')

    ax.text(0.3, 5.85, 'Widar3.0 (6 rooms):', fontsize=8.5, weight='bold')
    widar_doms = [('d1', False), ('d3', False), ('d5', False), ('d6', False), ('d7', True), ('d9', False)]
    for i, (d, is_test) in enumerate(widar_doms):
        face = COL_TEST if is_test else COL_LODO
        edge = 'red' if is_test else 'black'
        lw = 1.5 if is_test else 1.0
        rect = Rectangle((0.3 + i*0.42, 5.45), 0.4, 0.35,
                         facecolor=face, edgecolor=edge, linewidth=lw)
        ax.add_patch(rect)
        ax.text(0.5 + i*0.42, 5.625, d, ha='center', va='center', fontsize=7.5)
    ax.text(0.3, 5.20, '↳ 6 folds (one per held-out room)', fontsize=7.3, color='#444', style='italic')

    ax.text(0.3, 4.90, 'MMFi (4 envs):', fontsize=8.5, weight='bold')
    mmfi_envs = [('E1', False), ('E2', False), ('E3', True), ('E4', False)]
    for i, (e, is_test) in enumerate(mmfi_envs):
        face = COL_TEST if is_test else COL_LODO
        edge = 'red' if is_test else 'black'
        lw = 1.5 if is_test else 1.0
        rect = Rectangle((0.3 + i*0.42, 4.50), 0.4, 0.35,
                         facecolor=face, edgecolor=edge, linewidth=lw)
        ax.add_patch(rect)
        ax.text(0.5 + i*0.42, 4.675, e, ha='center', va='center', fontsize=7.5)
    ax.text(0.3, 4.25, '↳ 4 folds (one per held-out env)', fontsize=7.3, color='#444', style='italic')

    ax.add_patch(Rectangle((3.0, 4.55), 0.25, 0.22, facecolor=COL_LODO, edgecolor='black', linewidth=0.8))
    ax.text(3.30, 4.66, 'Train domain', va='center', fontsize=7.5)
    ax.add_patch(Rectangle((4.4, 4.55), 0.25, 0.22, facecolor=COL_TEST, edgecolor='red', linewidth=1.2))
    ax.text(4.70, 4.66, 'Test domain (held out)', va='center', fontsize=7.5)

    # ============== Paired × Multi-Seed Grid ==============
    ax.text(5.85, 6.35, '6. Paired × Multi-Seed Evaluation', fontsize=10, weight='bold', color=COL_EVAL)
    ax.text(5.85, 6.15, '5 seeds × 6 folds = 30 paired samples (Widar)', fontsize=8, color='#444', style='italic')

    n_seed, n_fold = 5, 6
    grid_x, grid_y = 5.95, 4.55
    cell_w, cell_h = 0.35, 0.32
    for s in range(n_seed):
        for f in range(n_fold):
            x = grid_x + f * cell_w
            y = grid_y + (n_seed - 1 - s) * cell_h
            face = COL_TEST if f == 4 else '#E8E8E8'
            edge = 'red' if f == 4 else '#999'
            rect = Rectangle((x, y), cell_w-0.03, cell_h-0.03,
                             facecolor=face, edgecolor=edge, linewidth=0.7)
            ax.add_patch(rect)
    fold_labels = ['d1', 'd3', 'd5', 'd6', 'd7', 'd9']
    for f in range(n_fold):
        ax.text(grid_x + f * cell_w + cell_w/2, grid_y - 0.18,
                fold_labels[f], ha='center', va='top', fontsize=7, color='#444')
    for s in range(n_seed):
        ax.text(grid_x - 0.15, grid_y + (n_seed - 1 - s) * cell_h + cell_h/2,
                f's{s}', ha='right', va='center', fontsize=7, color='#444')
    # 之前的 "30 acc samples" 标签方案：
    #   - v1: (8.25, 5.35) → 与 Paired t-test 框重叠 0.087 sq.in
    #   - v2: (8.65, 4.00) → 与 MDE_ROW1 重叠
    # 决定删除：GRID_SUB "5 seeds × 6 folds = 30 paired samples (Widar)" 已包含此信息

    # ============== 🆕 嵌入 Violin Plot (8 conditions × 2 datasets) ==============
    # 位置：figure 顶部右侧 (11.5, 6.8) - (14.85, 9.0)
    # 实测 fig.add_axes([..., bottom*0.74, height*0.24]) 渲染位置 ax y=6.79-9.24 (与 t-SNE box 间距 0.79 inch)
    # fig.add_axes 的 fig fraction 是基于 fig bbox_inches='tight' 后的实际渲染高度 (~9.17 inch) 而非 figsize
    # 实测 violin plot 实际高度 ≈ fig_fraction × 9.17 × 1.115 (实测 0.27 fig_fraction 渲染 3.014 inch)
    # 故 violin plot fig fraction height 应小一些以匹配代码预期高度
    violin_left = 11.5 / 15.0   # = 0.767
    violin_bottom = 7.4 / 10.0  # = 0.74 (实测对应 ax y=6.79，让 violin plot 实际位置在 t-SNE box 上方 0.79 inch)
    violin_width = 3.35 / 15.0  # = 0.223
    violin_height = 2.4 / 10.0  # = 0.24 (实测渲染高度 ≈ 2.45 inch, 略大于代码预期)
    violin_ax = fig.add_axes([violin_left, violin_bottom, violin_width, violin_height])

    # 加载 violin 数据
    vd = json.load(open(os.path.join(OUT_DIR, '_violin_data.json'), encoding='utf-8'))

    # 4 方法 × 2 数据集 = 8 violins
    methods = ['Multi-Rx', 'Multi-scale', 'RMS-agg', 'Single-Rx']
    method_keys = [('Widar_Multi-Rx', 'MMFi_Multi-Rx'),
                   ('Widar_multiscale', 'MMFi_multiscale'),
                   ('Widar_RMS-agg', 'MMFi_RMS-agg'),
                   ('Widar_Single-Rx', 'MMFi_Single-Rx')]
    method_colors = [COL_MR, COL_MS, COL_RMS, COL_SR]

    # 为 violin plot 准备数据
    widar_data = [vd[k[0]] for k in method_keys]
    mmfi_data = [vd[k[1]] for k in method_keys]

    # 创建位置
    n_meth = 4
    positions_widar = np.arange(1, n_meth+1)
    positions_mmfi = np.arange(1, n_meth+1) + 0.45  # 偏移让两个数据集的小提琴并排

    # 画 Widar violins (在位置 1-4)
    parts_w = violin_ax.violinplot(widar_data, positions=positions_widar,
                                    widths=0.4, showmeans=True, showmedians=False, showextrema=False)
    for i, pc in enumerate(parts_w['bodies']):
        pc.set_facecolor(method_colors[i])
        pc.set_edgecolor('black')
        pc.set_alpha(0.75)
        pc.set_linewidth(0.8)
    if 'cmeans' in parts_w:
        parts_w['cmeans'].set_color('black')
        parts_w['cmeans'].set_linewidth(1.5)

    # 画 MMFi violins (在位置 1.45-4.45)
    parts_m = violin_ax.violinplot(mmfi_data, positions=positions_mmfi,
                                    widths=0.4, showmeans=True, showmedians=False, showextrema=False)
    for i, pc in enumerate(parts_m['bodies']):
        pc.set_facecolor(method_colors[i])
        pc.set_edgecolor('black')
        pc.set_alpha(0.45)
        pc.set_linewidth(0.8)
        # 加斜线 hatch 表示 MMFi
        pc.set_hatch('///')
    if 'cmeans' in parts_m:
        parts_m['cmeans'].set_color('black')
        parts_m['cmeans'].set_linewidth(1.5)

    # 在 violin plot 上叠加真实数据点 (scatter)
    np.random.seed(42)
    for i, (w_d, m_d) in enumerate(zip(widar_data, mmfi_data)):
        # Widar 数据点
        x_w = positions_widar[i] + np.random.uniform(-0.08, 0.08, len(w_d))
        violin_ax.scatter(x_w, w_d, s=10, color=method_colors[i], edgecolor='black',
                          linewidth=0.3, alpha=0.85, zorder=4)
        # MMFi 数据点
        x_m = positions_mmfi[i] + np.random.uniform(-0.08, 0.08, len(m_d))
        violin_ax.scatter(x_m, m_d, s=10, color=method_colors[i], edgecolor='black',
                          linewidth=0.3, alpha=0.85, zorder=4)

    # X 轴标签
    violin_ax.set_xticks([1.225, 2.225, 3.225, 4.225])
    violin_ax.set_xticklabels(methods, fontsize=7.5)
    violin_ax.tick_params(axis='x', length=0)
    violin_ax.set_xlim(0.5, 5.0)

    # Y 轴（扩大上界至 50 容纳 ★ Widar best 标注 + 47.52 实际峰值）
    violin_ax.set_ylabel('Test accuracy (%)', fontsize=8)
    violin_ax.set_ylim(0, 50)
    violin_ax.tick_params(axis='y', labelsize=7)
    violin_ax.grid(axis='y', linestyle=':', linewidth=0.5, color='#888', alpha=0.5)
    violin_ax.set_axisbelow(True)

    # 标题（pad=14 给 ★ 标注留 bbox 空间）
    violin_ax.set_title('Acc distribution: Widar (n=30) ■ vs MMFi (n=20) ▨',
                        fontsize=8.5, weight='bold', pad=14)

    # 在 Multi-Rx (Widar) 顶端标 ★ Widar best —— 用 bbox 框起来避免与 violin 顶端数据点视觉重叠
    violin_ax.annotate('★ Widar best',
                       xy=(1, 47.5), xytext=(1, 45),
                       fontsize=8, color=COL_MR, weight='bold', ha='center', va='center',
                       arrowprops=dict(arrowstyle='->', color=COL_MR, lw=1.2),
                       bbox=dict(boxstyle='round,pad=0.25', facecolor='#FFE8E8',
                                 edgecolor=COL_MR, linewidth=1.0, alpha=0.95),
                       zorder=10)

    # 在 Multi-scale (MMFi) 顶端标 ★ MMFi best —— bbox 框避免与 Multi-scale Widar violin 顶部数据点视觉重叠
    violin_ax.annotate('★ MMFi best',
                       xy=(2.45, 11), xytext=(2.45, 13.0),
                       fontsize=8, color=COL_MS, weight='bold', ha='center', va='center',
                       arrowprops=dict(arrowstyle='->', color=COL_MS, lw=1.2),
                       bbox=dict(boxstyle='round,pad=0.25', facecolor='#E8F0FF',
                                 edgecolor=COL_MS, linewidth=1.0, alpha=0.95),
                       zorder=10)

    # INVERSION 反转说明框 — 放在主图底部 "10 methods" 上方 (远离 violin plot)
    ax.text(7.5, 1.85,
            '⟹ INVERSION:  on Widar3.0 Multi-Rx wins (★ Widar best)  |  on MMFi Multi-scale wins (★ MMFi best) — no single "best representation" exists',
            fontsize=8, color='#222', ha='center', va='center', style='italic',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#FFFACD',
                      edgecolor='red', linewidth=1.2))

    # ============== Statistical Inference ==============
    ax.text(9.5, 6.35, '7. Statistical Inference', fontsize=10, weight='bold', color=COL_STAT)
    ax.text(9.5, 6.15, 'Paired tests + effect size + power', fontsize=8, color='#444', style='italic')

    # 4 stat method boxes — 重新分配位置避免与 violin plot X 轴 labels 视觉接触
    # violin plot 底部 X 轴 labels 大约在 fig y=6.4 (位于 violin plot 6.5-9.5 之下约 0.1 inch)
    # Paired t-test: 左边
    # Cohen's d: 中间
    # t-SNE diagnostic: 右边 (移到 violin plot X 轴 labels 下方仍有间距的位置)
    box(ax, 8.85, 5.45, 1.30, 0.55, "Paired t-test", COL_STAT, fontsize=8)
    box(ax, 10.25, 5.45, 1.30, 0.55, "Cohen's d", COL_STAT, fontsize=8)

    # t-SNE diagnostic — 第三个 stat box (放在右边) — 已移到 violin plot 右侧不被遮挡
    # violin plot 范围 x=11.5-14.85 y=6.8-9.5；t-SNE box 范围 x=11.65-12.95 y=5.45-6.00 间距 0.8 inch
    box(ax, 11.65, 5.45, 1.30, 0.55, "t-SNE\ndiagnostic", COL_STAT, fontsize=8)

    # ====== 新增：MDE formula derivation box ======
    formula_y = 4.80
    ax.add_patch(FancyBboxPatch(
        (8.85, formula_y), 2.70, 0.55,
        boxstyle="round,pad=0.02,rounding_size=0.04",
        linewidth=1.2, edgecolor='black', facecolor='#E8F4F8', zorder=2))
    ax.text(10.20, formula_y + 0.27,
            'MDE  =  $(t_{\\alpha/2, n-1} + t_{\\beta, n-1}) \\times \\sigma_{\\mathrm{within}} \\,/\\, \\sqrt{n}$',
            ha='center', va='center', fontsize=7.5, color='#0B3D5C', weight='bold', zorder=3)
    ax.text(10.20, formula_y + 0.02,
            '= 2.91 × $\\sigma_{\\mathrm{within}}$ / $\\sqrt{n}$  (n=30 paired)',
            ha='center', va='center', fontsize=7, color='#0B3D5C', style='italic', zorder=3)

    # 箭头：grid → 4 stat boxes (3 个箭头，原来只有 2 个，t-SNE 箭头缺失)
    # A8: grid → Paired t-test
    arrow(ax, grid_x + n_fold * cell_w + 0.05, grid_y + n_seed * cell_h/2,
          9.50, 6.00, color='black', lw=1.0)
    # A9: grid → Cohen's d
    arrow(ax, grid_x + n_fold * cell_w + 0.05, grid_y + n_seed * cell_h/2,
          10.90, 6.00, color='black', lw=1.0)
    # A10 (新增，之前缺失！): grid → t-SNE diagnostic
    arrow(ax, grid_x + n_fold * cell_w + 0.05, grid_y + n_seed * cell_h/2,
          12.30, 6.00, color='black', lw=1.0)

    # ====== σ_within → MDE 数值实例表 ======
    inst_y = 4.20
    ax.text(8.85, inst_y + 0.28, 'MDE numerical instantiation (Widar LODO, 30 paired):',
            ha='left', va='center', fontsize=7.5, weight='bold', color=COL_MDE)

    # 表头
    ax.add_patch(Rectangle((8.85, inst_y), 4.10, 0.20,
                           facecolor='#444', edgecolor='black', linewidth=0.8))
    ax.text(9.40, inst_y + 0.10, 'representation', ha='center', va='center',
            fontsize=7.5, color='white', weight='bold')
    ax.text(10.20, inst_y + 0.10, 'σ_within (pp)', ha='center', va='center',
            fontsize=7.5, color='white', weight='bold')
    ax.text(11.10, inst_y + 0.10, '= MDE (pp)', ha='center', va='center',
            fontsize=7.5, color='white', weight='bold')
    ax.text(12.45, inst_y + 0.10, 'detectable?', ha='center', va='center',
            fontsize=7, color='white', weight='bold')

    rows = [
        ('★ Multi-Rx', '8.22', '4.37', '✓ Δ=+7.69pp > MDE 4.37pp'),
        ('RMS-agg',    '1.50', '0.80', '✓ Δ<1pp < MDE 0.80pp'),
        ('Single-Rx',  '2.10', '1.12', '✓ Δ<1pp < MDE 1.12pp'),
    ]
    for i, (repr_name, sigma, mde, verdict) in enumerate(rows):
        y = inst_y - 0.18 - i * 0.18
        if i == 0:
            ax.add_patch(Rectangle((8.85, y), 4.10, 0.18,
                                   facecolor='#FFE0E0', edgecolor='#888', linewidth=0.5))
        else:
            ax.add_patch(Rectangle((8.85, y), 4.10, 0.18,
                                   facecolor='#F8F8F8', edgecolor='#CCC', linewidth=0.5))
        weight = 'bold' if i == 0 else 'normal'
        color = COL_REPR_OURS if i == 0 else '#222'
        ax.text(9.40, y + 0.09, repr_name, ha='center', va='center',
                fontsize=7.5, weight=weight, color=color)
        ax.text(10.20, y + 0.09, sigma, ha='center', va='center', fontsize=7.5)
        ax.text(11.10, y + 0.09, mde, ha='center', va='center',
                fontsize=7.5, weight=weight, color=color)
        ax.text(12.45, y + 0.09, verdict, ha='center', va='center', fontsize=6.8)

    # ====== Verdict logic box (下移到 violin plot 下方 1.5 单位，避免 bbox 边缘重叠) ======
    ax.text(13.2, 2.95,
            'Verdict logic:\n'
            '  • |d| < 0.3  &  |Δ| < MDE  →  NOT REAL\n'
            '  • |d| ≥ 0.8  &  Δ > MDE  →  REAL & REPRODUCIBLE',
            ha='center', va='top', fontsize=7.5, color='#222',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#FFF8DC', edgecolor='#888', linewidth=0.8))

    # 箭头：formula box → verdict
    arrow(ax, 11.2, 4.80, 13.2, 4.05, color=COL_MDE, lw=1.0, style='--')

    # ============== 🆕 反转发现 callout 框 (左下角，紧邻 LODO) ==============
    ax.text(7.5, 3.55,
            '>>> Key observation (§8): Representation choice is DATASET-DEPENDENT.\n'
            '    >> On Widar3.0:  Multi-Rx (keep A=9) wins  → +7.69pp vs RMS-agg (d=1.71)\n'
            '    >> On MMFi:        Multi-scale (hop ∈{8,16,32}) wins  → +1.20pp vs Multi-Rx (d=0.59)\n'
            '    No single "best representation" exists across datasets. The choice depends on\n'
            '    the inherent separability of the BVP domain manifold (see t-SNE, §8.3).',
            ha='center', va='top', fontsize=7.5, color='#222',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#FFFACD',
                      edgecolor='red', linewidth=1.5))

    # ============== Footer: 10 methods ==============
    ax.text(7.5, 1.30, '10 methods compared (each method × 30 paired samples on Widar / 20 on MMFi):',
            ha='center', va='center', fontsize=9, weight='bold')
    methods_text = (
        'Architectural:  LeNet | LeNet+Attn | LeNet+Attn+MHA | ResNet8     '
        '·    Algorithmic:  DANN (GRL) | Snapshot×3 | Mixup | Reweighting | Pretrain\n'
        'Representational (key axis):  Widar-orig (RMS h=16)  |  Multi-scale (hop ∈{8,16,32})  |  '
        '★ Multi-Rx (keep A=9)  |  RMS-agg (vanilla)  |  Single-Rx (A=1, ref)'
    )
    ax.text(7.5, 0.55, methods_text, ha='center', va='center', fontsize=7.8, color='#222',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#F5F5F5', edgecolor='#888', linewidth=0.8))

    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, 'fig0_pipeline.png')
    plt.savefig(out_path, bbox_inches='tight', facecolor='white')
    print(f'[OK] Saved → {out_path}')


if __name__ == '__main__':
    make_fig0()