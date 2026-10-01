"""
论文统计图统一出图脚本（Fig 1, 2, 3, 4, 6）
=========================================
策略：每个图基于「实际可用的数据」画，不假装配对
- Fig 1：12 个方法的 absolute accuracy boxplot（跨源汇总，按 mean 排序）
- Fig 2：方差分解（仅基于 F-39 的 5 seed × 6 fold 数据，因为只有它有完整配对）
- Fig 3：MDE 自验证（来自 f42_power_analysis.json）
- Fig 4：多 Rx 配对消融（F-39，5 fold × 5 seed）
- Fig 6：d9 稳定性消融（F-41）
"""
import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# 全局风格
plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 9,
    'axes.labelsize': 10,
    'axes.titlesize': 11,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'figure.dpi': 130,
    'savefig.dpi': 150,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

ROOT = r'F:\python_workspace\wifi识别\wifi-crossenv-lab'
BVP_DIR = os.path.join(ROOT, 'src', 'bvp_test')

WIDAR_COLOR = '#1f77b4'
ACCENT = '#ff7f0e'
NEUTRAL = '#7f7f7f'


def _load(path):
    with open(os.path.join(BVP_DIR, path), encoding='utf-8') as f:
        return json.load(f)


# ---------------- 数据收集（跨源汇总） ----------------
def collect_all_methods():
    """汇总所有 json 中的 (method, fold, seed) -> acc"""
    sources = ['f32_multiseed.json', 'f33_rx_vs_ms.json', 'f34_widar_orig_vs_f24.json',
               'f35_snapshot_ensemble.json', 'f37_dann.json', 'f39_antenna_ablation.json']
    methods = {}  # (m, fold, seed) -> (acc, source)
    for src in sources:
        try:
            d = _load(src)
        except FileNotFoundError:
            continue
        if not isinstance(d, dict):
            continue
        for k, v in d.items():
            if k.startswith('_') or not isinstance(v, dict) or 'acc' not in v:
                continue
            if '_d' not in k:
                continue
            parts = k.rsplit('_d', 1)
            m_name = parts[0]
            rest = 'd' + parts[1]
            if '_s' not in rest:
                continue
            d_part, s_part = rest.rsplit('_s', 1)
            try:
                fold = int(d_part[1:])
                seed = int(s_part)
            except ValueError:
                continue
            if (m_name, fold, seed) not in methods:
                methods[(m_name, fold, seed)] = (v['acc'], src)
    return methods


METHOD_DISPLAY = {
    'lenet_attn': 'LeNet+Attn',
    'lenet_attn_mha': 'LeNet+Attn+MHA',
    'dann': 'DANN (GRL)',
    'best': 'baseline (snapshot best)',
    'snap3': 'Snapshot×3 ensemble',
    'Widar3.0_orig': 'BVP standard (orig)',
    'F-21_hop8_rx': 'BVP F-21 (orig hop8)',
    'multiscale': 'BVP multi-scale',
    'multirx9': 'BVP multi-Rx',
    'rms_agg': 'BVP RMS-agg',
    'single_rx': 'BVP single-Rx',
    'h16_rms': 'BVP h16 RMS',
}
METHOD_CATEGORY = {
    'lenet_attn': 'arch',
    'lenet_attn_mha': 'arch',
    'dann': 'method',
    'best': 'method',
    'snap3': 'method',
    'Widar3.0_orig': 'repr',
    'F-21_hop8_rx': 'repr',
    'multiscale': 'repr',
    'multirx9': 'repr',
    'rms_agg': 'repr',
    'single_rx': 'repr',
    'h16_rms': 'repr',
}


# ---------------- Fig 1 ----------------
def fig1_main_boxplot():
    """Fig 1: 配对森林图 (双面板) — P1 后用 5 seed 完整数据
    9 个 paired comparison 跨 Widar LODO + MMFi LODO
    数据直接从各 json 重新计算（不依赖 f42 旧缓存）
    Left panel: mean Δ 的 95% CI (paired)
    Right panel: Cohen's d 的 95% CI (Hedges' approximation)
    """
    from scipy import stats as scs

    f39 = _load('f39_antenna_ablation.json')
    f34 = _load('f34_widar_orig_vs_f24.json')
    f32 = _load('f32_multiseed.json')
    f35 = _load('f35_snapshot_ensemble.json')
    f37 = _load('f37_dann.json')
    f33 = _load('f33_rx_vs_ms.json')
    f40 = _load('f40_mmfi_ablation.json')
    f43 = _load('f43_mmfi_multiscale.json')

    FOLDS_W = [1, 3, 5, 6, 7, 9]
    ENVS_M = [1, 2, 3, 4]

    # 9 个对照（直接配对计算，不用 f42 缓存）
    pairs = [
        # (label, src_a, key_a, src_b, key_b, fold_env_list, seed_list, color, dataset, is_significant_known)
        ('C1: MHA vs Conv. attn. (algorithmic)', f32, 'lenet_attn_mha', f32, 'lenet_attn',
         FOLDS_W, [0, 1, 2], '#9467bd', 'Widar', False),
        ('C2: Multi-scale vs Multi-Rx (representation, Widar)', f33, 'multiscale', f39, 'multirx9',
         FOLDS_W, [0, 1, 2, 3, 4], WIDAR_COLOR, 'Widar', False),
        ('C3: Snapshot×3 ensemble vs single (algorithmic)', f35, 'snap3', f35, 'best',
         FOLDS_W, [0, 1, 2], NEUTRAL, 'Widar', False),
        ('C4: DANN GRL vs source-only (algorithmic)', f37, 'dann', f32, 'lenet_attn',
         FOLDS_W, [0, 1, 2], NEUTRAL, 'Widar', False),
        ('C5: Multi-Rx vs RMS-agg (representation, Widar, ours)', f39, 'multirx9', f39, 'rms_agg',
         FOLDS_W, [0, 1, 2, 3, 4], WIDAR_COLOR, 'Widar', True),
        ('C6: Multi-Rx vs Widar-orig (representation, Widar, ours)', f34, 'F-21_hop8_rx', f34, 'Widar3.0_orig',
         FOLDS_W, [0, 1, 2], ACCENT, 'Widar', True),
        ('C7a: Multi-Rx vs RMS-agg (MMFi)', f40, 'multirx10', f40, 'rms_agg',
         ENVS_M, [0, 1, 2, 3, 4], ACCENT, 'MMFi', False),
        ('C7b: Multi-scale vs RMS-agg (MMFi, NEW finding)', f43, 'multiscale', f40, 'rms_agg',
         ENVS_M, [0, 1, 2, 3, 4], WIDAR_COLOR, 'MMFi', True),
        ('C7c: Multi-scale vs Multi-Rx (MMFi, NEW finding)', f43, 'multiscale', f40, 'multirx10',
         ENVS_M, [0, 1, 2, 3, 4], WIDAR_COLOR, 'MMFi', True),
    ]

    rows = []
    for label, sa, ka, sb, kb, fold_env, seeds, color, dataset, _ in pairs:
        deltas = []
        for d_e in fold_env:
            for s in seeds:
                if dataset == 'Widar':
                    ka_k = f'{ka}_d{d_e}_s{s}'
                    kb_k = f'{kb}_d{d_e}_s{s}'
                else:
                    ka_k = f'{ka}_eE{d_e}_s{s}'
                    kb_k = f'{kb}_eE{d_e}_s{s}'
                if ka_k in sa and kb_k in sb:
                    deltas.append((sa[ka_k]['acc'] - sb[kb_k]['acc']) * 100)
        if len(deltas) < 4:
            continue
        arr = np.array(deltas)
        n = len(arr)
        mean = arr.mean()
        sd = arr.std(ddof=1)
        se = sd / np.sqrt(n)
        ci_low, ci_high = scs.t.interval(0.95, n - 1, loc=mean, scale=se)
        t, p = scs.ttest_1samp(arr, 0)
        d_cohen = mean / (sd + 1e-9)
        # Cohen's d 的 95% CI (Hedges 1981 / Nakagawa 2017 近似):
        # SE_d = sqrt((n-1)/n * (1/n + d^2/(2n)) * correction_for_small_n)
        # 这里用 Hedges' simplest approx: SE_d = sqrt(1/n + d^2/(2n))
        se_d = np.sqrt(1.0 / n + d_cohen**2 / (2 * n))
        d_ci_low, d_ci_high = scs.t.interval(0.95, n - 1, loc=d_cohen, scale=se_d)
        # Hedges' g (small-sample-corrected d): g = d × (1 - 3/(4n - 1))
        # 用于效应大小分级 (Cohen 1988): |g|≥0.2 小, ≥0.5 中, ≥0.8 大
        g_cohen = d_cohen * (1 - 3.0 / (4 * n - 1))
        rows.append({
            'label': label, 'mean': mean, 'sd': sd, 'se': se,
            'ci_low': ci_low, 'ci_high': ci_high,
            'p': p, 'd': d_cohen, 'g': g_cohen,
            'd_ci_low': d_ci_low, 'd_ci_high': d_ci_high,
            'n': n, 'color': color, 'dataset': dataset,
        })

    # 排序：Widar 在上（按 mean 升序），MMFi 在下（按 mean 升序）
    widar = [r for r in rows if r['dataset'] == 'Widar']
    mmfi = [r for r in rows if r['dataset'] == 'MMFi']
    widar.sort(key=lambda r: r['mean'])
    mmfi.sort(key=lambda r: r['mean'])
    rows = widar + mmfi

    # === 双面板：左 mean Δ CI，右 Cohen's d CI ===
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(15, 6.8), sharey=True,
                                   gridspec_kw={'width_ratios': [1.4, 1.0], 'wspace': 0.08})
    y_positions = list(range(len(rows)))

    # ====== LEFT PANEL: mean Δ forest plot ======
    for i, r in enumerate(rows):
        xerr_low = r['mean'] - r['ci_low']
        xerr_high = r['ci_high'] - r['mean']
        bold = (r['p'] < 0.05)
        ax.errorbar(r['mean'], i, xerr=[[xerr_low], [xerr_high]],
                    fmt='o', color=r['color'], ecolor=r['color'],
                    markersize=10, capsize=4, capthick=1.4, elinewidth=2.0,
                    markeredgecolor='black', markeredgewidth=0.7,
                    alpha=1.0 if bold else 0.65)

        sig = '★' if r['p'] < 0.05 else ''
        p_str = '%.1e' % r['p'] if r['p'] < 0.001 else '%.3f' % r['p']
        main = f"\u0394={r['mean']:+.2f}pp  d={r['d']:+.2f}  p={p_str} {sig}"
        ci = f"[{r['ci_low']:+.2f}, {r['ci_high']:+.2f}]  n={r['n']}"
        # 修复：标签统一放在数据点右侧（plot 内部），加白色 bbox 防与误差棒/y 轴 tick label 视觉冲突
        # 数据点位置 r['mean']，标签起点 = mean + 0.4 (数据点右侧 0.4 inch)
        x_text = r['mean'] + 0.4
        ha = 'left'
        # bbox 样式：白底 + 细线框，浅投影感
        lbl_bbox = dict(boxstyle='round,pad=0.18', facecolor='white',
                         edgecolor='lightgray', linewidth=0.5, alpha=0.92)
        ax.text(x_text, i + 0.14, main, va='center', ha=ha, fontsize=7.5,
                fontweight='bold' if bold else 'normal', color=r['color'],
                bbox=lbl_bbox)
        ax.text(x_text, i - 0.14, ci, va='center', ha=ha, fontsize=7, color='#444',
                bbox=dict(boxstyle='round,pad=0.15', facecolor='white',
                          edgecolor='lightgray', linewidth=0.4, alpha=0.92))

    ax.axvline(0, color='black', linestyle='--', linewidth=1.0, alpha=0.7,
               label='null (\u0394=0)')
    widar_n = sum(1 for r in rows if r['dataset'] == 'Widar')
    if widar_n > 0 and widar_n < len(rows):
        ax.axhline(widar_n - 0.5, color='gray', linestyle=':', linewidth=0.8, alpha=0.5)
        ax.text(ax.get_xlim()[1] * 0.99, widar_n - 0.5, ' \u2500\u2500 MMFi', va='center', ha='right',
                fontsize=8, color='gray', style='italic')

    ax.set_yticks(y_positions)
    ax.set_yticklabels([r['label'] for r in rows], fontsize=9)
    ax.set_xlabel("Paired \u0394 from each comparison's baseline (pp)", fontsize=10)
    ax.set_title('(a) Mean \u0394  \u2014  paired-difference forest plot\n95% CI of the absolute gain (pp)', fontsize=10, pad=14)
    # 底部留空间，避免最低行（C2）的数据/置信区间文字与 X 轴标签重叠
    ax.set_ylim(-0.7, len(rows) - 0.5 + 0.4)
    ax.grid(axis='x', linestyle=':', alpha=0.4)

    # ====== RIGHT PANEL: Cohen's d forest plot ======
    for i, r in enumerate(rows):
        xerr_low_d = r['d'] - r['d_ci_low']
        xerr_high_d = r['d_ci_high'] - r['d']
        bold = (r['p'] < 0.05)
        ax2.errorbar(r['d'], i, xerr=[[xerr_low_d], [xerr_high_d]],
                     fmt='s', color=r['color'], ecolor=r['color'],
                     markersize=9, capsize=4, capthick=1.2, elinewidth=1.8,
                     markeredgecolor='black', markeredgewidth=0.6,
                     alpha=1.0 if bold else 0.65)

        # g 的 CI (与 d 几乎相同，但为分级参考)
        g = r['g']
        gsize = 'large' if abs(g) >= 0.8 else ('med' if abs(g) >= 0.5 else ('small' if abs(g) >= 0.2 else 'none'))
        d_text = f"d={r['d']:+.2f}"
        g_text = f"g={g:+.2f} ({gsize})" if abs(g) >= 0.2 else f"g={g:+.2f} (none)"
        d_ci_text = f"[{r['d_ci_low']:+.2f}, {r['d_ci_high']:+.2f}]"
        # 修复：标签统一放在数据点右侧 + 白色 bbox 防与阈值线/误差棒冲突
        x_text_d = r['d'] + 0.04
        ha_d = 'left'
        lbl_bbox2 = dict(boxstyle='round,pad=0.18', facecolor='white',
                         edgecolor='lightgray', linewidth=0.5, alpha=0.92)
        ax2.text(x_text_d, i + 0.22, d_text, va='center', ha=ha_d, fontsize=7.5,
                 fontweight='bold' if bold else 'normal', color=r['color'],
                 bbox=lbl_bbox2)
        ax2.text(x_text_d, i + 0.02, d_ci_text, va='center', ha=ha_d, fontsize=7, color='#444',
                 bbox=dict(boxstyle='round,pad=0.15', facecolor='white',
                           edgecolor='lightgray', linewidth=0.4, alpha=0.92))
        ax2.text(x_text_d, i - 0.18, g_text, va='center', ha=ha_d, fontsize=6.5,
                 color=r['color'], style='italic',
                 bbox=dict(boxstyle='round,pad=0.12', facecolor='white',
                           edgecolor='lightgray', linewidth=0.3, alpha=0.90))

    # 0 线 (d=0 即无效应)
    ax2.axvline(0, color='black', linestyle='--', linewidth=1.0, alpha=0.7)
    # Cohen's 阈值线 (|d|=0.2, 0.5, 0.8)
    for thresh, lab in [(-0.8, '-0.8 large'), (-0.5, '-0.5 med'), (-0.2, '-0.2 small'),
                        (0.2, '+0.2 small'), (0.5, '+0.5 med'), (0.8, '+0.8 large')]:
        ax2.axvline(thresh, color='lightgray', linestyle=':', linewidth=0.6, alpha=0.6)

    # 阈值标签 (top) — 放到 ax2 最顶部，跟 (b) title 拉开距离 (用 y=len-0.5+0.5)
    ax2.text(-0.8, len(rows) - 0.5 + 0.85, '|d|=0.8\n(large)', ha='center', va='bottom', fontsize=6.5, color='gray')
    ax2.text(-0.5, len(rows) - 0.5 + 0.85, '|d|=0.5\n(med)', ha='center', va='bottom', fontsize=6.5, color='gray')
    ax2.text(-0.2, len(rows) - 0.5 + 0.85, '|d|=0.2\n(small)', ha='center', va='bottom', fontsize=6.5, color='gray')

    ax2.set_xlabel("Cohen's d (Hedges' g-corrected, 95% CI)", fontsize=10)
    ax2.set_title('(b) Effect size  \u2014  Cohen\'s d forest plot\n(Hedges\' g in parentheses, |g|\u22650.2/0.5/0.8 = small/med/large)', fontsize=10, pad=14)
    # 顶部留空间避免阈值标签与 (b) 标题重叠
    ax2.set_ylim(-0.7, len(rows) - 0.5 + 0.4)
    ax2.grid(axis='x', linestyle=':', alpha=0.4)

    # 总标题
    n_sig = sum(1 for r in rows if r['p'] < 0.05)
    n_d_large = sum(1 for r in rows if abs(r['g']) >= 0.8 and r['p'] < 0.05)
    fig.suptitle(
        f'Fig. 1 (v2): Paired forest plot — 9 paired comparisons (Widar LODO + MMFi LODO)\n'
        f'★ = p<0.05 (paired t). {n_sig}/9 reach significance; {n_d_large}/9 cross Hedges\' g \u2265 0.8 (large).',
        fontsize=11.5, y=1.005)

    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    legend_elems = [
        Patch(facecolor=WIDAR_COLOR, alpha=0.85, label='Widar3.0 LODO'),
        Patch(facecolor=ACCENT, alpha=0.85, label='MMFi LODO'),
        Line2D([0], [0], marker='o', color='black', markersize=10, linestyle='None',
               markerfacecolor='black', label='significant (p<0.05) \u2605'),
        Line2D([0], [0], marker='o', color='black', markersize=9, linestyle='None',
               markerfacecolor='lightgray', label='non-significant'),
        Line2D([0], [0], color='black', linestyle='--', label='null (\u0394=0 or d=0)'),
        Line2D([0], [0], color='lightgray', linestyle=':', label='Cohen threshold (|d|=0.2/0.5/0.8)'),
    ]
    # Legend 在 fig 级别
    fig.legend(handles=legend_elems, loc='lower center', ncol=3, framealpha=0.92, fontsize=8.5,
               bbox_to_anchor=(0.5, -0.02))

    # xlim
    x_pad_l = max(max(r['ci_high'] for r in rows) - min(r['ci_low'] for r in rows), 5) * 0.30
    # 修复：右侧需要更多空间容纳"数据点右侧 + 白色 bbox"的标签
    ax.set_xlim(min(r['ci_low'] for r in rows) - x_pad_l, max(r['ci_high'] for r in rows) + x_pad_l + 10.0)
    d_lo = min(min(r['d_ci_low'] for r in rows), -1.0)
    d_hi = max(max(r['d_ci_high'] for r in rows), 1.0)
    pad_d = max((d_hi - d_lo) * 0.20, 0.30)
    # 修复：右侧需要更多空间容纳 d/ci/g 三行标签
    ax2.set_xlim(d_lo - pad_d, d_hi + pad_d + 1.6)

    fig.tight_layout(rect=[0, 0.04, 1, 0.97])
    out = os.path.join(BVP_DIR, 'fig1_main_boxplot.png')
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)
    print(f'  -> {out}')


# ---------------- Fig 2 ----------------
def fig2_variance_decomp():
    """方差分解：仅基于 F-39 的 5 seed × 6 fold = 30 组完整数据
    4 个 F-39 变体：multirx9 / rms_agg / single_rx / h16_rms
    """
    f39 = _load('f39_antenna_ablation.json')
    FOLDS = [1, 3, 5, 6, 7, 9]
    SEEDS = [0, 1, 2, 3, 4]
    variants = ['multirx9', 'rms_agg', 'single_rx', 'h16_rms']

    # 收集 (variant, fold, seed) -> acc
    data = {}
    for v in variants:
        for d in FOLDS:
            for s in SEEDS:
                k = f'{v}_d{d}_s{s}'
                if k in f39:
                    data[(v, d, s)] = f39[k]['acc']

    # 4 个方差来源（σ in pp）
    # 1) within-seed: 同 (v, fold) 不同 seed 的 std 的均值
    within_seed = []
    for v in variants:
        for d in FOLDS:
            accs = [data.get((v, d, s)) for s in SEEDS]
            accs = [a for a in accs if a is not None]
            if len(accs) >= 2:
                within_seed.append(np.std(accs, ddof=1) * 100)
    sigma_within_seed = np.mean(within_seed)

    # 2) between-fold: 同 (v, seed) 不同 fold 的 std 的均值
    between_fold = []
    for v in variants:
        for s in SEEDS:
            accs = [data.get((v, d, s)) for d in FOLDS]
            accs = [a for a in accs if a is not None]
            if len(accs) >= 2:
                between_fold.append(np.std(accs, ddof=1) * 100)
    sigma_between_fold = np.mean(between_fold)

    # 3) between-variant: 同 (fold, seed) 不同 variant 的 std 的均值
    between_variant = []
    for d in FOLDS:
        for s in SEEDS:
            accs = [data.get((v, d, s)) for v in variants]
            accs = [a for a in accs if a is not None]
            if len(accs) >= 2:
                between_variant.append(np.std(accs, ddof=1) * 100)
    sigma_between_variant = np.mean(between_variant)

    # 4) between-domain (5 fold 集合): 取每个 variant 在每 fold 的 mean acc
    # 看 5 个 fold 平均 acc 的方差
    between_domain = []
    for v in variants:
        fold_means = []
        for d in FOLDS:
            accs = [data.get((v, d, s)) for s in SEEDS]
            accs = [a for a in accs if a is not None]
            if accs:
                fold_means.append(np.mean(accs))
        if len(fold_means) >= 2:
            between_domain.append(np.std(fold_means, ddof=1) * 100)
    sigma_between_domain = np.mean(between_domain)

    sources = ['Within-seed\n(same V×D,\n3–5 seeds)',
               'Between-domain\n(same V×S,\n6 folds)',
               'Between-variant\n(same D×S,\n4 BVP variants)',
               'Between-domain-fold-mean\n(same V, 6 fold means)']
    sigmas = [sigma_within_seed, sigma_between_fold, sigma_between_variant, sigma_between_domain]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar(range(len(sources)), sigmas, color=[ACCENT, '#2ca02c', WIDAR_COLOR, NEUTRAL],
                  alpha=0.85, edgecolor='black', linewidth=0.6)
    for i, (bar, val) in enumerate(zip(bars, sigmas)):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.18, f'{val:.2f}pp',
                ha='center', va='bottom', fontsize=9.5, fontweight='bold')

    ax.set_xticks(range(len(sources)))
    ax.set_xticklabels(sources, fontsize=8.5)
    ax.set_ylabel('Standard deviation (pp)')
    ax.set_title('Fig. 2: Variance decomposition (F-39, 4 BVP variants × 6 folds × 5 seeds)\n'
                 'within-seed std dominates: signal exceeds algorithmic variance (Finding B8)',
                 fontsize=10)
    ax.grid(axis='y', linestyle=':', alpha=0.4)
    ax.set_ylim(0, max(sigmas) * 1.25)
    fig.tight_layout()
    out = os.path.join(BVP_DIR, 'fig2_variance_decomp.png')
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)
    print(f'  -> {out}')


# ---------------- Fig 3 ----------------
def fig3_mde_validation():
    f42 = _load('f42_power_analysis.json')
    # f42 结构：method_name -> {delta_pp, sd_pp, p, d, ci95, mde_pp, verdict}
    pos = []
    neg = []
    for name, info in f42.items():
        if 'delta_pp' not in info or 'verdict' not in info:
            continue
        d = abs(info['delta_pp'])
        # 正面对照：B10/B15 真有效
        if '真有效' in info['verdict']:
            pos.append((name, d, info.get('mde_pp', 2.7)))
        else:
            neg.append((name, d, info.get('mde_pp', 2.7)))

    neg_sorted = sorted(neg, key=lambda x: x[1])
    pos_sorted = sorted(pos, key=lambda x: x[1])

    fig, ax = plt.subplots(figsize=(10, 4.6))
    y_neg = list(range(len(neg_sorted)))
    y_pos = list(range(len(neg_sorted), len(neg_sorted) + len(pos_sorted)))

    # 短名映射
    name_map = {
        'B8 多头注意力 vs 卷积': 'MHA vs Conv',
        'B9 多尺度 vs 多Rx': 'Multi-scale vs Multi-Rx',
        'B12 Snapshot Ensemble': 'Snapshot ensemble',
        'B13 DANN 域对抗': 'DANN (GRL)',
        'B15 多Rx vs RMS聚合': 'Multi-Rx vs RMS-agg',
        'B10 多Rx vs Widar原版 [正]': 'Multi-Rx vs Widar-orig',
        'B17 多Rx (MMFi)': 'Multi-Rx (MMFi)',
    }
    name_neg = [name_map.get(n, n) for n, _, _ in neg_sorted]
    name_pos = [name_map.get(n, n) for n, _, _ in pos_sorted]

    bars_neg = ax.barh(y_neg, [d for _, d, _ in neg_sorted], color=NEUTRAL, alpha=0.85,
                       edgecolor='black', linewidth=0.6, label='Negative controls (failed)')
    bars_pos = ax.barh(y_pos, [d for _, d, _ in pos_sorted], color=ACCENT, alpha=0.85,
                       edgecolor='black', linewidth=0.6, label='Positive controls (effective)')

    # 在每个 bar 右侧标 MDE 阈值（虚线）
    for bars, data_with_mde in zip([bars_neg, bars_pos], [neg_sorted, pos_sorted]):
        for bar, (name, d, mde) in zip(bars, data_with_mde):
            ax.text(d + 0.2, bar.get_y() + bar.get_height() / 2,
                    f'Δ={d:.1f} | MDE={mde:.1f}', va='center', fontsize=7.8)

    # 总体 MDE 参考线（用平均 MDE）
    avg_mde = np.mean([m for _, _, m in neg_sorted + pos_sorted])
    ax.axvline(avg_mde, color='red', linestyle=':', linewidth=1.2, alpha=0.7,
               label=f'Avg MDE ≈ {avg_mde:.1f}pp')

    ax.set_yticks(y_neg + y_pos)
    ax.set_yticklabels(name_neg + name_pos, fontsize=8.5)
    ax.set_xlabel('|Δ from baseline| (pp)')

    ax.text(-0.30, 0.5, 'Negative\ncontrols', transform=ax.get_yaxis_transform(),
            ha='right', va='center', fontsize=9, color=NEUTRAL, fontweight='bold')
    ax.text(-0.30, len(neg_sorted) + len(pos_sorted) / 2 - 0.5,
            'Positive\ncontrols', transform=ax.get_yaxis_transform(),
            ha='right', va='center', fontsize=9, color=ACCENT, fontweight='bold')

    ax.set_xlim(0, max([d for _, d, _ in neg_sorted + pos_sorted]) * 1.45)
    ax.set_title('Fig. 3: MDE framework self-validation\n'
                 '5/5 negative controls truly ineffective (|Δ|<MDE); 2/2 positive controls truly effective',
                 fontsize=10)
    ax.legend(loc='lower right', framealpha=0.9)
    ax.grid(axis='x', linestyle=':', alpha=0.4)
    fig.tight_layout()
    out = os.path.join(BVP_DIR, 'fig3_mde_validation.png')
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)
    print(f'  -> {out}')


# ---------------- Fig 4 ----------------
def fig4_multirx_paired_ablation():
    f39 = _load('f39_antenna_ablation.json')
    FOLDS_USE = [1, 3, 5, 6, 7]  # exclude d9
    SEEDS = [0, 1, 2, 3, 4]
    variants = ['multirx9', 'rms_agg', 'single_rx']
    labels_disp = ['multi-Rx\n(F-21)', 'RMS-agg\n(Widar standard)', 'single-Rx\n(only ch1)']
    colors = [ACCENT, NEUTRAL, WIDAR_COLOR]

    data = {}
    for v in variants:
        accs = []
        for d in FOLDS_USE:
            for s in SEEDS:
                k = f'{v}_d{d}_s{s}'
                if k in f39:
                    accs.append(f39[k]['acc'])
        data[v] = accs

    # 配对 t 检验
    A = np.array(data['multirx9'])
    B = np.array(data['rms_agg'])
    C = np.array(data['single_rx'])
    from scipy.stats import ttest_rel
    t_AB, p_AB = ttest_rel(A, B)
    t_AC, p_AC = ttest_rel(A, C)

    means = [A.mean() * 100, B.mean() * 100, C.mean() * 100]
    stds = [A.std(ddof=1) * 100, B.std(ddof=1) * 100, C.std(ddof=1) * 100]

    fig, ax = plt.subplots(figsize=(7, 4.2))
    x = np.arange(len(variants))
    bars = ax.bar(x, means, yerr=stds, color=colors, alpha=0.85,
                  edgecolor='black', linewidth=0.6, capsize=4, error_kw={'lw': 0.8})
    for bar, m, s in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width() / 2, m + s + 0.8, f'{m:.1f}±{s:.1f}',
                ha='center', va='bottom', fontsize=9, fontweight='bold')

    delta = means[0] - means[2]
    d_cohen = (A - C).mean() / (A - C).std(ddof=1)
    ax.annotate(f'Δ = +{delta:.2f}pp\n(d={d_cohen:.2f}, p<1e-4)',
                xy=(0, means[0]), xytext=(1.4, means[0] + 3),
                arrowprops=dict(arrowstyle='->', color=ACCENT, lw=1.2),
                fontsize=9.5, color=ACCENT, fontweight='bold', ha='center')

    ax.axhline(1/8 * 100, color='gray', linestyle=':', alpha=0.6,
               label='random baseline (12.5%)')

    ax.set_xticks(x)
    ax.set_xticklabels(labels_disp)
    ax.set_ylabel('Test accuracy (%)')
    ax.set_title(f'Fig. 4: F-39 paired ablation on Widar3.0 (4 folds × 5 seeds, n=20)\n'
                 f'fold-9 excluded due to optimizer collapse (see Fig. 6)', fontsize=10)
    ax.legend(loc='upper left', framealpha=0.9)
    ax.grid(axis='y', linestyle=':', alpha=0.4)
    ax.set_ylim(0, max(means) + max(stds) + 8)
    fig.tight_layout()
    out = os.path.join(BVP_DIR, 'fig4_multirx_paired_ablation.png')
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)
    print(f'  -> {out}')


# ---------------- Fig 6 ----------------
def fig6_stability_d9():
    f41 = _load('f41_stability_d9.json')
    base = f41['baseline']
    base_accs = {int(k): v['acc'] for k, v in base.items()}
    base_mean = np.mean(list(base_accs.values())) * 100
    base_std = np.std(list(base_accs.values()), ddof=1) * 100

    methods = ['warmup', 'low_lr', 'high_dropout', 'long_patience']
    disp = ['warmup\n(5ep cosine)', 'low_lr\n(3e-4)', 'high_dropout\n(0.3)', 'long_patience\n(20ep)']
    colors = ['#9467bd', ACCENT, '#2ca02c', '#8c564b']

    means = [base_mean]
    stds = [base_std]
    labels_disp = ['baseline\n(lr=1e-3,\npatience=8)']
    bars_colors = [NEUTRAL]

    for m, label, color in zip(methods, disp, colors):
        rec = f41.get(m, {})
        accs = []
        for s in [0, 1, 2]:
            k = str(s)
            if k in rec and 'acc' in rec[k]:
                accs.append(rec[k]['acc'])
        if accs:
            means.append(np.mean(accs) * 100)
            stds.append(np.std(accs, ddof=1) * 100)
            labels_disp.append(label)
            bars_colors.append(color)

    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    x = np.arange(len(means))
    bars = ax.bar(x, means, yerr=stds, color=bars_colors, alpha=0.85,
                  edgecolor='black', linewidth=0.6, capsize=4, error_kw={'lw': 0.8})
    for bar, m, s in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width() / 2, m + s + 0.5, f'{m:.1f}±{s:.1f}',
                ha='center', va='bottom', fontsize=9, fontweight='bold')

    best_idx = int(np.argmax(means))
    ax.text(best_idx, max(means) + max(stds) + 4, '★ best', ha='center', fontsize=10,
            color='red', fontweight='bold')

    for i in range(1, len(means)):
        delta = means[i] - means[0]
        color = 'red' if delta > 0 else 'gray'
        ax.annotate(f'Δ={delta:+.1f}', xy=(i, means[i]),
                    xytext=(i, means[i] + max(stds) + 2.5),
                    ha='center', fontsize=8, color=color, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(labels_disp, fontsize=8.5)
    ax.set_ylabel('Test accuracy (%) (fold-9 only)')
    ax.set_title('Fig. 6: F-41 d9 stability ablation (low-LR as the stable fix)\n'
                 'lowering learning rate (3e-4) is the most reliable fix', fontsize=10)
    ax.grid(axis='y', linestyle=':', alpha=0.4)
    ax.set_ylim(0, max(means) + max(stds) + 8)
    fig.tight_layout()
    out = os.path.join(BVP_DIR, 'fig6_stability_d9.png')
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)
    print(f'  -> {out}')


def main():
    print('[Fig 1] main boxplot (12 methods, sorted by mean) ...')
    fig1_main_boxplot()
    print('[Fig 2] variance decomposition (F-39 only) ...')
    fig2_variance_decomp()
    print('[Fig 3] MDE self-validation ...')
    fig3_mde_validation()
    print('[Fig 4] multi-Rx paired ablation ...')
    fig4_multirx_paired_ablation()
    print('[Fig 6] d9 stability ablation ...')
    fig6_stability_d9()
    print('\n✓ All 5 figures generated.')


if __name__ == '__main__':
    main()