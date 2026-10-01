"""
DFRF Phase 1: 数据档案生成脚本
==============================

为 MMFi / Widar3.0 / CSIDA 三个数据集生成统一格式的数据认知报告（dossier）。
**纯 NumPy + matplotlib，无任何模型**，只回答"数据本身长什么样"。

输出：
  src/bvp_test/dossier/data_dossier_<dataset>.md
  src/bvp_test/dossier/figs/<dataset>_{inventory,signal,domain_pca,class_pca}.png

计算内容：
  1. 样本清单（per-class / per-domain / per-subject 计数 + cross-tab）
  2. 信号特征（NaN/Inf 比例、幅度分布）
  3. 域结构（per-domain 均值特征 PCA 2D 可视化）
  4. 类别可分性（PCA 2D 着色 by class）

DFRF 规则（写入 MEMORY.md）：
  - 进入 Phase 2 / Phase 3 前**必须**先有对应数据集的 dossier.md
  - 否则视为"无数据认知的盲训"，结论不可信

用法：
  python bench_dossier.py mmfi      # 单数据集
  python bench_dossier.py widar3    # 单数据集
  python bench_dossier.py csida     # 单数据集
  python bench_dossier.py all       # 三个一起跑
  python bench_dossier.py all 500   # 自定义 sample_n=500
"""

from __future__ import annotations
import sys, os, time
from pathlib import Path
from collections import Counter
import numpy as np
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from wfcslab.data.mmfi import make_mmfi_dataset
from wfcslab.data.widar import make_widar_dataset
from wfcslab.data.csida import make_csida_dataset


REPO_ROOT = Path(__file__).parent.parent
# Data is at the project root (one level above REPO_ROOT=src/..), not inside wifi-crossenv-lab
PROJECT_ROOT = REPO_ROOT.parent
BVP_TEST = REPO_ROOT / 'src' / 'bvp_test'
OUT = BVP_TEST / 'dossier'
OUT.mkdir(exist_ok=True)
FIGS = OUT / 'figs'
FIGS.mkdir(exist_ok=True)
DATA = PROJECT_ROOT / 'data' / 'processed'

PATHS = {
    'mmfi':      BVP_TEST / '_mmfi_full.h5',
    'widar3':    DATA / 'widar3.0' / 'csi_50k.h5',         # BROKEN — ges 语义错配 (rename -> _broken_csi_50k.h5)
    'widar_v2':  DATA / 'widar3.0' / 'csi_50k_v2.h5',     # NEW — canonical 5-gesture (13 zip, ges=1..5, 158.5k pool → 50k subset)
    'csida':     DATA / 'csida' / 'csi_env.h5',
}
LOADER_NAME = {
    'mmfi': 'mmfi.make_mmfi_dataset',
    'widar3': 'wfcslab.data.widar.make_widar_dataset (BROKEN)',
    'widar_v2': 'wfcslab.data.widar.make_widar_dataset (csi_50k_v2.h5, canonical 5-gesture set per raw_dir 实测)',
    'csida': 'wfcslab.data.csida.make_csida_dataset',
}


def log(msg):
    s = f'[{time.strftime("%H:%M:%S")}] {msg}'
    print(s, flush=True)


def save_fig(fig, name):
    p = FIGS / name
    fig.tight_layout()
    fig.savefig(p, dpi=80, bbox_inches='tight')
    plt.close(fig)
    return p


def load(name):
    """Load dataset as CSIDataset. Widar uses lazy h5py to avoid 50k OOM."""
    if name == 'mmfi':
        ds = make_mmfi_dataset(PATHS['mmfi'])
    elif name in ('widar3', 'widar_v2'):
        ds = make_widar_dataset(PATHS[name], load_X=False)  # lazy h5py
    elif name == 'csida':
        ds = make_csida_dataset(PATHS['csida'], domain_axis='env')
    else:
        raise ValueError(name)
    return ds


def _read_one(X, i):
    """Read sample i, returning np.ndarray even when X is lazy h5py Dataset."""
    if isinstance(X, np.ndarray):
        return X[i]
    return np.asarray(X[i])  # h5py


def inventory(ds):
    """Per-class / per-domain / per-subject counts + cross-tab (class × domain)."""
    y = ds.y.astype(int)
    doms = ds.domains.astype(int)
    subs = ds.subjects.astype(int)
    n = len(y)
    classes = sorted(set(y.tolist()))
    domains = sorted(set(doms.tolist()))
    subjects = sorted(set(subs.tolist()))

    per_class = {int(c): int((y == c).sum()) for c in classes}
    per_domain = {int(d): int((doms == d).sum()) for d in domains}
    per_subject = {int(s): int((subs == s).sum()) for s in subjects}

    cross = np.zeros((len(classes), len(domains)), dtype=int)
    for i, c in enumerate(classes):
        for j, d in enumerate(domains):
            cross[i, j] = int(((y == c) & (doms == d)).sum())

    return {
        'n': n,
        'n_classes': len(classes),
        'n_domains': len(domains),
        'n_subjects': len(subjects),
        'per_class': per_class,
        'per_domain': per_domain,
        'per_subject': per_subject,
        'class_x_domain': cross,
        'classes': classes,
        'domains': domains,
        'subjects': subjects,
    }


def class_balance_stats(inv):
    cnts = list(inv['per_class'].values())
    if not cnts:
        return None
    return {
        'min': min(cnts), 'max': max(cnts),
        'mean': sum(cnts) / len(cnts),
        'ratio': max(cnts) / max(min(cnts), 1),
    }


def plot_inventory(inv, name):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    cs = inv['classes']
    cnts = [inv['per_class'][c] for c in cs]
    axes[0, 0].bar(range(len(cs)), cnts, color='steelblue')
    axes[0, 0].set_xlabel('class'); axes[0, 0].set_ylabel('count')
    axes[0, 0].set_title(f'Per-class sample count (n_classes={inv["n_classes"]})')

    ds = inv['domains']
    cnts = [inv['per_domain'][d] for d in ds]
    axes[0, 1].bar(range(len(ds)), cnts, color='coral')
    axes[0, 1].set_xlabel('domain'); axes[0, 1].set_ylabel('count')
    axes[0, 1].set_title(f'Per-domain sample count (n_domains={inv["n_domains"]})')

    ss = inv['subjects']
    cnts = [inv['per_subject'][s] for s in ss]
    axes[1, 0].bar(range(len(ss)), cnts, color='seagreen')
    axes[1, 0].set_xlabel('subject'); axes[1, 0].set_ylabel('count')
    axes[1, 0].set_title(f'Per-subject sample count (n_subjects={inv["n_subjects"]})')

    cross = inv['class_x_domain']
    im = axes[1, 1].imshow(cross, cmap='Blues', aspect='auto')
    axes[1, 1].set_xlabel('domain'); axes[1, 1].set_ylabel('class')
    axes[1, 1].set_title('Class × Domain (counts)')
    plt.colorbar(im, ax=axes[1, 1], fraction=0.046)

    return save_fig(fig, f'{name}_inventory.png')


def signal_stats(X, sample_idx, y):
    """Per-sample signal stats on a sample subset."""
    per_sample = []
    nan_count = inf_count = total = 0
    for i in sample_idx:
        x = _read_one(X, i)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        amp = np.abs(x)
        per_sample.append({
            'idx': int(i),
            'mean': float(amp.mean()),
            'std': float(amp.std()),
            'min': float(amp.min()),
            'max': float(amp.max()),
            'class': int(y[i]),
        })
        # NaN/Inf check on raw (before nan_to_num)
        x_raw = _read_one(X, i)
        nan_count += int(np.isnan(x_raw).sum())
        inf_count += int(np.isinf(x_raw).sum())
        total += int(x_raw.size)

    arr = np.array([[s['mean'], s['std'], s['min'], s['max']] for s in per_sample])
    return {
        'n_sampled': len(per_sample),
        'mean_amp': {'mean': float(arr[:, 0].mean()), 'std': float(arr[:, 0].std())},
        'std_amp':  {'mean': float(arr[:, 1].mean()), 'std': float(arr[:, 1].std())},
        'nan_frac': float(nan_count / max(total, 1)),
        'inf_frac': float(inf_count / max(total, 1)),
        'per_sample': per_sample,
    }


def plot_signal_stats(stats, name):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    means = [s['mean'] for s in stats['per_sample']]
    stds  = [s['std']  for s in stats['per_sample']]

    axes[0].hist(means, bins=30, color='steelblue', edgecolor='black')
    axes[0].set_xlabel('mean amplitude')
    axes[0].set_ylabel('count')
    axes[0].set_title(f'Per-sample mean amplitude (μ={stats["mean_amp"]["mean"]:.3f})')

    axes[1].hist(stds, bins=30, color='coral', edgecolor='black')
    axes[1].set_xlabel('std amplitude')
    axes[1].set_ylabel('count')
    axes[1].set_title(f'Per-sample std amplitude (μ={stats["std_amp"]["mean"]:.3f})')

    return save_fig(fig, f'{name}_signal.png')


def make_features(X, sample_idx):
    """Per-sample feature: per-antenna mean amplitude. (N, A) float32."""
    feats = []
    for i in sample_idx:
        x = _read_one(X, i)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        amp = np.abs(x).mean(axis=(1, 2))  # (A,)
        feats.append(amp.astype(np.float32))
    return np.stack(feats)


def pca2(feats):
    """2D PCA via SVD."""
    fc = feats - feats.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(fc, full_matrices=False)
    return fc @ Vt[:2].T, S


def domain_pca(feats, domains, sample_idx, name):
    emb, _ = pca2(feats)
    fig, ax = plt.subplots(figsize=(7, 5))
    doms_arr = np.array([int(domains[i]) for i in sample_idx])
    scatter = ax.scatter(emb[:, 0], emb[:, 1], c=doms_arr, cmap='tab10', s=20, alpha=0.7)
    ax.set_xlabel('PC1'); ax.set_ylabel('PC2')
    ax.set_title(f'{name}: PCA of per-antenna mean amplitude (color=domain)')
    plt.colorbar(scatter, ax=ax, label='domain')
    return save_fig(fig, f'{name}_domain_pca.png'), emb


def class_pca(feats, y, sample_idx, name):
    emb, _ = pca2(feats)
    fig, ax = plt.subplots(figsize=(7, 5))
    y_arr = np.array([int(y[i]) for i in sample_idx])
    n_class = len(set(y_arr.tolist()))
    cmap = 'tab20' if n_class <= 20 else 'tab20' if n_class <= 40 else 'hsv'
    scatter = ax.scatter(emb[:, 0], emb[:, 1], c=y_arr, cmap=cmap, s=20, alpha=0.7)
    ax.set_xlabel('PC1'); ax.set_ylabel('PC2')
    ax.set_title(f'{name}: PCA of per-antenna mean amplitude (color=class, n={n_class})')
    plt.colorbar(scatter, ax=ax, label='class')
    return save_fig(fig, f'{name}_class_pca.png'), emb


def write_markdown(name, inv, sig_stats, cb, ds_total, ds_source_size):
    cnts = [(c, inv['per_class'][c]) for c in inv['classes']]
    cnts_sorted = sorted(cnts, key=lambda x: -x[1])

    subset_pct = (ds_total / ds_source_size * 100) if ds_source_size else 100.0

    lines = []
    lines.append(f'# Data Dossier: {name.upper()}')
    lines.append('')
    lines.append(f'**Generated**: {time.strftime("%Y-%m-%d %H:%M:%S")}  ')
    lines.append(f'**Source h5**: `{PATHS[name]}`  ')
    lines.append(f'**Loader**: `{LOADER_NAME[name]}`  ')
    lines.append(f'**Sample size**: {inv["n"]} (cached subset of source dataset)')
    if ds_source_size and ds_source_size != ds_total:
        lines.append(f'**Subset ratio**: {ds_total} / {ds_source_size} = {subset_pct:.2f}%')
    if name == 'widar_v2':
        lines.append('')
        lines.append('> **DFRF note**: This is the rebuilt cache using Widar3.0 **canonical 5-gesture set**')
        lines.append('> (Push&Pull, Sweep, Clap, Slide, Draw-O H) restricted to 13 zips (158,500 source samples).')
        lines.append('> raw_dir 实测 **不存在 9 类同质集**（ges=9 在所有 zip 中缺失；ges=6..8 仅 2 个 zip）。')
        lines.append('> See `WIDAR_TRUTH.md` and `WIDAR_LABELING_CRISIS.md`.')
    elif name == 'widar3':
        lines.append('')
        lines.append('> **⚠️ BROKEN cache**: ges is per-zip-index, not global class label. See `WIDAR_TRUTH.md`.')
        lines.append('> Use `widar_v2` for valid experiments.')
    lines.append('')

    # 1. Sample inventory
    lines.append('## 1. Sample inventory')
    lines.append('')
    lines.append(f'- n = **{inv["n"]}** samples')
    lines.append(f'- n_classes = **{inv["n_classes"]}** (range [{min(inv["classes"])}, {max(inv["classes"])}])')
    lines.append(f'- n_domains = **{inv["n_domains"]}** (range [{min(inv["domains"])}, {max(inv["domains"])}])')
    lines.append(f'- n_subjects = **{inv["n_subjects"]}** (range [{min(inv["subjects"])}, {max(inv["subjects"])}])')
    lines.append(f'- **Class balance**: min={cb["min"]}, max={cb["max"]}, mean={cb["mean"]:.1f}, **ratio={cb["ratio"]:.2f}** (max/min)')
    lines.append('')
    lines.append('### Top 5 / Bottom 5 classes by count')
    lines.append('')
    lines.append('| rank | class | count |')
    lines.append('|---|---|---|')
    for i, (c, n) in enumerate(cnts_sorted[:5]):
        lines.append(f'| top{i+1} | {c} | {n} |')
    if len(cnts_sorted) > 10:
        lines.append('| ... | ... | ... |')
        for i, (c, n) in enumerate(cnts_sorted[-5:]):
            lines.append(f'| bot{i+1} | {c} | {n} |')
    lines.append('')

    # 2. Signal characteristics
    lines.append('## 2. Signal characteristics')
    lines.append('')
    lines.append(f'Sampled {sig_stats["n_sampled"]} random samples.')
    lines.append('')
    lines.append(f'- **NaN fraction**: {sig_stats["nan_frac"]:.6f}')
    lines.append(f'- **Inf fraction**: {sig_stats["inf_frac"]:.6f}')
    lines.append(f'- **Mean amplitude (per sample)**: μ = {sig_stats["mean_amp"]["mean"]:.3f}, σ = {sig_stats["mean_amp"]["std"]:.3f}')
    lines.append(f'- **Std amplitude (per sample)**:  μ = {sig_stats["std_amp"]["mean"]:.3f}, σ = {sig_stats["std_amp"]["std"]:.3f}')
    lines.append('')
    lines.append(f'![signal](figs/{name}_signal.png)')
    lines.append('')

    # 3. Inventory charts
    lines.append('## 3. Sample inventory charts')
    lines.append('')
    lines.append(f'![inventory](figs/{name}_inventory.png)')
    lines.append('')

    # 4. Domain PCA
    lines.append('## 4. Domain structure (PCA, color=domain)')
    lines.append('')
    lines.append(f'![domain_pca](figs/{name}_domain_pca.png)')
    lines.append('')

    # 5. Class PCA
    lines.append('## 5. Class structure (PCA, color=class)')
    lines.append('')
    lines.append(f'![class_pca](figs/{name}_class_pca.png)')
    lines.append('')
    lines.append('Observations:')
    lines.append('- If points cluster by domain in (4) but spread in (5) → env signature dominates, class signal is weak')
    lines.append('- If points cluster by class in (5) → class signal is recoverable from mean amplitude features')
    lines.append('- If neither clusters clearly → need richer features (BVP / phase) or finer-grained task')
    lines.append('')

    # 6. Cross-tab
    lines.append('## 6. Cross-tab: class × domain')
    lines.append('')
    header = '| class | ' + ' | '.join(f'd{d}' for d in inv['domains']) + ' | total |'
    lines.append(header)
    lines.append('|' + '---|' * (len(inv['domains']) + 2))
    cross = inv['class_x_domain']
    for i, c in enumerate(inv['classes']):
        row = cross[i]
        total = row.sum()
        lines.append(f'| {c} | ' + ' | '.join(str(v) for v in row) + f' | **{total}** |')
    lines.append('')

    # 7. Decision hooks
    lines.append('## 7. Decision hooks for Phase 2 / Phase 3')
    lines.append('')
    lines.append(f'- `class_balance_ratio = {cb["ratio"]:.2f}`  → if > 5: per-class sampling / class-weighting needed')
    lines.append(f'- `n_classes = {inv["n_classes"]}` and `avg_per_class = {cb["mean"]:.0f}`  → if n_classes > 10 AND avg < 50: consider grain sweep in Phase 2')
    lines.append(f'- `n_domains = {inv["n_domains"]}`  → if < 4: LODO has low statistical power; minimum 5 seeds')
    lines.append(f'- `NaN fraction = {sig_stats["nan_frac"]:.6f}`  → if > 0.001: pre-cleaning before Phase 3')
    lines.append('')

    lines.append('---')
    lines.append(f'Generated by `src/bench_dossier.py` — DFRF Phase 1 (NO MODEL).')

    p = OUT / f'data_dossier_{name}.md'
    with open(p, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    return p


def main(name, sample_n=300, source_size=None):
    log(f'=== Dossier: {name} ===')
    log(f'Loading {PATHS[name]}')
    ds = load(name)
    X = ds.X
    log(f'  X type: {type(X).__name__}, shape: {X.shape}, dtype: {X.dtype}')
    log(f'  y: {ds.y.shape}, domains: {ds.domains.shape}, subjects: {ds.subjects.shape}')

    log('Computing inventory...')
    inv = inventory(ds)
    cb = class_balance_stats(inv)
    log(f'  classes={inv["n_classes"]}, domains={inv["n_domains"]}, subjects={inv["n_subjects"]}')
    log(f'  class balance: min={cb["min"]}, max={cb["max"]}, ratio={cb["ratio"]:.2f}')

    log('Plotting inventory...')
    plot_inventory(inv, name)

    # sample subset
    n_total = X.shape[0]
    rng = np.random.RandomState(0)
    sample_idx = rng.choice(n_total, min(sample_n, n_total), replace=False)
    log(f'Sampled {len(sample_idx)} / {n_total} for signal/PCA stats')

    log('Computing signal stats...')
    sig = signal_stats(X, sample_idx, ds.y)
    log(f'  NaN frac={sig["nan_frac"]:.6f}, Inf frac={sig["inf_frac"]:.6f}')
    log(f'  mean_amp μ={sig["mean_amp"]["mean"]:.3f}, std_amp μ={sig["std_amp"]["mean"]:.3f}')

    log('Plotting signal stats...')
    plot_signal_stats(sig, name)

    log('Building per-sample features (mean amplitude per antenna)...')
    feats = make_features(X, sample_idx)
    log(f'  feats shape: {feats.shape}')

    log('Computing domain PCA...')
    _, emb_dom = domain_pca(feats, ds.domains, sample_idx, name)

    log('Computing class PCA...')
    _, emb_cls = class_pca(feats, ds.y, sample_idx, name)

    log('Writing markdown report...')
    report_path = write_markdown(name, inv, sig, cb, ds_total=n_total, ds_source_size=source_size)
    log(f'  Report: {report_path}')

    log(f'=== Done: {name} ===\n')
    return report_path


# Reference source dataset sizes (for subset_ratio disclosure)
SOURCE_SIZES = {
    'mmfi':      1134,    # 1134 wifi-csi dirs in MMFi (some NaN-filtered)
    'widar3':    163650,  # full Widar3.0 (after dedup) — BROKEN (ges 语义错配)
    'widar_v2':  158500,  # canonical 5-gesture set per raw_dir 实测 (13 zips, ges=1..5)
    'csida':     2844,    # full CSIDA (no subset)
}


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: bench_dossier.py <mmfi|widar3|widar_v2|csida|all> [sample_n=300]')
        sys.exit(1)
    target = sys.argv[1]
    sample_n = int(sys.argv[2]) if len(sys.argv) > 2 else 300
    if target == 'all':
        for name in ['mmfi', 'widar_v2', 'csida']:  # widar3 intentionally excluded (broken)
            try:
                main(name, sample_n=sample_n, source_size=SOURCE_SIZES.get(name))
            except Exception as e:
                print(f'!! Error in {name}: {e}')
                import traceback
                traceback.print_exc()
    else:
        main(target, sample_n=sample_n, source_size=SOURCE_SIZES.get(target))
