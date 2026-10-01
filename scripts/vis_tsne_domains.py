"""
域特征可视化（t-SNE）：Widar3.0 + MMFi 跨数据集对比
=====================================================
- Widar3.0：用 BVP rx64 多通道特征，时间维平均 → (12000, 9, 60, 33)
- MMFi：用 raw CSI 时间维平均 → (1079, 30, 114)
- 抽样策略：每个 (domain, gesture) 取前 N 个样本，保证可视密度
- 三张图：
  (a) Widar by domain —— 6 个 ENV 是否分簇
  (b) Widar by gesture —— 跨域同 gesture 是否可聚
  (c) MMFi by env —— 4 个 ENV 是否可分（直接解释 B17 地板效应）
  (d) 联合 t-SNE —— 两数据集分布差异
"""
import os
import sys
import time
import json
import numpy as np
import h5py
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = r'F:\python_workspace\wifi识别\wifi-crossenv-lab'
BVP_DIR = os.path.join(ROOT, 'src', 'bvp_test')
WIDAR_H5 = r'F:\python_workspace\wifi识别\data\processed\widar3.0\csi.h5'
MMFI_H5 = os.path.join(BVP_DIR, '_mmfi_full.h5')

OUT_DIR = BVP_DIR  # 直接放图
os.makedirs(OUT_DIR, exist_ok=True)

# 抽样参数
WIDAR_DOMAINS = [1, 3, 5, 6, 7, 9]
WIDAR_GESTURES = [1, 2, 3, 4, 5, 6]  # Widar 共 8 类，取前 6
MMFI_GESTURES = list(range(6))  # MMFi 共 27 类，取前 6
N_PER = 30
SEED = 0


def load_widar():
    """加载 Widar BVP 特征 + 标签"""
    print('[Widar] 加载 BVP rx64 特征...')
    bvp = np.load(os.path.join(BVP_DIR, '_cache_bvp_rx64.npy'), mmap_mode='r')
    print(f'  BVP shape = {bvp.shape}')
    print('[Widar] 加载标签...')
    f = h5py.File(WIDAR_H5, 'r')
    y = f['y'][:]
    domains = f['domains'][:]
    subjects = f['subjects'][:]
    f.close()
    return bvp, y, domains, subjects


def load_mmfi():
    """加载 MMFi raw CSI + 标签"""
    print('[MMFi] 加载 raw CSI...')
    f = h5py.File(MMFI_H5, 'r')
    X = f['X'][:]
    y = f['y'][:]
    domains = f['domains'][:]
    subjects = f['subjects'][:]
    f.close()
    print(f'  X shape = {X.shape}, y unique = {len(set(y.tolist()))}, domains unique = {sorted(set(domains.tolist()))}')
    return X, y, domains, subjects


def sample_widar(bvp, y, domains, subjects):
    """每个 (domain, gesture) 取前 N 个"""
    samples, d_list, g_list, s_list = [], [], [], []
    for d in WIDAR_DOMAINS:
        for g in WIDAR_GESTURES:
            idx = np.where((domains == d) & (y == g))[0][:N_PER]
            if len(idx) == 0:
                continue
            samples.append(idx)
            d_list.extend([d] * len(idx))
            g_list.extend([g] * len(idx))
            s_list.extend(subjects[idx].tolist())
    samples = np.concatenate(samples)
    return samples, np.array(d_list), np.array(g_list), np.array(s_list)


def sample_mmfi(X, y, domains, subjects):
    """每个 (env, gesture) 取前 N 个"""
    samples, e_list, g_list, s_list = [], [], [], []
    for e in [0, 1, 2, 3]:
        for g in MMFI_GESTURES:
            idx = np.where((domains == e) & (y == g))[0][:N_PER]
            if len(idx) == 0:
                continue
            samples.append(idx)
            e_list.extend([e] * len(idx))
            g_list.extend([g] * len(idx))
            s_list.extend(subjects[idx].tolist())
    samples = np.concatenate(samples)
    return samples, np.array(e_list), np.array(g_list), np.array(s_list)


def widar_to_features(bvp, idx):
    """Widar BVP (12000, 9, 60, 33, 25) → (N, 9, 60, 33) → (N, 17820)"""
    X = bvp[idx].mean(axis=-1)  # 时间维平均
    X = X.reshape(len(idx), -1)
    return np.nan_to_num(X).astype(np.float32)


def mmfi_to_features(X, idx):
    """MMFi raw (1079, 30, 512, 114) → (N, 30, 114) → (N, 3420)
    含未清洗 inf/nan → log1p + clip + nan_to_num 三重保险
    """
    amp = np.abs(X[idx]).mean(axis=-2)  # 时间维平均 → (N, 30, 114)
    amp = np.log1p(amp.astype(np.float64))  # log 缩放
    amp = np.clip(amp, -10, 20)  # 截断极端值
    X = amp.reshape(len(idx), -1)
    X = np.nan_to_num(X, nan=0.0, posinf=10.0, neginf=-10.0)
    return X.astype(np.float32)


def run_tsne(X, perplexity=30, n_iter=1000):
    """PCA 50 → t-SNE 2"""
    t0 = time.time()
    pca = PCA(n_components=min(50, X.shape[1] - 1, X.shape[0] - 1), random_state=SEED)
    X_pca = pca.fit_transform(X)
    print(f'  PCA: {X.shape[1]} → 50, explained_var_ratio sum = {pca.explained_variance_ratio_.sum():.3f}, 用时 {time.time()-t0:.1f}s')
    t0 = time.time()
    tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter, random_state=SEED, init='pca')
    X_emb = tsne.fit_transform(X_pca)
    print(f'  t-SNE: 用时 {time.time()-t0:.1f}s')
    return X_emb


def make_legend(ax, colors, names, title):
    handles = [Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[i], markersize=8, label=names[i])
               for i in range(len(names))]
    ax.legend(handles=handles, title=title, loc='best', fontsize=8, title_fontsize=9, framealpha=0.9)


def plot_widar(X_emb, d_arr, g_arr, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    # 左图：按 domain 染色
    domain_colors = plt.cm.tab10(np.linspace(0, 1, len(WIDAR_DOMAINS)))
    for i, d in enumerate(WIDAR_DOMAINS):
        m = d_arr == d
        axes[0].scatter(X_emb[m, 0], X_emb[m, 1], c=[domain_colors[i]], alpha=0.55, s=18, edgecolors='none')
    axes[0].set_title(f'Widar3.0 · colored by domain (6 ENV)', fontsize=12)
    axes[0].set_xlabel('t-SNE 1'); axes[0].set_ylabel('t-SNE 2')
    make_legend(axes[0], domain_colors, [f'ENV{d}' for d in WIDAR_DOMAINS], 'Domain')
    # 右图：按 gesture 染色
    g_colors = plt.cm.Set2(np.linspace(0, 1, len(WIDAR_GESTURES)))
    for i, g in enumerate(WIDAR_GESTURES):
        m = g_arr == g
        axes[1].scatter(X_emb[m, 0], X_emb[m, 1], c=[g_colors[i]], alpha=0.55, s=18, edgecolors='none')
    axes[1].set_title(f'Widar3.0 · colored by gesture (6 classes)', fontsize=12)
    axes[1].set_xlabel('t-SNE 1'); axes[1].set_ylabel('t-SNE 2')
    make_legend(axes[1], g_colors, [f'G{g}' for g in WIDAR_GESTURES], 'Gesture')
    fig.suptitle('Widar3.0 · Domain gap visualization (multirx BVP features)', fontsize=13, fontweight='bold')
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f'  保存 {out_path}')


def plot_mmfi(X_emb, e_arr, g_arr, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    e_colors = plt.cm.tab10(np.linspace(0, 1, 4))
    for i, e in enumerate([0, 1, 2, 3]):
        m = e_arr == e
        axes[0].scatter(X_emb[m, 0], X_emb[m, 1], c=[e_colors[i]], alpha=0.65, s=22, edgecolors='none')
    axes[0].set_title('MMFi · colored by env (4 ENV)', fontsize=12)
    axes[0].set_xlabel('t-SNE 1'); axes[0].set_ylabel('t-SNE 2')
    make_legend(axes[0], e_colors, ['E1', 'E2', 'E3', 'E4'], 'Env')
    g_colors = plt.cm.Set2(np.linspace(0, 1, len(MMFI_GESTURES)))
    for i, g in enumerate(MMFI_GESTURES):
        m = g_arr == g
        axes[1].scatter(X_emb[m, 0], X_emb[m, 1], c=[g_colors[i]], alpha=0.65, s=22, edgecolors='none')
    axes[1].set_title('MMFi · colored by gesture (6 classes of 27)', fontsize=12)
    axes[1].set_xlabel('t-SNE 1'); axes[1].set_ylabel('t-SNE 2')
    make_legend(axes[1], g_colors, [f'G{g}' for g in MMFI_GESTURES], 'Gesture')
    fig.suptitle('MMFi · Domain gap visualization (raw CSI amplitude)', fontsize=13, fontweight='bold')
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f'  保存 {out_path}')


def plot_joint(Xw_emb, Xm_emb, out_path):
    fig, ax = plt.subplots(1, 1, figsize=(11, 8))
    ax.scatter(Xw_emb[:, 0], Xw_emb[:, 1], c='#1f77b4', alpha=0.55, s=14, label=f'Widar3.0 (N={len(Xw_emb)})', edgecolors='none')
    ax.scatter(Xm_emb[:, 0], Xm_emb[:, 1], c='#d62728', alpha=0.7, s=18, label=f'MMFi (N={len(Xm_emb)})', edgecolors='none', marker='^')
    ax.set_title('Cross-dataset feature distribution (joint t-SNE)\nWidar3.0 ↔ MMFi', fontsize=12, fontweight='bold')
    ax.set_xlabel('t-SNE 1'); ax.set_ylabel('t-SNE 2')
    ax.legend(loc='best', fontsize=10, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f'  保存 {out_path}')


def main():
    # ============ Widar ============
    bvp, y_w, dom_w, sub_w = load_widar()
    idx_w, d_arr, g_arr, _ = sample_widar(bvp, y_w, dom_w, sub_w)
    Xw = widar_to_features(bvp, idx_w)
    print(f'[Widar] 抽样 {len(idx_w)} 个样本 → features shape {Xw.shape}')
    Xw_emb = run_tsne(Xw, perplexity=30, n_iter=1500)
    plot_widar(Xw_emb, d_arr, g_arr, os.path.join(OUT_DIR, 'fig_tsne_widar.png'))

    # ============ MMFi ============
    X_mmfi, y_m, dom_m, sub_m = load_mmfi()
    idx_m, e_arr, g_arr_m, _ = sample_mmfi(X_mmfi, y_m, dom_m, sub_m)
    Xm = mmfi_to_features(X_mmfi, idx_m)
    print(f'[MMFi] 抽样 {len(idx_m)} 个样本 → features shape {Xm.shape}')
    Xm_emb = run_tsne(Xm, perplexity=30, n_iter=1000)
    plot_mmfi(Xm_emb, e_arr, g_arr_m, os.path.join(OUT_DIR, 'fig_tsne_mmfi.png'))

    # ============ 联合 ============
    # 各自 PCA 50 后再拼接（维度不同时的跨数据集 t-SNE 标准做法）
    print(f'[Joint] 各自 PCA 50 后拼接 ...')
    pca_w = PCA(n_components=50, random_state=SEED)
    Xw_pca = pca_w.fit_transform(Xw)
    pca_m = PCA(n_components=50, random_state=SEED)
    Xm_pca = pca_m.fit_transform(Xm)
    Xj = np.vstack([Xw_pca, Xm_pca])
    print(f'  联合 N={len(Xj)}, 维度 {Xj.shape[1]}')
    t0 = time.time()
    tsne_j = TSNE(n_components=2, perplexity=30, n_iter=1500, random_state=SEED, init='pca')
    Xj_emb = tsne_j.fit_transform(Xj)
    print(f'  联合 t-SNE 用时 {time.time()-t0:.1f}s')
    nw = len(Xw_emb)
    plot_joint(Xj_emb[:nw], Xj_emb[nw:], os.path.join(OUT_DIR, 'fig_tsne_joint.png'))

    # ============ 保存数据 ============
    out_json = os.path.join(OUT_DIR, 'f44_tsne_embeddings.json')
    data = {
        '_summary': {
            'widar': {'n': len(idx_w), 'feats': Xw.shape[1], 'perplexity': 30},
            'mmfi': {'n': len(idx_m), 'feats': Xm.shape[1], 'perplexity': 30},
            'joint': {'n': len(Xj), 'feats': Xj.shape[1], 'perplexity': 30},
        },
        'widar': {
            'idx': idx_w.tolist(),
            'domain': d_arr.tolist(),
            'gesture': g_arr.tolist(),
            'emb_x': Xw_emb[:, 0].tolist(),
            'emb_y': Xw_emb[:, 1].tolist(),
        },
        'mmfi': {
            'idx': idx_m.tolist(),
            'env': e_arr.tolist(),
            'gesture': g_arr_m.tolist(),
            'emb_x': Xm_emb[:, 0].tolist(),
            'emb_y': Xm_emb[:, 1].tolist(),
        },
        'joint': {
            'emb_x': Xj_emb[:, 0].tolist(),
            'emb_y': Xj_emb[:, 1].tolist(),
            'dataset': ['Widar'] * nw + ['MMFi'] * (len(Xj) - nw),
        },
    }
    with open(out_json, 'w', encoding='utf-8') as fp:
        json.dump(data, fp, ensure_ascii=False)
    print(f'  保存 {out_json}')


if __name__ == '__main__':
    main()