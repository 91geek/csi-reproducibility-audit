"""
F-49: MMFi root-cause diagnostic experiment.

Question: MMFi CBAM LODO test_acc ≈ random (0.037-0.054). Is this:
  A) "Cannot learn": train_acc also ≈ random → task itself is unlearnable
     (bad labels, task mismatch, or signal absent) → DA/DG/SSL all moot
  B) "Overfit to env": train_acc high, test_acc low → env features leak
     into the learned classifier → DA/DG/SSL appropriate
  C) "Mixed": partial learning but severe env-specific overfitting

Protocol: 1 env (E2, chosen because C10v2 in F-40 showed highest variance) ×
          1 seed × 2 backbones (LeNetCSI_Attn vs CBAM+ResNet18) × 1 variant (multirx10).

Outputs (saved to src/bvp_test/f49_diag/):
  - train_curve.png     : per-epoch train_acc vs val_acc vs test_acc
  - confusion_matrix.png: 27x27 normalized confusion matrix (test)
  - per_class_f1.png    : per-class F1 bar chart, sorted descending
  - env_tsne.png        : t-SNE of train/test features colored by env id

Decision rule printed at end:
  if max(train_acc) < 0.10  -> A (cannot learn)
  elif max(train_acc) - test_acc > 0.20 -> B (env leakage)
  else -> C (mixed)

Wall-clock budget: 30 min on RTX 2080 Ti.
"""

from __future__ import annotations
import json, time, sys, os, argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, Dataset
import h5py

sys.path.insert(0, str(Path(__file__).parent))
from wfcslab.data.mmfi import make_mmfi_dataset
from wfcslab.data.base import lodo_splits_with_val
from wfcslab.models.backbones import CBAMResNet18, LeNetCSI_Attn
from wfcslab.signal.doppler_gpu import compute_bvp_gpu
from wfcslab.engine.trainer import set_perf_flags
import matplotlib.pyplot as plt
# sklearn unavailable in this env → numpy-only fallbacks used below

set_perf_flags()
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

OUT = Path(__file__).parent / 'bvp_test' / 'f49_diag'
OUT.mkdir(parents=True, exist_ok=True)
LOG = OUT / 'log.txt'

# Same hyperparams as F-40 / F-48 for fair comparability
EPOCHS = 30
BATCH = 64
LR = 1e-3
WD = 1e-4
PATIENCE = 8
TEST_ENV = 'E2'
SEED = 0


def log(msg):
    s = f'[{time.strftime("%H:%M:%S")}] {msg}'
    print(s, flush=True)
    with open(LOG, 'a', encoding='utf-8') as f:
        f.write(s + '\n')


def build_model(name, in_ant):
    # BVP shape (1080, A, 114, 33, 29) -> (A, in_subcarriers=114, in_freq=33, in_time=29)
    if name == 'lenet_attn':
        m = LeNetCSI_Attn(
            in_antennas=in_ant, in_subcarriers=114, in_freq=33, in_time=29,
            out_dim=27, width=32, hidden=128, n_heads=4, dropout=0.4)
    elif name == 'cbam':
        m = CBAMResNet18(
            in_antennas=in_ant, in_subcarriers=114, in_freq=33, in_time=29,
            out_dim=27, width=64, hidden=256, dropout=0.4)
    else:
        raise ValueError(name)
    return m.to(DEVICE)


def extract_features(model, bvp_h5, indices, batch=64):
    """Return penultimate features for t-SNE using forward hook on self.head.

    Both LeNetCSI_Attn and CBAMResNet18 follow: backbone(x) -> z -> self.head(z).
    We hook self.head[0] to capture z (input to head).

    `bvp_h5` is a h5py dataset object (bvp_ds from BVP_H5).
    `indices` is an array of sample indices into bvp_h5.
    """
    model.eval()
    captured = {}

    def hook(module, inp, out):
        captured['z'] = inp[0].detach()

    handle = model.head.register_forward_hook(hook)
    feats = []
    try:
        with torch.no_grad():
            for i in range(0, len(indices), batch):
                batch_idx = indices[i:i+batch]
                xb = torch.from_numpy(np.asarray(bvp_h5[batch_idx], dtype=np.float32)).to(DEVICE)
                _ = model(xb)
                z = captured['z'].view(captured['z'].size(0), -1).cpu().numpy()
                feats.append(z)
    finally:
        handle.remove()
    return np.concatenate(feats, axis=0)


def train_one(name, in_ant, bvp_h5_path, y, train_idx, val_idx, test_idx):
    torch.manual_seed(SEED); np.random.seed(SEED)
    model = build_model(name, in_ant)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    scaler = torch.amp.GradScaler('cuda')

    train_ds = MMFiBvpLazyDataset(bvp_h5_path, train_idx)
    val_ds   = MMFiBvpLazyDataset(bvp_h5_path, val_idx)
    test_ds  = MMFiBvpLazyDataset(bvp_h5_path, test_idx)
    tr_ld = DataLoader(train_ds, batch_size=BATCH, shuffle=True, num_workers=0)
    va_ld = DataLoader(val_ds, batch_size=BATCH, shuffle=False, num_workers=0)
    te_ld = DataLoader(test_ds, batch_size=BATCH, shuffle=False, num_workers=0)

    history = {'train_acc': [], 'val_acc': [], 'test_acc': [], 'epoch': []}
    best_val, best_state, bad = -1.0, None, 0
    t0 = time.time()
    for ep in range(EPOCHS):
        model.train()
        correct, total = 0, 0
        for xb, yb in tr_ld:
            xb, yb = xb.to(DEVICE, non_blocking=True), yb.to(DEVICE, non_blocking=True)
            opt.zero_grad()
            with torch.amp.autocast('cuda'):
                out = model(xb)
                loss = F.cross_entropy(out, yb)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update()
            sched.step()
            pred = out.argmax(1)
            correct += (pred == yb).sum().item(); total += yb.size(0)
        train_acc = correct / total

        def _eval(ld):
            model.eval(); c, t = 0, 0
            with torch.no_grad():
                for xb, yb in ld:
                    xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                    with torch.amp.autocast('cuda'):
                        out = model(xb)
                    c += (out.argmax(1) == yb).sum().item(); t += yb.size(0)
            return c / t
        val_acc = _eval(va_ld)
        test_acc = _eval(te_ld)

        history['epoch'].append(ep)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['test_acc'].append(test_acc)
        log(f'  ep{ep:02d} train={train_acc:.3f} val={val_acc:.3f} test={test_acc:.3f}')

        if val_acc > best_val + 1e-4:
            best_val, best_state, bad = val_acc, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= PATIENCE:
                log(f'  early stop @ ep{ep} (best val={best_val:.3f})')
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    # Final test eval with best model
    model.eval(); preds, gts = [], []
    with torch.no_grad():
        for xb, yb in te_ld:
            xb = xb.to(DEVICE)
            with torch.amp.autocast('cuda'):
                out = model(xb)
            preds.append(out.argmax(1).cpu().numpy()); gts.append(yb.numpy())
    preds = np.concatenate(preds); gts = np.concatenate(gts)
    final_test_acc = (preds == gts).mean()
    train_t = time.time() - t0
    log(f'  FINAL test={final_test_acc:.3f} best_val={best_val:.3f} ({train_t:.0f}s)')

    # Extract features for t-SNE — read from h5 lazy
    bvp_ds = h5py.File(bvp_h5_path, 'r')['bvp']
    feats = extract_features(model, bvp_ds, test_idx)
    bvp_ds.file.close()
    return model, history, preds, gts, feats, final_test_acc


def main():
    if LOG.exists(): LOG.unlink()
    cache_name = getattr(sys.modules[__name__], 'CACHE_PATH', '_mmfi_full.h5')
    log(f'[start] F-49 root-cause diagnostic | test_env={TEST_ENV} seed={SEED}')
    log(f'[out] {OUT}')
    log(f'[cache] {cache_name}')

    # Load + BVP (same as F-40/F-48 for fair compare)
    log(f'[data] loading {cache_name}')
    ds = make_mmfi_dataset(Path(__file__).parent / 'bvp_test' / cache_name)
    X = ds.X[:, :10]  # (N, 10, T, 114) complex — slice to first 10 antennas
    y = ds.y
    domains = ds.domains
    log(f'  raw X={X.shape} y={y.shape} domains={domains.shape} n_class={int(y.max())+1}')

    # Detect v2 cache (per-segment max_time=96 vs v1=512). Save raw T for chunked padding later.
    raw_T = X.shape[2]
    target_T = 512  # BVP STFT expects T=512 → in_time=29 = (512-64)/16+1
    if raw_T < target_T:
        log(f'  [pad-plan] cache T={raw_T} < target {target_T}; will zero-pad per chunk in BVP stage')
    elif raw_T > target_T:
        log(f'  [warn] cache T={raw_T} > target {target_T}; center-cropping X in place')
        start = (raw_T - target_T) // 2
        X = X[:, :, start:start + target_T, :].copy()

    log('[bvp] compute_bvp_gpu n_fft=64 hop=16 (chunked via torch)')
    # v2 bvp size = N * 10 * 114 * 33 * 29 * 4 bytes ≈ 67 GB. Stream-write to h5.
    BVP_H5 = OUT.parent / f'_mmfi_v2_bvp.h5'
    if BVP_H5.exists(): BVP_H5.unlink()
    chunk_size = 96
    n_total = X.shape[0]
    log(f'  [bvp] streaming to {BVP_H5.name} ({n_total} samples, chunk={chunk_size})')
    with h5py.File(BVP_H5, 'w') as f:
        f.create_dataset('bvp', shape=(n_total, 10, 114, 33, 29),
                         dtype='float32', chunks=(chunk_size, 10, 114, 33, 29))
        f.create_dataset('y', data=y.astype(np.int64))
        f.create_dataset('domains', data=domains.astype(np.int64))
    with h5py.File(BVP_H5, 'r+') as f:
        ds_bvp = f['bvp']
        for i in range(0, n_total, chunk_size):
            chunk = X[i:i+chunk_size]
            # v2 cache: pad each in-CPU chunk to target_T to match v1 model contract.
            # RAM cost: chunk_size × 10 × target_T × 114 × 8 bytes ≈ 4.5 MB per chunk (safe)
            if chunk.shape[2] < target_T:
                pad = np.zeros((chunk.shape[0], chunk.shape[1],
                                target_T - chunk.shape[2], chunk.shape[3]),
                               dtype=chunk.dtype)
                chunk = np.concatenate([chunk, pad], axis=2)
            chunk = np.nan_to_num(chunk, nan=0.0, posinf=0.0, neginf=0.0)
            t = torch.from_numpy(np.ascontiguousarray(chunk)).to(DEVICE)
            b = compute_bvp_gpu(t, n_fft=64, hop=16, keep_antenna=True,
                                batch_n=32, return_device='cpu').numpy()
            b = np.nan_to_num(b, nan=0.0, posinf=0.0, neginf=0.0)
            ds_bvp[i:i+chunk_size] = b
            del t, b
            torch.cuda.empty_cache()
            if (i // chunk_size) % 10 == 0:
                log(f'  [bvp] {i}/{n_total} ({i/n_total*100:.1f}%)')
    log(f'  [bvp] wrote {BVP_H5}')
    log(f'  [bvp] size on disk: {BVP_H5.stat().st_size / 1e9:.2f} GB')

    # LODO split (lodo_splits_with_val returns list of dicts per test domain)
    splits = lodo_splits_with_val(ds, val_mode='random', val_frac=0.2, seed=SEED,
                                  group_by_subject=True)
    test_env_idx = ds.domain_names.index(TEST_ENV)
    sp = splits[test_env_idx]
    train_idx = np.asarray(sp['train_idx'])
    val_idx = np.asarray(sp['val_idx'])
    test_idx = np.asarray(sp['test_idx'])
    log(f'[split] test_env={TEST_ENV}(idx={test_env_idx}) train={len(train_idx)} '
        f'val={len(val_idx)} test={len(test_idx)}')

    results = {}
    for name in ['lenet_attn', 'cbam']:
        log(f'[train] {name} in_ant=10')
        _, hist, pr, gt, feats, final = train_one(
            name, 10, BVP_H5, y, train_idx, val_idx, test_idx)
        results[name] = {
            'history': hist,
            'preds': pr.tolist(),
            'gts': gt.tolist(),
            'feats': feats.tolist(),
            'final_test_acc': final,
        }

    # Save raw results
    with open(OUT / 'results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f)
    log(f'[save] {OUT / "results.json"}')

    # ---- Generate the 4 figures ----
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        # sklearn unavailable — use numpy fallbacks for confusion_matrix / f1 / 2D embedding
    except ImportError as e:
        log(f'[skip plots] matplotlib not available: {e}')
        return

    def confusion_matrix_np(y_true, y_pred, n_classes):
        cm = np.zeros((n_classes, n_classes), dtype=np.int64)
        for t, p in zip(y_true, y_pred):
            cm[t, p] += 1
        return cm

    def f1_per_class_np(y_true, y_pred, n_classes):
        f1s = np.zeros(n_classes)
        for c in range(n_classes):
            tp = ((y_true == c) & (y_pred == c)).sum()
            fp = ((y_true != c) & (y_pred == c)).sum()
            fn = ((y_true == c) & (y_pred != c)).sum()
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1s[c] = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        return f1s

    def pca2_np(X, n_components=2):
        # X: (N, D) centered then SVD
        X = X - X.mean(axis=0, keepdims=True)
        U, S, Vt = np.linalg.svd(X, full_matrices=False)
        return X @ Vt[:n_components].T

    fig_dir = OUT / 'figs'
    fig_dir.mkdir(exist_ok=True)

    # 1) train curve
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, name in zip(axes, ['lenet_attn', 'cbam']):
        h = results[name]['history']
        ax.plot(h['epoch'], h['train_acc'], 'o-', label='train')
        ax.plot(h['epoch'], h['val_acc'], 's-', label='val')
        ax.plot(h['epoch'], h['test_acc'], '^-', label='test')
        ax.set_title(f'{name} (final test={results[name]["final_test_acc"]:.3f})')
        ax.set_xlabel('epoch'); ax.set_ylabel('accuracy')
        ax.set_ylim(0, 1); ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(fig_dir / 'train_curve.png', dpi=100); plt.close()
    log('[plot] train_curve.png')

    # 2) confusion matrix (test) - both models side by side
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, name in zip(axes, ['lenet_attn', 'cbam']):
        cm = confusion_matrix_np(np.array(results[name]['gts']),
                                 np.array(results[name]['preds']), 27)
        cm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
        im = ax.imshow(cm, cmap='Blues', vmin=0, vmax=0.3)
        ax.set_title(f'{name}'); ax.set_xlabel('predicted'); ax.set_ylabel('true')
        plt.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout(); fig.savefig(fig_dir / 'confusion_matrix.png', dpi=100); plt.close()
    log('[plot] confusion_matrix.png')

    # 3) per-class F1
    fig, ax = plt.subplots(figsize=(12, 4))
    width = 0.4
    x = np.arange(27)
    for i, name in enumerate(['lenet_attn', 'cbam']):
        f1s = f1_per_class_np(np.array(results[name]['gts']),
                              np.array(results[name]['preds']), 27)
    ax.set_xticks(x + width/2); ax.set_xticklabels([str(i) for i in range(27)], fontsize=8)
    ax.set_xlabel('class'); ax.set_ylabel('F1'); ax.set_ylim(0, 1); ax.legend(); ax.grid(axis='y', alpha=0.3)
    fig.tight_layout(); fig.savefig(fig_dir / 'per_class_f1.png', dpi=100); plt.close()
    log('[plot] per_class_f1.png')

    # 4) 2D embedding (sklearn TSNE unavailable → numpy PCA fallback) of test features
    #    colored by class label. If features cluster by class → some learnable signal;
    #    if features mix → even the head's penultimate representation is uninformative.
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, name in zip(axes, ['lenet_attn', 'cbam']):
        feats = np.array(results[name]['feats'])
        # Subsample to 200 for speed
        if len(feats) > 200:
            idx = np.random.RandomState(0).choice(len(feats), 200, replace=False)
            feats_s = feats[idx]; y_s = np.array(results[name]['gts'])[idx]
        else:
            feats_s = feats; y_s = np.array(results[name]['gts'])
        emb = pca2_np(feats_s, 2)
        ax.scatter(emb[:, 0], emb[:, 1], c=y_s, cmap='tab20', s=20, alpha=0.7)
        ax.set_title(f'{name} test features PCA (color=class)')
        ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout(); fig.savefig(fig_dir / 'env_tsne.png', dpi=100); plt.close()
    log('[plot] env_tsne.png')

    # ---- Decision rule ----
    la = results['lenet_attn']; ca = results['cbam']
    la_max_tr = max(la['history']['train_acc'])
    ca_max_tr = max(ca['history']['train_acc'])
    la_test = la['final_test_acc']
    ca_test = ca['final_test_acc']
    log(f'')
    log(f'=== DIAGNOSIS ===')
    log(f'LeNet: max_train={la_max_tr:.3f}  test={la_test:.3f}  gap={la_max_tr-la_test:+.3f}')
    log(f'CBAM : max_train={ca_max_tr:.3f}  test={ca_test:.3f}  gap={ca_max_tr-ca_test:+.3f}')
    la_cls = classify(la_max_tr, la_test)
    ca_cls = classify(ca_max_tr, ca_test)
    log(f'LeNet classification: {la_cls}')
    log(f'CBAM classification : {ca_cls}')

    if la_cls.startswith('A') and ca_cls.startswith('A'):
        log('=> A: BOTH models fail to learn. Task itself unlearnable.')
        log('   Action: audit labels, simplify task, check signal presence')
    elif la_cls.startswith('B') and ca_cls.startswith('B'):
        log('=> B: BOTH models overfit to env features. Domain methods may help.')
        log('   Action: try IRM/V-REx; visualize what env features leak')
    else:
        log('=> MIXED: one model overfits, other does not. Investigate why.')


def classify(max_train, test_acc):
    if max_train < 0.10:
        return 'A (cannot learn)'
    elif (max_train - test_acc) > 0.20:
        return 'B (overfit to env)'
    else:
        return 'C (mixed/partial)'


class MMFiBvpLazyDataset(Dataset):
    """Lazy dataset reading bvp chunks from a precomputed h5 file.

    v2 cache bvp is 16,447 * 10 * 114 * 33 * 29 * 4 bytes ≈ 67 GB, too big to fit in RAM.
    h5py reads only the requested index range, so peak RAM is one batch.
    """
    def __init__(self, h5_path, indices):
        self.h5_path = str(h5_path)
        self._h5 = None
        self.indices = np.asarray(indices, dtype=np.int64)

    def _ensure_open(self):
        if self._h5 is None:
            self._h5 = h5py.File(self.h5_path, 'r')
            self._bvp = self._h5['bvp']
            self._y = self._h5['y']

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        self._ensure_open()
        idx = int(self.indices[i])
        x = torch.from_numpy(np.asarray(self._bvp[idx], dtype=np.float32))
        y = torch.tensor(int(self._y[idx]), dtype=torch.long)
        return x, y

    def __del__(self):
        if self._h5 is not None:
            try: self._h5.close()
            except Exception: pass


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--cache', default='_mmfi_full.h5',
                   help='cache file (default: _mmfi_full.h5 v1; v2 use _mmfi_v2.h5)')
    args = p.parse_args()
    # Stash into a module global so main() can read it
    bench_f49_mmfi_diag = sys.modules[__name__]
    bench_f49_mmfi_diag.CACHE_PATH = args.cache
    main()