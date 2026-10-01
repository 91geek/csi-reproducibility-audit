"""
F-50: MMFi signal-presence audit (4 sub-diagnostics).

Follow-up to F-49 root-cause diagnosis. Goal is to decide *which* of
(a) class granularity / (b) signal strength / (c) input pipeline /
(d) label integrity is the binding constraint on MMFi LODO accuracy.

Outputs (saved to src/bvp_test/f50_signal/):
  - label_integrity.txt   : per-env per-class counts, missing-class flags (d)
  - variance_report.txt   : within-class vs between-class variance ratio (b)
  - coarse_train_curve.png: LeNet trained on 5-grouped action targets (a)
  - input_compare.png     : raw CSI vs BVP for one action across envs (c)
  - F50_FINDINGS.md       : consolidated 4-finding report

Wall-clock budget: ~75 min on RTX 2080 Ti.
"""
from __future__ import annotations
import json, time, sys, argparse
from pathlib import Path
import numpy as np
import torch
import h5py
from torch.utils.data import DataLoader, TensorDataset, Dataset

sys.path.insert(0, str(Path(__file__).parent))
from wfcslab.data.mmfi import make_mmfi_dataset
from wfcslab.models.backbones import LeNetCSI_Attn
from wfcslab.signal.doppler_gpu import compute_bvp_gpu
from wfcslab.engine.trainer import set_perf_flags
import matplotlib.pyplot as plt

set_perf_flags()
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
OUT = Path(__file__).parent / 'bvp_test' / 'f50_signal'
OUT.mkdir(parents=True, exist_ok=True)
LOG = OUT / '_run.log'

SEED = 0
TEST_ENV = 'E2'  # same as F-49 for continuity
N_COARSE = 5

# Index-based 5-group mapping (NOTE: action semantic names not available in
# project — this grouping is positional and documented as a caveat)
COARSE_GROUPS = {
    0: list(range(0, 6)),    # group A (likely locomotion/early)
    1: list(range(6, 12)),   # group B
    2: list(range(12, 18)),  # group C
    3: list(range(18, 24)),  # group D
    4: list(range(24, 27)),  # group E (smaller tail)
}


def log(msg):
    ts = time.strftime('%H:%M:%S')
    line = f'[{ts}] {msg}'
    print(line, flush=True)
    with LOG.open('a', encoding='utf-8') as f:
        f.write(line + '\n')


def build_bvp(raw_h5_or_X, cache_name='_mmfi_full.h5'):
    """Compute BVP via streaming h5 write; return h5py handle for lazy reads.

    Accepts either:
      - h5py dataset (raw_h5_or_X['X'] shape (N, A, T, S) complex) → reads chunked
      - np.ndarray (legacy v1 path; loads full X to RAM)

    v2 path:
      - check `_mmfi_v2_bvp.h5` next to cache; reuse if exists
      - else compute chunked via torch and stream-write to disk (avoids 67 GB RAM)
    v1 path:
      - in-memory np.zeros (old behavior, fails on big caches)
    """
    is_v2 = 'v2' in cache_name
    bvp_h5_path = (Path(__file__).parent / 'bvp_test' / cache_name).with_name(
        cache_name.replace('.h5', '_bvp.h5'))
    if is_v2 and bvp_h5_path.exists():
        log(f'[bvp] reusing existing {bvp_h5_path.name}')
        f = h5py.File(bvp_h5_path, 'r')
        log(f'  shape={f["bvp"].shape} dtype={f["bvp"].dtype}')
        return f

    # Resolve h5 dataset (lazy) vs numpy array (legacy)
    if hasattr(raw_h5_or_X, 'file'):  # h5py Dataset
        ds_X = raw_h5_or_X
        n = ds_X.shape[0]
    else:
        ds_X = None
        n = raw_h5_or_X.shape[0]

    log(f'[bvp] compute on N={n} (chunked torch → streaming h5)')
    f = h5py.File(bvp_h5_path, 'w')
    f.create_dataset('bvp', shape=(n, 10, 114, 33, 29), dtype='float32',
                     chunks=(96, 1, 114, 33, 29), compression='lzf')
    t0 = time.time()
    for i in range(0, n, 96):
        if ds_X is not None:
            chunk = ds_X[i:i+96, :10]  # (96, 10, T, 114) complex; lazy h5 read
        else:
            chunk = raw_h5_or_X[i:i+96, :10]
        chunk = np.nan_to_num(chunk, nan=0.0, posinf=0.0, neginf=0.0)
        t = torch.from_numpy(np.ascontiguousarray(chunk)).to(DEVICE)
        b = compute_bvp_gpu(t, n_fft=64, hop=16, keep_antenna=True,
                            batch_n=32, return_device='cpu').numpy()
        b = np.nan_to_num(b, nan=0.0, posinf=0.0, neginf=0.0)
        f['bvp'][i:i+96] = b
        del t, b, chunk
        torch.cuda.empty_cache()
        if i % 960 == 0:
            log(f'  [bvp] {i}/{n} ({100*i/n:.1f}%) {time.time()-t0:.1f}s')
    f.close()
    log(f'[bvp] wrote {bvp_h5_path} ({bvp_h5_path.stat().st_size/1e9:.2f} GB)')
    return h5py.File(bvp_h5_path, 'r')


class BvpLazyDataset(Dataset):
    """Reads (bvp_i, y_i) from a streaming h5 file; per-sample random access."""
    def __init__(self, h5_path, indices, labels, target_T=29):
        self.f = h5py.File(h5_path, 'r')
        self.indices = np.asarray(indices, dtype=np.int64)
        self.labels = np.asarray(labels, dtype=np.int64)
        self.target_T = target_T
    def __len__(self):
        return len(self.indices)
    def __getitem__(self, k):
        i = int(self.indices[k])
        x = self.f['bvp'][i]  # (10, 114, 33, 29) float32
        return torch.from_numpy(np.ascontiguousarray(x)), int(self.labels[k])
    def close(self):
        self.f.close()


# ============================================================
# (d) LABEL INTEGRITY
# ============================================================
def audit_label_integrity(ds, OUT):
    log('=== (d) LABEL INTEGRITY ===')
    y = np.array(ds.y); dom = np.array(ds.domains)
    n_classes = int(y.max()) + 1
    env_names = ds.domain_names  # ['E1','E2','E3','E4']
    n_envs = len(env_names)

    lines = []
    lines.append('MMFi Label Integrity Report')
    lines.append('=' * 60)
    lines.append(f'n_total_samples = {len(y)}')
    lines.append(f'n_classes       = {n_classes}')
    lines.append(f'n_envs          = {n_envs}  ({env_names})')
    lines.append('')

    # Per-env per-class counts
    header = 'class | total | ' + ' '.join(f'{e:>5}' for e in env_names) + ' | flag'
    lines.append(header)
    lines.append('-' * len(header))
    issues = 0
    for c in range(n_classes):
        total = int((y == c).sum())
        per_env = [int(((y == c) & (dom == e_idx)).sum()) for e_idx in range(n_envs)]
        flag = []
        if total == 0:
            flag.append('GLOBAL_MISSING'); issues += 1
        if any(v == 0 for v in per_env):
            flag.append('ENV_MISSING'); issues += 1
        if max(per_env) - min(per_env) > max(per_env) * 0.5:
            flag.append('IMBALANCED'); issues += 1
        flag_str = ','.join(flag) if flag else 'OK'
        lines.append(f'  {c:2d}  | {total:5d} | ' + ' '.join(f'{v:5d}' for v in per_env) + f' | {flag_str}')

    lines.append('')
    lines.append(f'Total integrity issues: {issues}')
    lines.append('')

    # Class imbalance summary
    cls_totals = np.array([int((y == c).sum()) for c in range(n_classes)])
    lines.append(f'Class count: min={cls_totals.min()} max={cls_totals.max()} '
                 f'mean={cls_totals.mean():.1f} std={cls_totals.std():.1f}')
    lines.append(f'Imbalance ratio max/min = {cls_totals.max()/cls_totals.min():.2f}')

    text = '\n'.join(lines)
    (OUT / 'label_integrity.txt').write_text(text, encoding='utf-8')
    log(f'  wrote label_integrity.txt ({issues} issues)')
    return issues


# ============================================================
# (b) WITHIN/BETWEEN VARIANCE
# ============================================================
def audit_variance(bvp_h5, y, OUT):
    """Streaming variance on bvp h5 file (avoids 67 GB RAM).

    Args:
        bvp_h5: h5py.File handle (must have 'bvp' dataset)
        y: (N,) int labels
    """
    log('=== (b) WITHIN/BETWEEN VARIANCE (streaming h5) ===')
    n = bvp_h5['bvp'].shape[0]
    F_dim = int(np.prod(bvp_h5['bvp'].shape[1:]))
    log(f'  N={n} F_dim={F_dim}')

    n_classes = int(y.max()) + 1

    # Pass 1: per-class sum & sum-of-squares
    class_sum = np.zeros((n_classes, F_dim), dtype=np.float64)
    class_sqsum = np.zeros((n_classes, F_dim), dtype=np.float64)
    class_size = np.zeros(n_classes, dtype=np.int64)

    CHUNK = 256
    t0 = time.time()
    for s in range(0, n, CHUNK):
        e = min(s + CHUNK, n)
        x = bvp_h5['bvp'][s:e].reshape(e - s, F_dim).astype(np.float64)
        yc = y[s:e]
        for c in range(n_classes):
            mask = (yc == c)
            if mask.any():
                class_sum[c] += x[mask].sum(axis=0)
                class_sqsum[c] += (x[mask] ** 2).sum(axis=0)
                class_size[c] += int(mask.sum())
        del x
        if s % (CHUNK * 10) == 0:
            log(f'  [scan] {s}/{n} ({100*s/n:.1f}%) {time.time()-t0:.1f}s')

    # Per-class mean
    class_mean = np.zeros_like(class_sum)
    valid = class_size > 0
    class_mean[valid] = class_sum[valid] / class_size[valid, None]

    # Global mean
    total_size = int(class_size.sum())
    global_mean = class_sum.sum(axis=0) / total_size

    # Within-class SSE
    total_within_sse = 0.0
    for c in range(n_classes):
        if class_size[c] > 1:
            sse = class_sqsum[c] - class_size[c] * (class_mean[c] ** 2)
            sse = np.maximum(sse, 0.0)
            total_within_sse += float(sse.sum())
    total_within_var = total_within_sse / (total_size - n_classes)

    # Between-class variance
    total_between_ssb = 0.0
    for c in range(n_classes):
        if class_size[c] > 0:
            diff = class_mean[c] - global_mean
            total_between_ssb += class_size[c] * float((diff ** 2).sum())
    total_between_var = total_between_ssb / (n_classes - 1)

    ratio = total_between_var / (total_within_var + 1e-12)

    text = (
        f'MMFi Signal-Presence (Within-vs-Between) Report\n'
        f'{'=' * 60}\n'
        f'Feature dim (flattened BVP) = {F_dim}\n'
        f'Total within-class variance  = {total_within_var:.6f}\n'
        f'Total between-class variance = {total_between_var:.6f}\n'
        f'Ratio (between/within)       = {ratio:.4f}\n'
        f'\n'
        f'Interpretation:\n'
        f'  ratio >> 1: between-class var dominates → classes are separable in BVP\n'
        f'  ratio ~ 1: noise floor, no class structure\n'
        f'  ratio << 1: within-class dominates → labels are noise\n'
        f'\n'
        f'Random-baseline signal-to-noise for {n_classes}-class problem: 1.0 (no signal)\n'
        f'Observed ratio = {ratio:.4f}\n'
    )
    (OUT / 'variance_report.txt').write_text(text, encoding='utf-8')
    log(f'  within={total_within_var:.4f} between={total_between_var:.4f} ratio={ratio:.4f}')
    return ratio


# ============================================================
# (a) COARSE CLASS RETRAIN
# ============================================================
def coarse_class_train(bvp_h5, y, dom, ds, OUT, cache_name='_mmfi_full.h5'):
    """(a) Coarse 5-class retrain. Reads BVP via lazy h5 (no 67 GB RAM).
    """
    log('=== (a) COARSE 5-CLASS RETRAIN (lazy h5) ===')
    # Map 27-class labels to 5-group labels
    label_map = np.zeros(int(y.max()) + 1, dtype=np.int64)
    for grp, members in COARSE_GROUPS.items():
        for c in members:
            label_map[c] = grp
    y_coarse = label_map[y]
    log(f'  27→5 mapping: class sizes per group = '
        f'{[int((y_coarse==g).sum()) for g in range(N_COARSE)]}')

    # Same LODO split as F-49
    test_env_idx = ds.domain_names.index(TEST_ENV)
    train_mask = (dom != test_env_idx)
    val_mask = (dom == test_env_idx)
    rng = np.random.RandomState(SEED)
    val_indices = rng.choice(np.where(val_mask)[0], size=int(val_mask.sum() * 0.2), replace=False)
    val_set = set(val_indices.tolist())
    test_indices = np.array([i for i in np.where(val_mask)[0] if i not in val_set])

    train_idx = np.where(train_mask)[0]
    val_idx = val_indices
    test_idx = test_indices
    log(f'  train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}')

    bvp_h5_path = bvp_h5.filename
    tr_ds = BvpLazyDataset(bvp_h5_path, train_idx, y_coarse[train_idx])
    va_ds = BvpLazyDataset(bvp_h5_path, val_idx,   y_coarse[val_idx])
    te_ds = BvpLazyDataset(bvp_h5_path, test_idx,  y_coarse[test_idx])

    bs = 64
    tr_ld = DataLoader(tr_ds, batch_size=bs, shuffle=True, drop_last=False, num_workers=0)
    va_ld = DataLoader(va_ds, batch_size=bs, shuffle=False, num_workers=0)
    te_ld = DataLoader(te_ds, batch_size=bs, shuffle=False, num_workers=0)

    model = LeNetCSI_Attn(in_antennas=10, in_subcarriers=114, in_freq=33, in_time=29,
                          out_dim=N_COARSE).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=30)
    scaler = torch.amp.GradScaler('cuda')
    crit = torch.nn.CrossEntropyLoss()

    history = {'epoch': [], 'train_acc': [], 'val_acc': [], 'test_acc': []}
    best_val = 0.0; best_state = None; patience = 0
    PATIENCE = 8

    for ep in range(30):
        model.train(); n_corr = 0; n_tot = 0
        for xb, yb in tr_ld:
            xb = xb.to(DEVICE); yb = yb.to(DEVICE)
            opt.zero_grad()
            with torch.amp.autocast('cuda'):
                out = model(xb); loss = crit(out, yb)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update()
            n_corr += (out.argmax(1) == yb).sum().item(); n_tot += yb.size(0)
        train_acc = n_corr / n_tot
        sched.step()

        model.eval(); va_corr = 0; va_tot = 0
        with torch.no_grad():
            for xb, yb in va_ld:
                xb = xb.to(DEVICE); yb = yb.to(DEVICE)
                with torch.amp.autocast('cuda'):
                    out = model(xb)
                va_corr += (out.argmax(1) == yb).sum().item(); va_tot += yb.size(0)
        val_acc = va_corr / va_tot

        te_corr = 0; te_tot = 0
        with torch.no_grad():
            for xb, yb in te_ld:
                xb = xb.to(DEVICE); yb = yb.to(DEVICE)
                with torch.amp.autocast('cuda'):
                    out = model(xb)
                te_corr += (out.argmax(1) == yb).sum().item(); te_tot += yb.size(0)
        test_acc = te_corr / te_tot

        history['epoch'].append(ep); history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc); history['test_acc'].append(test_acc)
        log(f'  ep{ep:02d} train={train_acc:.3f} val={val_acc:.3f} test={test_acc:.3f}')

        if val_acc > best_val:
            best_val = val_acc; best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= PATIENCE:
                log(f'  early stop @ ep{ep} (best val={best_val:.3f})')
                break

    tr_ds.close(); va_ds.close(); te_ds.close()

    # Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(history['epoch'], history['train_acc'], 'o-', label='train')
    ax.plot(history['epoch'], history['val_acc'], 's-', label='val')
    ax.plot(history['epoch'], history['test_acc'], '^-', label='test')
    ax.axhline(1.0 / N_COARSE, color='gray', linestyle='--', alpha=0.5, label=f'random (1/{N_COARSE})')
    ax.set_xlabel('epoch'); ax.set_ylabel('accuracy')
    ax.set_title(f'F-50 (a): LeNet trained on 5-grouped actions '
                 f'(final test={history["test_acc"][-1]:.3f})')
    ax.set_ylim(0, 1); ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(OUT / 'coarse_train_curve.png', dpi=100); plt.close()
    log('  wrote coarse_train_curve.png')

    return history['test_acc'][-1], best_val


# ============================================================
# (c) INPUT PIPELINE COMPARE
# ============================================================
def input_compare(raw_h5, bvp_h5, y, dom, ds, OUT, cache_name='_mmfi_full.h5'):
    """(c) Input pipeline compare. Reads 1 raw sample (T=96, pad to 512) + bvp from h5.
    """
    log('=== (c) INPUT PIPELINE COMPARE (lazy h5) ===')
    c_class = 0
    log(f'  visualizing class {c_class} across all envs')
    fig, axes = plt.subplots(2, 4, figsize=(16, 6))

    for col, env_idx in enumerate(range(len(ds.domain_names))):
        idxs = np.where((y == c_class) & (dom == env_idx))[0]
        if len(idxs) == 0:
            continue
        i = idxs[0]
        # RAW from h5: shape (10, T, 114) complex; pad T to 512
        raw = raw_h5['X'][i, 0]  # (T, 114) complex; T=96 for v2
        T = raw.shape[0]
        if T < 512:
            pad = np.zeros((512 - T, raw.shape[1]), dtype=raw.dtype)
            raw = np.concatenate([raw, pad], axis=0)
        raw_amp = np.abs(raw)

        # BVP from h5
        bvp_i = bvp_h5['bvp'][i, 0]  # (114, 33, 29) float32
        bvp_spec = bvp_i.mean(axis=0)  # (33, 29) avg over subcarriers

        ax = axes[0, col]
        im = ax.imshow(raw_amp.T, aspect='auto', origin='lower', cmap='viridis')
        ax.set_title(f'{ds.domain_names[env_idx]} raw amp (|CSI|)')
        ax.set_xlabel('time (samples)'); ax.set_ylabel('subcarrier')
        plt.colorbar(im, ax=ax, fraction=0.046)

        ax = axes[1, col]
        im = ax.imshow(bvp_spec.T, aspect='auto', origin='lower', cmap='viridis')
        ax.set_title(f'{ds.domain_names[env_idx]} BVP (avg-over-subcarrier)')
        ax.set_xlabel('time frames'); ax.set_ylabel('doppler bin')
        plt.colorbar(im, ax=ax, fraction=0.046)

    fig.suptitle(f'F-50 (c): class={c_class} — raw |CSI| spectrogram (top) vs BVP (bottom)',
                 fontsize=13)
    fig.tight_layout(); fig.savefig(OUT / 'input_compare.png', dpi=100); plt.close()
    log('  wrote input_compare.png')


# ============================================================
# MAIN
# ============================================================
def main():
    if LOG.exists(): LOG.unlink()
    cache_name = getattr(sys.modules[__name__], 'CACHE_PATH', '_mmfi_full.h5')

    log('[start] F-50 MMFi signal-presence audit')
    log(f'[out] {OUT}')
    log(f'[cache] {cache_name}')

    # Open cache as h5 directly (avoid loading whole X into RAM)
    cache_path = Path(__file__).parent / 'bvp_test' / cache_name
    log(f'[data] opening {cache_name} as h5 (lazy)')
    raw_h5 = h5py.File(cache_path, 'r')
    n_total = raw_h5['X'].shape[0]
    log(f'  raw X={raw_h5["X"].shape} (lazy)')

    # Use mmfi.py to get y, dom, domain_names without loading X
    ds = make_mmfi_dataset(cache_path)
    y = np.array(ds.y); dom = np.array(ds.domains)
    log(f'  y={y.shape} dom={dom.shape} n_class={int(y.max())+1}')

    # (d) Label integrity (no compute needed)
    n_issues = audit_label_integrity(ds, OUT)

    # BVP via streaming (returns h5 handle) — pass h5 dataset, not numpy slice
    bvp_h5 = build_bvp(raw_h5['X'], cache_name=cache_name)

    # (b) Variance on BVP (streaming)
    var_ratio = audit_variance(bvp_h5, y, OUT)

    # (a) Coarse-class retrain (lazy h5)
    coarse_test, coarse_best_val = coarse_class_train(
        bvp_h5, y, dom, ds, OUT, cache_name=cache_name)
    log(f'  (a) coarse-class final test={coarse_test:.3f} (vs random 1/5={0.2:.3f})')

    # (c) Input compare (lazy h5)
    input_compare(raw_h5, bvp_h5, y, dom, ds, OUT, cache_name=cache_name)

    # Summary
    summary = {
        'n_issues': n_issues,
        'variance_ratio_between_over_within': var_ratio,
        'coarse_test_acc': coarse_test,
        'coarse_best_val_acc': coarse_best_val,
        'random_baseline_5class': 1.0 / N_COARSE,
    }
    (OUT / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    log(f'[done] summary: {summary}')

    bvp_h5.close(); raw_h5.close()


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--cache', default='_mmfi_full.h5',
                   help='cache file (default: _mmfi_full.h5 v1; v2 use _mmfi_v2.h5)')
    args = p.parse_args()
    sys.modules[__name__].CACHE_PATH = args.cache
    main()