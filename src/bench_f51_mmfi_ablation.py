"""
F-51: MMFi representation ablation + granularity sweep + action-name mapping.

Three sub-experiments in one script:

  (1) Action-name mapping (loaded from JSON; semantic groupings used below)
  (2) Granularity sweep on BVP: 2 / 4 / 5 / 8 / 12 classes — find the
      "maximum learnable granularity" for MMFi LODO.
  (3) raw |CSI| vs BVP on the best-granularity task from (2). Same
      backbone, same hyperparameters. If raw > BVP, the BVP
      representation is the binding constraint (corroborates F-50 (c)).

Outputs (saved to src/bvp_test/f51_ablation/):
  - actions.json            (1) verified 27-name mapping
  - granularity_sweep.png   (2) test_acc vs granularity curve
  - raw_vs_bvp.png          (3) train curves for both representations
  - raw_vs_bvp.json         (3) summary numbers
  - F51_FINDINGS.md         consolidated report

Wall-clock budget: ~30 min on RTX 2080 Ti.
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
OUT = Path(__file__).parent / 'bvp_test' / 'f51_ablation'
OUT.mkdir(parents=True, exist_ok=True)
LOG = OUT / '_run.log'
SEED = 0
TEST_ENV = 'E2'

# ============================================================
# (1) Action name mapping — verified from arXiv 2305.10345 + official MMFi GitHub
# ============================================================
ACTIONS = [
    ('A01', 'Stretching and relaxing',   'rehab'),
    ('A02', 'Chest expansion horizontal', 'daily'),
    ('A03', 'Chest expansion vertical',   'daily'),
    ('A04', 'Twist left',                 'daily'),
    ('A05', 'Twist right',                'daily'),
    ('A06', 'Mark time',                  'rehab'),
    ('A07', 'Limb extension left',        'rehab'),
    ('A08', 'Limb extension right',       'rehab'),
    ('A09', 'Lunge left-front',           'rehab'),
    ('A10', 'Lunge right-front',          'rehab'),
    ('A11', 'Limb extension both',        'rehab'),
    ('A12', 'Squat',                      'rehab'),
    ('A13', 'Raising hand left',          'daily'),
    ('A14', 'Raising hand right',         'daily'),
    ('A15', 'Lunge left side',            'rehab'),
    ('A16', 'Lunge right side',           'rehab'),
    ('A17', 'Waving hand left',           'daily'),
    ('A18', 'Waving hand right',          'daily'),
    ('A19', 'Picking up things',          'daily'),
    ('A20', 'Throwing left side',         'daily'),
    ('A21', 'Throwing right side',        'daily'),
    ('A22', 'Kicking left side',          'daily'),
    ('A23', 'Kicking right side',         'daily'),
    ('A24', 'Body extension left',        'rehab'),
    ('A25', 'Body extension right',       'rehab'),
    ('A26', 'Jumping up',                 'rehab'),
    ('A27', 'Bowing',                     'daily'),
]

# ============================================================
# Coarse groupings — all semantic, derived from the action names above
# ============================================================
def make_group(assign):
    """assign: list of length 27, group index per action."""
    return np.asarray(assign, dtype=np.int64)

# 2-class: Daily vs Rehabilitation (paper's protocol 1 vs 2)
GRP_2 = make_group([0 if cat == 'daily' else 1 for _, _, cat in ACTIONS])

# 4-class: Daily-Upper, Daily-Lower, Rehab-Upper, Rehab-Lower
# Define via action names
def _grp4():
    out = []
    for code, name, cat in ACTIONS:
        is_lower = any(k in name.lower() for k in ['lunge', 'squat', 'kick', 'body extension', 'jumping'])
        is_upper = any(k in name.lower() for k in ['chest', 'twist', 'raising', 'waving', 'picking', 'throwing', 'stretch', 'mark time', 'limb extension'])
        # Daily group: 0 upper, 1 lower
        # Rehab group: 2 upper, 3 lower
        if cat == 'daily':
            out.append(0 if is_upper else 1)
        else:
            out.append(2 if is_upper else 3)
    return make_group(out)

GRP_4 = _grp4()

# 5-class: same as F-50 (positional fallback documented)
GRP_5 = make_group([0]*6 + [1]*6 + [2]*6 + [3]*6 + [4]*3)

# 8-class: by category × limb
def _grp8():
    out = []
    for code, name, cat in ACTIONS:
        lname = name.lower()
        if cat == 'daily':
            if any(k in lname for k in ['chest', 'twist']):
                out.append(0)  # daily torso
            elif any(k in lname for k in ['raising', 'waving']):
                out.append(1)  # daily hand
            elif any(k in lname for k in ['picking', 'throwing']):
                out.append(2)  # daily object
            else:
                out.append(3)  # daily other (bowing, kick)
        else:
            if any(k in lname for k in ['limb extension']):
                out.append(4)  # rehab limb ext
            elif any(k in lname for k in ['lunge', 'squat']):
                out.append(5)  # rehab lower
            elif any(k in lname for k in ['body extension']):
                out.append(6)  # rehab body
            else:
                out.append(7)  # rehab other (stretch, mark time, jumping)
    return make_group(out)

GRP_8 = _grp8()

# 12-class: by category × finer body part
def _grp12():
    out = []
    for code, name, cat in ACTIONS:
        lname = name.lower()
        if cat == 'daily':
            if 'chest' in lname:
                out.append(0)  # chest
            elif 'twist' in lname:
                out.append(1)  # twist
            elif 'raising' in lname:
                out.append(2)  # raising
            elif 'waving' in lname:
                out.append(3)  # waving
            elif 'picking' in lname:
                out.append(4)  # picking
            elif 'throwing' in lname:
                out.append(5)  # throwing
            elif 'kick' in lname:
                out.append(6)  # kick
            else:
                out.append(7)  # bowing
        else:
            if 'lunge' in lname:
                out.append(8)   # lunge
            elif 'squat' in lname:
                out.append(9)   # squat
            elif 'body extension' in lname:
                out.append(10)  # body ext
            else:
                out.append(11)  # stretch / mark time / limb ext / jumping
    return make_group(out)

GRP_12 = _grp12()

GRANULARITIES = {
    '2_class_daily_vs_rehab': GRP_2,
    '4_class_daily-rehab_x_upper-lower': GRP_4,
    '5_class_positional': GRP_5,
    '8_class_semantic_finer': GRP_8,
    '12_class_semantic_finest': GRP_12,
}


def log(msg):
    ts = time.strftime('%H:%M:%S')
    line = f'[{ts}] {msg}'
    print(line, flush=True)
    with LOG.open('a', encoding='utf-8') as f:
        f.write(line + '\n')


def write_actions_json():
    p = OUT / 'actions.json'
    obj = [
        {'index': i, 'code': code, 'name': name, 'category': cat}
        for i, (code, name, cat) in enumerate(ACTIONS)
    ]
    p.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding='utf-8')
    log(f'[actions] wrote {p} (verified from arXiv 2305.10345 + MMFi GitHub)')


def build_bvp(raw_h5_or_X, cache_name='_mmfi_full.h5'):
    """Compute BVP via streaming h5; return h5 handle for lazy reads.
    Reuses `_mmfi_v2_bvp.h5` if exists (F-49/F-50 already wrote it).
    """
    is_v2 = 'v2' in cache_name
    bvp_h5_path = (Path(__file__).parent / 'bvp_test' / cache_name).with_name(
        cache_name.replace('.h5', '_bvp.h5'))
    if is_v2 and bvp_h5_path.exists():
        log(f'[bvp] reusing existing {bvp_h5_path.name}')
        return h5py.File(bvp_h5_path, 'r')

    if hasattr(raw_h5_or_X, 'file'):
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
            chunk = ds_X[i:i+96, :10]
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


def make_raw_magnitude_spectrogram(raw_h5, cache_name='_mmfi_full.h5'):
    """Stream raw |CSI| magnitude + time-pool to h5. Returns h5 handle.
    raw input is (N, A, T, S) complex from v2 cache (T=96 for v2).
    Output: (N, A, n_out, S) float32 where n_out = T//16.
    """
    raw_h5_path = raw_h5.filename
    mag_h5_path = (Path(raw_h5_path).parent / cache_name).with_name(
        cache_name.replace('.h5', '_rawmag.h5'))
    if mag_h5_path.exists():
        log(f'[raw] reusing existing {mag_h5_path.name}')
        return h5py.File(mag_h5_path, 'r')

    n = raw_h5['X'].shape[0]
    A = 10
    T = raw_h5['X'].shape[2]
    S = raw_h5['X'].shape[3]
    factor = 16
    n_out = T // factor
    if n_out == 0:
        # T too small for factor=16; use factor=8
        factor = 8
        n_out = T // factor
    log(f'[raw] streaming |CSI| magnitude + time-pool factor={factor} → '
        f'({n},{A},{n_out},{S})')

    f = h5py.File(mag_h5_path, 'w')
    f.create_dataset('raw_mag', shape=(n, A, n_out, S), dtype='float32',
                     chunks=(96, 1, n_out, S), compression='lzf')
    t0 = time.time()
    for i in range(0, n, 96):
        x = raw_h5['X'][i:i+96, :A]  # (96, A, T, S) complex
        amp = np.abs(x).astype(np.float32)
        pooled = amp[:, :, :n_out*factor, :].reshape(x.shape[0], A, n_out, factor, S).mean(axis=3)
        f['raw_mag'][i:i+96] = pooled
        del x, amp, pooled
        if i % (96 * 10) == 0:
            log(f'  [raw] {i}/{n} ({100*i/n:.1f}%) {time.time()-t0:.1f}s')
    f.close()
    log(f'[raw] wrote {mag_h5_path} ({mag_h5_path.stat().st_size/1e9:.2f} GB)')
    return h5py.File(mag_h5_path, 'r')


class _LazyDataset(Dataset):
    """Reads a single h5 dataset, indexed by `indices`, returns (x, y_int).
    Lazy per-sample access; lazy transform (e.g., reshape to 5D for raw) applied on read."""
    def __init__(self, h5_path, indices, labels, dset_name, shape_5d=None):
        self.f = h5py.File(h5_path, 'r')
        self.indices = np.asarray(indices, dtype=np.int64)
        self.labels = np.asarray(labels, dtype=np.int64)
        self.dset_name = dset_name
        # If raw 4D, on read we reshape to (A, F=1, T, S) then transpose to (A, S, F=1, T)
        self.shape_5d = shape_5d
    def __len__(self):
        return len(self.indices)
    def __getitem__(self, k):
        i = int(self.indices[k])
        x = self.f[self.dset_name][i]  # native shape
        if self.shape_5d is not None:
            # raw 4D → 5D (A, S, F=1, T) for LeNetCSI_Attn
            x = x.reshape(self.shape_5d).transpose(0, 3, 1, 2).copy()  # (A, S, F=1, T)
        else:
            x = np.ascontiguousarray(x)
        return torch.from_numpy(x), int(self.labels[k])
    def close(self):
        self.f.close()


def lodo_split(ds, y_coarse, dom):
    test_env_idx = ds.domain_names.index(TEST_ENV)
    train_mask = (dom != test_env_idx)
    val_mask = (dom == test_env_idx)
    rng = np.random.RandomState(SEED)
    val_indices = rng.choice(np.where(val_mask)[0], size=int(val_mask.sum() * 0.2), replace=False)
    val_set = set(val_indices.tolist())
    test_indices = np.array([i for i in np.where(val_mask)[0] if i not in val_set])
    train_idx = np.where(train_mask)[0]
    return train_idx, val_indices, test_indices


def train_one(X_or_h5, y_coarse, ds, dom, n_classes, label, epochs=30, batch=64, patience=8):
    """Train one granularity sweep point.

    `X_or_h5` may be:
      - h5py.File handle (has 'bvp' or 'raw_mag' dataset) → lazy DataLoader
      - np.ndarray (legacy v1 path) → in-memory slicing

    5D inputs (BVP): shape (N, A, S, F, T); treat as-is.
    4D inputs (raw): shape (N, A, T, S); reshape per-sample to (A, S, F=1, T) inside dataset.
    """
    train_idx, val_idx, test_idx = lodo_split(ds, y_coarse, dom)
    log(f'  [{label}] train={len(train_idx)} val={len(val_idx)} test={len(test_idx)} n_classes={n_classes}')

    # Detect input type
    if hasattr(X_or_h5, 'file') or hasattr(X_or_h5, 'keys'):
        # h5py handle
        dset_name = 'bvp' if 'bvp' in X_or_h5 else 'raw_mag'
        h5 = X_or_h5
        sample_shape = h5[dset_name].shape[1:]
        # bvp per-sample: (A=10, S=114, F=33, T=29) — 4 dims
        # raw per-sample: (A=10, T=6,   S=114)   — 3 dims
        if len(sample_shape) == 4:
            # bvp
            in_ant, in_s, in_f, in_t = sample_shape
            shape_5d = None
        else:
            # raw 3D (A, T, S) → reshape to 4D (A, F=1, T, S) then transpose to (A, S, F=1, T)
            in_ant, in_t, in_s = sample_shape
            in_f = 1
            shape_5d = (in_ant, 1, in_t, in_s)
        h5_path = h5.filename
        tr_ds = _LazyDataset(h5_path, train_idx, y_coarse[train_idx], dset_name, shape_5d=shape_5d)
        va_ds = _LazyDataset(h5_path, val_idx,   y_coarse[val_idx],   dset_name, shape_5d=shape_5d)
        te_ds = _LazyDataset(h5_path, test_idx,  y_coarse[test_idx],  dset_name, shape_5d=shape_5d)
        close_ds = True
    else:
        # legacy numpy path
        X = X_or_h5
        close_ds = False
        Xtr = X[train_idx]; ytr = y_coarse[train_idx]
        Xva = X[val_idx];   yva = y_coarse[val_idx]
        Xte = X[test_idx];  yte = y_coarse[test_idx]
        in_ant = X.shape[1]
        in_s = X.shape[3]
        in_f = X.shape[2]
        if X.ndim == 5:
            in_t = X.shape[4]
        else:
            in_t = X.shape[2]; in_s = X.shape[3]; in_f = 1
            X5 = X.reshape(X.shape[0], X.shape[1], 1, in_t, in_s)
            X5 = np.transpose(X5, (0, 1, 4, 2, 3)).astype(np.float32)
            Xtr = X5[train_idx]; ytr = y_coarse[train_idx]
            Xva = X5[val_idx];   yva = y_coarse[val_idx]
            Xte = X5[test_idx];  yte = y_coarse[test_idx]
        tr_ds = TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr))
        va_ds = TensorDataset(torch.from_numpy(Xva), torch.from_numpy(yva))
        te_ds = TensorDataset(torch.from_numpy(Xte), torch.from_numpy(yte))

    log(f'  in_ant={in_ant} in_s={in_s} in_f={in_f} in_t={in_t}')
    model = LeNetCSI_Attn(in_antennas=in_ant, in_subcarriers=in_s, in_freq=in_f, in_time=in_t,
                          out_dim=n_classes).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    scaler = torch.amp.GradScaler('cuda')
    crit = torch.nn.CrossEntropyLoss()

    nw = 0 if close_ds else 0
    tr_ld = DataLoader(tr_ds, batch_size=batch, shuffle=True, drop_last=False, num_workers=nw)
    va_ld = DataLoader(va_ds, batch_size=batch, shuffle=False, num_workers=nw)
    te_ld = DataLoader(te_ds, batch_size=batch, shuffle=False, num_workers=nw)

    history = {'epoch': [], 'train_acc': [], 'val_acc': [], 'test_acc': []}
    best_val = 0.0; best_state = None; pat = 0

    for ep in range(epochs):
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

        def _eval(ld):
            model.eval(); c = 0; t = 0
            with torch.no_grad():
                for xb, yb in ld:
                    xb = xb.to(DEVICE); yb = yb.to(DEVICE)
                    with torch.amp.autocast('cuda'):
                        out = model(xb)
                    c += (out.argmax(1) == yb).sum().item(); t += yb.size(0)
            return c / t
        val_acc = _eval(va_ld)
        test_acc = _eval(te_ld)

        history['epoch'].append(ep); history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc); history['test_acc'].append(test_acc)

        if val_acc > best_val:
            best_val = val_acc; best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            pat = 0
        else:
            pat += 1
            if pat >= patience:
                break

    if close_ds:
        tr_ds.close(); va_ds.close(); te_ds.close()

    final_test = history['test_acc'][-1]
    final_train = history['train_acc'][-1]
    log(f'  [{label}] final train={final_train:.3f} test={final_test:.3f} '
        f'best_val={best_val:.3f} (random={1.0/n_classes:.3f})')
    return {'history': history, 'final_test': final_test, 'final_train': final_train,
            'best_val': best_val, 'n_classes': n_classes}


def main():
    if LOG.exists(): LOG.unlink()
    cache_name = getattr(sys.modules[__name__], 'CACHE_PATH', '_mmfi_full.h5')
    log('[start] F-51: representation ablation + granularity sweep + action-name mapping')
    log(f'[cache] {cache_name}')

    write_actions_json()

    log(f'[data] loading {cache_name} (lazy h5)')
    cache_path = Path(__file__).parent / 'bvp_test' / cache_name
    raw_h5 = h5py.File(cache_path, 'r')
    n_total = raw_h5['X'].shape[0]
    log(f'  raw X={raw_h5["X"].shape} (lazy)')

    ds = make_mmfi_dataset(cache_path)  # still load y/dom from mmfi.py
    y27 = np.array(ds.y); dom = np.array(ds.domains)
    log(f'  y={y27.shape} dom={dom.shape} n_class={int(y27.max())+1}')

    # BVP via streaming (reuses F-49/F-50 cache)
    bvp = build_bvp(raw_h5['X'], cache_name=cache_name)

    # Raw magnitude via streaming h5 write
    X_raw_mag = make_raw_magnitude_spectrogram(raw_h5, cache_name=cache_name)

    # ============================================================
    # (2) Granularity sweep on BVP
    # ============================================================
    log('=== (2) GRANULARITY SWEEP on BVP ===')
    sweep_results = {}
    for label, grp in GRANULARITIES.items():
        n_classes = int(grp.max()) + 1
        # Map y27 to coarse labels via the per-index assignment
        y_coarse = grp[y27]
        # Class-size printout
        sizes = [int((y_coarse == c).sum()) for c in range(n_classes)]
        log(f'  [{label}] class sizes = {sizes}')
        res = train_one(bvp, y_coarse, ds, dom, n_classes, label, epochs=30)
        sweep_results[label] = {'test': res['final_test'], 'train': res['final_train'],
                                 'best_val': res['best_val'], 'n_classes': n_classes,
                                 'history': res['history']}

    # Plot sweep
    fig, ax = plt.subplots(figsize=(10, 5))
    labels = list(sweep_results.keys())
    tests = [sweep_results[l]['test'] for l in labels]
    trains = [sweep_results[l]['train'] for l in labels]
    nc = [sweep_results[l]['n_classes'] for l in labels]
    rand = [1.0 / n for n in nc]
    x = np.arange(len(labels))
    width = 0.3
    ax.bar(x - width, trains, width, label='train', color='steelblue')
    ax.bar(x,         tests, width, label='test',  color='darkorange')
    ax.bar(x + width, rand,  width, label='random',color='gray', alpha=0.6)
    for i, (tr, te, r, n) in enumerate(zip(trains, tests, rand, nc)):
        ax.text(i - width,  tr + 0.01, f'{tr:.2f}', ha='center', fontsize=8)
        ax.text(i,         te + 0.01, f'{te:.2f}', ha='center', fontsize=8)
        ax.text(i + width, r  + 0.01, f'{r:.2f}',  ha='center', fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels([f'{l}\n(n={nc[i]})' for i, l in enumerate(labels)],
                                          fontsize=8, rotation=0)
    ax.set_ylabel('accuracy'); ax.set_ylim(0, 1.0)
    ax.set_title('F-51 (2): MMFi granularity sweep on BVP — LeNetCSI_Attn, multirx10, E2 LODO')
    ax.legend(); ax.grid(axis='y', alpha=0.3)
    fig.tight_layout(); fig.savefig(OUT / 'granularity_sweep.png', dpi=100); plt.close()
    log('  wrote granularity_sweep.png')

    # Pick best granularity (highest test_acc)
    best_label = max(sweep_results, key=lambda l: sweep_results[l]['test'])
    best_n = sweep_results[best_label]['n_classes']
    log(f'[best granularity] {best_label} (n_classes={best_n}, test={sweep_results[best_label]["test"]:.3f})')

    # ============================================================
    # (3) raw |CSI| vs BVP at the best granularity
    # ============================================================
    log('=== (3) raw |CSI| vs BVP at best granularity ===')
    grp_best = GRANULARITIES[best_label]
    y_best = grp_best[y27]

    log('  [raw] training LeNet on raw magnitude spectrogram...')
    res_raw = train_one(X_raw_mag, y_best, ds, dom, best_n,
                        f'raw_mag_{best_label}', epochs=30)
    log('  [bvp] training LeNet on BVP at same granularity...')
    res_bvp = train_one(bvp, y_best, ds, dom, best_n,
                        f'bvp_{best_label}', epochs=30)

    # Plot train curves
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for ax, res, name in zip(axes, [res_raw, res_bvp], ['raw |CSI| magnitude', 'BVP']):
        h = res['history']
        ax.plot(h['epoch'], h['train_acc'], 'o-', label='train')
        ax.plot(h['epoch'], h['val_acc'], 's-', label='val')
        ax.plot(h['epoch'], h['test_acc'], '^-', label='test')
        ax.axhline(1.0 / best_n, color='gray', linestyle='--', alpha=0.5,
                   label=f'random (1/{best_n})')
        ax.set_title(f'{name} (final test={res["final_test"]:.3f})')
        ax.set_xlabel('epoch'); ax.set_ylabel('accuracy')
        ax.set_ylim(0, 1); ax.legend(); ax.grid(alpha=0.3)
    fig.suptitle(f'F-51 (3): raw |CSI| vs BVP at {best_label} (n_classes={best_n})', fontsize=12)
    fig.tight_layout(); fig.savefig(OUT / 'raw_vs_bvp.png', dpi=100); plt.close()
    log('  wrote raw_vs_bvp.png')

    # Save summary JSON
    summary = {
        'best_granularity_label': best_label,
        'best_granularity_n_classes': best_n,
        'granularity_sweep': {k: {'test': v['test'], 'train': v['train'],
                                  'best_val': v['best_val'], 'n_classes': v['n_classes']}
                              for k, v in sweep_results.items()},
        'raw_vs_bvp_at_best_granularity': {
            'raw_mag': {'final_train': res_raw['final_train'], 'final_test': res_raw['final_test'],
                        'best_val': res_raw['best_val']},
            'bvp':     {'final_train': res_bvp['final_train'], 'final_test': res_bvp['final_test'],
                        'best_val': res_bvp['best_val']},
        },
    }
    (OUT / 'raw_vs_bvp.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    log(f'[done] summary saved. raw_test={res_raw["final_test"]:.3f} bvp_test={res_bvp["final_test"]:.3f}')

    # Close h5 handles
    bvp.close(); X_raw_mag.close(); raw_h5.close()


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--cache', default='_mmfi_full.h5',
                   help='cache file (default: _mmfi_full.h5 v1; v2 use _mmfi_v2.h5)')
    args = p.parse_args()
    sys.modules[__name__].CACHE_PATH = args.cache
    main()