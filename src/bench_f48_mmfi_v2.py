"""
F-48 v2: Paired LeNetCSI_Attn vs CBAM+ResNet18 on MMFi (16447-segment natural distribution).

Goal: paper §5 core numbers — Δ(LeNet − CBAM) on the v2 real data, paired t-test
      over multiple seeds.

Protocol:
  - 1 env (E2 LODO; same as F-49 v2 / F-50 v2 / F-51 v2 for continuity)
  - 3 seeds (0, 1, 2) × 2 backbones (LeNetCSI_Attn, CBAM+ResNet18)
  - 1 input variant (multirx10 = first 10 antennas)
  - BVP input from `_mmfi_v2_bvp.h5` (reused from F-49/F-50)
  - 6 runs total (~5h on RTX 2080 Ti)

Outputs (saved to src/bvp_test/f48_v2/):
  - per_seed.json       : {seed: {backbone: {test_acc, val_acc, train_time}}}
  - paired_ttest.txt    : t-statistic, p-value, Cohen's d, CI95
  - F48_V2_REPORT.md    : consolidated numbers + paper §5 suggestion

Statistical convention:
  - LeNet − CBAM > 0 means LeNet better (consistent with 50k Widar)
  - LeNet − CBAM < 0 means CBAM better (the v2 reversal hypothesis)
"""
from __future__ import annotations
import json, time, sys, argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import h5py

sys.path.insert(0, str(Path(__file__).parent))
from wfcslab.data.mmfi import make_mmfi_dataset
from wfcslab.models.backbones import LeNetCSI_Attn, CBAMResNet18
from wfcslab.engine.trainer import set_perf_flags
from torch.utils.data import DataLoader, Dataset

set_perf_flags()
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

OUT = Path(__file__).parent / 'bvp_test' / 'f48_v2'
OUT.mkdir(parents=True, exist_ok=True)
LOG = OUT / '_run.log'

TEST_ENV = 'E2'
SEEDS = [0, 1, 2]
BACKBONES = ['LeNetCSI_Attn', 'CBAMResNet18']
EPOCHS = 30
PATIENCE = 8
BATCH = 64
LR = 1e-3
WD = 1e-4
EVAL_BS = 64


def log(m: str) -> None:
    ts = time.strftime('%H:%M:%S')
    line = f'[{ts}] {m}'
    print(line, flush=True)
    with LOG.open('a', encoding='utf-8') as f:
        f.write(line + '\n')


class _BvpLazy(Dataset):
    def __init__(self, h5_path, indices, labels):
        self.f = h5py.File(h5_path, 'r')
        self.indices = np.asarray(indices, dtype=np.int64)
        self.labels = np.asarray(labels, dtype=np.int64)
    def __len__(self):
        return len(self.indices)
    def __getitem__(self, k):
        i = int(self.indices[k])
        x = self.f['bvp'][i]
        return torch.from_numpy(np.ascontiguousarray(x)), int(self.labels[k])
    def close(self):
        self.f.close()


def lodo_split(ds, dom):
    """E2 LODO; 20% of E2 as val, rest as test."""
    test_env_idx = ds.domain_names.index(TEST_ENV)
    train_mask = (dom != test_env_idx)
    val_mask = (dom == test_env_idx)
    rng = np.random.RandomState(0)  # split is deterministic across seeds
    val_indices = rng.choice(np.where(val_mask)[0], size=int(val_mask.sum() * 0.2), replace=False)
    val_set = set(val_indices.tolist())
    test_indices = np.array([i for i in np.where(val_mask)[0] if i not in val_set])
    train_indices = np.where(train_mask)[0]
    return train_indices, val_indices, test_indices


def train_one(bvp_h5_path, y, dom, ds, n_classes, backbone, seed, in_A, in_S, in_F, in_T):
    torch.manual_seed(seed); np.random.seed(seed)
    train_idx, val_idx, test_idx = lodo_split(ds, dom)
    log(f'  split: train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}')

    tr_ds = _BvpLazy(bvp_h5_path, train_idx, y[train_idx])
    va_ds = _BvpLazy(bvp_h5_path, val_idx,   y[val_idx])
    te_ds = _BvpLazy(bvp_h5_path, test_idx,  y[test_idx])

    tr_ld = DataLoader(tr_ds, batch_size=BATCH, shuffle=True, drop_last=False)
    va_ld = DataLoader(va_ds, batch_size=BATCH, shuffle=False)
    te_ld = DataLoader(te_ds, batch_size=BATCH, shuffle=False)

    if backbone == 'LeNetCSI_Attn':
        model = LeNetCSI_Attn(in_antennas=in_A, in_subcarriers=in_S,
                              in_freq=in_F, in_time=in_T,
                              out_dim=n_classes).to(DEVICE)
    elif backbone == 'CBAMResNet18':
        model = CBAMResNet18(in_antennas=in_A, in_subcarriers=in_S,
                             in_freq=in_F, in_time=in_T,
                             out_dim=n_classes, width=64, hidden=256, dropout=0.4).to(DEVICE)
    else:
        raise ValueError(backbone)

    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    scaler = torch.amp.GradScaler('cuda', enabled=(DEVICE == 'cuda'))
    crit = torch.nn.CrossEntropyLoss()

    history = {'epoch': [], 'train_acc': [], 'val_acc': [], 'test_acc': []}
    best_val = 0.0; best_state = None; pat = 0
    t0 = time.time()
    for ep in range(EPOCHS):
        model.train(); n_corr = 0; n_tot = 0
        for xb, yb in tr_ld:
            xb = xb.to(DEVICE); yb = torch.tensor(yb, dtype=torch.long, device=DEVICE)
            opt.zero_grad()
            with torch.amp.autocast('cuda', enabled=(DEVICE == 'cuda')):
                out = model(xb); loss = crit(out, yb)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update()
            n_corr += (out.argmax(1) == yb).sum().item(); n_tot += yb.size(0)
        train_acc = n_corr / n_tot
        sched.step()

        # Val + test eval
        model.eval(); va_corr = 0; va_tot = 0; te_corr = 0; te_tot = 0
        with torch.no_grad():
            for xb, yb in va_ld:
                xb = xb.to(DEVICE); yb = torch.tensor(yb, dtype=torch.long, device=DEVICE)
                with torch.amp.autocast('cuda', enabled=(DEVICE == 'cuda')):
                    out = model(xb)
                va_corr += (out.argmax(1) == yb).sum().item(); va_tot += yb.size(0)
            for xb, yb in te_ld:
                xb = xb.to(DEVICE); yb = torch.tensor(yb, dtype=torch.long, device=DEVICE)
                with torch.amp.autocast('cuda', enabled=(DEVICE == 'cuda')):
                    out = model(xb)
                te_corr += (out.argmax(1) == yb).sum().item(); te_tot += yb.size(0)
        val_acc = va_corr / va_tot; test_acc = te_corr / te_tot

        history['epoch'].append(ep); history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc); history['test_acc'].append(test_acc)

        if val_acc > best_val:
            best_val = val_acc; best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            pat = 0
        else:
            pat += 1
            if pat >= PATIENCE: break

    # Restore best state and re-evaluate test (best-val-state convention)
    if best_state is not None:
        model.load_state_dict({k: v.to(DEVICE) for k, v in best_state.items()})
    model.eval(); te_corr = 0; te_tot = 0
    with torch.no_grad():
        for xb, yb in te_ld:
            xb = xb.to(DEVICE); yb = torch.tensor(yb, dtype=torch.long, device=DEVICE)
            with torch.amp.autocast('cuda', enabled=(DEVICE == 'cuda')):
                out = model(xb)
            te_corr += (out.argmax(1) == yb).sum().item(); te_tot += yb.size(0)
    final_test = te_corr / te_tot

    tr_ds.close(); va_ds.close(); te_ds.close()
    train_dt = time.time() - t0
    log(f'  [{backbone} seed={seed}] best_val={best_val:.3f} '
        f'best_state_test={final_test:.3f} params={n_params/1e6:.2f}M train={train_dt:.0f}s')
    return {'best_val': best_val, 'test': final_test,
            'train_time_sec': train_dt, 'n_params': n_params, 'history': history}


def main():
    if LOG.exists(): LOG.unlink()
    cache_name = getattr(sys.modules[__name__], 'CACHE_PATH', '_mmfi_v2.h5')
    log(f'[start] F-48 v2 paired LeNet vs CBAM on MMFi real data')
    log(f'[cache] {cache_name}')

    bvp_h5_path = Path(__file__).parent / 'bvp_test' / cache_name.replace('.h5', '_bvp.h5')
    if not bvp_h5_path.exists():
        log(f'[ERROR] {bvp_h5_path} not found; run F-49 v2 first to build BVP cache')
        sys.exit(1)

    log(f'[bvp] reusing {bvp_h5_path.name}')
    f = h5py.File(bvp_h5_path, 'r')
    sample_shape = f['bvp'].shape[1:]  # (A=10, S=114, F=33, T=29)
    f.close()
    in_A, in_S, in_F, in_T = sample_shape
    log(f'  shape={sample_shape}')

    cache_path = Path(__file__).parent / 'bvp_test' / cache_name
    ds = make_mmfi_dataset(cache_path)
    y = np.array(ds.y); dom = np.array(ds.domains)
    n_classes = int(y.max()) + 1
    log(f'  y={y.shape} dom={dom.shape} n_class={n_classes}')

    # Run all combinations
    results = {}  # {seed: {backbone: {test, val, train_time}}}
    for seed in SEEDS:
        results[seed] = {}
        for backbone in BACKBONES:
            log(f'\n========== seed={seed} backbone={backbone} ==========')
            res = train_one(bvp_h5_path.as_posix(), y, dom, ds, n_classes, backbone, seed,
                            in_A, in_S, in_F, in_T)
            results[seed][backbone] = {
                'test': res['test'], 'val': res['best_val'],
                'train_time_sec': res['train_time_sec'],
                'n_params_M': res['n_params'] / 1e6,
                'history': res['history'],
            }
            # Save after each run (resume-safe)
            (OUT / 'per_seed.json').write_text(json.dumps(results, indent=2), encoding='utf-8')

    # Paired t-test
    deltas = []
    for seed in SEEDS:
        d = results[seed]['LeNetCSI_Attn']['test'] - results[seed]['CBAMResNet18']['test']
        deltas.append(d)
        log(f'  seed={seed}: LeNet={results[seed]["LeNetCSI_Attn"]["test"]:.4f} '
            f'CBAM={results[seed]["CBAMResNet18"]["test"]:.4f} Δ(LeNet−CBAM)={d:+.4f}')
    deltas = np.array(deltas)
    mean_d = float(deltas.mean())
    std_d = float(deltas.std(ddof=1)) if len(deltas) > 1 else 0.0
    n = len(deltas)
    if std_d > 0:
        from scipy import stats
        t_stat, p_val = stats.ttest_1samp(deltas, popmean=0)
        # Cohen's d for paired = mean(d) / std(d)
        d_eff = mean_d / std_d
        # CI95 for mean
        if n > 1:
            t_crit = stats.t.ppf(0.975, df=n - 1)
            ci_low = mean_d - t_crit * std_d / np.sqrt(n)
            ci_high = mean_d + t_crit * std_d / np.sqrt(n)
        else:
            ci_low = ci_high = mean_d
    else:
        t_stat, p_val, d_eff = 0.0, 1.0, 0.0
        ci_low = ci_high = mean_d

    summary = {
        'n_seeds': n,
        'delta_lenet_minus_cbam_mean': mean_d,
        'delta_lenet_minus_cbam_std': std_d,
        't_statistic': float(t_stat),
        'p_value': float(p_val),
        'cohens_d_paired': float(d_eff),
        'ci95_low': float(ci_low),
        'ci95_high': float(ci_high),
        'interpretation': (
            'LeNet > CBAM' if mean_d > 0 and p_val < 0.05 else
            ('CBAM > LeNet' if mean_d < 0 and p_val < 0.05 else
             'no significant difference at α=0.05')
        ),
        'raw_deltas': deltas.tolist(),
        'per_seed': {str(s): results[s] for s in SEEDS},
    }
    (OUT / 'paired_ttest.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    log(f'\n=== PAIRED T-TEST (n={n} seeds, E2 LODO, multirx10) ===')
    log(f'  Δ(LeNet − CBAM) mean = {mean_d:+.4f}  std = {std_d:.4f}')
    log(f'  t = {t_stat:.3f}  p = {p_val:.4f}  Cohen\'s d = {d_eff:+.3f}')
    log(f'  CI95 = [{ci_low:+.4f}, {ci_high:+.4f}]')
    log(f'  → {summary["interpretation"]}')

    # Plot per-seed deltas
    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 4))
        for i, d in enumerate(deltas):
            color = 'steelblue' if d > 0 else 'darkorange'
            ax.bar(i, d, color=color, alpha=0.8)
            ax.text(i, d + (0.005 if d > 0 else -0.015), f'{d:+.3f}', ha='center', fontsize=9)
        ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
        ax.axhline(mean_d, color='red', linestyle='-', alpha=0.5, label=f'mean Δ={mean_d:+.3f}')
        ax.set_xticks(range(n)); ax.set_xticklabels([f'seed {s}' for s in SEEDS])
        ax.set_ylabel('Δ(LeNet − CBAM) test_acc')
        ax.set_title(f'F-48 v2: paired LeNet vs CBAM on MMFi 16447 seg (E2 LODO)\n'
                     f't={t_stat:.2f} p={p_val:.3f} d={d_eff:+.2f} CI95=[{ci_low:+.3f},{ci_high:+.3f}]')
        ax.legend(); ax.grid(axis='y', alpha=0.3)
        fig.tight_layout(); fig.savefig(OUT / 'paired_deltas.png', dpi=100); plt.close()
        log(f'  wrote paired_deltas.png')
    except Exception as e:
        log(f'  [plot warn] {e}')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--cache', default='_mmfi_v2.h5')
    args = p.parse_args()
    sys.modules[__name__].CACHE_PATH = args.cache
    main()
