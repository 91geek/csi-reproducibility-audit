# -*- coding: utf-8 -*-
"""F-40 v2 · MMFi 跨数据集验证（修正输入：A=10 物理天线）

MMFi 的"30 通道"实际是 3 瞬时帧 × 10 物理天线 reshape 而成。
前 3 通道只含 3 个非零时间样本，喂 BVP STFT 会得到垃圾。
⇒ 只用前 10 通道（10 物理天线），与 Widar3.0 (9 vs 1) 真正对等。

3 变体 × 4 fold × 3 seed = 36 run
  V1 multirx10  (N, 10, S, F, T')  10 物理天线原样保留
  V2 rms_agg    (N,  1, S, F, T')  10 → 1 RMS
  V3 single_ch  (N,  1, S, F, T')  只取第 0 物理天线
"""
from __future__ import annotations

import gc
import json
import os
import sys
import time

import numpy as np
import torch
from scipy import stats as scs

sys.path.insert(0, 'src')

from wfcslab.data.mmfi import make_mmfi_dataset
from wfcslab.data.base import lodo_splits_with_val
from wfcslab.engine.trainer import (TrainCfg, train_one, set_perf_flags,
                                    _batched_predict)
from wfcslab.signal.doppler_gpu import compute_bvp_gpu

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BT = os.path.join(ROOT, 'src/bvp_test')
CACHE = os.path.join(BT, '_mmfi_full.h5')
OUT_JSON = os.path.join(BT, 'f40_mmfi_ablation.json')

ENV_NAMES = ['E1', 'E2', 'E3', 'E4']
SEEDS = [0, 1, 2, 3, 4]
EPOCHS, PATIENCE, BATCH = 30, 8, 64
HOP, N_FFT = 16, 64
A_PICK = 10                       # 只取前 10 物理天线（忽略前 3 通道的瞬时帧）
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def log(m: str) -> None:
    print(m, flush=True)


def bvp_block(X_complex: np.ndarray, batch_n: int = 96) -> np.ndarray:
    """(N, A, T, S) complex64 -> (N, A, S, F, T') float32."""
    n = X_complex.shape[0]
    out = None
    for i in range(0, n, batch_n):
        chunk = X_complex[i:i + batch_n]
        # NaN/Inf 保护：CSIphase 含 NaN，cache 时已污染
        chunk = np.nan_to_num(chunk, nan=0.0, posinf=0.0, neginf=0.0)
        t = torch.from_numpy(np.ascontiguousarray(chunk)).to(DEVICE)
        b = compute_bvp_gpu(t, n_fft=N_FFT, hop=HOP,
                            keep_antenna=True, batch_n=32, return_device='cpu').numpy()
        b = np.nan_to_num(b, nan=0.0, posinf=0.0, neginf=0.0)
        if out is None:
            out = np.zeros((n, *b.shape[1:]), dtype=np.float32)
        out[i:i + batch_n] = b
        del t, b
        torch.cuda.empty_cache()
    return out


def to_variants(X_bvp: np.ndarray):
    return {
        'multirx10': X_bvp.copy(),
        'rms_agg': np.sqrt((X_bvp ** 2).mean(axis=1, keepdims=True)),
        'single_ch': X_bvp[:, 0:1],
    }


def main():
    set_perf_flags(tf32=True, matmul_precision="high")
    log(f'[device] {DEVICE}')
    if DEVICE == 'cuda':
        log(f'[GPU] {torch.cuda.get_device_name(0)}, free {torch.cuda.mem_get_info(0)[0]/1e9:.2f} GB')

    log(f'\n[data] loading {CACHE}')
    ds = make_mmfi_dataset(CACHE)
    X, y, dom = ds.X, ds.y, ds.domains.astype(np.int64)
    log(f'  raw X={X.shape}  dtype={X.dtype}  n_class={len(set(y.tolist()))}  '
        f'dom={sorted(set(dom.tolist()))}')

    # 只取前 A_PICK 通道（10 物理天线）
    X = X[:, :A_PICK]
    log(f'  trimmed to first {A_PICK} antennas: X={X.shape}')

    results = json.load(open(OUT_JSON, encoding='utf-8')) if os.path.exists(OUT_JSON) else {}
    log(f'  already done: {len(results)}')

    for env_name in ENV_NAMES:
        test_env = ENV_NAMES.index(env_name)
        for seed in SEEDS:
            todo = [v for v in ('multirx10', 'rms_agg', 'single_ch')
                    if f'{v}_e{env_name}_s{seed}' not in results
                    or 'acc' not in results[f'{v}_e{env_name}_s{seed}']]
            if not todo:
                continue

            log(f'\n========== E={env_name}  seed={seed}  ({len(todo)}/3 to do) ==========')
            all_splits = lodo_splits_with_val(ds, val_mode='random', val_frac=0.2, seed=seed)
            candidates = [s for s in all_splits if s['test_domain'] == test_env]
            assert candidates, f"E={env_name} 找不到 split"
            split = candidates[0]
            tr_idx, va_idx, te_idx = split['train_idx'], split['val_idx'], split['test_idx']
            n_tr, n_va, n_te = len(tr_idx), len(va_idx), len(te_idx)
            log(f'  split: train={n_tr} val={n_va} test={n_te}')

            needed = np.concatenate([tr_idx, va_idx, te_idx])
            X_sub = X[needed]
            log(f'  BVP 计算 (sub={X_sub.shape[0]}  hop={HOP} n_fft={N_FFT}) ...')
            t0 = time.time()
            X_bvp_sub = bvp_block(X_sub)
            log(f'  BVP shape={X_bvp_sub.shape}  mem={X_bvp_sub.nbytes/1e9:.2f} GB  '
                f'用时 {time.time()-t0:.1f}s')

            idx_map = {int(o): i for i, o in enumerate(needed)}
            tr_idx_l = np.array([idx_map[int(i)] for i in tr_idx])
            va_idx_l = np.array([idx_map[int(i)] for i in va_idx])
            te_idx_l = np.array([idx_map[int(i)] for i in te_idx])

            cfg = TrainCfg(epochs=EPOCHS, patience=PATIENCE, batch_size=BATCH, lr=1e-3,
                           weight_decay=1e-4, optimizer='adam', scheduler='cosine',
                           eval_batch_size=128, seed=seed, device=DEVICE,
                           use_amp=(DEVICE == 'cuda'))

            for vname in ('multirx10', 'rms_agg', 'single_ch'):
                key = f'{vname}_e{env_name}_s{seed}'
                if key in results and 'acc' in results[key]:
                    continue
                Xv = to_variants(X_bvp_sub)[vname]
                log(f'  [{key}] X={Xv.shape} ...')
                t0 = time.time()

                out = train_one(
                    Xtr=np.ascontiguousarray(Xv[tr_idx_l]), ytr=y[tr_idx],
                    Xval=np.ascontiguousarray(Xv[va_idx_l]), yval=y[va_idx],
                    model_name='lenet_attn', task='classification', cfg=cfg, model_kwargs={},
                )
                model = out['model']
                Xte_t = torch.from_numpy(np.ascontiguousarray(Xv[te_idx_l])).float().to(DEVICE)
                model.eval()
                with torch.no_grad():
                    logits = _batched_predict(model, Xte_t, batch_size=128,
                                              use_amp=False, amp_dtype=torch.float16)
                pred = logits.argmax(dim=1).cpu().numpy()
                acc = float((pred == y[te_idx]).mean())

                history = out.get('history', [])
                val_accs = [h.get('acc', 0) for h in history if 'acc' in h]
                best_val = max(val_accs) if val_accs else 0.0

                results[key] = {
                    'acc': acc, 'val_acc': best_val,
                    'best_ep': int(out.get('best_epoch', 0)),
                    'shape': list(Xv.shape), 'n_train': n_tr, 'n_val': n_va, 'n_test': n_te,
                    't_sec': time.time() - t0,
                }
                log(f'    -> acc={acc:.4f} val={best_val:.4f} ep={out.get("best_epoch", 0)} '
                    f't={time.time()-t0:.0f}s')
                with open(OUT_JSON, 'w', encoding='utf-8') as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)
                del model, Xte_t
                torch.cuda.empty_cache()
                gc.collect()

            del X_bvp_sub
            gc.collect()

    # ---- 汇总 ----
    log('\n' + '=' * 60)
    log('F-40 MMFi 汇总（A=10 物理天线, 同架构 LeNetCSI_Attn, 只改输入）')
    log('=' * 60)
    VARS = ['multirx10', 'rms_agg', 'single_ch']
    for v in VARS:
        accs = [results[f'{v}_e{e}_s{s}']['acc']
                for e in ENV_NAMES for s in SEEDS
                if f'{v}_e{e}_s{s}' in results and 'acc' in results[f'{v}_e{e}_s{s}']]
        if accs:
            log(f'  {v:<11} mean={np.mean(accs):.4f} ± {np.std(accs, ddof=1):.4f}  n={len(accs)}')
    log('\n[配对检验]')
    for a, b in [('rms_agg', 'multirx10'), ('single_ch', 'multirx10'),
                 ('rms_agg', 'single_ch')]:
        A, B = [], []
        for e in ENV_NAMES:
            for s in SEEDS:
                ka, kb = f'{a}_e{e}_s{s}', f'{b}_e{e}_s{s}'
                if ka in results and kb in results and 'acc' in results[ka]:
                    A.append(results[ka]['acc']); B.append(results[kb]['acc'])
        if len(A) >= 3:
            A, B = np.array(A), np.array(B)
            D = B - A
            t, p = scs.ttest_rel(B, A)
            d = D.mean() / (D.std(ddof=1) + 1e-12)
            log(f'  {b} - {a}:  Δ={D.mean()*100:+.2f}pp  p={p:.4f}  d={d:+.3f}  '
                f'wins/losses={int((D>0).sum())}/{int((D<0).sum())}  n={len(A)}')
    log('[done]')


if __name__ == '__main__':
    main()
