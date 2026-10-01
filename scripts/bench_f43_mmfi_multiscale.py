"""P1-3 · MMFi 补完整 recipe：多尺度（F-24 口径）

方向 A 的 A-2 对照（多 Rx vs WIDAR_ORIG）在 MMFi 上已由 F-40 覆盖：
  F-40 的 rms_agg = hop16 + RMS 聚合 = WIDAR_ORIG 口径
  F-40 的 multirx10 = hop16 + 保留天线

但 A-4 消融「F24_MULTISCALE（多尺度）」在 MMFi 上缺失。本脚本补齐：
  multiscale = hop16, n_fft=64+128 沿 S 维拼接, keep_antenna=True

与 Widar F-24 对齐（F-24 也是 n_fft=64+128 沿 S 拼接 + keep_antenna=True），
唯一差异是 hop（Widar 用 8，此处用 16 与 F-40 的 MMFi 其他变体保持一致；
B15 已证明 hop 无效，故 hop 差异不影响结论）。

变体 × 4 env × 5 seed = 20 run（断点续传）
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
OUT_JSON = os.path.join(BT, 'f43_mmfi_multiscale.json')

ENV_NAMES = ['E1', 'E2', 'E3', 'E4']
SEEDS = [0, 1, 2, 3, 4]
EPOCHS, PATIENCE, BATCH = 30, 8, 64
HOP, N_FFTS = 16, (64, 128)
A_PICK = 10
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def log(m: str) -> None:
    print(m, flush=True)


def multiscale_block(X_complex: np.ndarray, batch_n: int = 64) -> np.ndarray:
    """(N, A, T, S) complex64 -> 多尺度 BVP (N, A, S*2, F_min, T_min) float32."""
    n = X_complex.shape[0]
    out = None
    for i in range(0, n, batch_n):
        chunk = X_complex[i:i + batch_n]
        chunk = np.nan_to_num(chunk, nan=0.0, posinf=0.0, neginf=0.0)
        t = torch.from_numpy(np.ascontiguousarray(chunk)).to(DEVICE)
        specs = []
        for n_fft in N_FFTS:
            s = compute_bvp_gpu(t, n_fft=n_fft, hop=HOP, keep_antenna=True,
                                batch_n=16, return_device='cpu').numpy()
            s = np.nan_to_num(s, nan=0.0, posinf=0.0, neginf=0.0)
            specs.append(s)
        F_min = min(s.shape[-2] for s in specs)
        T_min = min(s.shape[-1] for s in specs)
        b = np.concatenate([s[..., :F_min, :T_min] for s in specs], axis=2)
        if out is None:
            out = np.zeros((n, *b.shape[1:]), dtype=np.float32)
        out[i:i + batch_n] = b
        del t, b, specs
        torch.cuda.empty_cache()
    return out


def main():
    set_perf_flags(tf32=True, matmul_precision="high")
    log(f'[device] {DEVICE}')
    if DEVICE == 'cuda':
        log(f'[GPU] {torch.cuda.get_device_name(0)}, '
            f'free {torch.cuda.mem_get_info(0)[0]/1e9:.2f} GB')

    log(f'\n[data] loading {CACHE}')
    ds = make_mmfi_dataset(CACHE)
    X, y, dom = ds.X, ds.y, ds.domains.astype(np.int64)
    X = X[:, :A_PICK]
    log(f'  trimmed to first {A_PICK} antennas: X={X.shape}  '
        f'n_class={len(set(y.tolist()))}')

    results = json.load(open(OUT_JSON, encoding='utf-8')) if os.path.exists(OUT_JSON) else {}
    log(f'  already done: {len(results)}')

    total = len(ENV_NAMES) * len(SEEDS)
    done = 0
    t_global = time.time()

    for env_name in ENV_NAMES:
        test_env = ENV_NAMES.index(env_name)
        for seed in SEEDS:
            key = f'multiscale_e{env_name}_s{seed}'
            if key in results and 'acc' in results[key]:
                done += 1
                continue

            log(f'\n========== E={env_name}  seed={seed} ==========')
            all_splits = lodo_splits_with_val(ds, val_mode='random',
                                              val_frac=0.2, seed=seed)
            candidates = [s for s in all_splits if s['test_domain'] == test_env]
            split = candidates[0]
            tr_idx, va_idx, te_idx = split['train_idx'], split['val_idx'], split['test_idx']
            n_tr, n_va, n_te = len(tr_idx), len(va_idx), len(te_idx)
            log(f'  split: train={n_tr} val={n_va} test={n_te}')

            needed = np.concatenate([tr_idx, va_idx, te_idx])
            X_sub = X[needed]
            log(f'  多尺度 BVP 计算 (sub={X_sub.shape[0]} hop={HOP} '
                f'n_fft={N_FFTS}) ...')
            t0 = time.time()
            X_bvp = multiscale_block(X_sub)
            log(f'  BVP shape={X_bvp.shape}  mem={X_bvp.nbytes/1e9:.2f} GB  '
                f'用时 {time.time()-t0:.1f}s')

            idx_map = {int(o): i for i, o in enumerate(needed)}
            tr_l = np.array([idx_map[int(i)] for i in tr_idx])
            va_l = np.array([idx_map[int(i)] for i in va_idx])
            te_l = np.array([idx_map[int(i)] for i in te_idx])

            cfg = TrainCfg(epochs=EPOCHS, patience=PATIENCE, batch_size=BATCH,
                           lr=1e-3, weight_decay=1e-4, optimizer='adam',
                           scheduler='cosine', eval_batch_size=128, seed=seed,
                           device=DEVICE, use_amp=(DEVICE == 'cuda'))

            out = train_one(
                Xtr=np.ascontiguousarray(X_bvp[tr_l]), ytr=y[tr_idx],
                Xval=np.ascontiguousarray(X_bvp[va_l]), yval=y[va_idx],
                model_name='lenet_attn', task='classification', cfg=cfg,
                model_kwargs={},
            )
            model = out['model']
            Xte_t = torch.from_numpy(np.ascontiguousarray(X_bvp[te_l])).float().to(DEVICE)
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
                'shape': list(X_bvp.shape), 'n_train': n_tr, 'n_val': n_va,
                'n_test': n_te, 't_sec': time.time() - t0,
            }
            done += 1
            el = time.time() - t_global
            log(f'  [{done:02d}/{total}] {key} acc={acc:.4f} val={best_val:.4f} '
                f'ep={out.get("best_epoch", 0)} ETA={el/max(done,1)*(total-done)/60:.1f}min')
            with open(OUT_JSON, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            del model, Xte_t, X_bvp
            torch.cuda.empty_cache()
            gc.collect()

    # ---- 汇总 ----
    accs = [results[f'multiscale_e{e}_s{s}']['acc']
            for e in ENV_NAMES for s in SEEDS
            if f'multiscale_e{e}_s{s}' in results
            and 'acc' in results[f'multiscale_e{e}_s{s}']]
    if accs:
        log('\n' + '=' * 60)
        log(f'MMFi 多尺度 multiscale: mean={np.mean(accs):.4f} '
            f'± {np.std(accs, ddof=1):.4f}  n={len(accs)}')
    log('[done]')


if __name__ == '__main__':
    main()
