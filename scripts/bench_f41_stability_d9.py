"""F-41 · d9 优化崩溃的稳定性改进消融 —— 能否把多 Rx 在难域上"救回来"？

背景
====
B16 确诊：F-39 里 multirx9 在 d9 域上 best_epoch 停在 1（s1/s2），
三 seed 一致地低（0.25/0.23/0.22），而同域 rms_agg 能训到 15-29 轮。
判定为「高容量多通道输入在小/难域上的系统性优化停滞」，而非表示失效。

本实验要回答：这是不是可修复的？四种单变量稳定性改进，哪个能救回 d9？

实验设计（单变量，固定 d9 fold × 3 seed）
========================================
基线来自 F-39 已跑结果（multirx9_d9_s{0,1,2}）：
    lr=1e-3, patience=8, dropout=0.3, scheduler=cosine, 无 warmup

四种改进（相对基线各只改一个因素）：
  A. warmup         : 前 5 epoch 线性 LR 预热（0.1×lr → lr）
  B. low_lr         : lr 1e-3 → 3e-4
  C. high_dropout   : dropout 0.3 → 0.5（降低有效容量）
  D. long_patience  : patience 8 → 16（给更多恢复机会）

若某改进能显著拉升 d9（best_epoch 恢复正常 + acc 提升），
再做第二阶段「全 6 fold 无害性验证」。

协议
====
- 只 d9 fold × 3 seed = 3 组/变体，共 12 组（基线复用 F-39）
- 变体统一 multirx9 输入 (N,9,30,33,25)，架构 LeNetCSI_Attn, width=16
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import torch
from scipy import stats as scs

sys.path.insert(0, 'src')

from wfcslab.data.widar import make_widar_dataset
from wfcslab.data.base import lodo_splits_with_val
from wfcslab.engine.trainer import TrainCfg, train_one, set_perf_flags

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BT = os.path.join(ROOT, 'src/bvp_test')
OUT_JSON = os.path.join(BT, 'f41_stability_d9.json')
F39_JSON = os.path.join(BT, 'f39_antenna_ablation.json')

CACHE_RX8 = os.path.join(BT, '_cache_bvp_rx64.npy')   # (N, 9, 30, 33, 25)

FOLD = 9
SEEDS = [0, 1, 2]

# 配置名 -> (改哪个字段, 值)。基线是 F-39 的 multirx9_d9_*
VARIANTS = {
    "warmup":        dict(warmup_epochs=5, warmup_start_factor=0.1),
    "low_lr":        dict(lr=3e-4),
    "high_dropout":  dict(dropout=0.5),
    "long_patience": dict(patience=16),
}

BASE_CFG = dict(epochs=30, batch_size=64, lr=1e-3, dropout=0.3,
                patience=8, scheduler="cosine")


def log(m: str) -> None:
    print(m, flush=True)


def load_baseline():
    """从 F-39 结果读 multirx9_d9_s{0,1,2} 作为基线。"""
    data = json.load(open(F39_JSON, encoding='utf-8'))
    out = {}
    for s in SEEDS:
        key = f'multirx9_d{FOLD}_s{s}'
        rec = data.get(key, {})
        if 'acc' in rec:
            out[s] = {"acc": rec["acc"], "val_acc": rec.get("val_acc"),
                      "best_ep": rec.get("epoch"), "source": "F39"}
    return out


def main():
    if '--smoke' in sys.argv:
        log('[smoke] 只跑 seed=0')

    set_perf_flags(tf32=True, matmul_precision="high")
    log(f'[GPU] {torch.cuda.get_device_name(0)}, '
        f'free {torch.cuda.mem_get_info(0)[0]/1e9:.2f} GB')

    ds = make_widar_dataset(h5_path=r'F:\python_workspace\wifi识别\data\processed\widar3.0\csi.h5')
    splits = lodo_splits_with_val(ds, val_mode='random', val_frac=0.2, seed=0)
    splits_by_d = {s['test_domain']: s for s in splits}
    sp = splits_by_d[FOLD]
    tr_i, va_i, te_i = sp['train_idx'], sp['val_idx'], sp['test_idx']
    log(f'[data] N={len(ds.y)}, d{FOLD}: train={len(tr_i)} val={len(va_i)} '
        f'test={len(te_i)}')

    X = np.load(CACHE_RX8, mmap_mode='r')
    X5 = np.asarray(X[:, :, :, :, :])   # (N, 9, 30, 33, 25)
    Xtr = X5[tr_i].astype(np.float32)
    Xva = X5[va_i].astype(np.float32)
    Xte = X5[te_i].astype(np.float32)
    ytr, yva, yte = ds.y[tr_i], ds.y[va_i], ds.y[te_i]
    mu, std = Xtr.mean(), Xtr.std() + 1e-8
    Xtr = (Xtr - mu) / std
    Xva = (Xva - mu) / std
    Xte = (Xte - mu) / std
    log(f'[cache] multirx9 Xtr={Xtr.shape} (已标准化)')

    # 基线
    results = {"baseline": load_baseline()}
    if results["baseline"]:
        bs = [results["baseline"][s]["acc"] for s in SEEDS if s in results["baseline"]]
        log(f'[baseline] F-39 multirx9_d9 acc={[round(x,4) for x in bs]}')

    seeds_to_run = SEEDS if '--smoke' not in sys.argv else [SEEDS[0]]
    total = len(VARIANTS) * len(seeds_to_run)
    done = 0
    t_global = time.time()

    for vname, overrides in VARIANTS.items():
        results.setdefault(vname, {})
        for seed in seeds_to_run:
            key = str(seed)
            if key in results[vname] and 'acc' in results[vname][key]:
                done += 1
                continue
            merged = {**BASE_CFG, **overrides}
            cfg = TrainCfg(seed=seed, device="cuda", use_amp=True,
                           compile_model=False, verbose=False,
                           show_progress=False, **merged)
            torch.cuda.reset_peak_memory_stats()
            t0 = time.time()
            try:
                res = train_one(Xtr, ytr, Xva, yva, model_name="lenet_attn",
                                task="classification", cfg=cfg,
                                model_kwargs={"width": 16})
                model = res["model"]
                from wfcslab.engine.trainer import predict
                out_dim = int(np.max(ytr)) + 1
                logits = predict(model, Xte, task="classification",
                                 out_dim=out_dim, device="cuda",
                                 batch_size=256, use_amp=True)
                pred = np.asarray(logits).argmax(axis=1)
                acc = float((pred == yte).mean())
                results[vname][key] = {
                    "acc": acc, "val_acc": float(res["best_metric"]),
                    "best_ep": int(res["best_epoch"]),
                    "time_s": round(time.time() - t0, 1),
                    "cfg": {**BASE_CFG, **overrides},
                }
                done += 1
                el = time.time() - t_global
                log(f'  [{done:02d}/{total}] {vname}_s{seed} test_acc={acc:.4f} '
                    f'val={res["best_metric"]:.4f} ep={res["best_epoch"]} '
                    f't={time.time()-t0:.0f}s')
                del model, res
            except Exception as e:
                log(f'  [FAIL] {vname}_s{seed}: {e}')
                results[vname][key] = {"error": str(e)}
            torch.cuda.empty_cache()
            with open(OUT_JSON, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)

    # ---------------- 统计 ----------------
    log('\n' + '=' * 74)
    log('F-41 d9 稳定性消融：四种改进 vs 基线（multirx9, 只 d9 fold）')
    log('=' * 74)
    base = results["baseline"]
    base_acc = np.array([base[s]["acc"] for s in SEEDS if s in base])
    base_ep = [base[s]["best_ep"] for s in SEEDS if s in base]
    log(f'  {"baseline":<14s} acc={base_acc.mean():.4f}±{base_acc.std(ddof=1):.4f}  '
        f'best_ep={base_ep}')

    summary = {}
    for vname in VARIANTS:
        rec = results.get(vname, {})
        accs = [rec[s]["acc"] for s in SEEDS if s in rec and "acc" in rec]
        eps = [rec[s]["best_ep"] for s in SEEDS if s in rec and "acc" in rec]
        if not accs:
            log(f'  {vname:<14s} 数据不足')
            continue
        accs = np.array(accs)
        log(f'  {vname:<14s} acc={accs.mean():.4f}±{accs.std(ddof=1):.4f}  '
            f'best_ep={eps}')
        # 配对 vs baseline
        pairs = [(rec[s]["acc"], base[s]["acc"]) for s in SEEDS
                 if s in rec and "acc" in rec and s in base]
        if len(pairs) >= 2:
            A = np.array([p[1] for p in pairs])   # baseline
            B = np.array([p[0] for p in pairs])   # 改进
            D = B - A
            t, p = scs.ttest_rel(B, A)
            r = {
                "n": len(A), "base_mean": float(A.mean()),
                "variant_mean": float(B.mean()),
                "diff_pp": float(D.mean() * 100),
                "p_ttest": float(p),
                "cohens_d": float(D.mean() / (D.std(ddof=1) + 1e-12)),
                "wins": int((D > 0).sum()), "losses": int((D < 0).sum()),
            }
            summary[vname] = r
            log(f'    vs baseline: Δ={r["diff_pp"]:+6.2f}pp  p={r["p_ttest"]:.4f}  '
                f'd={r["cohens_d"]:+.3f}  wins/losses={r["wins"]}/{r["losses"]}')

    results['_summary'] = summary
    with open(OUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    log(f'\n[done] {OUT_JSON}')


if __name__ == '__main__':
    main()
