"""F-39 · B10 机制分析 Part B：受控下游消融 —— 天线维信息到底有没有用？

为什么必须做这个？
==================
Part A（数据层统计）给出反直觉结果：多 Rx 表示的**逐元素** Fisher 判别力
只有 RMS 聚合的 0.20×。原因是 RMS 是 9 天线**相干叠加**，有信噪比增益，
单元素当然更强。但那不能回答真正的问题：

    那 9 个天线通道里「非相干 / 互补」的那部分信息，
    下游模型到底能不能利用？

这只能靠**受控下游实验**回答。

实验设计（单变量）
==================
四个变体**全部使用同一个模型 LeNetCSI_Attn**，输入统一成 5D (N, A, S, F, T')，
唯一区别是 A 通道的构造方式：

  A. multirx9   : (N, 9, 30, 33, 25)  9 根天线原样保留       [F-21 的做法]
  B. rms_agg    : (N, 1, 30, 33, 25)  9 根天线 RMS 聚合成 1  [Widar3.0 的做法]
  C. single_rx  : (N, 1, 30, 33, 25)  只取第 1 根天线        [不用聚合、也不用多天线]
  D. h16_rms    : (N, 1, 30, 33, 13)  hop=16 + RMS 聚合      [Widar3.0 原版]

对照逻辑：
  A vs B → 纯「天线聚合方式」效应：同样 hop、同样模型，
           9 通道 vs 聚合 1 通道。★本实验的核心
  A vs C → 纯「多天线」价值：9 根 vs 1 根，都不聚合
  C vs B → 纯「聚合」价值：1 根 vs 9 根聚合
  B vs D → 纯「hop / 时间分辨率」效应：都是 RMS 1 通道

这把 F-34/B10 那个「三因素同时变」的 +8.14pp 彻底拆开。

协议
====
- LODO：6 fold (d1,d3,d5,d6,d7,d9) × 3 seed = 18 组/变体，共 72 组
- 与 F-32/F-33/F-34 完全相同的 TrainCfg（width=16, 30 epoch, patience=8）
- 配对 t 检验 + Cohen's d + Wilcoxon
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
OUT_JSON = os.path.join(BT, 'f39_antenna_ablation.json')

CACHE_RX8 = os.path.join(BT, '_cache_bvp_rx64.npy')              # (N, 9, 30, 33, 25)
CACHE_H8_RMS = os.path.join(BT, '_cache_bvp_h8_rms.npy')         # (N, 30, 33, 25)
CACHE_H16_RMS = os.path.join(BT, '_cache_bvp_widar_orig_h16.npy')  # (N, 30, 33, 13)

FOLDS = [1, 3, 5, 6, 7, 9]
SEEDS = [0, 1, 2, 3, 4]
EPOCHS, PATIENCE = 30, 8
RX_PICK = 0          # single_rx 用第几根天线


def log(m: str) -> None:
    print(m, flush=True)


def build_variants():
    """构造 4 个变体。为省内存，逐个构造、用完释放。"""
    return ["multirx9", "rms_agg", "single_rx", "h16_rms"]


def load_variant(name: str):
    """返回 (X_5d, 描述)。只加载必要的数据。"""
    if name == "multirx9":
        X = np.load(CACHE_RX8, mmap_mode='r')
        X5 = np.asarray(X[:, :, :, :, :])            # (N, 9, 30, 33, 25)
        return X5, "(hop=8, keep_antenna=True) A=9"
    if name == "single_rx":
        X = np.load(CACHE_RX8, mmap_mode='r')
        X5 = np.asarray(X[:, RX_PICK:RX_PICK + 1, :, :, :])   # (N, 1, 30, 33, 25)
        return X5, f"(hop=8, only Rx#{RX_PICK}) A=1"
    if name == "rms_agg":
        X = np.load(CACHE_H8_RMS, mmap_mode='r')
        X5 = np.asarray(X)[:, None, :, :, :]         # (N, 1, 30, 33, 25)
        return X5, "(hop=8, RMS aggregated) A=1"
    if name == "h16_rms":
        X = np.load(CACHE_H16_RMS, mmap_mode='r')
        X5 = np.asarray(X)[:, None, :, :, :]         # (N, 1, 30, 33, 13)
        return X5, "(hop=16, RMS aggregated) A=1"
    raise ValueError(name)


def main():
    global FOLDS, SEEDS
    # 冒烟模式：--smoke 只跑第 1 个 fold × 第 1 个 seed
    if '--smoke' in sys.argv:
        FOLDS = [FOLDS[0]]
        SEEDS = [SEEDS[0]]

    set_perf_flags(tf32=True, matmul_precision="high")
    log(f'[GPU] {torch.cuda.get_device_name(0)}, '
        f'free {torch.cuda.mem_get_info(0)[0]/1e9:.2f} GB')
    if '--smoke' in sys.argv:
        log(f'[smoke] folds={FOLDS} seeds={SEEDS}')

    ds = make_widar_dataset(h5_path=r'F:\python_workspace\wifi识别\data\processed\widar3.0\csi.h5')
    splits = lodo_splits_with_val(ds, val_mode='random', val_frac=0.2, seed=0)
    splits_by_d = {s['test_domain']: s for s in splits}
    log(f'[data] N={len(ds.y)}, folds={FOLDS}')

    results = json.load(open(OUT_JSON, encoding='utf-8')) if os.path.exists(OUT_JSON) else {}
    variants = build_variants()
    total = len(variants) * len(FOLDS) * len(SEEDS)
    t_global = time.time()
    done = 0

    for vname in variants:
        log(f'\n========== 变体 {vname} ==========')
        X5, desc = load_variant(vname)
        log(f'  {desc}  shape={X5.shape}')

        for fold in FOLDS:
            for seed in SEEDS:
                key = f'{vname}_d{fold}_s{seed}'
                if key in results and 'acc' in results[key]:
                    done += 1
                    continue

                sp = splits_by_d[fold]
                tr_i, va_i, te_i = sp['train_idx'], sp['val_idx'], sp['test_idx']
                Xtr = X5[tr_i].astype(np.float32)
                Xva = X5[va_i].astype(np.float32)
                Xte = X5[te_i].astype(np.float32)
                ytr, yva, yte = ds.y[tr_i], ds.y[va_i], ds.y[te_i]

                mu, std = Xtr.mean(), Xtr.std() + 1e-8
                Xtr = (Xtr - mu) / std
                Xva = (Xva - mu) / std
                Xte = (Xte - mu) / std

                cfg = TrainCfg(epochs=EPOCHS, batch_size=64, lr=1e-3, dropout=0.3,
                               patience=PATIENCE, seed=seed, device="cuda",
                               use_amp=True, compile_model=False,
                               verbose=False, show_progress=False)
                torch.cuda.reset_peak_memory_stats()
                t0 = time.time()
                try:
                    res = train_one(Xtr, ytr, Xva, yva, model_name="lenet_attn",
                                    task="classification", cfg=cfg,
                                    model_kwargs={"width": 16})
                    model = res["model"]

                    # ---- 测试集精度（batched，避免显存爆掉）----
                    # 注意：predict() 内部会处理 device / AMP / 分批
                    from wfcslab.engine.trainer import predict
                    out_dim = int(np.max(ytr)) + 1
                    logits = predict(model, Xte, task="classification",
                                     out_dim=out_dim, device="cuda",
                                     batch_size=256, use_amp=True)
                    pred = np.asarray(logits).argmax(axis=1)
                    acc = float((pred == yte).mean())
                    results[key] = {
                        "acc": acc, "val_acc": float(res["best_metric"]),
                        "epoch": int(res["best_epoch"]),
                        "time_s": round(time.time() - t0, 1),
                        "peak_gpu_mb": float(torch.cuda.max_memory_allocated() / 1e6),
                        "fold": fold, "seed": seed, "variant": vname,
                    }
                    done += 1
                    el = time.time() - t_global
                    log(f'  [{done:02d}/{total}] {key} test_acc={acc:.4f} '
                        f'val={res["best_metric"]:.4f} ep={res["best_epoch"]} '
                        f't={time.time()-t0:.0f}s ETA={el/max(done,1)*(total-done)/60:.1f}min')
                    del model, res
                except Exception as e:
                    log(f'  [FAIL] {key}: {e}')
                    results[key] = {"error": str(e), "variant": vname,
                                    "fold": fold, "seed": seed}
                torch.cuda.empty_cache()
                with open(OUT_JSON, 'w', encoding='utf-8') as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)

        del X5
        torch.cuda.empty_cache()

    # ---------------- 统计 ----------------
    log('\n' + '=' * 74)
    log('F-39 汇总：天线维机制消融（同架构 LeNetCSI_Attn，只改输入构造）')
    log('=' * 74)
    for v in variants:
        vs = [results.get(f'{v}_d{f}_s{s}', {}).get('acc')
              for f in FOLDS for s in SEEDS]
        vs = [x for x in vs if x is not None]
        if vs:
            log(f'  {v:<11s} mean={np.mean(vs):.4f} ± {np.std(vs, ddof=1):.4f}  n={len(vs)}')

    def paired(a, b):
        A, B = [], []
        for f in FOLDS:
            for s in SEEDS:
                x = results.get(f'{a}_d{f}_s{s}', {}).get('acc')
                y = results.get(f'{b}_d{f}_s{s}', {}).get('acc')
                if x is not None and y is not None:
                    A.append(x); B.append(y)
        if len(A) < 3:
            return None
        A, B = np.array(A), np.array(B)
        D = B - A
        t, p = scs.ttest_rel(B, A)
        r = {
            "n": len(A), "a_mean": float(A.mean()), "b_mean": float(B.mean()),
            "diff_pp": float(D.mean() * 100), "sd_pp": float(D.std(ddof=1) * 100),
            "p_ttest": float(p),
            "cohens_d": float(D.mean() / (D.std(ddof=1) + 1e-12)),
            "wins": int((D > 0).sum()), "losses": int((D < 0).sum()),
        }
        try:
            w, pw = scs.wilcoxon(B, A)
            r["p_wilcoxon"] = float(pw)
        except Exception:
            pass
        return r

    log('\n[配对检验]  (b - a，正=后者更好)')
    comparisons = [
        ("multirx9", "rms_agg", "★ 纯天线聚合效应：9 通道 vs RMS 聚合 (hop 都=8)"),
        ("multirx9", "single_rx", "纯多天线价值：9 根 vs 1 根（都不聚合）"),
        ("single_rx", "rms_agg", "纯聚合价值：1 根 vs 9 根聚合"),
        ("rms_agg", "h16_rms", "纯 hop 效应：hop8 vs hop16（都 RMS 1 通道）"),
        ("multirx9", "h16_rms", "总效应：F-21 vs Widar3.0 原版（对齐 F-34/B10）"),
    ]
    summary = {}
    for a, b, note in comparisons:
        r = paired(a, b)
        if r is None:
            log(f'  {b} - {a}: 数据不足')
            continue
        summary[f'{b}_vs_{a}'] = {**r, "note": note}
        log(f'  {b:<10s} - {a:<10s}: {r["diff_pp"]:+6.2f} pp  '
            f'p={r["p_ttest"]:.4f}  d={r["cohens_d"]:+.3f}  '
            f'wins/losses={r["wins"]}/{r["losses"]}   # {note}')

    results['_summary'] = summary
    with open(OUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    log(f'\n[done] {OUT_JSON}')


if __name__ == '__main__':
    main()
