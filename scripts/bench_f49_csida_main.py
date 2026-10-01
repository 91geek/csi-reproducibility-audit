# -*- coding: utf-8 -*-
"""
F-49 · CSIDA 主对照实验（跨数据集验证 Part 3）
================================================

为什么用 CSIDA
==============
Widar3.0 与 MMFi 的协议对比已经在 F-39/F-40 跑完，给出关键发现：
- Widar: Multi-Rx 显著 +7.69pp (B10/B15)
- MMFi: Multi-Rx 不显著，多尺度反而 +0.83-1.20pp (B17/C7)
- "最优表示是数据集依赖的" (Dataset-Dependent Optimality)

但只有 2 个数据集下这个结论**。CSIDA 提供关键第三点**：
- A=3（少 3-4×）—— 多 Rx 表示空间受限
- S=114（4× 多）—— 多尺度拼接可消化的频段更多
- env / loc / user 三轴**都**可做 LODO（每个轴都跨 user/act 平衡）
- 6 类动作，分布与 Widar 一样均衡

实验设计
========
3 个 BVP recipe × 2 fold (env LODO) × 3 seed = 18 runs

变体说明
--------
- widar_orig:  hop=16, RMS, n_fft=64     (4D 输入)   → LeNet
- f21_multirx: hop=8,  keep_antenna=True   (5D 输入)   → LeNetCSI_Attn
- f24_multiscale: hop=8, nfft=64+128 (5D 输入)        → LeNetCSI_Attn

主对照
======
C8: f21_multirx vs widar_orig      （多 Rx 价值，A=3）
C9: f24_multiscale vs widar_orig   （多尺度价值）
C10: f24_multiscale vs f21_multirx  （多尺度 vs 多 Rx 在 CSIDA 上）

预期与含义
==========
- 若 C8 不显著：Multi-Rx 在 A=3 上已无额外信息（多 Rx 价值依赖通道数）
- 若 C9 显著：多尺度在 S=114 的细粒度频段上独立有效
- 若 C10 多尺度胜：与 MMFi 一致（Dataset-Dependent Optimality 第 3 个证据点）
- 若 C10 多 Rx 胜：与 Widar 一致（多 Rx 是普适最优）

输出
====
src/bvp_test/f49_csida_main.json
src/bvp_test/f49_csida_summary.md

用法
====
    python scripts/bench_f49_csida_main.py            # 完整跑
    python scripts/bench_f49_csida_main.py --smoke    # 1 fold × 1 seed
    python scripts/bench_f49_csida_main.py --domain-axis loc
    python scripts/bench_f49_csida_main.py --domain-axis user
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict

import numpy as np
import torch
from scipy import stats as scs

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from wfcslab.data.csida import make_csida_dataset
from wfcslab.data.base import lodo_splits_with_val
from wfcslab.engine.trainer import TrainCfg, train_one, set_perf_flags, predict

BT = os.path.join(ROOT, "src/bvp_test")
os.makedirs(BT, exist_ok=True)

DEFAULT_H5 = r"F:/python_workspace/wifi识别/data/processed/csida/csi_env.h5"
BVP_DIR = r"F:/python_workspace/wifi识别/data/processed/csida"

RECIPES = {
    # name on disk    : (npy file suffix,        model_name,         expects_5d)
    "widar_orig":      ("bvp_widar_orig.npy",     "lenet",            False),
    "f21_multirx":     ("bvp_f21_multi_rx.npy",   "lenet_attn",       True),
    "f24_multiscale":  ("bvp_f24_multiscale.npy", "lenet_attn",       True),
}

EPOCHS, PATIENCE = 30, 8


def log(m: str) -> None:
    print(m, flush=True)


def load_bvp(recipe_name: str, force_4d_to_5d: bool = False) -> np.ndarray:
    """读 npy 落盘的 BVP。"""
    npy, _, expects_5d = RECIPES[recipe_name]
    path = os.path.join(BVP_DIR, npy)
    if not os.path.exists(path):
        raise FileNotFoundError(f"缺少 BVP cache: {path}\n请先跑 scripts/feat_csida_bvp.py")
    X = np.load(path, mmap_mode='r')
    X = np.asarray(X, dtype=np.float32)
    if X.ndim == 4 and force_4d_to_5d:
        # (n, S, F, T') -> (n, 1, S, F, T')
        X = X[:, None, :, :, :]
    return X


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--h5", default=DEFAULT_H5)
    p.add_argument("--domain-axis", default="env", choices=["env", "loc", "user"])
    p.add_argument("--folds", default="", help="逗号分隔的 test domain；空=全部")
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--epochs", type=int, default=EPOCHS)
    p.add_argument("--patience", type=int, default=PATIENCE)
    p.add_argument("--out", default=os.path.join(BT, "f49_csida_main.json"))
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()

    set_perf_flags(tf32=True, matmul_precision="high")
    log(f"[GPU] {torch.cuda.get_device_name(0)}, free {torch.cuda.mem_get_info(0)[0]/1e9:.2f} GB")

    # 数据集
    ds = make_csida_dataset(args.h5, domain_axis=args.domain_axis)
    log(ds.summary())
    log(f"[axis] domain_axis={args.domain_axis},  n_folds={ds.n_domains}")

    # LODO 划分
    splits = lodo_splits_with_val(ds, val_mode="random", val_frac=0.2, seed=0)
    splits_by_d = {int(s["test_domain"]): s for s in splits}
    folds = list(splits_by_d.keys())
    if args.folds:
        folds = [int(x) for x in args.folds.split(",") if x.strip()]
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    if args.smoke:
        folds = folds[:1]
        seeds = seeds[:1]
    log(f"[splits] folds={folds}, seeds={seeds}, total={len(folds) * len(seeds) * len(RECIPES)} runs")

    # 已有结果（断点续传）
    results = json.load(open(args.out, encoding="utf-8")) if os.path.exists(args.out) else {}
    total = len(RECIPES) * len(folds) * len(seeds)
    done = sum(1 for k, v in results.items()
               if k != "_summary" and isinstance(v, dict) and "acc" in v)
    t_global = time.time()

    for vname, (_, model_name, expects_5d) in RECIPES.items():
        log(f"\n========== 变体 {vname}  (model={model_name}) ==========")
        X5 = load_bvp(vname, force_4d_to_5d=False)
        if X5.ndim == 4 and expects_5d:
            X5 = X5[:, None, :, :, :]  # 4D -> 5D
        log(f"  X={X5.shape} dtype={X5.dtype}  ({X5.nbytes/1e9:.2f} GB)")

        for fold in folds:
            sp = splits_by_d[fold]
            tr_i, va_i, te_i = sp["train_idx"], sp["val_idx"], sp["test_idx"]
            ytr, yva, yte = ds.y[tr_i], ds.y[va_i], ds.y[te_i]

            for seed in seeds:
                key = f"{vname}_d{fold}_s{seed}"
                if key in results and "acc" in results[key]:
                    done += 1
                    continue

                # 归一化（按 train 集）
                mu = float(X5[tr_i].mean())
                sigma = float(X5[tr_i].std() + 1e-8)
                Xtr = ((X5[tr_i] - mu) / sigma).astype(np.float32)
                Xva = ((X5[va_i] - mu) / sigma).astype(np.float32)
                Xte = ((X5[te_i] - mu) / sigma).astype(np.float32)

                cfg = TrainCfg(
                    epochs=args.epochs, batch_size=64, lr=1e-3, dropout=0.3,
                    patience=args.patience, seed=seed, device="cuda",
                    use_amp=True, compile_model=False,
                    verbose=False, show_progress=False,
                )
                torch.cuda.reset_peak_memory_stats()
                t0 = time.time()
                try:
                    res = train_one(
                        Xtr, ytr, Xva, yva,
                        model_name=model_name, task="classification",
                        cfg=cfg, model_kwargs={"width": 16} if model_name == "lenet_attn" else None,
                    )
                    model = res["model"]
                    out_dim = int(np.max(ytr)) + 1
                    logits = predict(
                        model, Xte, task="classification",
                        out_dim=out_dim, device="cuda",
                        batch_size=256, use_amp=True,
                    )
                    pred = np.asarray(logits).argmax(axis=1)
                    acc = float((pred == yte).mean())
                    results[key] = {
                        "acc": acc,
                        "val_acc": float(res["best_metric"]),
                        "epoch": int(res["best_epoch"]),
                        "time_s": round(time.time() - t0, 1),
                        "peak_gpu_mb": float(torch.cuda.max_memory_allocated() / 1e6),
                        "fold": fold, "seed": seed, "variant": vname,
                        "domain_axis": args.domain_axis,
                    }
                    done += 1
                    el = time.time() - t_global
                    log(f"  [{done:02d}/{total}] {key:<26} test_acc={acc:.4f} "
                        f"val={res['best_metric']:.4f} ep={res['best_epoch']} "
                        f"t={time.time()-t0:.0f}s ETA={el/max(done,1)*(total-done)/60:.1f}min")
                    del model, res
                except Exception as e:
                    log(f"  [FAIL] {key}: {e}")
                    results[key] = {"error": str(e), "variant": vname,
                                    "fold": fold, "seed": seed}
                torch.cuda.empty_cache()
                with open(args.out, "w", encoding="utf-8") as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)

        del X5
        torch.cuda.empty_cache()

    # ---------------- 统计 ----------------
    log("\n" + "=" * 74)
    log(f"F-49 汇总（CSIDA · domain_axis={args.domain_axis}）")
    log("=" * 74)
    for v in RECIPES:
        vs = []
        for f in folds:
            for s in seeds:
                x = results.get(f"{v}_d{f}_s{s}", {}).get("acc")
                if x is not None:
                    vs.append(x)
        if vs:
            log(f"  {v:<18s} mean={np.mean(vs):.4f} ± {np.std(vs, ddof=1):.4f}  n={len(vs)}")

    def paired(a, b):
        A, B = [], []
        for f in folds:
            for s in seeds:
                x = results.get(f"{a}_d{f}_s{s}", {}).get("acc")
                y = results.get(f"{b}_d{f}_s{s}", {}).get("acc")
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

    log("\n[配对检验]  (b - a，正=后者更好)")
    comparisons = [
        ("widar_orig", "f21_multirx",    "C8  多 Rx 价值：F21 vs Widar-orig (A=3)"),
        ("widar_orig", "f24_multiscale", "C9  多尺度价值：F24 vs Widar-orig"),
        ("f21_multirx", "f24_multiscale","C10 多尺度 vs 多 Rx：Dataset-Dep Optimality Test"),
    ]
    summary = {"domain_axis": args.domain_axis, "folds": folds, "seeds": seeds}
    for a, b, note in comparisons:
        r = paired(a, b)
        if r is None:
            log(f"  {b} - {a}: 数据不足")
            continue
        summary[f"{b}_vs_{a}"] = {**r, "note": note}
        log(f"  {b:<14s} - {a:<14s}: {r['diff_pp']:+6.2f} pp  "
            f"p={r['p_ttest']:.4f}  d={r['cohens_d']:+.3f}  "
            f"wins/losses={r['wins']}/{r['losses']}   # {note}")

    # 加按域（fold）拆分的结果，验证域难度
    by_fold = {}
    for v in RECIPES:
        per = {}
        for f in folds:
            xs = [results.get(f"{v}_d{f}_s{s}", {}).get("acc")
                  for s in seeds]
            xs = [x for x in xs if x is not None]
            if xs:
                per[f] = {"mean": float(np.mean(xs)), "n": len(xs)}
        by_fold[v] = per
    summary["by_fold"] = by_fold

    results["_summary"] = summary
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    log(f"\n[done] {args.out}")


if __name__ == "__main__":
    main()
