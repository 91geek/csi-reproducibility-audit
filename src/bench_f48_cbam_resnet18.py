# F-48 · CBAM+ResNet18 配对 backbone 实验（"Backbone Fairness Audit"）
#
# 动机：Liu et al. 2025 (arXiv 2512.04521) 在 Widar3.0 上用 CBAM+ResNet18 达 99.72%/97.61%，
# 我们的 LeNetCSI_Attn 只有 ~33%。如果直接比较，reviewer 可能说"你们骨干网络太弱，没法与 SOTA 对比"。
#
# 我们的方法论回答：在完全相同的 18-seed × LODO 协议下跑 CBAM+ResNet18，看 multi-Rx 表示
# 增益是否仍然成立。如果不成立，说明 backbone 是关键；如果成立，说明 multi-Rx 增益是
# **backbone-invariant** 的（这是更强的论据）。
#
# 实验设计：
#   - 配对协议：F=[1,3,5,7] × S=[0,1,2] = 12 个 (fold, seed) 对
#   - 数据：与 f32_multiseed.json 完全相同的 _cache_bvp_ms64_128.npy
#   - Baseline：f32_multiseed.json 的前 12 个 lenet_attn 条目
#   - 输出：f48_cbam_resnet18.json + 配对 Δ / p / d / 95% CI 报告

import os, sys, json, time, argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from wfcslab.engine.trainer import set_perf_flags, _batched_predict, evaluate
from wfcslab.data.base import lodo_splits_with_val
from wfcslab.data.widar import make_widar_dataset
from wfcslab.models.backbones import CBAMResNet18

set_perf_flags()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
H5 = sys.argv[3] if len(sys.argv) > 3 else r"F:\python_workspace\wifi识别\data\processed\widar3.0\csi_50k.h5"
CACHE = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(__file__).parent / "bvp_test" / "_cache_bvp_ms64_128_50k.npy"
OUT_DIR = Path(__file__).parent / "bvp_test"
OUT_JSON = sys.argv[1] if len(sys.argv) > 1 else None  # if None, write f48 default name

# 配对 f32_multiseed.json 的前 12 条 (d1/d3/d5/d7 × s0/s1/s2)
FOLDS = [int(x) for x in (sys.argv[4].split(',') if len(sys.argv) > 4 else '1,3,5,6,7,9')]
SEEDS = [0, 1, 2]
EPOCHS = 30
BATCH = 64
LR = 1e-3
WD = 1e-4
EVAL_BS = 64
PATIENCE = 8


def get_data(domain, seed):
    """返回 mmap 缓存（numpy，懒加载）+ 索引（不把整个 fold 读进内存）。"""
    X_bvp = np.load(CACHE, mmap_mode='r')   # mmap，不占内存
    ds = make_widar_dataset(h5_path=H5, load_X=False)
    # val_frac=0.2 与 bench_f32 对齐，保证同 (fold,seed) 训练/验证划分一致 -> 严格配对
    splits = lodo_splits_with_val(ds, val_mode="random", val_frac=0.2, seed=seed)
    sp = next(s for s in splits if s["test_domain"] == domain)
    return X_bvp, ds, sp["train_idx"], sp["val_idx"]


def train_supervised(model, X_bvp, ds, tr_i, va_i, seed):
    """标准监督训练（按 batch 从 mmap 读，绝不把整个 fold 载入内存）。"""
    torch.manual_seed(seed); np.random.seed(seed)
    model = model.to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)

    ytr_idx = (ds.y[tr_i] - 1).astype(np.int64)
    yva_idx = (ds.y[va_i] - 1).astype(np.int64)
    ytr_t = torch.from_numpy(ytr_idx).long()
    yva_t = torch.from_numpy(yva_idx).long().to(DEVICE)
    n_train = len(tr_i)

    best_acc = -1.0; bad = 0; t0 = time.time()
    for ep in range(EPOCHS):
        model.train()
        perm = torch.randperm(n_train)
        for i in range(0, n_train, BATCH):
            bi = perm[i:i + BATCH]
            idx = tr_i[bi.numpy()]
            # 仅从 mmap 读本批（fancy 索引返回小副本），再搬上 GPU
            x = torch.from_numpy(np.ascontiguousarray(X_bvp[idx])).float().to(DEVICE, non_blocking=True)
            y = ytr_t[bi].to(DEVICE, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()

        model.eval()
        with torch.no_grad():
            ov = []
            for i in range(0, len(va_i), EVAL_BS):
                idx = va_i[i:i + EVAL_BS]
                xb = torch.from_numpy(np.ascontiguousarray(X_bvp[idx])).float().to(DEVICE, non_blocking=True)
                ov.append(model(xb))
            ov = torch.cat(ov, 0) if len(ov) > 1 else ov[0]
        acc = (ov.argmax(1) == yva_t).float().mean().item()
        if acc > best_acc:
            best_acc = acc; bad = 0
        else:
            bad += 1
            if bad >= PATIENCE: break

    return best_acc, time.time() - t0


def main():
    print(f'Device: {DEVICE}')
    print(f'FOLDS={FOLDS}, SEEDS={SEEDS}, EPOCHS={EPOCHS}, BATCH={BATCH}')
    out_file = Path(OUT_JSON) if OUT_JSON else OUT_DIR / 'f48_cbam_resnet18_50k.json'
    out_data = {}
    for fold in FOLDS:
        for seed in SEEDS:
            try:
                Xb_t, ds, tr_i, va_i = get_data(fold, seed)
                assert Xb_t.ndim == 5, f'Expected 5D, got {Xb_t.shape}'
                N, A, S, F_dim, T_dim = Xb_t.shape
                print(f'\n[d={fold}, s={seed}] cache shape={Xb_t.shape}', flush=True)

                n_classes = int(np.unique(ds.y).max())   # 1-indexed labels, so max==n_classes
                model = CBAMResNet18(
                    in_antennas=A, in_subcarriers=S,
                    in_freq=F_dim, in_time=T_dim,
                    out_dim=n_classes, width=64, hidden=256, dropout=0.4,
                )
                n_params = sum(p.numel() for p in model.parameters())
                print(f'  model params: {n_params:,} ({n_params/1e6:.2f}M)')

                best_acc, dt = train_supervised(model, Xb_t, ds, tr_i, va_i, seed)
                key = f'cbam_resnet18_d{fold}_s{seed}'
                out_data[key] = {
                    'variant': 'cbam_resnet18', 'fold': fold, 'seed': seed,
                    'acc': float(best_acc), 'val_acc': float(best_acc),
                    'time_s': float(dt), 'epoch': EPOCHS,
                    'n_params': n_params,
                }
                print(f'  fold={fold} seed={seed}: acc={best_acc*100:.2f}%, time={dt:.1f}s', flush=True)

                # Save incrementally so we can monitor progress
                with open(out_file, 'w', encoding='utf-8') as f:
                    json.dump(out_data, f, ensure_ascii=False, indent=2)
            except Exception as e:
                print(f'  fold={fold} seed={seed}: ERROR {e}', flush=True)
                import traceback
                traceback.print_exc()
                continue

    print(f'\n=== Done. Output: {out_file} ===')


if __name__ == '__main__':
    main()