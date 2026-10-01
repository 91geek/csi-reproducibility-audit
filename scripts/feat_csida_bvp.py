# -*- coding: utf-8 -*-
"""
CSIDA BVP 落盘脚本（CSIDA 专用：分批上传避免 OOM）
==================================================

CSIDA X = (2844, 3, 512, 114) complex64 ≈ 3.98 GB
GPU 11.8 GB，wfcslab.extract_bvp_features 会先一次性把 X 装 GPU 再切 batch，
装完再 .float() 临时张量直接 OOM。

本脚本**不依赖** extract_bvp_features，**自己实现分块上传**：
- 每次从 h5 读 nb 样本（nb=64）→ 上传 GPU → 算 BVP → 落回 CPU 拼到内存结果
- 用 np.empty 预分配结果数组，避免反复 cat

每个 recipe 独立成 npy。
"""

from __future__ import annotations

import os
import sys
import argparse
import time
import json
import gc
import logging

import numpy as np
import h5py
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "src"))
sys.path.insert(0, ROOT)

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")
log = logging.getLogger("csida_bvp")


def csida_bvp_chunked(
    X: np.ndarray,           # (n, A, T, S) complex64
    n_fft: int, hop: int, keep_antenna: bool,
    device: str = "cuda",
    batch_n: int = 64,
    n_ffts_for_multiscale: tuple = (64,),
    out_dtype=np.float32,
) -> np.ndarray:
    """CSIDA 友好的 BVP：分批上传 GPU，每批算完直接拼 CPU。"""
    n, A, T, S = X.shape
    if device == "cuda" and not torch.cuda.is_available():
        log.warning("CUDA 不可用，回退 CPU")
        device = "cpu"

    # 先 probe 形状（n=8 GPU 上前 8 个样本）
    if device == "cuda":
        Xt = torch.as_tensor(X[:8], dtype=torch.complex64, device="cuda")
        amp = Xt.abs().to(torch.float32)
        if True:  # dc_removal
            amp = amp - amp.mean(dim=-2, keepdim=True)
        flat = amp.permute(0, 1, 3, 2).reshape(-1, T).contiguous()
        win = torch.hann_window(n_fft, periodic=True, device="cuda", dtype=torch.float32)
        spec = torch.stft(flat, n_fft=n_fft, hop_length=hop, win_length=n_fft,
                          window=win, center=False, return_complex=True)
        F_dim, Tp = spec.shape[-2], spec.shape[-1]
        del Xt, amp, flat, spec, win
        torch.cuda.empty_cache()

    # 多尺度：把每个 n_fft 的 F/T 算出来，取最小
    F_dims, T_dims = [], []
    for nf in n_ffts_for_multiscale:
        F_dims.append(nf // 2 + 1)  # 默认 complex=True 形状 (F, T)
        # T' = 1 + (T - n_fft) / hop
        T_dims.append(1 + (T - nf) // hop)
    F_min, T_min = min(F_dims), min(T_dims)
    S_out = S * len(n_ffts_for_multiscale)

    # 预分配
    if keep_antenna:
        out_shape = (n, A, S_out, F_min, T_min)
    else:
        out_shape = (n, S_out, F_min, T_min)
    log.info("  输出预分配: %s  (%.2f GB float32)", out_shape,
             np.prod(out_shape) * 4 / 1e9)
    out = np.empty(out_shape, dtype=out_dtype)

    # 分批
    t0 = time.time()
    last_t = t0
    for i in range(0, n, batch_n):
        end = min(i + batch_n, n)
        nb = end - i
        if device == "cuda":
            Xt = torch.as_tensor(X[i:end], dtype=torch.complex64, device="cuda")
        else:
            Xt = torch.as_tensor(X[i:end], dtype=torch.complex64, device="cpu")

        # 每个 n_fft 单独算
        scales = []
        for nf in n_ffts_for_multiscale:
            amp = Xt.abs().to(torch.float32)
            if True:  # dc_removal
                amp = amp - amp.mean(dim=-2, keepdim=True)
            flat = amp.permute(0, 1, 3, 2).reshape(nb * A * S, T).contiguous()
            if device == "cuda":
                win = torch.hann_window(nf, periodic=True, device="cuda", dtype=torch.float32)
            else:
                win = torch.hann_window(nf, periodic=True, device="cpu", dtype=torch.float32)
            spec = torch.stft(flat, n_fft=nf, hop_length=hop, win_length=nf,
                              window=win, center=False, return_complex=True)
            spec = spec.abs().pow(2).reshape(nb, A, S, spec.shape[-2], spec.shape[-1])
            spec = spec[..., :F_min, :T_min]   # 对齐
            if keep_antenna:
                spec = torch.log(spec + 1e-9)
            else:
                spec = torch.sqrt((spec ** 2).mean(dim=1))
                spec = torch.log(spec + 1e-9)
            scales.append(spec)
            del amp, flat, win, spec
        # 拼多尺度 — cat 在子载波维(S)而不是 batch 维
        # keep_antenna=True:  每个 scale 是 (nb, A, S, F, T),  cat dim=2 → (nb, A, S*k, F, T)
        # keep_antenna=False: 每个 scale 是 (nb, S, F, T),     cat dim=1 → (nb, S*k, F, T)
        cat_dim = 2 if keep_antenna else 1
        merged = torch.cat(scales, dim=cat_dim)
        del merged, scales, Xt
        if device == "cuda":
            torch.cuda.empty_cache()

        # 进度
        if time.time() - last_t > 3.0 or end == n:
            rate = nb / (time.time() - last_t + 1e-9)
            eta = (n - end) / max(rate, 1e-9)
            log.info("    [%4d/%4d]  %.1f samples/s  ETA %.0fs  (%.1fs elapsed)",
                     end, n, rate, eta, time.time() - t0)
            last_t = time.time()

    log.info("  完成: %.1fs  output shape %s", time.time() - t0, out.shape)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--h5", default=r"F:/python_workspace/wifi识别/data/processed/csida/csi_env.h5")
    p.add_argument("--out_dir", default=r"F:/python_workspace/wifi识别/data/processed/csida")
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--batch_n", type=int, default=64)
    p.add_argument("--max_time", type=int, default=512,
                   help="切到多长帧（CSIDA 原始 1800）")
    p.add_argument("--max_n", type=int, default=None,
                   help="限制最大样本数（调试）")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # 1) 读 cache
    log.info("Loading %s ...", args.h5)
    t0 = time.time()
    with h5py.File(args.h5, "r") as f:
        n_max = int(f["X"].shape[0])
        if args.max_n is not None:
            n_max = min(n_max, args.max_n)
        # 分批读 X
        Xs = []
        for i in range(0, n_max, 1024):
            end = min(i + 1024, n_max)
            Xs.append(np.asarray(f["X"][i:end], dtype=np.complex64))
        X = np.concatenate(Xs, axis=0)
        y = np.asarray(f["y"][:n_max], dtype=np.int64)
        domains = np.asarray(f["domains"][:n_max], dtype=np.int64)
        subjects = np.asarray(f["subjects"][:n_max], dtype=np.int64)
        try:
            env_all = np.asarray(f["env_all"][:n_max], dtype=np.int64)
            loc_all = np.asarray(f["loc_all"][:n_max], dtype=np.int64)
        except KeyError:
            env_all = domains
            loc_all = np.zeros_like(domains)
    log.info("Loaded X=%s (%.2f GB)  %.1fs", X.shape, X.nbytes/1e9, time.time()-t0)

    # 2) 三个 recipe
    recipes = {
        "widar_orig":     dict(n_fft=64,  hop=16, keep_antenna=False, n_ffts=(64,)),
        "f21_multi_rx":   dict(n_fft=64,  hop=8,  keep_antenna=True,  n_ffts=(64,)),
        "f24_multiscale": dict(n_fft=64,  hop=8,  keep_antenna=True,  n_ffts=(64, 128)),
    }

    for name, cfg in recipes.items():
        out_npy = os.path.join(args.out_dir, f"bvp_{name}.npy")
        meta_json = os.path.join(args.out_dir, f"bvp_{name}.meta.json")
        if os.path.exists(out_npy):
            log.info("[%s] 已存在，跳过", name)
            continue
        log.info(">>> %s  n_fft=%s  hop=%d  keep_antenna=%s",
                 name, cfg["n_ffts"], cfg["hop"], cfg["keep_antenna"])
        out = csida_bvp_chunked(
            X,
            n_fft=cfg["n_fft"], hop=cfg["hop"],
            keep_antenna=cfg["keep_antenna"],
            n_ffts_for_multiscale=cfg["n_ffts"],
            device=args.device, batch_n=args.batch_n,
        )
        np.save(out_npy, out.astype(np.float32))
        with open(meta_json, "w", encoding="utf-8") as f:
            json.dump({
                "recipe": name,
                "X_shape_in": list(X.shape),
                "out_shape": list(out.shape),
                "min": float(out.min()), "max": float(out.max()),
                "n_ffts": list(cfg["n_ffts"]),
                "hop": cfg["hop"],
                "keep_antenna": cfg["keep_antenna"],
            }, f, indent=2, ensure_ascii=False)
        log.info("saved: %s  (%.2f GB)", out_npy, out.nbytes/1e9)
        del out
        gc.collect()
        if args.device == "cuda":
            torch.cuda.empty_cache()

    # 3) labels
    labels_npz = os.path.join(args.out_dir, "labels.npz")
    if not os.path.exists(labels_npz):
        np.savez_compressed(
            labels_npz,
            y=y, domains=domains, subjects=subjects,
            env_all=env_all, loc_all=loc_all,
        )
        log.info("saved labels: %s", labels_npz)


if __name__ == "__main__":
    main()
