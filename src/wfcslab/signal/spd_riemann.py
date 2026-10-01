"""
SPD Riemannian 协方差 + Log-Euclidean 投影 —— 几何先验替代统计对齐。

动机
----
F-03~F-04 ComBat / F-17~F-18 EA / F-19~F-20 KNN-MMD 全部反指，根因是：
统计对齐破坏协方差矩阵的**几何结构**。

跨房间 WiFi CSI 的真实变化规律：
- 房间 A: 协方差矩阵 C_A = U diag(λ_A) U^T
- 房间 B: 协方差矩阵 C_B = U diag(λ_B) U^T
- **特征向量 U 不变**（空间几何），**特征值 λ 变化**（能量）
- ComBat/EA 改变了特征向量 → 反指
- KNN-MMD 在 PCA 空间也无法恢复 → 反指
- SPD Riemannian：**只对齐特征值（log-euclidean mean）**，保留特征向量 → 解

参考论文
--------
- [2608.16134] MRieHy: Riemannian 均值对齐多天分布
- [2606.25456] GBWAtt: Bures-Wasserstein SPD 自注意力
- [2607.01279] I²RiMA: 频点 SPD + 切线空间 + 时序注意力

实现
----
对每个 (n, F_band)，把 (A, S*T) reshape 成 (A, samples) → 计算 (A, A) 协方差 → logm → 提取上三角 → (n, F, A*(A+1)/2) 特征
"""
from __future__ import annotations

import torch

__all__ = ["spd_covariance_per_freq", "logm_spd", "spd_upper_triangle", "compute_spd_features"]


def spd_covariance_per_freq(
    x: torch.Tensor,                # (n, A, S, F, T') float32
    eps: float = 1e-6,
) -> torch.Tensor:
    """对每个频率带 f 计算 (A, A) 协方差矩阵。

    输入形状
    --------
    (n, A=9, S=30, F=33, T'=25)

    计算方式
    --------
    把每个频率带 f 看作一个"通道组"：
    - 沿 S 和 T' 维收集样本（共 S*T'=30*25=750 个）
    - 沿 A 维（9 个天线）计算协方差
    - 公式：C = (1/(S*T'-1)) * Σ (X_s - μ)(X_s - μ)^T

    返回
    ----
    cov : (n, F=33, A=9, A=9) float32，对称正定（SPD）矩阵
    """
    n, A, S, F, Tp = x.shape
    # 重新排列：把 S 和 T' 合并为"样本"维
    # 目标: (n, F, A, samples)  其中 samples = S*Tp = 750
    x_perm = x.permute(0, 3, 1, 2, 4).reshape(n, F, A, S * Tp).contiguous()  # (n, F, A, samples)

    # 去均值
    mu = x_perm.mean(dim=-1, keepdim=True)                            # (n, F, A, 1)
    x_centered = x_perm - mu                                            # (n, F, A, samples)

    # 协方差: C = (1/(N-1)) X_centered X_centered^T
    cov = torch.einsum("nfat,nfbt->nfab", x_centered, x_centered) / max(S * Tp - 1, 1)
    # cov: (n, F, A, A)

    # 加正则保证 SPD
    eye = torch.eye(A, device=x.device, dtype=x.dtype).unsqueeze(0).unsqueeze(0)
    cov = cov + eps * eye
    return cov


def logm_spd(C: torch.Tensor) -> torch.Tensor:
    """对 SPD 矩阵做矩阵对数（C 必须是对称正定）。

    公式
    ----
    C = U diag(λ) U^T  →  logm(C) = U diag(log(λ)) U^T

    参数
    ----
    C : (..., A, A)  对称正定

    返回
    ----
    L : (..., A, A)  实对称（logm(C)）
    """
    # 用 eigh 保证数值稳定性（对称矩阵专用）
    eigvals, eigvecs = torch.linalg.eigh(C)                           # (..., A), (..., A, A)
    # 强制 ≥ eps 防止 log(0)
    eigvals = torch.clamp(eigvals, min=1e-10)
    log_eigvals = torch.log(eigvals)                                   # (..., A)
    # L = U diag(log λ) U^T
    L = eigvecs @ torch.diag_embed(log_eigvals) @ eigvecs.transpose(-1, -2)
    return L


def spd_upper_triangle(L: torch.Tensor) -> torch.Tensor:
    """提取 SPD log 矩阵的上三角（含对角线），作为平铺特征。

    参数
    ----
    L : (..., A, A)  实对称（logm SPD）

    返回
    ----
    flat : (..., A*(A+1)/2)
    """
    # 取上三角（含对角线）
    A = L.shape[-1]
    triu_indices = torch.triu_indices(A, A, offset=0, device=L.device)
    flat = L[..., triu_indices[0], triu_indices[1]]                    # (..., A*(A+1)/2)
    return flat


def compute_spd_features(
    x: torch.Tensor,                  # (n, A, S, F, T') float32
    use_logm: bool = True,
) -> torch.Tensor:
    """一站式 SPD 几何特征提取。

    流程
    ----
    1) 每个频率带 f 计算 (A, A) 协方差矩阵
    2) Log-Euclidean 投影（矩阵对数）→ 实对称
    3) 上三角平铺 → (n, F, A*(A+1)/2)

    返回
    ----
    feats : (n, F * A*(A+1)/2) float32
            默认 A=9 → 33 * 45 = 1485 维特征
    """
    cov = spd_covariance_per_freq(x)                                   # (n, F, A, A)
    if use_logm:
        L = logm_spd(cov)                                              # (n, F, A, A)
    else:
        L = cov
    feats = spd_upper_triangle(L)                                      # (n, F, A*(A+1)/2)
    n, F_dim, F2 = feats.shape
    return feats.reshape(n, F_dim * F2)                                # (n, F * A*(A+1)/2)


def compute_spd_per_time(x: torch.Tensor, use_logm: bool = True) -> torch.Tensor:
    """对每个时间帧 T' 计算 (A, A) 协方差（与 compute_spd_features 不同的是沿 T 维）。

    输入
    ----
    x : (n, A, S, F, T') float32

    返回
    ----
    feats : (n, T' * A*(A+1)/2)
    """
    n, A, S, F, Tp = x.shape
    # 沿 S 和 F 维收集样本，共 S*F = 990 个；每帧是一个 (A=9) 向量
    x_perm = x.permute(0, 4, 1, 2, 3).reshape(n, Tp, A, S * F).contiguous()  # (n, T', A, S*F)
    mu = x_perm.mean(dim=-1, keepdim=True)
    x_centered = x_perm - mu
    cov = torch.einsum("ntas,ntbs->ntab", x_centered, x_centered) / max(S * F - 1, 1)
    eye = torch.eye(A, device=x.device, dtype=x.dtype).unsqueeze(0).unsqueeze(0)
    cov = cov + 1e-6 * eye
    if use_logm:
        L = logm_spd(cov)
    else:
        L = cov
    feats = spd_upper_triangle(L)
    n, T_dim, F2 = feats.shape
    return feats.reshape(n, T_dim * F2)
