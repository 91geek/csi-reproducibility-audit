"""
评测指标。

姿态任务（MM-Fi 的口径）
------------------------
MPJPE    Mean Per Joint Position Error，逐关节欧氏误差的均值，单位毫米。
         这是最直接的指标，但会同时惩罚"整体位置偏了"和"姿态本身错了"。
PA-MPJPE Procrustes Aligned MPJPE。先做相似变换（旋转+平移+缩放）对齐再算误差，
         只衡量"姿态形状"对不对，剔除根关节绝对位置的影响。
         **跨域评测必须同时报这两个** —— 域偏移会同时移动绝对位置，
         只看 MPJPE 会分不清是"跨域校准没做好"还是"姿态本身没学好"。
PCK@thr  Percentage of Correct Keypoints，误差小于阈值 thr 的关节占比。
         常用 PCK@50mm。

活动/手势识别
-------------
Accuracy 与 Macro-F1。类别不平衡时（MM-Fi 的 27 类、Widar 的多域分布）必须看
Macro-F1，单看 Accuracy 会被大类掩盖。
"""

from __future__ import annotations

import numpy as np

__all__ = ["mpjpe", "pa_mpjpe", "pck", "accuracy", "macro_f1", "evaluate"]


def _as_np(x) -> np.ndarray:
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=np.float64)


def mpjpe(pred, gt, reduction: str = "mean") -> float:
    """pred/gt: (..., J, 3)。返回平均逐关节误差（与输入同单位）。"""
    p, g = _as_np(pred), _as_np(gt)
    err = np.linalg.norm(p - g, axis=-1)              # (..., J)
    if reduction == "mean":
        return float(err.mean())
    if reduction == "none":
        return err
    raise ValueError(reduction)


def _procrustes_align(p: np.ndarray, g: np.ndarray) -> np.ndarray:
    """把预测姿态 p 用相似变换对齐到 gt（去均值 + 最优旋转 + 最优缩放）。"""
    # p, g: (J, 3)
    mu_p, mu_g = p.mean(axis=0), g.mean(axis=0)
    pc, gc = p - mu_p, g - mu_g
    # 最优正交旋转（Kabsch）
    H = pc.T @ gc
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1.0, 1.0, d])
    R = Vt.T @ D @ U.T
    # 最优缩放
    scale = np.trace(np.diag(S) @ D) / (pc ** 2).sum()
    return (scale * (R @ pc.T).T) + mu_g


def pa_mpjpe(pred, gt) -> float:
    """Procrustes 对齐后的 MPJPE。"""
    p, g = _as_np(pred), _as_np(gt)
    if p.ndim == 2:
        return float(np.linalg.norm(_procrustes_align(p, g) - g, axis=-1).mean())
    out = []
    for i in range(p.shape[0]):
        out.append(np.linalg.norm(_procrustes_align(p[i], g[i]) - g[i], axis=-1).mean())
    return float(np.mean(out))


def pck(pred, gt, threshold: float = 50.0) -> float:
    """误差小于 threshold 的关节占比（0~1）。threshold 与输入同单位。"""
    p, g = _as_np(pred), _as_np(gt)
    err = np.linalg.norm(p - g, axis=-1)
    return float((err < threshold).mean())


def accuracy(logits, y) -> float:
    lg = _as_np(logits)
    pred = lg.argmax(axis=-1) if lg.ndim > 1 else (lg > 0).astype(int)
    return float((pred == _as_np(y).ravel()).mean())


def macro_f1(logits, y, n_classes: int | None = None) -> float:
    lg = _as_np(logits)
    pred = lg.argmax(axis=-1) if lg.ndim > 1 else (lg > 0).astype(int)
    y = _as_np(y).ravel().astype(int)
    if n_classes is None:
        n_classes = int(max(pred.max(), y.max())) + 1
    f1s = []
    for c in range(n_classes):
        tp = ((pred == c) & (y == c)).sum()
        fp = ((pred == c) & (y != c)).sum()
        fn = ((pred != c) & (y == c)).sum()
        if tp + fp == 0 or tp + fn == 0:
            f1s.append(0.0)
            continue
        prec, rec = tp / (tp + fp), tp / (tp + fn)
        f1s.append(0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec))
    return float(np.mean(f1s))


def evaluate(task: str, logits, y, thresholds=(50.0,)) -> dict:
    """统一评测入口。返回指标字典。"""
    if task == "classification":
        return {"acc": accuracy(logits, y), "macro_f1": macro_f1(logits, y)}
    if task == "pose":
        out = {
            "mpjpe": mpjpe(logits, y),
            "pa_mpjpe": pa_mpjpe(logits, y),
        }
        for t in thresholds:
            out[f"pck@{t:g}"] = pck(logits, y, t)
        return out
    raise ValueError(f"未知 task: {task}")
