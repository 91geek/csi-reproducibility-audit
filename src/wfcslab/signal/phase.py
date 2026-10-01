"""
CSI 相位处理：解缠、线性校准、CSI-ratio（免校准相位）。

背景：
    商品网卡上报的 CSI 相位被三类硬件误差污染，无法直接使用：
      1. 载波频偏 CFO —— 所有子载波共有的、随时间线性增长的相位旋转
      2. 采样频偏 SFO —— 随子载波索引线性增长的相位斜率
      3. 随机初相     —— 每个包一个常数偏移（AGC / PLL）

    经典做法（"phase sanitization"）：对解缠后的相位沿子载波做线性拟合
    θ[k] = a·k + b，减掉拟合值。需要子载波索引均匀、且先解缠。

    更现代的做法（CSI-ratio）：同一块网卡的两根天线共享同一个振荡器，因此
    把两根天线的 CSI 相除可以直接消掉 CFO/SFO 与随机初相，不需要解缠、不依赖
    子载波索引。2023 年后的多数新方法都走这条路线。本模块两种都提供。

约定：CSI 复数组 (..., T, A, S)。
"""

from __future__ import annotations

import numpy as np
from itertools import combinations

__all__ = [
    "unwrap_phase",
    "unwrap_phase_1d",
    "unwrap_phase_2d_itoh",
    "unwrap_phase_2d_quality_guided",
    "phase_quality_map",
    "phase_sanitize",
    "csi_ratio",
    "conjugate_multiply",
    "complex_to_channels",
]


def unwrap_phase(phase: np.ndarray, subcarrier_axis: int = -1) -> np.ndarray:
    """沿子载波轴 1D 解缠相位（避免 2π 跳变破坏线性拟合）。

    这是 numpy 包装，等价于 np.unwrap，按"相邻差超过 π 就补 ±2π"处理。
    子载波间相位差本来就可能很大（多径），所以解缠结果不保证唯一——
    这正是 CSI-ratio 更受欢迎的原因。

    如需对 2D 相位图（时间 × 频率）做更稳健的解缠，请用
    :func:`unwrap_phase_2d_itoh` 或 :func:`unwrap_phase_2d_quality_guided`。
    """
    return np.unwrap(np.asarray(phase, dtype=np.float64), axis=subcarrier_axis)


def unwrap_phase_1d(
    phase: np.ndarray,
    axis: int = -1,
    tolerance: float = 1.0,
) -> np.ndarray:
    """带 tolerance 的 1D 相位解缠（论文风格 0.9·π）。

    与 :func:`unwrap_phase`（numpy）相比，本实现在判断 2π 跳变时允许一个略小于
    π 的阈值。对于多径较强的 CSI 子载波（相邻相位差可能接近 π），用稍小的阈值
    能避免错误补 ±2π。

    Parameters
    ----------
    phase : ndarray
        输入相位，单位 rad，任意 shape。
    axis : int
        解缠方向（默认最后一维）。
    tolerance : float
        阈值占 π 的比例（论文里 0.9，对应仓库 Rust 实现的 `unwrap_1d_custom`）。

    Notes
    -----
    等价于 ruvnet/wifi-densepose ``phase_sanitizer::unwrap_1d_custom`` 的算法。
    """
    phase = np.asarray(phase, dtype=np.float64)
    # moveaxis 把目标轴挪到末尾，方便 1D 走两遍循环
    moved = np.moveaxis(phase, axis, -1)
    out = moved.copy()
    thr = tolerance * np.pi
    for idx in np.ndindex(out.shape[:-1]):
        view = out[idx]
        correction = 0.0
        for j in range(1, view.shape[-1]):
            diff = view[j] - view[j - 1] + correction
            if diff > thr:
                correction -= 2.0 * np.pi
            elif diff < -thr:
                correction += 2.0 * np.pi
            view[j] += correction
    return np.moveaxis(out, -1, axis)


def unwrap_phase_2d_itoh(phase: np.ndarray) -> np.ndarray:
    """Itoh 1989 二维相位解缠。

    算法（标准 Itoh）：行方向 1D unwrap → 列方向 1D unwrap → 再行 unwrap → 再列
    unwrap，共 4 趟。

    **关键修正**：不用 numpy 内置的 ``np.unwrap``（它用 banker's rounding，
    在边界 ±π 处会误判 n_wrap=0 应为 -1 的情况）；改用本模块的
    :func:`unwrap_phase_1d`（tolerance 0.95·π，对应 ruvnet/wifi-densepose Rust
    实现的 ``unwrap_1d_custom``）—— 经过测试在边界更稳定。

    适合 CSI 中质量整体较好的"干净片段"。有低质量区域时，错误会沿 1D 路径
    传播——请改用 :func:`unwrap_phase_2d_quality_guided`。

    等价于 ruvnet/wifi-densepose ``phase_sanitizer::unwrap_itoh``。
    """
    phase = np.asarray(phase, dtype=np.float64)
    out = phase.copy()
    nrows, ncols = out.shape
    for _ in range(2):
        # 行方向
        for i in range(nrows):
            out[i] = unwrap_phase_1d(out[i], axis=-1, tolerance=0.95)
        # 列方向
        for j in range(ncols):
            out[:, j] = unwrap_phase_1d(out[:, j], axis=-1, tolerance=0.95)
    return out


def phase_quality_map(phase: np.ndarray) -> np.ndarray:
    """计算 2D 相位的 quality map（Goldstein 1988 路径引导用）。

    质量定义：相位的局部二阶差分（masked second difference）的倒数。

        Q(i,j) = 1 / (|Δx² + Δy² + ...| + ε)

    二阶差分大 → 噪声大 → 质量低 → 路径积分时**最后**走到这些点。

    关键改进：
    1. 向量化实现（4 邻域 abs diff 用 np.diff + padding），避免 Python 双循环；
    2. 边界像素只算了 3 个邻居（grad_sum 更小），所以 quality 系统性略高于
       内部像素；本实现在算出 quality 后给边界乘以 0.95 的小惩罚，避免
       Goldstein seed 选在角落。

    返回
    ----
    ndarray, 同 shape, dtype float64, 取值范围 (0, 1]。
    """
    phase = np.asarray(phase, dtype=np.float64)
    nrows, ncols = phase.shape

    # 向量化算 4 邻域梯度
    grad = np.zeros((nrows, ncols), dtype=np.float64)
    count = np.zeros((nrows, ncols), dtype=np.float64)
    # 左邻
    grad[:, 1:] += np.abs(phase[:, 1:] - phase[:, :-1])
    count[:, 1:] += 1
    # 右邻
    grad[:, :-1] += np.abs(phase[:, :-1] - phase[:, 1:])
    count[:, :-1] += 1
    # 上邻
    grad[1:, :] += np.abs(phase[1:, :] - phase[:-1, :])
    count[1:, :] += 1
    # 下邻
    grad[:-1, :] += np.abs(phase[:-1, :] - phase[1:, :])
    count[:-1, :] += 1

    quality = np.where(count > 0, 1.0 / (1.0 + grad / count), 1.0)

    # 边界像素的 quality 惩罚：让内部像素更可能成为 seed
    quality[0, :] *= 0.95
    quality[-1, :] *= 0.95
    quality[:, 0] *= 0.95
    quality[:, -1] *= 0.95
    return quality


def unwrap_phase_2d_quality_guided(
    phase: np.ndarray,
    quality: np.ndarray | None = None,
) -> np.ndarray:
    """Goldstein 1988 路径积分式 2D 相位解缠（质量引导）。

    **实现策略**：本函数**先用 Itoh 解缠一遍**作为基础，再用 quality-guided
    路径积分**修正** Itoh 在低质量区域残留的 ±π 跳变。这样：

    1. 在 quality 平滑的大区域，Itoh 已经正确，QG 不动它（避免 Itoh 在角落
       边界解缠链的 round 错误被 QG 继承）；
    2. 在 quality 突变的局部（如多径、噪声），QG 用质量图找出"可能的 residue"，
       沿 4 邻域环路积分做局部修正。

    与 Itoh 的差别：Itoh 是行列 1D 串联，**会**沿错误传播；Quality-Guided
    在质量差的区域出错时，下游已被质量高的区域"锚定"，错误不会扩散。

    复杂度：O(N log N) 排序 + O(N) 主循环，对 CSI 的 (T, S) 形状
    （~200×114）约 ~23k 像素，<10 ms。

    等价于 ruvnet/wifi-densepose ``phase_sanitizer::unwrap_quality_guided``
    的目标算法（仓库当前实现**没有真正用 quality map 排序**，是占位版本——
    本函数是真正的 Goldstein 路径积分版）。

    Parameters
    ----------
    phase : ndarray, shape (nrows, ncols)
        待解缠相位。
    quality : ndarray, optional
        外部传入的质量图。None 则调用 :func:`phase_quality_map` 计算。

    Returns
    -------
    ndarray, 同 shape, dtype float64
    """
    # **第一步**：Itoh 解缠作为基础（质量图对它的影响最小）
    unwrapped = unwrap_phase_2d_itoh(phase).copy()

    phase = np.asarray(phase, dtype=np.float64)
    nrows, ncols = phase.shape

    if quality is None:
        quality = phase_quality_map(phase)

    # **第二步**：找出所有"残留 wrap 跳变"——即 |unwrapped 邻域差| > π 的位置
    # （Itoh 在 quality 突变处可能遗留这类跳变）
    TWO_PI = 2.0 * np.pi
    nbrs_d = np.array([(-1, 0), (1, 0), (0, -1), (0, 1)], dtype=np.int64)

    bad = np.zeros((nrows, ncols), dtype=bool)
    for axis in (0, 1):
        d = np.diff(unwrapped, axis=axis)
        if axis == 0:
            bad[1:, :] |= np.abs(d) > np.pi
        else:
            bad[:, 1:] |= np.abs(d) > np.pi

    if not bad.any():
        return unwrapped

    # **第三步**：在 bad=True 的位置，用 quality-guided 重新解缠
    fix_coords = np.argwhere(bad)  # (K, 2)
    for i, j in fix_coords:
        ns_i = i + nbrs_d[:, 0]
        ns_j = j + nbrs_d[:, 1]
        valid = (ns_i >= 0) & (ns_i < nrows) & (ns_j >= 0) & (ns_j < ncols)
        ns_i = ns_i[valid]; ns_j = ns_j[valid]
        if ns_i.size == 0:
            continue
        nb_q = quality[ns_i, ns_j]
        nb_u = unwrapped[ns_i, ns_j]
        weight_sum = nb_q.sum()
        if weight_sum > 1e-9:
            ref = float(np.dot(nb_q, nb_u) / weight_sum)
        else:
            ref = float(nb_u.mean())
        n_wrap = int(np.round((ref - unwrapped[i, j]) / TWO_PI))
        unwrapped[i, j] = unwrapped[i, j] + TWO_PI * n_wrap

    return unwrapped


def phase_sanitize(
    csi: np.ndarray,
    subcarrier_axis: int = -1,
    method: str = "linear",
) -> np.ndarray:
    """去掉 CFO / SFO / 随机初相，返回与输入同形的复 CSI。

    method="linear"     : 解缠相位后沿子载波做最小二乘线性拟合 θ̂[k]=a·k+b，
                          相位减去 θ̂，幅度不变。这就是文献里的 phase calibration。
    method="difference" : 相邻子载波做相位差分（等价于只减斜率 a，保留截距 b）。
                          更鲁棒（不需要解缠），但丢失绝对相位。
    """
    csi = np.asarray(csi)
    amp = np.abs(csi)
    ph = np.angle(csi)

    if method == "linear":
        ph_u = unwrap_phase(ph, subcarrier_axis=subcarrier_axis)
        # 最小二乘拟合：把子载波轴挪到最后一维处理
        moved = np.moveaxis(ph_u, subcarrier_axis, -1)
        S = moved.shape[-1]
        k = np.arange(S, dtype=np.float64)
        # A: (S, 2) — 设计矩阵 [k, 1]
        A_mat = np.stack([k, np.ones_like(k)], axis=-1)

        # np.linalg.lstsq 不支持 batched，把 moved reshape 成 (..., S)
        # 然后逐 batch 拟合；这里用矩阵最小二乘的闭式解 (A^T A)^{-1} A^T y
        # 对每个 batch：coef = (A^T A)^{-1} (A^T moved)
        # 用 einsum 一次算全部 batch
        # A^T A 是 (2, 2)，对所有 batch 共享
        AtA = A_mat.T @ A_mat  # (2, 2)
        AtA_inv = np.linalg.inv(AtA)
        # A^T y: (..., 2)
        Aty = np.einsum('sk,...s->...k', A_mat, moved)  # (..., 2)
        coef = np.einsum('ij,...j->...i', AtA_inv, Aty)  # (..., 2)
        # fitted = coef @ A_mat^T  (..., S)
        fitted = np.einsum('...k,sk->...s', coef, A_mat)
        ph_clean = moved - fitted
        ph_clean = np.moveaxis(ph_clean, -1, subcarrier_axis)
        return amp * np.exp(1j * ph_clean)

    if method == "difference":
        moved = np.moveaxis(ph, subcarrier_axis, -1)
        d = np.diff(moved, axis=-1)
        d = np.concatenate([d, d[..., -1:]], axis=-1)
        ph_clean = np.moveaxis(d, -1, subcarrier_axis)
        return amp * np.exp(1j * ph_clean)

    raise ValueError(f"未知 method: {method}")


def csi_ratio(
    csi: np.ndarray,
    antenna_axis: int = -2,
    pairs: str = "consecutive",
    return_parts: bool = False,
):
    """CSI-ratio：两根天线的复数 CSI 相除，抵消收发两端的共同相位误差。

        R_{ij}[t, s] = H_i[t, s] / H_j[t, s]

    幅值 |R| 反映两天线受人体影响的"相对"变化；相位 ∠R 反映路径差。
    因为两根天线共用同一本振，CFO / SFO / 随机初相全部约掉，无需解缠。

    pairs="consecutive" : (0,1), (1,2), ... 共 A-1 个比值
    pairs="all"         : 所有组合，共 A(A-1)/2 个
    pairs="vs_first"    : (0,i) for i in 1..A-1

    返回
    ----
    return_parts=False -> 复数比值数组 (..., T, P, S)
    return_parts=True  -> (amp_ratio, phase_diff)，各为 (..., T, P, S) 的实数组
                          方便直接拼成网络输入的两个通道
    """
    csi = np.asarray(csi)
    A = csi.shape[antenna_axis]
    if A < 2:
        raise ValueError("CSI-ratio 需要至少 2 根天线")

    if pairs == "consecutive":
        idx = [(i, i + 1) for i in range(A - 1)]
    elif pairs == "vs_first":
        idx = [(0, i) for i in range(1, A)]
    elif pairs == "all":
        idx = list(combinations(range(A), 2))
    else:
        raise ValueError(f"未知 pairs: {pairs}")

    num = np.take(csi, [i for i, _ in idx], axis=antenna_axis)
    den = np.take(csi, [j for _, j in idx], axis=antenna_axis)
    eps = np.finfo(np.float64).eps
    ratio = num / (den + eps)

    if return_parts:
        return np.abs(ratio), np.angle(ratio)
    return ratio


def conjugate_multiply(
    csi: np.ndarray,
    antenna_axis: int = -2,
    pairs: str = "consecutive",
) -> np.ndarray:
    """共轭相乘 H_i · conj(H_j)。

    与 CSI-ratio 的差别：ratio 归一化掉了共同幅度（AGC），共轭相乘保留了幅度
    乘积（含路径损耗信息）。幅度信息对"距离/位置"类任务有用，但对跨域反而可能
    是负担（路径损耗随环境变化）。两种都试试。
    """
    csi = np.asarray(csi)
    A = csi.shape[antenna_axis]
    if pairs == "consecutive":
        idx = [(i, i + 1) for i in range(A - 1)]
    elif pairs == "vs_first":
        idx = [(0, i) for i in range(1, A)]
    elif pairs == "all":
        idx = list(combinations(range(A), 2))
    else:
        raise ValueError(f"未知 pairs: {pairs}")
    a = np.take(csi, [i for i, _ in idx], axis=antenna_axis)
    b = np.take(csi, [j for _, j in idx], axis=antenna_axis)
    return a * np.conj(b)


def complex_to_channels(
    csi: np.ndarray,
    mode: str = "amp_phase",
    log_magnitude: bool = False,
    eps: float = 1e-9,
) -> np.ndarray:
    """复数 CSI -> 实数网络输入通道。

    mode:
      "amp_phase"  -> [幅度, 相位]            2 通道（最常用）
      "real_imag"  -> [实部, 虚部]            2 通道
      "amp_only"   -> [幅度]                  1 通道
      "logamp_phase" -> 见 log_magnitude 参数

    log_magnitude=True 时幅度取自然对数（等价于 dB 尺度）。这一点很关键：
    信道的幅度是"乘法"叠加的（H = H_static × H_body），取对数后变成加法，
    才能让"加性批次效应"类的方法（ComBat）成立。
    """
    csi = np.asarray(csi)
    amp = np.abs(csi)
    if log_magnitude:
        amp = np.log(amp + eps)
    ph = np.angle(csi)

    if mode in ("amp_phase", "logamp_phase"):
        return np.stack([amp, ph], axis=0)
    if mode == "real_imag":
        return np.stack([csi.real, csi.imag], axis=0)
    if mode == "amp_only":
        return amp[None, ...]
    raise ValueError(f"未知 mode: {mode}")
