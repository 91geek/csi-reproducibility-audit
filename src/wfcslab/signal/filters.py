"""
时频域常规滤波：Hampel 去野点、Savitzky-Golay 平滑、Butterworth 带通、多普勒谱。

这些都是各论文 method 章节里的"零件"，几乎不单独成文。这里统一实现，方便做消融。
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import median_filter
from scipy.signal import butter, filtfilt, savgol_filter

__all__ = [
    "hampel",
    "savgol",
    "butter_bandpass",
    "doppler_spectrum",
    "dfs_profile",
]


def hampel(
    x: np.ndarray,
    window: int = 5,
    n_sigmas: float = 3.0,
    time_axis: int = -3,
) -> np.ndarray:
    """Hampel 滤波器：中位数 + MAD 判据，替换脉冲型野点（AGC 跳变、丢包尖峰）。

    判据：|x[t] - med[t]| > n_sigmas * 1.4826 * MAD[t]  ->  用 med[t] 替换。
    1.4826 是把 MAD 校准到正态分布标准差的比例因子。
    对复数分别处理实部与虚部（等价处理幅度和相位的野点）。
    """
    x = np.asarray(x)
    out = np.empty_like(x)
    for part in ("real", "imag"):
        arr = getattr(x, part)()
        med = median_filter(arr, size=window, mode="nearest", axes=time_axis)
        if np.iscomplexobj(x):
            absdev = np.abs(arr - med)
        else:
            absdev = np.abs(arr - med)
        mad = median_filter(absdev, size=window, mode="nearest", axes=time_axis)
        thresh = n_sigmas * 1.4826 * mad
        cleaned = np.where(absdev > thresh, med, arr)
        if part == "real":
            out = cleaned.astype(out.dtype) if not np.iscomplexobj(x) else cleaned + 0j
        else:
            out = out + 1j * cleaned
    return out


def savgol(
    x: np.ndarray,
    window: int = 11,
    polyorder: int = 3,
    axis: int = -3,
) -> np.ndarray:
    """Savitzky-Golay 平滑：局部多项式拟合，比滑动平均更好地保住峰值形状。

    window 必须是奇数且 > polyorder。对复数分别处理实部虚部。
    """
    x = np.asarray(x)
    if window % 2 == 0:
        window += 1
    if window <= polyorder:
        raise ValueError("window 必须大于 polyorder")
    if np.iscomplexobj(x):
        re = savgol_filter(x.real, window, polyorder, axis=axis)
        im = savgol_filter(x.imag, window, polyorder, axis=axis)
        return re + 1j * im
    return savgol_filter(x, window, polyorder, axis=axis)


def butter_bandpass(
    x: np.ndarray,
    low: float = 0.5,
    high: float = 40.0,
    fs: float = 100.0,
    order: int = 4,
    time_axis: int = -3,
) -> np.ndarray:
    """零相位 Butterworth 带通（filtfilt，无相位失真）。

    典型取值：人体活动 0.5~40 Hz（fs 需 > 2*high）。
    呼吸 0.2~0.5 Hz、心跳 0.8~2 Hz 需要单独设低频段。
    """
    if high >= fs / 2:
        raise ValueError(f"high={high} 必须小于奈奎斯特频率 {fs/2}")
    b, a = butter(order, [low / (fs / 2), high / (fs / 2)], btype="band")
    x = np.asarray(x)
    if np.iscomplexobj(x):
        re = filtfilt(b, a, x.real, axis=time_axis)
        im = filtfilt(b, a, x.imag, axis=time_axis)
        return re + 1j * im
    return filtfilt(b, a, x, axis=time_axis)


def doppler_spectrum(
    x: np.ndarray,
    fs: float = 100.0,
    time_axis: int = -3,
    shift: bool = True,
) -> np.ndarray:
    """沿慢时间做 FFT，得到多普勒谱（复数，保留正负频率方向）。

    人体运动在正/负多普勒上的分布不对称（朝向/远离收发器），这个不对称性
    就是 Widar 系列 DFS（Doppler Frequency Shift）特征的基础。
    """
    X = np.fft.fft(np.asarray(x), axis=time_axis)
    if shift:
        return np.fft.fftshift(X, axes=time_axis)
    return X


def dfs_profile(
    x: np.ndarray,
    fs: float = 100.0,
    time_axis: int = -3,
    remove_static: bool = True,
) -> np.ndarray:
    """DFS 功率谱（幅度），可选先去掉零多普勒的静态能量。

    这是雷达/声纳里比 CSI 幅度更抗环境变化的表征——Widar3.0 的 BVP 和
    MORIC 的 delay-Doppler 分解都基于这一点。作为对比基线很有价值。
    """
    X = np.fft.fft(np.asarray(x), axis=time_axis)
    X = np.fft.fftshift(X, axes=time_axis)
    if remove_static:
        z_idx = X.shape[time_axis] // 2  # 零频位置（fftshift 后居中）
        sl = [slice(None)] * X.ndim
        sl[time_axis] = z_idx
        X[tuple(sl)] = 0
    return np.abs(X)
