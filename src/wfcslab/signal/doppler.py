"""
Body Velocity Profile（BVP）—— 动作的微多普勒频谱图。

为什么 WiFi 感知必须先做时频分析
================================
原始复数 CSI H(t, s) 是 30 个子载波在 ~kHz 包率下的**时域复信号**。直接把
H(t) 喂给 CNN，等于让模型从 30 条载波的正弦/余弦曲线里自己学"这是哪个动作"——
理论上可解，但需要大量数据 + 复杂结构。

更高效的做法（2020 年后的论文主流，Widar3.0 / EI / SignFi 都用）：
把每个 (subcarrier, antenna) 对沿慢时间做 STFT，取 |STFT|^2 作为**谱能量**。
这条谱能量沿多普勒频率轴的分布（= body velocity profile，BVP）才是动作的
"指纹"：手往哪挥、挥多快，对应不同的多普勒偏移。这正是 radar signal
processing 里 micro-Doppler signature 的标准定义。

实现要点
--------
- DC removal：减去时间均值，等价于 MTI filter（去静态杂波）
- Hann window：STFT 标配，旁瓣衰减 ~31 dB
- |STFT|^2：能量谱，比 |STFT| 更突出动作段
- antenna aggregation：RMS（几何平均）合并多天线谱
- subcarrier aggregation：保留全部 30 子载波（BVP 对频率选择性衰落敏感，
  不能简单平均）。CNN 端会自己学子载波维的权重
- log 压缩：和 csi_ratio+log 链路一致，把乘法叠加（H = H_static * H_body）
  变成加法叠加（log H = log H_static + log H_body），与 ComBat 等加性批次
  效应方法对齐

形状约定
--------
    输入  X: (n, A, T, S) complex64   A=天线, T=慢时间, S=子载波
    输出  X: (n, C, F, T') float32    C=1（已聚合成单通道）, F=频率bin, T'=时间步
"""

from __future__ import annotations

import numpy as np

__all__ = ["stft1d", "compute_bvp", "compute_phase_bvp"]


# --------------------------------------------------------------------------
def _hann_window(n_fft: int) -> np.ndarray:
    """周期 Hann 窗（左右端点相等），与 torch.stft 默认行为一致。"""
    if n_fft == 1:
        return np.ones(1, dtype=np.float32)
    return np.sqrt(np.hanning(n_fft).astype(np.float32))


def stft1d(
    x: np.ndarray,
    n_fft: int = 64,
    hop: int = 16,
    window: np.ndarray | None = None,
    center: bool = False,
) -> np.ndarray:
    """对最后一维做 STFT，返回 |STFT|^2 能量谱。

    参数
    ----
    x      : (..., T)  任意实/复数信号
    n_fft  : FFT 长度
    hop    : 帧移
    window : 自定义窗（默认 Hann）
    center : 是否在两端补零使帧中心对齐

    返回
    ----
    out    : (..., F, T')   F = n_fft//2 + 1（单边谱），T' = 帧数
    """
    x = np.asarray(x)
    if window is None:
        window = _hann_window(n_fft)
    window = window.astype(np.float32)
    if center:
        pad = n_fft // 2
        pad_shape = (*x.shape[:-1], pad)
        x = np.concatenate([np.zeros(pad_shape, x.dtype), x,
                            np.zeros(pad_shape, x.dtype)], axis=-1)
    n_frames = 1 + (x.shape[-1] - n_fft) // hop if x.shape[-1] >= n_fft else 0
    if n_frames <= 0:
        raise ValueError(
            f"信号太短：T={x.shape[-1]} < n_fft={n_fft}，调小 n_fft 或增大 max_time"
        )
    # 滑窗：构造 (..., n_frames, n_fft) 切片
    strides = x.strides
    new_shape = (*x.shape[:-1], n_frames, n_fft)
    new_strides = (*strides[:-1], hop * strides[-1], strides[-1])
    frames = np.lib.stride_tricks.as_strided(
        x, shape=new_shape, strides=new_strides, writeable=False
    )
    # 加窗 + FFT
    spec = np.fft.rfft(frames * window, n=n_fft, axis=-1)   # (..., n_frames, F)
    spec = np.moveaxis(spec, -2, -1)                        # (..., F, n_frames)
    return (spec.real ** 2 + spec.imag ** 2).astype(np.float32)


# --------------------------------------------------------------------------
def compute_bvp(
    csi: np.ndarray,
    n_fft: int = 64,
    hop: int = 16,
    dc_removal: bool = True,
    eps: float = 1e-9,
    batch_n: int = 512,
    keep_antenna: bool = False,
) -> np.ndarray:
    """Body Velocity Profile：把 (n, A, T, S) 的复数 CSI 变成 (n, S, F, T') 谱图。

    流程
    ----
    1) 取 |H| 得到幅度（相位对动作判别信号弱，且保留相位会引入天线间几何噪声）
    2) 减时间均值（去静态杂波，=单天线 MTI）
    3) 对每个 (antenna, subcarrier) 沿时间做 STFT → (n, A, F, T')
    4) 天线维 RMS 聚合 → (n, S, F, T')  （注意 S 是子载波数，对应原始输入的最后一维）
    5) log 压缩 → (n, S, F, T')

    形状选择
    -------
    我们把子载波维和频率维合并到 backbone 的 "in_channels" 与 "n_subc" 维：
    约定输出 (n, C=S, F, T')，backbone 视为 (in_channels=S, n_time=F, n_subc=T')。

    这样 backbone 不需要任何新代码就能用上 BVP：LeNetCSI/CNNGRU 都接受 (B, C, T, S)。

    keep_antenna=False（默认）：输出 (n, S, F, T')，天线维 RMS 聚合；
    keep_antenna=True：输出 (n, A, S, F, T')，保留每个 Rx 的独立 BVP，
        让 backbone 用注意力学天线权重（论文 [2512.04521] SMSA）。

    内存
    ----
    中间产物 (n, A, S, F, T') 在 12000 样本下是 ~28 GB（float32）。所以按 n 维
    分 batch 处理，每批 5D 中间产物 ~1 GB，单进程不会 OOM。

    返回
    ----
    out : (n, S, F, T') float32（keep_antenna=False）
          或 (n, A, S, F, T') float32（keep_antenna=True）
    """
    csi = np.asarray(csi)
    if csi.ndim != 4:
        raise ValueError(f"BVP 输入需要 (n, A, T, S)，收到 shape={csi.shape}")
    n, A, T, S = csi.shape

    # 先跑一批空数据算 F / Tp，确定输出 shape（不分配大数组）
    # 取一个最小 batch（最多 16 样本）探测 shape
    probe = csi[: min(16, n)]
    amp_probe = np.abs(probe).astype(np.float32)
    if dc_removal:
        amp_probe = amp_probe - amp_probe.mean(axis=-2, keepdims=True)
    nb_p, A_p, T_p, S_p = amp_probe.shape
    flat_p = amp_probe.transpose(0, 1, 3, 2).reshape(nb_p * A_p * S_p, T_p)
    spec_p = stft1d(flat_p, n_fft=n_fft, hop=hop)
    F = spec_p.shape[-2]
    Tp = spec_p.shape[-1]

    out_shape = (n, A, S, F, Tp) if keep_antenna else (n, S, F, Tp)
    out = np.empty(out_shape, dtype=np.float32)

    batch_n = max(1, min(batch_n, n))
    for i in range(0, n, batch_n):
        end = min(i + batch_n, n)
        amp = np.abs(csi[i:end]).astype(np.float32)        # (nb, A, T, S)
        if dc_removal:
            amp = amp - amp.mean(axis=-2, keepdims=True)
        nb, _, _, _ = amp.shape
        flat = amp.transpose(0, 1, 3, 2).reshape(nb * A * S, T)
        spec = stft1d(flat, n_fft=n_fft, hop=hop)           # (nb*A*S, F, T')
        spec = spec.reshape(nb, A, S, F, Tp)
        if keep_antenna:
            # 保留每个 Rx 的独立 BVP，让 backbone 用注意力学天线权重
            spec = np.log(spec + eps)                        # (nb, A, S, F, T')
        else:
            # 天线维 RMS 聚合：sqrt(mean(|.|^2))，物理意义为各天线能量几何合并
            spec = np.sqrt((spec ** 2).mean(axis=1))        # (nb, S, F, T')
            spec = np.log(spec + eps)
        out[i:end] = spec.astype(np.float32)

    return out


# --------------------------------------------------------------------------
def _phase_sanitize(csi: np.ndarray) -> np.ndarray:
    """CSI 相位净化（conjugate multiplication）——去除随机相位偏移。

    论文 [2506.11616] Wi-CBR 标准做法：取相邻子载波 H[s] 与 H[s+1] 共轭相乘，
    抵消 CFO/sampling-timing offset 等公共相位误差，保留相对相位变化。

    输入 csi: (..., T, S) complex64
    输出    : (..., T, S-1) complex64
    """
    a = csi[..., :-1]
    b = csi[..., 1:]
    return a * np.conj(b)


def compute_phase_bvp(
    csi: np.ndarray,
    n_fft: int = 64,
    hop: int = 16,
    dc_removal: bool = True,
    eps: float = 1e-9,
    batch_n: int = 512,
    keep_antenna: bool = False,
) -> np.ndarray:
    """Phase BVP：先用 conjugate 净化相位，再做 STFT，输出能量谱。

    与 ``compute_bvp`` 流程一致，但用净化后的相位差异信号（实部）代替 |H|。
    文献 [2506.11616] Wi-CBR 把 DFS（=BVP）+ Phase 作为双流互补输入。

    输入 csi: (n, A, T, S) complex64
    输出    : (n, S-1, F, T')  float32（keep_antenna=False）
              (n, A, S-1, F, T') float32（keep_antenna=True）
    """
    csi = np.asarray(csi)
    if csi.ndim != 4:
        raise ValueError(f"Phase BVP 输入需要 (n, A, T, S)，收到 shape={csi.shape}")
    n, A, T, S = csi.shape

    # Conjugate sanitization
    csi_san = _phase_sanitize(csi)            # (n, A, T, S-1)
    # 注意：这里 S-1 维与 T 维不同名（不要混）
    S_san = S - 1

    # 探测 shape
    probe = csi_san[: min(16, n)]
    if dc_removal:
        probe = probe - probe.mean(axis=-2, keepdims=True)
    nb_p, A_p, T_p, S_p = probe.shape
    flat_p = probe.transpose(0, 1, 3, 2).reshape(nb_p * A_p * S_p, T_p)
    # 用实部做 STFT（conjugate 输出仍为复数，real/imag 都有信息；选 real 简化）
    spec_p = stft1d(flat_p.real.astype(np.float32), n_fft=n_fft, hop=hop)
    F = spec_p.shape[-2]
    Tp = spec_p.shape[-1]

    out_shape = (n, A, S - 1, F, Tp) if keep_antenna else (n, S - 1, F, Tp)
    out = np.empty(out_shape, dtype=np.float32)

    batch_n = max(1, min(batch_n, n))
    for i in range(0, n, batch_n):
        end = min(i + batch_n, n)
        x = csi_san[i:end].real.astype(np.float32)         # (nb, A, T, S-1)
        if dc_removal:
            x = x - x.mean(axis=-2, keepdims=True)
        nb = x.shape[0]
        flat = x.transpose(0, 1, 3, 2).reshape(nb * A * S_san, T)
        spec = stft1d(flat, n_fft=n_fft, hop=hop)          # (nb*A*S_san, F, T')
        spec = spec.reshape(nb, A, S_san, F, Tp)
        if keep_antenna:
            spec = np.log(spec + eps)
        else:
            spec = np.sqrt((spec ** 2).mean(axis=1))        # (nb, S_san, F, T')
            spec = np.log(spec + eps)
        out[i:end] = spec.astype(np.float32)

    return out