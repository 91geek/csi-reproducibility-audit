"""
倒谱域（cepstrum）与对数域处理 —— 把"信道"和"人体"从相乘变成相加。

为什么这是关键的一步
--------------------
CSI 的物理模型（简化）：

    H[t, f] = H_static(f) · ( 1 + Σ_p a_p(t) · exp(-j 2π f τ_p(t)) )
               ^^^^^^^^^^     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
               房间决定的        人体各散射点（关节）贡献的时变部分
               静态信道

注意：房间和人体是**相乘**的，不是相加的。

这直接决定了两件事：

1. 雷达/医学影像里"减掉批次效应"的方法（ComBat、EA、MPC）都假设信号是
   **加性**的：X = 信号 + 批次效应。直接对 CSI 幅度用 ComBat，模型假设就是错的。
   取对数后：log|H| = log|H_static| + log|1 + 人体项| —— 变成加法，假设才成立。

2. 沿子载波做 IFFT 得到 CIR（时延域），静态多径表现为固定位置的抽头；
   再取对数 -> IFFT 得到倒谱，静态部分集中在低倒频（quefrency），
   人体反射因为路径更长、且随时间变化，落在高倒频和时变分量上。

   语音识别里用完全相同的手法分离"声道（信道）"与"声源"：
       倒谱 = IFFT(log|X(f)|)，低倒频 = 声道包络，高倒频 = 基音。
   语音界的 CMVN / 倒谱均值归一化 / liftering 就是这套东西。

本模块把这套工具搬到 CSI 上。这是我认为本领域最被低估的一条迁移路径。
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "csi_to_cir",
    "cir_to_csi",
    "real_cepstrum",
    "complex_cepstrum",
    "lifter",
    "quefrency_to_delay",
    "delay_axis",
    "to_log_domain",
    "CepstralFrontEnd",
]


def csi_to_cir(csi: np.ndarray, subcarrier_axis: int = -1, n_fft: int | None = None,
               window: str | None = "hann") -> np.ndarray:
    """CSI（频域）-> CIR（时延域），沿子载波轴做 IFFT。

    加窗（默认 Hann）可以压低因有限带宽造成的旁瓣泄漏，代价是主瓣变宽
    （时延分辨率下降 ~2 倍）。做路径分离的消融时建议把 window 也列进表格。

    注意：商品网卡上报的子载波索引通常不是均匀连续的（有导频、保护子载波、以及
    40MHz 下的两段拼接）。这里按"均匀采样"近似处理，是标准简化；要精确建模需要
    按实际子载波索引补零重采样。
    """
    x = np.moveaxis(np.asarray(csi), subcarrier_axis, -1)
    S = x.shape[-1]
    if n_fft is None:
        n_fft = S
    if window is not None:
        w = getattr(np, f"{window}")(S) if window == "hann" else np.ones(S)
        x = x * w
    cir = np.fft.ifft(x, n=n_fft, axis=-1)
    return np.moveaxis(cir, -1, subcarrier_axis)


def cir_to_csi(cir: np.ndarray, subcarrier_axis: int = -1, n_out: int | None = None) -> np.ndarray:
    """CIR -> CSI，csi_to_cir 的逆操作。"""
    x = np.moveaxis(np.asarray(cir), subcarrier_axis, -1)
    csi = np.fft.fft(x, n=n_out, axis=-1)
    return np.moveaxis(csi, -1, subcarrier_axis)


def real_cepstrum(csi: np.ndarray, subcarrier_axis: int = -1, eps: float = 1e-9) -> np.ndarray:
    """实倒谱（只取幅度谱的对数）——最常用、最稳。

        c[q] = IFFT( log|H[f]| ) 的实部

    log|H| 是实序列，其 IFFT 具有共轭对称性，实部就是完整的倒谱。
    """
    x = np.moveaxis(np.asarray(csi), subcarrier_axis, -1)
    log_mag = np.log(np.abs(x) + eps)
    c = np.fft.ifft(log_mag, axis=-1).real
    return np.moveaxis(c, -1, subcarrier_axis)


def complex_cepstrum(csi: np.ndarray, subcarrier_axis: int = -1, eps: float = 1e-9) -> np.ndarray:
    """复倒谱（保留相位信息，需要先解缠）。

        log H = log|H| + j·unwrap(∠H)
        c[q]  = IFFT( log H )

    比实倒谱信息更全，但依赖解缠正确 —— 多径环境下解缠容易出错，
    所以实践上优先用实倒谱，复倒谱作为消融对照。
    """
    x = np.moveaxis(np.asarray(csi), subcarrier_axis, -1)
    log_amp = np.log(np.abs(x) + eps)
    ph = np.unwrap(np.angle(x), axis=-1)
    c = np.fft.ifft(log_amp + 1j * ph, axis=-1).real
    return np.moveaxis(c, -1, subcarrier_axis)


def lifter(
    c: np.ndarray,
    cutoff: int = 3,
    mode: str = "high",
    quefrency_axis: int = -1,
    taper: int = 0,
) -> np.ndarray:
    """倒谱域加窗（liftering = 在 quefrency 域滤波，谐音 "filtering" 的字母换序）。

    mode="high" : 置零低倒频（|q| <= cutoff），保留高倒频。
                  -> 抹掉房间的大尺度频率选择性（慢变的谱包络）= 房间指纹。
    mode="low"  : 保留低倒频，作为对照（理论上应该更差）。

    物理含义（重要）：
        倒频 q 对应时延 τ = q / B，B 为总带宽。
        低倒频 = 短时延 = LOS 与近处强反射 —— 正是房间结构决定的部分。
        高倒频 = 长时延 = 远处/微弱的反射 —— 包含人体散射。

    所以 high-pass liftering = 在时延域上"掐掉"房间的主结构。

    cutoff 的选取：典型 40MHz、114 子载波，B ≈ 35.6 MHz，
        τ_q = q / B ≈ 28.1 ns × q   ->   路径差 8.4 m × q
    cutoff=1~3 意味着抹掉 8~25 m 以内的路径差尺度，覆盖整个房间的强反射。
    """
    c = np.asarray(c)
    Q = c.shape[quefrency_axis]
    q = np.abs(np.fft.fftfreq(Q, d=1.0 / Q))  # 0,1,2,...,Q/2,...
    q = np.round(q).astype(int)

    keep = np.ones(Q, dtype=np.float64)
    if mode == "high":
        keep[q <= cutoff] = 0.0
        if taper > 0:
            band = (q > cutoff) & (q <= cutoff + taper)
            keep[band] = 0.5 - 0.5 * np.cos(np.pi * (q[band] - cutoff) / taper)
    elif mode == "low":
        keep[q > cutoff] = 0.0
    else:
        raise ValueError(f"未知 mode: {mode}")

    shape = [1] * c.ndim
    shape[quefrency_axis] = Q
    return c * keep.reshape(shape)


def quefrency_to_delay(q: int | np.ndarray, bandwidth_hz: float) -> np.ndarray:
    """倒频索引 -> 时延（秒）。τ = q / B。"""
    return np.asarray(q) / bandwidth_hz


def delay_axis(n_fft: int, subcarrier_spacing_hz: float) -> np.ndarray:
    """CIR 的时延坐标轴（秒）。Δτ = 1 / (N · Δf) = 1 / B。"""
    return np.arange(n_fft) / (n_fft * subcarrier_spacing_hz)


def to_log_domain(
    csi: np.ndarray,
    eps: float = 1e-9,
    use_phase: bool = True,
) -> np.ndarray:
    """复数 CSI -> 实值对数域双通道 [log|H|, ∠H]。

    这是给 ComBat / EA 等"加性批次效应"方法准备的入口：
    只在 log 幅度上，房间的乘法效应才变成加法。
    """
    csi = np.asarray(csi)
    log_amp = np.log(np.abs(csi) + eps)
    if use_phase:
        return np.stack([log_amp, np.angle(csi)], axis=0)
    return log_amp[None, ...]


# --------------------------------------------------------------------------
# 组合前端：一条配置驱动的完整特征流水线
# --------------------------------------------------------------------------
class CepstralFrontEnd:
    """把"对数域 / 倒谱域 / 时延域"的选择做成一个开关，便于消融。

    mode 说明
    ---------
    "raw"        : 不做变换，直接 [幅度, 相位]
    "log"        : [log 幅度, 相位]                      <- ComBat 的正确输入域
    "cir"        : 先变到时延域，再 [幅度, 相位]
    "cepstrum"   : 实倒谱 + 高通 liftering，单通道
    "log_ceps"   : 对数域 + 倒谱 liftering 后还原（最完整的桥接方案）

    推荐实验顺序：raw -> log -> cir -> log_ceps，每一步单独看跨域增益。
    """

    MODES = ("raw", "log", "cir", "cepstrum", "log_ceps")

    def __init__(
        self,
        mode: str = "log",
        lifter_cutoff: int = 3,
        lifter_mode: str = "high",
        lifter_taper: int = 2,
        window: str | None = "hann",
        bandwidth_hz: float = 35.6e6,
        eps: float = 1e-9,
    ):
        if mode not in self.MODES:
            raise ValueError(f"mode 必须是 {self.MODES} 之一")
        self.mode = mode
        self.lifter_cutoff = lifter_cutoff
        self.lifter_mode = lifter_mode
        self.lifter_taper = lifter_taper
        self.window = window
        self.bandwidth_hz = bandwidth_hz
        self.eps = eps

    def __call__(self, csi: np.ndarray, subcarrier_axis: int = -1) -> np.ndarray:
        m = self.mode
        csi = np.asarray(csi)

        if m == "raw":
            return np.stack([np.abs(csi), np.angle(csi)], axis=0)

        if m == "log":
            return to_log_domain(csi, eps=self.eps)

        if m == "cir":
            cir = csi_to_cir(csi, subcarrier_axis=subcarrier_axis, window=self.window)
            return np.stack([np.abs(cir), np.angle(cir)], axis=0)

        if m == "cepstrum":
            c = real_cepstrum(csi, subcarrier_axis=subcarrier_axis, eps=self.eps)
            c = lifter(c, cutoff=self.lifter_cutoff, mode=self.lifter_mode,
                       quefrency_axis=subcarrier_axis, taper=self.lifter_taper)
            return c[None, ...]  # 单通道

        # log_ceps: 对数幅度谱 -> 倒谱 -> liftering -> 还原到对数幅度谱
        x = np.moveaxis(csi, subcarrier_axis, -1)
        log_mag = np.log(np.abs(x) + self.eps)      # 乘法 -> 加法的关键一步
        c = np.fft.ifft(log_mag, axis=-1).real
        c = lifter(c, cutoff=self.lifter_cutoff, mode=self.lifter_mode,
                   quefrency_axis=-1, taper=self.lifter_taper)
        log_mag_clean = np.fft.fft(c.astype(complex), axis=-1).real
        log_mag_clean = np.moveaxis(log_mag_clean, -1, subcarrier_axis)
        return np.stack([log_mag_clean, np.angle(csi)], axis=0)

    def __repr__(self) -> str:  # pragma: no cover
        if self.mode in ("cepstrum", "log_ceps"):
            return (f"CepstralFrontEnd({self.mode}, lifter_cutoff={self.lifter_cutoff}, "
                    f"mode={self.lifter_mode})")
        return f"CepstralFrontEnd({self.mode})"
