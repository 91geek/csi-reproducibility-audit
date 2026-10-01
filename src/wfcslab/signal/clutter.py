"""
雷达式静态杂波抑制（MTI / STAP 家族）——移植到 WiFi CSI。

为什么需要：
    CSI 里绝大部分能量来自静止环境（墙、家具、地板的反射），人体运动只是叠加在
    强静态基线上的微弱扰动。换一个房间 = 整条静态基线全变 = 你精心训出来的扰动
    模式全部失效。这个问题在雷达里叫 clutter，1960 年代就有确定性的解法。

    本模块的所有方法都满足三条：无训练、无超参搜索、推理时可流式执行。
    因此可以作为"插件"插在任何已有方法（AdaPose / DT-Pose / RePos）前面。

约定：
    CSI 复数组形状 (..., T, A, S)
        T = 慢时间（包序号 / 帧）
        A = 天线（接收天线数，MM-Fi 为 3）
        S = 子载波（MM-Fi 为 114）
    默认 time axis = -3。

术语对照（雷达 -> WiFi）：
    range gate / fast time  ->  子载波 / 时延域（CSI -> CIR 的 IFFT）
    pulse / slow time       ->  包序号 T
    clutter                 ->  静止环境反射
    moving target           ->  人体
    MTI (Moving Target Indication)      ->  沿慢时间高通滤波
    STAP (Space-Time Adaptive Processing) -> 慢时间 x（天线 x 子载波）联合降秩
"""

from __future__ import annotations

import numpy as np
from typing import Optional, Tuple

__all__ = [
    "mti_delay_line",
    "mpc",
    "ema_clutter",
    "svd_clutter",
    "zero_doppler_notch",
    "ClutterFilter",
]


# --------------------------------------------------------------------------
# 内部工具：把慢时间轴挪到最前，展平成 (T, M)，处理完再还原
# --------------------------------------------------------------------------
def _time_first(x: np.ndarray, time_axis: int) -> Tuple[np.ndarray, tuple]:
    xm = np.moveaxis(np.asarray(x), time_axis, 0)
    shape = xm.shape
    return xm.reshape(shape[0], -1), shape


def _restore(x2: np.ndarray, shape: tuple, time_axis: int) -> np.ndarray:
    return np.moveaxis(x2.reshape(shape), 0, time_axis)


# --------------------------------------------------------------------------
# 1. 延迟线对消器（最经典的 MTI）
# --------------------------------------------------------------------------
def mti_delay_line(x: np.ndarray, order: int = 1, time_axis: int = -3) -> np.ndarray:
    """单/双延迟线对消器。

        order=1:  y[t] = x[t] - x[t-1]                   （一阶差分）
        order=2:  y[t] = x[t] - 2 x[t-1] + x[t-2]        （二阶差分）

    频率响应 H(f) = 1 - e^{-j2πf}（一阶），在 f=0（零多普勒 / 静止散射体）处有
    精确零点，即完全抑制完全静止的反射。二阶陷波更深更宽，但会连带削弱慢速
    运动分量——对"站姿微调""缓慢呼吸"这类任务要慎用。

    注意：这是一个高通滤波器，输出是"变化量"而非"信道本身"。若下游模型依赖
    绝对幅度信息（比如基于幅度衰减测距），不要单独用它。
    """
    x2, shape = _time_first(x, time_axis)
    T = x2.shape[0]
    if T <= order:
        raise ValueError(f"慢时间长度 T={T} 必须大于 order={order}")

    y = x2.copy()
    if order >= 1:
        y[1:] = x2[1:] - x2[:-1]
        y[0] = 0
    if order >= 2:
        y[2:] = x2[2:] - 2.0 * x2[1:-1] + x2[:-2]
        y[1] = 0
    return _restore(y, shape, time_axis)


# --------------------------------------------------------------------------
# 2. 均值相量对消（MPC，雷达里叫 Mean / DC Phasor Cancellation）
# --------------------------------------------------------------------------
def mpc(
    x: np.ndarray,
    time_axis: int = -3,
    estimator: str = "mean",
    eps: float = 1e-12,
) -> np.ndarray:
    """沿慢时间估计每个 (天线, 子载波) 单元的静态杂波相量并减掉。

        c[m] = mean_t  x[t, m]          （或 median）
        y[t, m] = x[t, m] - c[m]

    这是 WiFi 感知领域"背景去除 / PCA 背景去除"的复数版本，也是雷达里最朴素的
    杂波对消。与 PCA 背景去除的区别：PCA 是在 (T x M) 矩阵上做低秩逼近（见
    svd_clutter），MPC 只看秩 1（沿时间的常量）。

    estimator="median" 对偶发的 AGC 跳变、丢包尖峰更鲁棒（类似 Hampel 的思路）。
    """
    x2, shape = _time_first(x, time_axis)
    if estimator == "mean":
        c = x2.mean(axis=0, keepdims=True)
    elif estimator == "median":
        c = np.median(x2.real, axis=0, keepdims=True) + 1j * np.median(
            x2.imag, axis=0, keepdims=True
        )
    else:
        raise ValueError(f"未知 estimator: {estimator}")
    return _restore(x2 - c, shape, time_axis)


# --------------------------------------------------------------------------
# 3. 指数滑动平均自适应杂波估计（可流式执行）
# --------------------------------------------------------------------------
def ema_clutter(
    x: np.ndarray,
    alpha: float = 0.95,
    time_axis: int = -3,
    warmup: Optional[int] = None,
) -> np.ndarray:
    """用一阶 IIR 递归估计并跟踪缓慢漂移的杂波基线。

        c[t] = alpha * c[t-1] + (1 - alpha) * x[t]
        y[t] = x[t] - c[t]

    与 MPC 的区别：MPC 需要一整段数据才能算均值（batch / 非因果），EMA 是因果的，
    可以在线跑。alpha 越大记忆越长、陷波越窄（只抑制非常慢的漂移）。

    建议：alpha = 1 - 1/(tau * fs)，tau 为想抑制的杂波时间尺度（秒），fs 为包率。
    例如 fs=100Hz、想抑制 10 秒尺度以上的漂移 -> alpha = 1 - 1/1000 = 0.999。
    """
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha 必须在 (0, 1) 之间")
    x2, shape = _time_first(x, time_axis)
    T, M = x2.shape
    c = np.empty_like(x2)
    c[0] = x2[0]
    for t in range(1, T):
        c[t] = alpha * c[t - 1] + (1.0 - alpha) * x2[t]
    y = x2 - c
    if warmup is not None:  # 前 warmup 帧杂波估计不可靠，置零
        y[:warmup] = 0
    return _restore(y, shape, time_axis)


# --------------------------------------------------------------------------
# 4. 慢时间 SVD 降秩（STAP 的简化版）
# --------------------------------------------------------------------------
def svd_clutter(
    x: np.ndarray,
    rank: int = 1,
    time_axis: int = -3,
    return_basis: bool = False,
):
    """投影掉 X (T x M) 的前 rank 个主方向。

    物理依据：静止散射体被"所有"天线和子载波共同观测到，因此它在不同 (A, S)
    通道间高度相关，在数据矩阵里占据最大的奇异值方向；而人体散射只影响局部
    天线/子载波且随时间变化，能量分散在更小的奇异值上。投影掉前 rank 个方向
    = 简化的空时自适应处理（STAP）。

    rank=1 等价于 MPC 的加权版本；rank>1 能同时吃掉多个强静态反射簇。

    训练/推理一致性：这是 transductive 操作（需要整段数据）。部署时应先用一段
    空场景（或静止几秒）数据 fit 出基向量 V，再对后续数据 transform。
    """
    x2, shape = _time_first(x, time_axis)
    if x2.shape[0] < rank + 1:
        raise ValueError("慢时间长度不足以做该秩的分解")
    # 经济型 SVD：X = U diag(s) Vt
    _, _, Vt = np.linalg.svd(x2, full_matrices=False)
    basis = Vt[:rank]                      # (rank, M) 杂波子空间的正交基
    proj = basis.conj().T @ basis          # 投影矩阵 (M, M)
    y = x2 - x2 @ proj
    if return_basis:
        return _restore(y, shape, time_axis), basis
    return _restore(y, shape, time_axis)


# --------------------------------------------------------------------------
# 5. 零多普勒陷波（频域版 MTI，陷波形状可控）
# --------------------------------------------------------------------------
def zero_doppler_notch(
    x: np.ndarray,
    fs: float = 100.0,
    cutoff_hz: float = 0.5,
    time_axis: int = -3,
    taper: bool = True,
) -> np.ndarray:
    """沿慢时间做 FFT，把 |f| < cutoff_hz 的多普勒单元置零后反变换。

    与 mti_delay_line 的区别：MTI 是固定系数 FIR，陷波形状写死；这里可以精确
    指定"多慢算静止"。cutoff_hz 的选择要匹配人体运动的多普勒频率：

        f_d = 2 v / lambda，  lambda = c / f_c
        5.8 GHz -> lambda ≈ 5.17 cm
        v = 0.05 m/s（缓慢抬手） -> f_d ≈ 1.9 Hz
        v = 1.0  m/s（走路挥手） -> f_d ≈ 38.7 Hz

    所以 cutoff_hz = 0.5~2 Hz 能滤掉环境热漂移和极慢的家具位移，同时保住动作。
    注意这会同时滤掉"静止姿态"（如站立不动），评估时必须说明这一点。

    taper=True 时用余弦过渡带代替硬切，避免时域振铃（Gibbs 现象）。
    """
    x2, shape = _time_first(x, time_axis)
    T = x2.shape[0]
    Xf = np.fft.fft(x2, axis=0)
    freqs = np.fft.fftfreq(T, d=1.0 / fs)
    af = np.abs(freqs)

    keep = np.ones(T, dtype=np.float64)
    if taper:
        # |f| <= cutoff*0.5 全切，cutoff*0.5 ~ cutoff 余弦过渡，>cutoff 全留
        lo, hi = 0.5 * cutoff_hz, cutoff_hz
        band = (af > lo) & (af < hi)
        keep[band] = 0.5 - 0.5 * np.cos(np.pi * (af[band] - lo) / (hi - lo))
    keep[af <= 0.5 * cutoff_hz] = 0.0
    y = np.fft.ifft(Xf * keep[:, None], axis=0)
    return _restore(y, shape, time_axis)


# --------------------------------------------------------------------------
# 统一接口：便于在配置文件里一行切换
# --------------------------------------------------------------------------
class ClutterFilter:
    """静态杂波抑制的统一封装，支持 fit / transform 分离（部署友好）。

    用法
    ----
        cf = ClutterFilter(method="mpc", time_axis=-3)
        x_tr = cf.fit_transform(x_tr)      # 训练：batch 模式
        x_te = cf.transform(x_te)          # 测试：用 fit 出来的基（仅 svd 需要）

    参数
    ----
    method : {"none", "mti1", "mti2", "mpc", "mpc_median", "ema", "svd", "doppler"}
    rank   : svd 方法的杂波子空间秩
    alpha  : ema 方法的遗忘因子
    fs, cutoff_hz : doppler 方法的采样率与陷波截止
    """

    METHODS = ("none", "mti1", "mti2", "mpc", "mpc_median", "ema", "svd", "doppler")

    def __init__(
        self,
        method: str = "mpc",
        time_axis: int = -3,
        rank: int = 1,
        alpha: float = 0.95,
        fs: float = 100.0,
        cutoff_hz: float = 0.5,
        warmup: Optional[int] = None,
    ):
        if method not in self.METHODS:
            raise ValueError(f"method 必须是 {self.METHODS} 之一，收到 {method!r}")
        self.method = method
        self.time_axis = time_axis
        self.rank = rank
        self.alpha = alpha
        self.fs = fs
        self.cutoff_hz = cutoff_hz
        self.warmup = warmup
        self.basis_ = None

    def fit(self, x: np.ndarray) -> "ClutterFilter":
        """仅 svd 需要 fit：从（静止/训练）数据中估计杂波子空间基。"""
        if self.method == "svd":
            _, self.basis_ = svd_clutter(
                x, rank=self.rank, time_axis=self.time_axis, return_basis=True
            )
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        m = self.method
        if m == "none":
            return np.asarray(x)
        if m == "mti1":
            return mti_delay_line(x, order=1, time_axis=self.time_axis)
        if m == "mti2":
            return mti_delay_line(x, order=2, time_axis=self.time_axis)
        if m == "mpc":
            return mpc(x, time_axis=self.time_axis, estimator="mean")
        if m == "mpc_median":
            return mpc(x, time_axis=self.time_axis, estimator="median")
        if m == "ema":
            return ema_clutter(x, alpha=self.alpha, time_axis=self.time_axis,
                               warmup=self.warmup)
        if m == "doppler":
            return zero_doppler_notch(x, fs=self.fs, cutoff_hz=self.cutoff_hz,
                                      time_axis=self.time_axis)

        # svd：优先复用 fit 得到的基，保证训练/推理一致
        x2, shape = _time_first(x, self.time_axis)
        if self.basis_ is not None and self.basis_.shape[1] == x2.shape[1]:
            proj = self.basis_.conj().T @ self.basis_
            y = x2 - x2 @ proj
        else:
            y = svd_clutter(x, rank=self.rank, time_axis=self.time_axis)
            return y
        return _restore(y, shape, self.time_axis)

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        return self.fit(x).transform(x)

    def __repr__(self) -> str:  # pragma: no cover
        extra = ""
        if self.method == "svd":
            extra = f", rank={self.rank}"
        elif self.method == "ema":
            extra = f", alpha={self.alpha}"
        elif self.method == "doppler":
            extra = f", cutoff={self.cutoff_hz}Hz"
        return f"ClutterFilter({self.method}{extra})"
