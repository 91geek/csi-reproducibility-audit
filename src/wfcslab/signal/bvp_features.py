"""
BVP Features 统一高层 API
==========================

为什么需要这个模块
=================
A 的核心论点是：**BVP 算子的普适性 > 模型架构差异**。要让这个论点可被复现，
需要把 BVP 算子抽象成"高层 API + 预设配方"，让任何人在任何数据集上都能
用同样一行代码跑出可比的结果。

本模块提供：
1. ``extract_bvp_features()`` —— 统一接口，自动选 CPU/GPU，接受 CSI numpy
2. ``BVPRecipe`` 枚举 —— 三个预设配方，对应 F-21 / F-24 / Widar3.0 原版
3. ``list_recipes()`` —— 列出所有可用配方（给论文方法描述用）

为什么有 "F-21 / F-24 / Widar3.0 原版" 这三个配方
================================================
F-21 / F-24 是我们自己 ablation 出来的"偏离原版"的两个变体：
- F-21 = Widar3.0 原版 + keep_antenna=True (多 Rx 独立 BVP)
- F-24 = F-21 + 多尺度 (n_fft=64+128 拼接)

Widar3.0 原版 = [Zheng et al. 2019] 论文用的标准配置：
- hop=16, n_fft=64, RMS 聚合 (keep_antenna=False), log 压缩

用法
----
    from wfcslab.signal.bvp_features import extract_bvp_features, BVPRecipe

    X_bvp = extract_bvp_features(ds.X, recipe=BVPRecipe.WIDAR_ORIG)
    X_bvp = extract_bvp_features(ds.X, recipe=BVPRecipe.F21_MULTI_RX)
    X_bvp = extract_bvp_features(ds.X, recipe=BVPRecipe.F24_MULTISCALE)

输入约定
--------
    csi: (n, A, T, S) complex64   A=天线, T=慢时间, S=子载波
        （与 wfcslab.data.{widar,mmfi,signfi,..} loader 输出 shape 一致）

输出约定
--------
    widar_orig:    (n, S, F, T')    float32   单尺度+聚合
    f21_multi_rx:  (n, A, S, F, T') float32   单尺度+多 Rx
    f24_multiscale: (n, A, S*2, F_min, T_min) float32  多尺度+多 Rx（沿 S 拼接 64/128）

内存
----
CPU 版 12000 样本 5D 输出约 28GB（float32），按 batch_n 分块。
GPU 版 5D 输出若不切 batch_n 会 OOM，必须用 compute_bvp_gpu。
"""

from __future__ import annotations

import enum
import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


__all__ = ["BVPRecipe", "extract_bvp_features", "list_recipes"]


class BVPRecipe(enum.Enum):
    """BVP 预设配方。

    论文方法章节可以直接引用这张表，避免读者迷失在 n_fft/hop/keep_antenna 的组合中。
    """

    WIDAR_ORIG = "widar_orig"
    """Widar3.0 原版 [Zheng et al. 2019]：hop=16, n_fft=64, RMS 聚合, log 压缩。
    输出 (n, S, F, T')。
    """

    F21_MULTI_RX = "f21_multi_rx"
    """我们的 F-21 变体：hop=8, n_fft=64, keep_antenna=True（多 Rx 独立 BVP）。
    输出 (n, A, S, F, T')，需要 backbone 支持 5D 输入（如 LeNetCSI_Attn）。
    """

    F24_MULTISCALE = "f24_multiscale"
    """我们的 F-24 变体：F-21 + 多尺度（n_fft=64+128 沿 S 拼接）。
    输出 (n, A, S*2, F_min, T_min)，仍是 5D（keep_antenna=True），需 backbone 支持 5D。
    """


_RECIPES_SPEC: dict[BVPRecipe, dict] = {
    BVPRecipe.WIDAR_ORIG: dict(
        hop=16, n_ffts=(64,), keep_antenna=False,
        description="Widar3.0 原版: hop=16, n_fft=64, RMS 聚合, log 压缩",
    ),
    BVPRecipe.F21_MULTI_RX: dict(
        hop=8, n_ffts=(64,), keep_antenna=True,
        description="F-21 变体: hop=8, n_fft=64, keep_antenna=True (多 Rx)",
    ),
    BVPRecipe.F24_MULTISCALE: dict(
        hop=8, n_ffts=(64, 128), keep_antenna=True,
        description="F-24 变体: hop=8, n_fft=64+128 多尺度拼接, keep_antenna=True",
    ),
}


def list_recipes() -> list[tuple[str, str]]:
    """返回 ``[(recipe_name, description), ...]`` 列表，给论文 / 文档使用。"""
    return [(r.value, _RECIPES_SPEC[r]["description"]) for r in BVPRecipe]


# --------------------------------------------------------------------------
# 选 backend: GPU 优先，回退 CPU
# --------------------------------------------------------------------------
def _try_gpu_bvp(
    csi_t: "torch.Tensor",
    hop: int, n_ffts: tuple, keep_antenna: bool,
    batch_n: int,
) -> "torch.Tensor":
    """调用 GPU 版 BVP，必要时拼接多尺度。返回的是 CPU tensor（torch）。"""
    from .doppler_gpu import compute_bvp_gpu
    if len(n_ffts) == 1:
        return compute_bvp_gpu(
            csi_t, n_fft=n_ffts[0], hop=hop, keep_antenna=keep_antenna,
            batch_n=batch_n, return_device="cpu",
        )
    # 多尺度：每个 n_fft 单独算，再沿 S 维拼接
    specs = [
        compute_bvp_gpu(
            csi_t, n_fft=nf, hop=hop, keep_antenna=keep_antenna,
            batch_n=batch_n, return_device="cpu",
        )
        for nf in n_ffts
    ]
    F_min = min(s.shape[-2] for s in specs)
    T_min = min(s.shape[-1] for s in specs)
    return torch.cat([s[..., :F_min, :T_min] for s in specs], dim=1)


def _cpu_bvp(
    csi: np.ndarray,
    hop: int, n_ffts: tuple, keep_antenna: bool,
    batch_n: int,
) -> np.ndarray:
    """CPU 版 BVP，支持多尺度拼接。返回 numpy。"""
    from .doppler import compute_bvp
    if len(n_ffts) == 1:
        return compute_bvp(csi, n_fft=n_ffts[0], hop=hop,
                           keep_antenna=keep_antenna, batch_n=batch_n)
    specs = [
        compute_bvp(csi, n_fft=nf, hop=hop, keep_antenna=keep_antenna, batch_n=batch_n)
        for nf in n_ffts
    ]
    F_min = min(s.shape[-2] for s in specs)
    T_min = min(s.shape[-1] for s in specs)
    return np.concatenate([s[..., :F_min, :T_min] for s in specs], axis=1)


# --------------------------------------------------------------------------
# 高层 API
# --------------------------------------------------------------------------
def extract_bvp_features(
    csi: np.ndarray,
    recipe: BVPRecipe | str = BVPRecipe.WIDAR_ORIG,
    *,
    device: str = "auto",
    batch_n: int = 64,
    return_tensor: bool = False,
) -> np.ndarray:
    """根据 recipe 提取 BVP 谱图。

    参数
    ----
    csi : (n, A, T, S) complex64 numpy
    recipe : ``BVPRecipe`` 或字符串（"widar_orig" / "f21_multi_rx" / "f24_multiscale"）
    device : "auto" / "cuda" / "cpu"
        - "auto": 有 CUDA 就用 GPU，否则 CPU
        - "cuda": 强制 GPU（无 CUDA 会报错）
        - "cpu": 强制 CPU
    batch_n : GPU 时按样本切块；CPU 默认 512（一般不 OOM）
    return_tensor : 若 True 且 device=cuda，返回 torch.Tensor（仍在 GPU 上）；否则返回 numpy

    返回
    ----
    out : numpy ndarray（默认）或 torch.Tensor
        形状依 recipe 而定（见 BVPRecipe 文档）
    """
    if isinstance(recipe, str):
        recipe = BVPRecipe(recipe)

    if recipe not in _RECIPES_SPEC:
        raise ValueError(
            f"未知 recipe={recipe!r}，可选: {[r.value for r in BVPRecipe]}"
        )

    spec = _RECIPES_SPEC[recipe]
    hop = spec["hop"]
    n_ffts = spec["n_ffts"]
    keep_antenna = spec["keep_antenna"]

    csi = np.asarray(csi)
    if csi.ndim != 4:
        raise ValueError(f"BVP 输入需要 (n, A, T, S)，收到 shape={csi.shape}")

    # 决定 device
    use_gpu = False
    if device == "auto":
        try:
            import torch
            use_gpu = torch.cuda.is_available()
        except ImportError:
            use_gpu = False
    elif device == "cuda":
        use_gpu = True
    elif device != "cpu":
        raise ValueError(f"device 必须是 'auto'/'cuda'/'cpu'，收到 {device!r}")

    if use_gpu:
        import torch
        csi_t = torch.as_tensor(csi, dtype=torch.complex64, device="cuda")
        out_t = _try_gpu_bvp(csi_t, hop, n_ffts, keep_antenna, batch_n=batch_n)
        if return_tensor:
            return out_t
        return out_t.cpu().numpy()
    else:
        return _cpu_bvp(csi, hop, n_ffts, keep_antenna, batch_n=batch_n or 512)


# --------------------------------------------------------------------------
# 自测（仅在直接跑本文件时执行）
# --------------------------------------------------------------------------
if __name__ == "__main__":
    # 直接跑本文件时，相对 import 会失败；用 sys.path hack 切到 src/
    import sys, os
    HERE = os.path.dirname(os.path.abspath(__file__))
    SRC = os.path.abspath(os.path.join(HERE, "..", ".."))
    if SRC not in sys.path:
        sys.path.insert(0, SRC)
    from wfcslab.signal.bvp_features import (  # noqa: E402
        extract_bvp_features, list_recipes, BVPRecipe
    )

    print("Available BVP recipes:")
    for name, desc in list_recipes():
        print(f"  - {name}: {desc}")

    rng = np.random.default_rng(0)
    n, A, T, S = 32, 9, 256, 30
    csi = (rng.standard_normal((n, A, T, S))
           + 1j * rng.standard_normal((n, A, T, S))).astype(np.complex64)

    print(f"\nInput shape: {csi.shape}")
    for recipe in BVPRecipe:
        out = extract_bvp_features(csi, recipe=recipe, device="cpu", batch_n=16)
        print(f"  {recipe.value}: {out.shape}, dtype={out.dtype}, "
              f"range=[{out.min():.2f}, {out.max():.2f}]")
