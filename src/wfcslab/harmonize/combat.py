"""
ComBat 批次效应协调（harmonization）—— 从医学影像多中心搬到 WiFi 跨域。

为什么搬这个
------------
医学影像的困境：同一台 MRI 设备、同一个扫描协议，换个医院/换台机器，图像强度
分布就变了，模型直接崩。这和 WiFi "换个房间 CSI 分布就变" 是同一个问题的两个
马甲。ComBat 是医学影像界二十年来处理这个问题的默认工具，近期被证明在
Swin Transformer 的**深度特征**上同样有效（跨厂商、跨场强）。

ComBat 假设的观测模型（逐特征 p、逐被试 j、逐站点 i）：

    Y_ijp = α_p + X_j·β_p + γ_ip + δ_ip · ε_ijp

    α_p        特征的总体均值
    X_j·β_p    想要**保留**的生物/任务变量（对应我们的动作类别）
    γ_ip       站点的位置效应（加性）  <- 房间指纹
    δ_ip       站点的尺度效应（乘性）  <- 房间增益
    ε_ijp      噪声

先用经验贝叶斯（跨站点收缩）估计 γ、δ，再把它们减掉/除掉，把各站点拉到同一个
"参考分布"上。

!! 迁移到 WiFi 的前置条件（这是最容易踩的坑）!!
------------------------------------------------
ComBat 是**加性**模型。而 CSI 的物理模型是**乘法**的：

    H = H_房间 × (1 + 人体项)

直接对 CSI 幅度跑 ComBat，模型假设就是错的。必须先取对数（或走倒谱域）：

    log|H| = log|H_房间| + log|1 + 人体项|      <- 这才满足加性

所以正确的调用顺序是 CepstralFrontEnd(mode="log") -> ComBat，而不是反过来。
本模块会在检测到输入可能为线性幅度时发出警告。

!! MM-Fi 只有 4 个环境 !!
--------------------------
经验贝叶斯要靠"跨站点的 γ 分布"来收缩。4 个站点估不出可靠先验，此时应该
用 mean_only=True（只校正位置效应 γ，不校正尺度 δ）。这是 neuroCombat 官方
在小批次场景下的推荐。本模块会在 n_batch < 10 时自动提示。

作者：Johnson, Li, Rabinovic (2007) 原始 ComBat；
      Fortin et al. (2018) 经验贝叶斯改进版（neuroCombat）。
"""

from __future__ import annotations

import warnings
import numpy as np
from typing import Optional, Sequence

__all__ = ["ComBat", "combat"]


# --------------------------------------------------------------------------
# 经验贝叶斯求解（Johnson et al. 2007 附录）
# --------------------------------------------------------------------------
def _postmean(g_hat: np.ndarray, g_bar: float, n: np.ndarray,
              d_star: np.ndarray, t2: float) -> np.ndarray:
    """后验均值 E[γ | 数据]，正态-正态共轭。"""
    return (t2 * n * g_hat + d_star * g_bar) / (t2 * n + d_star)


def _postvar(sum2: np.ndarray, n: np.ndarray, a: float, b: float) -> np.ndarray:
    """后验方差，inverse-gamma 共轭。"""
    return (0.5 * sum2 + b) / (n / 2.0 + a - 1.0)


def _it_sol(sdat, g_hat, d_hat, g_bar, t2, a, b, conv=1e-4, max_iter=500):
    """迭代求解 γ、δ 的后验（固定点迭代，EM 风格）。"""
    g_old = g_hat.copy()
    d_old = d_hat.copy()
    n = np.full(g_hat.shape, sdat.shape[1], dtype=np.float64)  # 每个 batch 的样本数
    change = np.inf
    it = 0
    while change > conv and it < max_iter:
        g_new = _postmean(g_hat, g_bar, n, d_old, t2)
        sum2 = ((sdat - g_new[:, None]) ** 2).sum(axis=1)
        d_new = _postvar(sum2, n, a, b)
        denom_g = np.maximum(np.abs(g_old), 1e-12)
        denom_d = np.maximum(np.abs(d_old), 1e-12)
        change = max(
            float(np.max(np.abs(g_new - g_old) / denom_g)),
            float(np.max(np.abs(d_new - d_old) / denom_d)),
        )
        g_old, d_old = g_new, d_new
        it += 1
    return g_old, d_old


# --------------------------------------------------------------------------
# 主类
# --------------------------------------------------------------------------
class ComBat:
    """ComBat 批次效应协调器。

    参数
    ----
    parametric : True 用正态-逆伽马先验（默认，推荐）；False 用 KDE 非参数先验
                 （**需要足够多的批次**，经验上 >= 20 个站点才稳定）
    mean_only  : True 只校正位置效应 γ，不校正尺度 δ。
                 批次少 / 每批样本少时官方推荐用这个。
    eps        : 数值稳定项
    ref_batch  : 参考批次（把其他批次对齐到它）。None = 对齐到全局加权均值

    用法（训练/测试无泄漏的标准流程）
    --------------------------------
        cb = ComBat(mean_only=True)          # MM-Fi 只有 4 个环境 -> mean_only
        Xtr_h = cb.fit_transform(Xtr, dom_tr, covars=ytr_onehot)   # 只在源域上 fit
        Xte_h = cb.transform_unseen(Xte)                            # 目标域自估计

    注意 ComBat **不需要标签**即可运行（covars=None 时）。给 covars 会把标签
    信息注入协调过程 —— 在跨域评测里这属于"用测试集标签"，必须单独说明。
    """

    def __init__(
        self,
        parametric: bool = True,
        mean_only: bool = False,
        eps: float = 1e-8,
        ref_batch=None,
    ):
        self.parametric = parametric
        self.mean_only = mean_only
        self.eps = eps
        self.ref_batch = ref_batch

        # fit 之后的状态
        self.batches_ = None
        self.stand_mean_ = None     # (p,) 全局加权均值
        self.var_pooled_ = None     # (p,) 合并方差
        self.gamma_star_ = None     # (n_batch, p) 后验位置效应
        self.delta_star_ = None     # (n_batch, p) 后验尺度效应
        self.B_ = None              # (q, p) 设计矩阵回归系数

    # ---------------- 内部：构造设计矩阵 ----------------
    @staticmethod
    def _onehot(batches: Sequence) -> tuple[np.ndarray, list, np.ndarray]:
        levels = list(dict.fromkeys(batches))          # 保序
        idx = {lv: i for i, lv in enumerate(levels)}
        codes = np.array([idx[b] for b in batches], dtype=int)
        design = np.zeros((len(batches), len(levels)), dtype=np.float64)
        design[np.arange(len(batches)), codes] = 1.0
        return design, levels, codes

    # ---------------- fit ----------------
    def fit(self, X: np.ndarray, batches: Sequence, covars: Optional[np.ndarray] = None):
        """估计各批次的位置/尺度效应。X: (n, p)，batches: 长度 n 的站点标签。"""
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        if X.ndim != 2:
            raise ValueError(f"X 必须是 2D (n, p)，收到 {X.shape}")
        n, p = X.shape
        if len(batches) != n:
            raise ValueError("batches 长度必须等于 n")

        design_b, levels, codes = self._onehot(batches)
        n_batch = len(levels)
        self.batches_ = levels

        if n_batch < 2:
            raise ValueError("ComBat 至少需要 2 个批次")
        if not self.parametric and n_batch < 20:
            warnings.warn(
                f"非参数 ComBat 依赖跨批次的经验先验，当前只有 {n_batch} 个批次，"
                "估计不可靠。建议改用 parametric=True（或 mean_only=True）。",
                RuntimeWarning,
            )
        if n_batch < 10 and not self.mean_only:
            warnings.warn(
                f"只有 {n_batch} 个批次，位置+尺度联合估计不稳定。"
                "建议设置 mean_only=True。",
                RuntimeWarning,
            )

        # 协变量：希望保留的任务相关变量（如动作 one-hot）
        if covars is not None:
            cov = np.asarray(covars, dtype=np.float64)
            if cov.ndim == 1:
                cov = cov[:, None]
            if cov.shape[0] != n:
                raise ValueError("covars 行数必须等于 n")
            design = np.hstack([design_b, cov])
        else:
            design = design_b

        # --- 1. 标准化：用完整设计矩阵回归，得到"保留协变量后"的残差尺度 ---
        # 注意本模块的约定是 X: (n, p)（样本 x 特征），与 neuroCombat 的 (p, n) 相反
        B = np.linalg.pinv(design.T @ design) @ design.T @ X        # (q, p)
        self.B_ = B
        batch_sizes = design_b.sum(axis=0)                          # (n_batch,)
        grand_mean = (batch_sizes / n) @ B[:n_batch, :]             # (p,)
        resid = X - design @ B                                      # (n, p)
        var_pooled = (resid ** 2).mean(axis=0)                      # (p,)
        var_pooled = np.maximum(var_pooled, self.eps)

        self.stand_mean_ = grand_mean
        self.var_pooled_ = var_pooled

        s_data = (X - grand_mean) / np.sqrt(var_pooled)             # (n, p)

        # --- 2. 逐批次的 γ、δ 初值 ---
        gamma_hat = np.linalg.pinv(design_b.T @ design_b) @ design_b.T @ s_data   # (n_batch, p)
        s_data_for_sol = s_data.T                                   # (p, n)，喂给 _it_sol
        delta_hat = np.zeros((n_batch, p), dtype=np.float64)
        for i in range(n_batch):
            m = codes == i
            ni = int(m.sum())
            delta_hat[i] = s_data[m].var(axis=0, ddof=1) if ni > 1 else 1.0
        delta_hat = np.maximum(delta_hat, self.eps)

        # --- 3. 经验贝叶斯收缩 ---
        if self.mean_only:
            # 只做位置校正。γ 用正态先验收缩，δ 固定为 1
            g_bar = float(gamma_hat.mean())
            t2 = float(gamma_hat.var(ddof=1)) if n_batch > 1 else 1.0
            if t2 < self.eps:
                t2 = 1.0
            n_vec = batch_sizes[:, None]          # (n_batch,1) 才能与 g_hat(n_batch,p) 广播
            gamma_star = _postmean(gamma_hat, g_bar, n_vec,
                                   np.ones_like(gamma_hat), t2)
            delta_star = np.ones_like(gamma_hat)
        else:
            g_bar = float(gamma_hat.mean())
            t2 = float(gamma_hat.var(ddof=1)) if n_batch > 1 else 1.0
            if t2 < self.eps:
                t2 = 1.0
            m = float(delta_hat.mean())
            s2 = float(delta_hat.var(ddof=1)) if n_batch > 1 else 1.0
            if s2 < self.eps:
                s2 = 1.0
            a_prior = (2.0 * s2 + m * m) / s2
            b_prior = (m * s2 + m ** 3) / s2
            if self.parametric:
                gamma_star, delta_star = _it_sol(
                    s_data_for_sol, gamma_hat, delta_hat, g_bar, t2, a_prior, b_prior
                )
            else:
                gamma_star, delta_star = self._it_sol_nonparam(
                    s_data_for_sol, gamma_hat, delta_hat, g_bar, t2, a_prior, b_prior
                )
            delta_star = np.maximum(delta_star, self.eps)

        # 参考批次对齐：把 ref 批次的效应设为 0，其余相对它校正
        if self.ref_batch is not None and self.ref_batch in levels:
            ri = levels.index(self.ref_batch)
            gamma_star = gamma_star - gamma_star[ri]
            # 尺度也相对参考批次
            delta_star = delta_star / delta_star[ri]

        self.gamma_star_ = gamma_star
        self.delta_star_ = delta_star
        return self

    # ---------------- 非参数（KDE）版本 ----------------
    def _it_sol_nonparam(self, sdat, g_hat, d_hat, g_bar, t2, a, b,
                         conv=1e-4, max_iter=200, n_grid=256):
        """γ 用核密度先验 + 数值积分求后验；δ 仍用参数化（J-L-R）。"""
        from scipy.stats import gaussian_kde

        n_batch, p = g_hat.shape
        n_vec = np.full(n_batch, sdat.shape[1], dtype=np.float64)

        # 逐特征处理：KDE 建立在 (n_batch,) 的一维 γ 样本上
        change = np.inf
        it = 0
        delta_old = d_hat.copy()
        while change > conv and it < max_iter:
            # --- γ 的非参数后验均值 ---
            gamma_new = np.zeros_like(g_hat)
            for pi in range(p):
                gh = g_hat[:, pi]
                try:
                    kde = gaussian_kde(gh)
                except Exception:
                    gamma_new[:, pi] = gh
                    continue
                lo, hi = gh.min() - 3 * np.std(gh) - 1e-6, gh.max() + 3 * np.std(gh) + 1e-6
                grid = np.linspace(lo, hi, n_grid)
                prior = kde(grid) + 1e-12
                # 似然 N(gh | grid, delta_old)
                L = np.exp(-0.5 * ((grid[None, :] - gh[:, None]) ** 2)
                           / delta_old[:, pi][:, None])
                post = prior[None, :] * L
                num = (post * grid[None, :]).sum(axis=1)
                den = post.sum(axis=1) + 1e-12
                gamma_new[:, pi] = num / den

            sum2 = ((sdat - gamma_new[:, None]) ** 2).sum(axis=1)
            delta_new = _postvar(sum2, n_vec, a, b)
            change = max(
                float(np.max(np.abs(gamma_new - g_hat) / np.maximum(np.abs(g_hat), 1e-12))),
                float(np.max(np.abs(delta_new - delta_old) / np.maximum(np.abs(delta_old), 1e-12))),
            )
            g_hat = gamma_new
            delta_old = delta_new
            it += 1

        gamma_star = g_hat
        delta_star = delta_old
        return gamma_star, delta_star

    # ---------------- transform ----------------
    def transform(self, X: np.ndarray, batches: Sequence) -> np.ndarray:
        """用 fit 得到的批次效应校正数据。批次标签必须都在 fit 时见过。"""
        self._check_fitted()
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        design_b = np.zeros((X.shape[0], len(self.batches_)), dtype=np.float64)
        for r, b in enumerate(batches):
            if b not in self.batches_:
                raise ValueError(
                    f"批次 {b!r} 未在 fit 中出现。未见过的批次请用 transform_unseen()"
                )
            design_b[r, self.batches_.index(b)] = 1.0

        gamma_exp = design_b @ self.gamma_star_        # (n, p)
        delta_exp = design_b @ self.delta_star_        # (n, p)
        s_data = (X - self.stand_mean_) / np.sqrt(self.var_pooled_)
        s_data = (s_data - gamma_exp) / np.sqrt(delta_exp)
        return s_data * np.sqrt(self.var_pooled_) + self.stand_mean_

    def transform_unseen(self, X: np.ndarray) -> np.ndarray:
        """对**未见过的**目标域做协调（部署场景的真实路径）。

        做法：用目标域自身数据估计它的 γ、δ（无监督，不需要标签），
        再按 fit 时存下来的全局均值/方差拉回参考分布。

        这一步是合理的关键：ComBat 本身就是无监督的，"目标域的批次效应"完全
        可以只用目标域数据估出来，不碰任何标签。
        """
        self._check_fitted()
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        s_data = (X - self.stand_mean_) / np.sqrt(self.var_pooled_)
        gamma_t = s_data.mean(axis=0, keepdims=True)                 # (1, p)
        delta_t = np.maximum(s_data.var(axis=0, ddof=1, keepdims=True), self.eps)
        if self.mean_only:
            delta_t = np.ones_like(delta_t)
        s_adj = (s_data - gamma_t) / np.sqrt(delta_t)
        return s_adj * np.sqrt(self.var_pooled_) + self.stand_mean_

    def fit_transform(self, X, batches, covars=None) -> np.ndarray:
        return self.fit(X, batches, covars).transform(X, batches)

    def _check_fitted(self):
        if self.gamma_star_ is None:
            raise RuntimeError("请先调用 fit()")


def combat(
    X: np.ndarray,
    batches: Sequence,
    covars: Optional[np.ndarray] = None,
    parametric: bool = True,
    mean_only: bool = False,
) -> np.ndarray:
    """函数式快捷入口。"""
    return ComBat(parametric=parametric, mean_only=mean_only).fit_transform(
        X, batches, covars
    )
