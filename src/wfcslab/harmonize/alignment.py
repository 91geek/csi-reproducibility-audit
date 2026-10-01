"""
欧氏对齐（Euclidean Alignment, EA）与 CORAL —— 从 EEG 跨被试搬到 WiFi 跨环境。

为什么搬这个
------------
EEG 的困境：每个被试的头骨导电率、电极位置、阻抗都不同，导致同一任务的脑电
协方差结构人人不一样，模型换个被试就崩。这个"每人一个分布"的结构和 WiFi
"每房一个分布"完全一致。

EA 的全部内容只有三行矩阵运算（He & Wu, IEEE TNSRE 2020）：

    R̄   = (1/N) Σ_i  X_i X_iᵀ / n_i        # 所有训练被试的平均协方差
    R̄^{-1/2}                                # 特征值分解后开根号再取逆
    X̃_i = R̄^{-1/2} X_i                      # 白化到统一坐标系

它把一个二阶统计量层面（而不是一阶均值层面）的域差异消掉。相比对抗训练：
无超参、无训练、闭式解、几分钟跑完。EEG 界几乎是标配 baseline。

对 WiFi 的适配：EEG 里 X 是 (通道 × 时间点)，一个"被试"一份数据；
WiFi 里我们把一个"域"（房间）视为一个被试，X 是 (特征维 × 样本数)，
特征维 = 我们拼出来的 CSI 特征（天线 × 子载波 × 通道）。

另外提供 CORAL（Sun et al., 2016）：把源域重新染色到目标域的协方差结构，
这是域适应里最经典的闭式解 baseline。
"""

from __future__ import annotations

import numpy as np
from typing import Sequence, Optional

__all__ = [
    "covariance",
    "inv_sqrtm",
    "sqrtm_sym",
    "EuclideanAlignment",
    "CORAL",
    "KNNMMD",
]


def covariance(X: np.ndarray, shrinkage: float = 0.0) -> np.ndarray:
    """样本协方差。X: (d, n) —— d 特征维，n 样本数。

    shrinkage: 0~1，向单位矩阵收缩（Ridge 正则）。
    当 n < d 时（WiFi 里很常见：样本数少于子载波×天线数）协方差是奇异的，
    必须加收缩，否则 R^{-1/2} 不存在。
    """
    X = np.asarray(X, dtype=np.float64)
    d, n = X.shape
    Xc = X - X.mean(axis=1, keepdims=True)
    C = (Xc @ Xc.T) / max(n - 1, 1)
    if shrinkage > 0:
        C = (1 - shrinkage) * C + shrinkage * np.trace(C) / d * np.eye(d)
    return C


def _sym(M: np.ndarray) -> np.ndarray:
    return 0.5 * (M + M.T)


def inv_sqrtm(C: np.ndarray, eps: float = 1e-8, rcond: float = 1e-10) -> np.ndarray:
    """对称正定矩阵的逆平方根 C^{-1/2}（特征值分解实现）。

    特征值会被裁剪到 >= eps，避免数值爆炸（n < d 时小特征值会接近 0）。
    """
    C = _sym(np.asarray(C, dtype=np.float64))
    w, V = np.linalg.eigh(C)
    w = np.clip(w, eps, None)
    return (V / np.sqrt(w)) @ V.T


def sqrtm_sym(C: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """对称正定矩阵的平方根 C^{1/2}。"""
    C = _sym(np.asarray(C, dtype=np.float64))
    w, V = np.linalg.eigh(C)
    w = np.clip(w, eps, None)
    return (V * np.sqrt(w)) @ V.T


class EuclideanAlignment:
    """欧氏对齐。

    两种模式
    --------
    transductive（默认）: fit 时把所有域（含目标域）的协方差一起平均。
                         更简单，且完全无监督 —— 不需要目标域标签，
                         所以**不构成标签泄漏**，符合跨域评测规范。
    inductive (rEA)     : 只用源域算 R̄，对目标域单独用自己的协方差白化。
                         更贴近在线部署。

    参数
    ----
    mode      : "transductive" | "inductive"
    shrinkage : 协方差收缩系数；样本数 < 特征维时建议 0.1~0.5
    eps       : 特征值裁剪下限
    """

    def __init__(self, mode: str = "transductive", shrinkage: float = 0.1,
                 eps: float = 1e-8):
        if mode not in ("transductive", "inductive"):
            raise ValueError("mode 必须是 'transductive' 或 'inductive'")
        self.mode = mode
        self.shrinkage = shrinkage
        self.eps = eps
        self.R_inv_sqrt_: Optional[np.ndarray] = None
        self.per_domain_: dict = {}

    def fit(self, domains: Sequence[np.ndarray]) -> "EuclideanAlignment":
        """domains: 每个元素是一份 (d, n_i) 的数据矩阵（一个域/一个房间）。"""
        if len(domains) == 0:
            raise ValueError("至少需要一个域")

        if self.mode == "transductive":
            # 把所有域的协方差等权平均，再开逆平方根
            Cs = [covariance(X, self.shrinkage) for X in domains]
            R = np.mean(Cs, axis=0)
            self.R_inv_sqrt_ = inv_sqrtm(R, eps=self.eps)
        else:
            # inductive：每个域用自己的协方差白化（rEA，He & Wu 2022）
            self.per_domain_ = {}
            for i, X in enumerate(domains):
                self.per_domain_[i] = inv_sqrtm(
                    covariance(X, self.shrinkage), eps=self.eps
                )
        return self

    def transform(self, X: np.ndarray, domain_id: int = 0) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        if self.mode == "transductive":
            if self.R_inv_sqrt_ is None:
                raise RuntimeError("请先 fit()")
            return self.R_inv_sqrt_ @ X
        key = domain_id
        if key not in self.per_domain_:
            # 未见过的域：用它自己的协方差白化（无监督，合法）
            self.per_domain_[key] = inv_sqrtm(covariance(X, self.shrinkage), eps=self.eps)
        return self.per_domain_[key] @ X

    def fit_transform(self, domains: Sequence[np.ndarray]) -> list[np.ndarray]:
        self.fit(domains)
        if self.mode == "transductive":
            return [self.transform(X) for X in domains]
        return [self.transform(X, i) for i, X in enumerate(domains)]


class CORAL:
    """CORAL：把源域特征重新染色到目标域的协方差结构。

        X_src' = C_tgt^{1/2} · C_src^{-1/2} · X_src

    闭式解、无超参。跨域视觉里的经典 baseline，在 WiFi 上没人用过。
    通常配合一阶统计量对齐（减均值 + 除标准差）一起用，这里已内置。
    """

    def __init__(self, shrinkage: float = 0.1, eps: float = 1e-8,
                 standardize: bool = True):
        self.shrinkage = shrinkage
        self.eps = eps
        self.standardize = standardize

    def fit(self, X_src: np.ndarray, X_tgt: np.ndarray) -> "CORAL":
        self.mu_s_ = X_src.mean(axis=1, keepdims=True)
        self.mu_t_ = X_tgt.mean(axis=1, keepdims=True)
        Cs = covariance(X_src - self.mu_s_, self.shrinkage)
        Ct = covariance(X_tgt - self.mu_t_, self.shrinkage)
        self.A_ = sqrtm_sym(Ct, self.eps) @ inv_sqrtm(Cs, self.eps)
        self.sd_t_ = np.sqrt(np.clip(np.diag(Ct), self.eps, None))[:, None]
        self.sd_s_ = np.sqrt(np.clip(np.diag(Cs), self.eps, None))[:, None]
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        Z = X - self.mu_s_
        Z = self.A_ @ Z
        if self.standardize:
            Z = Z / self.sd_s_ * self.sd_t_
        return Z + self.mu_t_

    def fit_transform(self, X_src, X_tgt) -> np.ndarray:
        return self.fit(X_src, X_tgt).transform(X_src)


# --------------------------------------------------------------------------
class KNNMMD:
    """KNN-MMD：按 K 个最近邻做局部均值对齐。

    论文
    ----
    Kang et al., "KNN-MMD: A Domain Adaptation Method for Knowledge
    Graph Completion", AAAI 2025 (arXiv:2412.04783).

    核心思想
    --------
    全局协方差对齐（EA/CORAL/ComBat）把"每个域"视为一个高斯，但 WiFi 房间
    间的真实分布偏移**不是均匀的**——某些子载波上的多径结构高度域相关，
    另一些（动作主信号所在）几乎不漂移。全局对齐会"过校正"主信号，反而
    损伤判别信息（F-03/F-04 已观察到这点）。

    KNN-MMD 的解法：
    1) 对每个训练样本，按特征余弦相似度找它在目标域中的 K 个最近邻
    2) 对每个训练样本，把它「重染色」到这 K 个邻域样本的局部均值/方差
    3) 只对训练样本做变换；目标域原样不动

    这种"局部、对齐到目标域的近邻"是 transductive 的（用了目标域信息），
    但**不依赖任何标签**，符合跨域评测规范。

    参数
    ----
    k           : 最近邻个数。论文默认 5~10，10 在多数任务上最稳。
    eps         : 数值稳定项
    """

    def __init__(self, k: int = 10, eps: float = 1e-8):
        self.k = k
        self.eps = eps
        # fit 时存（用来 transform 训练样本；测试样本保持不变）
        self.X_src_ = None
        self.X_tgt_ = None

    def fit(self, X_src: np.ndarray, X_tgt: np.ndarray) -> "KNNMMD":
        """X_src: (d, n_src)，X_tgt: (d, n_tgt)。d=特征维。"""
        X_src = np.asarray(X_src, dtype=np.float64)
        X_tgt = np.asarray(X_tgt, dtype=np.float64)
        if X_src.shape[0] != X_tgt.shape[0]:
            raise ValueError(
                f"KNN-MMD 要求源/目标同特征维：got {X_src.shape[0]} vs {X_tgt.shape[0]}")
        self.X_src_ = X_src
        self.X_tgt_ = X_tgt
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """对 X（按行样本）做 KNN-MMD 对齐。

        简化实现
        --------
        由于特征维数 d 在 ComBat 130s/折那种量级下（~130 PCA 维），
        我们直接用 numpy 计算 pairwise 余弦相似度（d×n_src · d×n_tgt），
        复杂度 O(d · n_src · n_tgt)。10 折平均每折 ~5k 训练样本 vs ~750
        测试样本 = 3.75M 次内积，CPU < 1s。
        """
        if self.X_src_ is None:
            raise RuntimeError("请先 fit()")
        X = np.asarray(X, dtype=np.float64)
        # X 是 (d, n)，处理成 (n, d) 计算
        if X.shape[0] == self.X_src_.shape[0]:
            X_T = X.T  # (n, d)
        else:
            X_T = X

        # 余弦相似度：sim[i, j] = <X_T[i], X_tgt_T[j]> / (||X_T[i]|| * ||X_tgt_T[j]||)
        X_tgt_T = self.X_tgt_.T  # (n_tgt, d)
        x_norm = np.linalg.norm(X_T, axis=1, keepdims=True) + self.eps
        t_norm = np.linalg.norm(X_tgt_T, axis=1, keepdims=True) + self.eps
        sim = (X_T @ X_tgt_T.T) / (x_norm * t_norm.T)  # (n, n_tgt)
        # 取 top-k
        k = min(self.k, sim.shape[1])
        knn_idx = np.argpartition(-sim, kth=k - 1, axis=1)[:, :k]  # (n, k)
        # 邻居均值/方差
        knn = self.X_tgt_[:, knn_idx.T].transpose(1, 0, 2)  # (n, d, k) → transpose to (n, k, d)? Easier:
        # self.X_tgt_[:, knn_idx] 是 (d, n, k)，拆开
        neighbor = self.X_tgt_[:, knn_idx]                  # (d, n, k)
        local_mu = neighbor.mean(axis=2)                    # (d, n)
        local_sd = neighbor.std(axis=2) + self.eps          # (d, n)
        src_sd = X.std(axis=1, keepdims=True) + self.eps    # (d, 1)
        # 染色：x ← (x - μ_x) / σ_x · σ_tgt + μ_tgt
        X_new = (X - X.mean(axis=1, keepdims=True)) / src_sd * local_sd + local_mu
        return X_new.astype(np.float32)

    def fit_transform(self, X_src: np.ndarray, X_tgt: np.ndarray) -> np.ndarray:
        return self.fit(X_src, X_tgt).transform(X_src)
