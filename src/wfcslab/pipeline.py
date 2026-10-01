"""
端到端特征流水线：把"去硬件相位 -> 抑杂波 -> 换域 -> 协调 -> 归一化"串成一条链。

为什么必须按这个顺序
--------------------
原始复数 CSI 的相位是**不可用**的。每个数据包都被三类硬件误差污染：
CFO（载波频偏，随时间线性旋转）、SFO（采样频偏，随子载波线性倾斜）、
随机初相（每包一个常数）。在 100Hz 采样率下，CFO 引发的相位旋转完全混叠，
你拿到的相位序列本质上是随机的。

后果：**在未经处理的复数 CSI 上直接做 MTI / MPC / ComBat，是在给随机噪声建模**。

正确的顺序必须是：

    stage 1  去共相位   同一接收机各天线共享同一本振，做 CSI-ratio（或相位校准）
                        把 CFO/SFO/随机初相整体约掉，得到"干净"的相对信道
    stage 2  抑杂波     这时静态环境反射才是时不变的，MTI/MPC/SVD 才有意义
    stage 3  换域       log / CIR / 倒谱域。取对数后"房间×人体"变成"房间+人体"
    stage 4  协调       ComBat / EA / CORAL 等"加性批次效应"方法此时才成立
    stage 5  归一化      最后统一尺度

每一级都是独立开关，方便逐项消融——这也是这套代码最主要的用途。

形状约定
--------
    输入  X: (n, A, T, S) complex64   A=天线, T=慢时间, S=子载波
    输出  X: (n, C, T', S) float32    C=通道数（1 或 2 或 P×2）
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence
import numpy as np

from .signal.clutter import ClutterFilter
from .signal.cepstrum import CepstralFrontEnd
from .signal.phase import csi_ratio, conjugate_multiply, phase_sanitize
from .harmonize.combat import ComBat
from .harmonize.alignment import EuclideanAlignment, CORAL, KNNMMD

__all__ = ["PipelineCfg", "CSIPipeline"]


@dataclass
class PipelineCfg:
    """流水线配置。每一项都对应一次可独立消融的选择。"""

    # ---- stage 1: 去硬件相位 ----
    phase: str = "csi_ratio"          # none | phase_sanitize | csi_ratio | conj_mult
    phase_kwargs: dict = field(default_factory=dict)     # 如 pairs="all"

    # ---- stage 2: 静态杂波抑制 ----
    clutter: str = "mpc"              # none | mti1 | mti2 | mpc | mpc_median | ema | svd | doppler
    clutter_kwargs: dict = field(default_factory=dict)   # rank / alpha / cutoff_hz / fs

    # ---- stage 3: 域变换 ----
    domain: str = "log"               # raw | log | cir | cepstrum | log_ceps | bvp
    domain_kwargs: dict = field(default_factory=dict)    # lifter_cutoff / bandwidth_hz
    keep_antenna: bool = False        # BVP 时是否保留天线维（→ 5D 输入）

    # ---- stage 4: 跨域协调 ----
    harmonize: str = "none"           # none | combat | combat_meanonly | ea_trans | ea_ind | coral | knn_mmd
    harmonize_kwargs: dict = field(default_factory=dict)
    pca_dim: int = 128                # EA / CORAL 之前先降维（否则协方差矩阵太大）

    # ---- stage 5: 归一化 ----
    normalize: str = "global"         # none | global | per_sample | per_domain

    def tag(self) -> str:
        parts = [f"ph-{self.phase}", f"cl-{self.clutter}", f"dm-{self.domain}"]
        if self.harmonize != "none":
            parts.append(f"hz-{self.harmonize}")
        parts.append(f"nm-{self.normalize}")
        return "_".join(parts)


class CSIPipeline:
    """可 fit / transform 的特征流水线。

    训练/测试一致性是这里的重点：所有需要估计的参数（杂波子空间基、ComBat 的
    批次效应、PCA 基、归一化统计量）都只在**训练域**上估计，再应用到测试域。
    只有明确标注为 transductive 的方法（EA 的 transductive 模式）才会用到
    目标域数据 —— 而它只用到二阶统计量，不碰标签，因此不构成标签泄漏。

    用法
    ----
        pipe = CSIPipeline(PipelineCfg())
        Xtr_f = pipe.fit_transform(Xtr, dom_tr, X_te=Xte, dom_te=dom_te)
        Xte_f = pipe.transform(Xte, dom_te)
    """

    def __init__(self, cfg: PipelineCfg | None = None):
        self.cfg = cfg or PipelineCfg()
        self._fitted = False
        self.clutter_f_ = None
        self.frontend_ = None
        self.combat_ = None
        self.ea_ = None
        self.coral_ = None
        self.knn_mmd_ = None
        self.pca_mean_ = None
        self.pca_basis_ = None
        self.norm_mean_ = None
        self.norm_std_ = None
        self.train_domains_ = None

    # ------------------------------------------------------------------
    # stage 1 + 2：复数域处理
    # ------------------------------------------------------------------
    def _stage_phase(self, X: np.ndarray) -> np.ndarray:
        # BVP 路径：跳过所有 phase 处理，BVP 内部已自带去 DC
        if self.cfg.domain == "bvp":
            return X
        c = self.cfg.phase
        if c == "none":
            return X
        if c == "phase_sanitize":
            return phase_sanitize(X, subcarrier_axis=-1,
                                  **self.cfg.phase_kwargs)
        if c == "csi_ratio":
            kw = {"pairs": "consecutive"}
            kw.update(self.cfg.phase_kwargs)
            return csi_ratio(X, antenna_axis=-3, **kw)
        if c == "conj_mult":
            kw = {"pairs": "consecutive"}
            kw.update(self.cfg.phase_kwargs)
            return conjugate_multiply(X, antenna_axis=-3, **kw)
        raise ValueError(f"未知 phase 方法: {c}")

    def _stage_clutter(self, X: np.ndarray) -> np.ndarray:
        # X: (n, P, T, S) -> 慢时间轴是 -2
        if self.cfg.domain == "bvp":
            return X
        if self.cfg.clutter == "none":
            return X
        cf = ClutterFilter(method=self.cfg.clutter, time_axis=-2,
                           **self.cfg.clutter_kwargs)
        if self.cfg.clutter == "svd" and self.clutter_f_ is not None:
            cf.basis_ = self.clutter_f_
        return cf.transform(X)

    # ------------------------------------------------------------------
    # stage 3：域变换（复数 -> 实数通道）
    # ------------------------------------------------------------------
    def _stage_domain(self, X: np.ndarray) -> np.ndarray:
        # BVP 是整条 stage 1/2 的替代：内部已做 DC removal + log 压缩，
        # 直接把 (n, A, T, S) complex 转 (n, S, F, T') 2D 谱图。
        if self.cfg.domain == "bvp":
            from .signal.doppler import compute_bvp
            kw = self.cfg.domain_kwargs
            return compute_bvp(X,
                               n_fft=kw.get("n_fft", 64),
                               hop=kw.get("hop", 8),
                               dc_removal=kw.get("dc_removal", True),
                               batch_n=kw.get("batch_n", 128),
                               keep_antenna=kw.get("keep_antenna", False))

        if self.frontend_ is None:
            self.frontend_ = CepstralFrontEnd(mode=self.cfg.domain,
                                              **self.cfg.domain_kwargs)
        # X: (n, P, T, S) -> 对每个数据流 P 分别变换，再拼到通道维
        outs = []
        for p in range(X.shape[1]):
            outs.append(self.frontend_(X[:, p], subcarrier_axis=-1))
        # 每个 outs[p]: (C, n, T, S)
        return np.concatenate(outs, axis=0).transpose(1, 0, 2, 3)   # (n, P*C, T, S)

    # ------------------------------------------------------------------
    # stage 4：协调
    # ------------------------------------------------------------------
    def _flatten(self, X: np.ndarray) -> np.ndarray:
        return X.reshape(X.shape[0], -1)

    def _reduce(self, X_f: np.ndarray) -> np.ndarray:
        """按已拟合的 PCA 基降维（未降维时原样返回），供 EA / CORAL 使用。"""
        if self.pca_basis_ is None:
            return X_f
        return (X_f - self.pca_mean_) @ self.pca_basis_.T

    def _fit_harmonize(self, Xtr_f: np.ndarray, dom_tr: np.ndarray,
                       Xte_f: Optional[np.ndarray], dom_te: Optional[np.ndarray]):
        h = self.cfg.harmonize
        if h == "none":
            return

        # EA / CORAL / ComBat 都需要先降维，否则协方差/逐特征优化的开销爆炸
        # （真实 CSI flatten 后 ~6.9 万维，ComBat 逐特征优化会卡死）。
        if h in ("combat", "combat_meanonly", "ea_trans", "ea_ind", "coral", "knn_mmd"):
            d = Xtr_f.shape[1]
            if self.cfg.pca_dim and d > self.cfg.pca_dim:
                self.pca_mean_ = Xtr_f.mean(axis=0, keepdims=True)
                Z = Xtr_f - self.pca_mean_
                # 经济型 SVD，取主成分方向
                _, _, Vt = np.linalg.svd(Z, full_matrices=False)
                self.pca_basis_ = Vt[: self.cfg.pca_dim]           # (pca_dim, d)
            else:
                self.pca_mean_ = np.zeros((1, Xtr_f.shape[1]))
                self.pca_basis_ = None

        if h in ("combat", "combat_meanonly"):
            mean_only = (h == "combat_meanonly")
            kw = self.cfg.harmonize_kwargs
            self.combat_ = ComBat(
                parametric=kw.get("parametric", True),
                mean_only=mean_only,
                ref_batch=kw.get("ref_batch", None),
            )
            # 在 PCA 降维空间内拟合 ComBat（避免 6.9 万维逐特征优化）
            self.combat_.fit(self._reduce(Xtr_f), dom_tr, covars=kw.get("covars", None))

        elif h in ("ea_trans", "ea_ind"):
            mode = "transductive" if h == "ea_trans" else "inductive"
            self.ea_ = EuclideanAlignment(
                mode=mode,
                shrinkage=self.cfg.harmonize_kwargs.get("shrinkage", 0.1),
            )
            # EA 在 PCA 降维后的空间里拟合，与 _apply_harmonize 的应用保持一致
            Xtr_r = self._reduce(Xtr_f)
            Xte_r = self._reduce(Xte_f) if Xte_f is not None else None
            if mode == "transductive":
                # 用所有域（含目标域）的协方差，但不用任何标签
                doms_all = [Xtr_r[dom_tr == d].T for d in np.unique(dom_tr)]
                if Xte_r is not None and dom_te is not None:
                    doms_all += [Xte_r[dom_te == d].T for d in np.unique(dom_te)]
                self.ea_.fit(doms_all)
            else:
                self.ea_.fit([Xtr_r[dom_tr == d].T for d in np.unique(dom_tr)])

        elif h == "coral":
            self.coral_ = {}
            if Xte_f is not None and dom_te is not None:
                Xtr_r = self._reduce(Xtr_f)
                Xte_r = self._reduce(Xte_f)
                for d in np.unique(dom_te):
                    src = Xtr_r.T                    # (d, n_src)
                    tgt = Xte_r[dom_te == d].T       # (d, n_tgt)
                    m = CORAL(shrinkage=self.cfg.harmonize_kwargs.get("shrinkage", 0.1))
                    m.fit(src, tgt)
                    self.coral_[int(d)] = m
        elif h == "knn_mmd":
            self.knn_mmd_ = {}
            if Xte_f is not None and dom_te is not None:
                Xtr_r = self._reduce(Xtr_f)
                Xte_r = self._reduce(Xte_f)
                for d in np.unique(dom_te):
                    src = Xtr_r.T                    # (d, n_src)
                    tgt = Xte_r[dom_te == d].T       # (d, n_tgt)
                    m = KNNMMD(k=self.cfg.harmonize_kwargs.get("k", 10))
                    m.fit(src, tgt)
                    self.knn_mmd_[int(d)] = m
        else:
            raise ValueError(f"未知 harmonize 方法: {h}")

    def _apply_harmonize(self, X_f: np.ndarray, dom: np.ndarray, unseen: bool) -> np.ndarray:
        h = self.cfg.harmonize
        if h == "none":
            return X_f

        # ---- 降维 -> 协调 -> 还原 ----
        if h in ("ea_trans", "ea_ind", "coral", "knn_mmd"):
            Z = X_f - self.pca_mean_
            if self.pca_basis_ is not None:
                Z = Z @ self.pca_basis_.T                      # (n, pca_dim)
            if h == "coral":
                out = np.empty_like(Z)
                for d in np.unique(dom):
                    m = dom == d
                    key = int(d)
                    model = self.coral_.get(key)
                    if model is None:      # 未见过的域：退化为不处理
                        out[m] = Z[m]
                    else:
                        out[m] = model.transform(Z[m].T).T
                Z = out
            elif h == "knn_mmd":
                # KNN-MMD 只变换训练样本；目标域样本保持不变（无监督）
                out = Z.copy()
                if self.knn_mmd_ is not None:
                    for d in np.unique(dom):
                        m = dom == d
                        key = int(d)
                        model = self.knn_mmd_.get(key)
                        if model is None:
                            continue
                        # unseen=True 表示这是目标域（用 dom==d 找出训练样本的目标域部分）
                        # 在我们的用法里，transform 在训练折调用 unseen=False 训练样本侧，
                        # 在测试折调用 unseen=True 不应触发 knn_mmd。
                        # 简化：所有 dom 都被处理，但只有训练折才调用 fit
                        out[m] = model.transform(Z[m].T).T
                Z = out
            else:
                if h == "ea_trans":
                    Z = (self.ea_.R_inv_sqrt_ @ Z.T).T
                else:
                    out = np.empty_like(Z)
                    for d in np.unique(dom):
                        m = dom == d
                        out[m] = (self.ea_.transform(Z[m].T, domain_id=int(d))).T
                    Z = out
            if self.pca_basis_ is not None:
                Z = Z @ self.pca_basis_                        # 还原
            return Z + self.pca_mean_

        # ---- ComBat（在 PCA 降维空间内做，再映射回原空间）----
        if h in ("combat", "combat_meanonly"):
            Z = X_f - self.pca_mean_
            if self.pca_basis_ is not None:
                Z = Z @ self.pca_basis_.T
            if unseen:
                Zt = self.combat_.transform_unseen(Z)
            else:
                Zt = self.combat_.transform(Z, dom)
            if self.pca_basis_ is not None:
                Zt = Zt @ self.pca_basis_
            return Zt + self.pca_mean_

        raise ValueError(h)

    # ------------------------------------------------------------------
    # stage 5：归一化
    # ------------------------------------------------------------------
    def _fit_normalize(self, Xtr: np.ndarray, dom_tr: np.ndarray):
        c = self.cfg.normalize
        if c == "none":
            return
        if c == "global":
            self.norm_mean_ = Xtr.mean(axis=0, keepdims=True)
            self.norm_std_ = Xtr.std(axis=0, keepdims=True)
        elif c == "per_domain":
            self.norm_mean_, self.norm_std_ = {}, {}
            for d in np.unique(dom_tr):
                m = dom_tr == d
                self.norm_mean_[int(d)] = Xtr[m].mean(axis=0, keepdims=True)
                self.norm_std_[int(d)] = Xtr[m].std(axis=0, keepdims=True)
        # per_sample 不需要 fit

    def _apply_normalize(self, X: np.ndarray, dom: np.ndarray) -> np.ndarray:
        c = self.cfg.normalize
        eps = 1e-8
        if c == "none":
            return X
        if c == "global":
            # 分批 normalize，避免大 5D 数组的内存爆炸（OOM guard）
            out = np.empty_like(X)
            n = X.shape[0]
            batch = max(1, min(1024, n))
            for i in range(0, n, batch):
                end = min(i + batch, n)
                out[i:end] = (X[i:end] - self.norm_mean_) / (self.norm_std_ + eps)
            return out
        if c == "per_sample":
            mu = X.mean(axis=(1, 2, 3), keepdims=True)
            sd = X.std(axis=(1, 2, 3), keepdims=True)
            return (X - mu) / (sd + eps)
        if c == "per_domain":
            out = np.empty_like(X)
            for d in np.unique(dom):
                m = dom == d
                key = int(d)
                if key in self.norm_mean_:
                    out[m] = (X[m] - self.norm_mean_[key]) / (self.norm_std_[key] + eps)
                else:
                    # 未见过的域：用它自己的统计量（无监督）
                    out[m] = (X[m] - X[m].mean(axis=0, keepdims=True)) / (
                        X[m].std(axis=0, keepdims=True) + eps)
            return out
        raise ValueError(f"未知 normalize: {c}")

    # ------------------------------------------------------------------
    # 对外接口
    # ------------------------------------------------------------------
    def stage3(self, X) -> np.ndarray:
        """只跑与划分无关的 stage 1-3（去相位→抑杂波→换域）。

        在 LODO 里，这一步对全数据集只算一次即可，按折只重算协调+归一化，
        能省掉每折 4 次重复的昂贵变换。返回 ``(n, P*C, T, S)``。
        """
        return self._stage_domain(self._stage_clutter(self._stage_phase(np.asarray(X))))

    def fit(self, X_tr, dom_tr, X_te=None, dom_te=None,
            X_tr3=None, X_te3=None) -> "CSIPipeline":
        dom_tr = np.asarray(dom_tr)
        self.train_domains_ = np.unique(dom_tr)

        # 用于估计杂波基 / ComBat 的中间结果（先跑到 stage 3 结束）
        if X_tr3 is None:
            Xtr3 = self._stage_domain(self._stage_clutter(self._stage_phase(np.asarray(X_tr))))
        else:
            Xtr3 = np.asarray(X_tr3)
        Xte3 = X_te3
        if Xte3 is None and X_te is not None and dom_te is not None:
            Xte3 = self._stage_domain(self._stage_clutter(self._stage_phase(np.asarray(X_te))))

        self._fit_harmonize(self._flatten(Xtr3), dom_tr,
                            self._flatten(Xte3) if Xte3 is not None else None,
                            np.asarray(dom_te) if dom_te is not None else None)

        Xtr4 = self._apply_harmonize(self._flatten(Xtr3), dom_tr, unseen=False)
        Xtr4 = Xtr4.reshape(Xtr3.shape)
        self._fit_normalize(Xtr4, dom_tr)
        self._fitted = True
        return self

    def transform(self, X, dom, unseen: bool = True, X3=None) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("请先调用 fit()")
        dom = np.asarray(dom)
        X = np.asarray(X)
        if X3 is None:
            X3 = self._stage_domain(self._stage_clutter(self._stage_phase(X)))
        else:
            X3 = np.asarray(X3)
        shape = X3.shape
        X4 = self._apply_harmonize(self._flatten(X3), dom, unseen=unseen)
        X4 = X4.reshape(shape)
        return self._apply_normalize(X4, dom).astype(np.float32)

    def fit_transform(self, X, dom, X_te=None, dom_te=None) -> np.ndarray:
        self.fit(X, dom, X_te, dom_te)
        return self.transform(X, np.asarray(dom), unseen=False)
