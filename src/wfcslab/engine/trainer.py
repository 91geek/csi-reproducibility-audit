"""
训练循环。

刻意保持简单：这个库的目的不是刷 SOTA，而是**公平比较不同预处理/协调方法**。
所以每个方法都必须用同一套训练脚本、同一套超参搜索预算、同一种早停准则。

性能 / 可视化（2026-09-04 改进）
---------------------------------
- TF32 全局开启（Turing/Ampere 矩阵乘提速 ~1.5–2×）
- AMP FP16 autocast（GPU 上 1.5–2×，CPU 上自动 fallback 到 no-op）
- ``torch.compile`` 可选（PyTorch ≥ 2.0，编译模式 reduce-overhead）
- DataLoader pin_memory + non_blocking + 持久 worker（小 batch 友好）
- tqdm 实时进度条（每 epoch 显示 loss / val_acc / ETA）
- TensorBoard 标量日志（loss / val_acc / val_f1 / lr / epoch_time）
- CSV 落盘（每 epoch 一行，方便后续画图）
"""
from __future__ import annotations

import copy
import csv
import os
import time
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from ..models.backbones import build_model, build_model_attn
from .metrics import evaluate

__all__ = ["TrainCfg", "train_one", "predict", "set_perf_flags"]


# --------------------------------------------------------------------------
# 性能开关（全局，只设一次）
# --------------------------------------------------------------------------
_PERF_ONCE = False


def set_perf_flags(tf32: bool = True, matmul_precision: str = "high") -> None:
    """开启 TF32 / 设置 matmul 精度。多次调用是 no-op。"""
    global _PERF_ONCE
    if _PERF_ONCE:
        return
    _PERF_ONCE = True
    if tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True  # 自动挑最快卷积算法
    torch.set_float32_matmul_precision(matmul_precision)  # 'highest' | 'high' | 'medium'
    # 线程数（i9 8 核 → DataLoader + numpy 都用得上）
    n = os.cpu_count() or 4
    torch.set_num_threads(max(2, n // 2))   # 给 BLAS/DataLoader 留一半
    # set_num_interop_threads 只能用 getattr，否则本函数里的 torch 会被编译器
    # 当作局部变量（PEP 227 规则：函数体内任何 import 都让同名变量变为 local）
    _interop = getattr(torch, "set_num_interop_threads", None)
    if _interop is not None:
        try:
            _interop(max(2, n // 2))
        except Exception:
            pass


def _pick_device(device: str) -> torch.device:
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device)


def _fmt_eta(sec: float) -> str:
    sec = int(max(0, sec))
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h:d}h{m:02d}m{s:02d}s"
    if m:
        return f"{m:d}m{s:02d}s"
    return f"{s:d}s"


# --------------------------------------------------------------------------
@dataclass
class TrainCfg:
    epochs: int = 60
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 1e-4
    optimizer: str = "adamw"          # adamw | adam | sgd
    scheduler: str = "cosine"         # cosine | step | none | cyclic_snapshot
                                       # cyclic_snapshot: CosineAnnealingWarmRestarts 每 n 个
                                       # epoch 一次循环，配合 train_one(..., snapshot=True)
                                       # 收集多个 checkpoint 做 ensemble（F-35 验证）
    warmup_epochs: int = 0            # 线性 LR 预热轮数（0=关闭）。配合 scheduler
                                       # "cosine"/"step" 使用：前 warmup_epochs 轮 LR 从
                                       # warmup_start_factor*lr 线性升到 lr，之后走原调度。
                                       # 2026-09-07 新增：缓解高容量多通道输入在难域上的
                                       # 早期优化崩溃（F-41 / d9 稳定性实验）
    warmup_start_factor: float = 0.1  # 预热起始 LR 系数（相对 base lr）
    dropout: float = 0.5
    patience: int = 15                # 早停耐心（epoch）
    label_smoothing: float = 0.0
    grad_clip: float = 1.0
    device: str = "auto"
    seed: int = 0
    num_workers: int = 0              # 2026-09-06: 默认 0，避免 Windows 共享内存 IO 风暴
    eval_batch_size: int = 256        # 2026-09-06: 验证/测试分批推理批大小。
                                      # 原实现把整个验证集一次性前向（trainer 旧版 line449
                                      # `model(Xval_t)`），在多尺度 5D 输入下峰值显存达
                                      # 18~28 GB，远超 11 GB 显卡 -> Windows WDDM 溢出到
                                      # 系统内存 -> pagefile 狂写 -> D/E 盘 IO 打满、系统卡死。
                                      # 分批后峰值显存与训练 batch 同量级。
    pin_memory: bool = True
    prefetch_factor: int = 2
    verbose: bool = False

    # ---- 增强（2026-09-04 加入）----
    mixup_alpha: float = 0.0           # Mixup 强度；0 = 关闭，0.2/0.4 是常用值

    # ---- 类别不平衡处理（2026-09-04 加入）----
    loss_type: str = "ce"              # "ce" | "focal"
    focal_gamma: float = 2.0           # Focal Loss 的聚焦参数（γ=2 常用）
    class_weights: Optional[Sequence[float]] = None  # 类别权重（CE/Focal 的 α）
    balanced_sampler: bool = False     # True=用 WeightedRandomSampler 按类频反比采样

    # ---- 性能 / 可视化扩展（保留向后兼容默认值）----
    use_amp: bool = True              # GPU 上自动 FP16 autocast
    compile_model: bool = False       # torch.compile（首次有 ~10s 编译开销）
    tf32: bool = True
    matmul_precision: str = "high"
    show_progress: bool = True        # tqdm 实时进度条
    log_csv: Optional[str] = None     # 每 epoch 一行 CSV
    tb_dir: Optional[str] = None      # TensorBoard 目录（None=不写）
    progress_desc: str = ""           # tqdm 描述前缀，如 "[cfg0/fold3]"
    clear_log_on_start: bool = True   # TB/CSV 存在时覆盖


def _batched_predict(model: nn.Module, X: torch.Tensor, batch_size: int,
                     use_amp: bool, amp_dtype) -> torch.Tensor:
    """分批前向，返回拼接后的 logits。

    2026-09-06 新增：原实现把整个验证/测试集一次性前向，5D 多尺度输入下
    中间激活（尤其注意力的 N×seq² 矩阵）会撑爆显存并溢出到系统内存，
    导致 pagefile 狂写、系统卡死。改为批推理后峰值显存与训练同量级。
    """
    was_training = model.training
    model.eval()
    outs = []
    with torch.no_grad():
        for i in range(0, X.size(0), batch_size):
            with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
                outs.append(model(X[i:i + batch_size]))
    if was_training:
        model.train()
    return torch.cat(outs, dim=0) if len(outs) > 1 else outs[0]


def _batched_predict_dl(model: nn.Module, ds, batch_size: int,
                        use_amp: bool, amp_dtype, dev) -> torch.Tensor:
    """从懒加载 Dataset（mmap 共享）分批前向，返回拼接 logits。

    与 ``_batched_predict`` 的区别：输入是 Dataset 而非整块张量，每批只从磁盘
    读对应行，峰值显存与训练 batch 同量级，适合 50k 规模下大 fold 的验证集。
    Dataset 的 ``__getitem__`` 返回 ``(x, y)``，这里忽略 y。
    """
    from torch.utils.data import DataLoader
    was_training = model.training
    model.eval()
    outs = []
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    with torch.no_grad():
        for xb, _ in dl:
            xb = xb.to(dev, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
                outs.append(model(xb))
    if was_training:
        model.train()
    return torch.cat(outs, dim=0) if len(outs) > 1 else outs[0]


def _make_optimizer(model: nn.Module, cfg: TrainCfg):
    if cfg.optimizer == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                                 weight_decay=cfg.weight_decay)
    if cfg.optimizer == "adam":
        return torch.optim.Adam(model.parameters(), lr=cfg.lr,
                                weight_decay=cfg.weight_decay)
    if cfg.optimizer == "sgd":
        return torch.optim.SGD(model.parameters(), lr=cfg.lr, momentum=0.9,
                                weight_decay=cfg.weight_decay)
    raise ValueError(cfg.optimizer)


class FocalLoss(nn.Module):
    """二分类/多分类 Focal Loss（Lin et al. 2017）。

    loss = -alpha_c * (1 - p_c)^gamma * log(p_c)

    - 不依赖类内 one-hot，自带 softmax + NLL
    - 支持 per-class 权重 α（用 cfg.class_weights 传入）
    - 与 Mixup 兼容（输入是 soft target 的 logits 和样本混合标签还没实现；
      简单起见，当 use_mixup=True 时退化为按混合比加权和的硬标签版本）
    """

    def __init__(self, gamma: float = 2.0,
                 weight: Optional[torch.Tensor] = None,
                 label_smoothing: float = 0.0):
        super().__init__()
        self.gamma = float(gamma)
        self.register_buffer("weight", weight if weight is not None else None)
        self.label_smoothing = float(label_smoothing)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # logits: (B, C), target: (B,)  long
        log_p = torch.log_softmax(logits, dim=-1)
        # log_softmax + nll_loss 自动处理 label_smoothing
        # 但 nll_loss 不接受 weight + reduction='none' 时返回 (B,)
        # 直接手算 CE（避免 label_smoothing 与 weight 在 nll_loss 中互不兼容问题）
        n_classes = logits.size(-1)
        with torch.no_grad():
            true_dist = torch.zeros_like(logits)
            true_dist.fill_(self.label_smoothing / (n_classes - 1)
            if self.label_smoothing > 0 and n_classes > 1 else 0.0)
            true_dist.scatter_(1, target.unsqueeze(1), 1.0 - self.label_smoothing)

        p = log_p.exp()
        focal_weight = (1.0 - p) ** self.gamma
        # per-class alpha → 每个样本取自己类别的 α
        if self.weight is not None:
            alpha = self.weight.to(logits.device)[target].unsqueeze(1)  # (B, 1)
        else:
            alpha = 1.0
        loss_per_sample = -(alpha * focal_weight * true_dist * log_p).sum(dim=-1)
        return loss_per_sample.mean()


def _make_loss(cfg: TrainCfg, n_classes: int):
    """根据 cfg 返回合适的 loss（分类任务用 CE/Focal，姿态用 MSE）。

    注：weight_t 先注册为 buffer（FocalLoss）/ 直接传 nn.CrossEntropyLoss
    时挪到正确 device，避免 loss 计算时 device mismatch。
    """
    weight_t = None
    if cfg.class_weights is not None:
        w = np.asarray(cfg.class_weights, dtype=np.float32)
        if w.size != n_classes:
            raise ValueError(
                f"class_weights 长度 {w.size} ≠ 类别数 {n_classes}")
        weight_t = torch.as_tensor(w, dtype=torch.float32)
    if cfg.loss_type == "focal":
        return FocalLoss(gamma=cfg.focal_gamma, weight=weight_t,
                         label_smoothing=cfg.label_smoothing)
    # CE weight 也要 .to(dev) 之后再用，train_one 里会处理
    return nn.CrossEntropyLoss(weight=weight_t,
                               label_smoothing=cfg.label_smoothing)


def _make_balanced_sampler(y: np.ndarray, n_classes: int,
                            class_weights: Optional[Sequence[float]] = None):
    """按类别频次反比做 WeightedRandomSampler（解决类不平衡）。"""
    from torch.utils.data import WeightedRandomSampler
    counts = np.bincount(y.astype(np.int64), minlength=n_classes).astype(np.float64)
    # 每个样本的采样权重 = 1 / 其类别频次，再乘以可选的 class_weights 调整
    if class_weights is not None:
        cw = np.asarray(class_weights, dtype=np.float64)
        per_class_w = cw / (counts + 1e-9)
    else:
        per_class_w = 1.0 / (counts + 1e-9)
    sample_w = per_class_w[y.astype(np.int64)]
    sample_w = sample_w / sample_w.sum() * len(sample_w)  # 归一到 ~1
    return WeightedRandomSampler(
        weights=torch.as_tensor(sample_w, dtype=torch.double),
        num_samples=len(sample_w),
        replacement=True,
    )


def _to_tensor(a, dtype=torch.float32) -> torch.Tensor:
    if isinstance(a, torch.Tensor):
        return a.to(dtype)
    return torch.as_tensor(np.asarray(a), dtype=dtype)


def _try_compile(model: nn.Module, cfg: TrainCfg) -> nn.Module:
    """torch.compile with graceful fallback."""
    if not cfg.compile_model or not hasattr(torch, "compile"):
        return model
    try:
        return torch.compile(model, mode="reduce-overhead")
    except Exception as e:
        print(f"  [warn] torch.compile 失败（{e!r}），退回 eager mode")
        return model


def train_one(
    Xtr: np.ndarray,
    ytr: np.ndarray,
    Xval: np.ndarray,
    yval: np.ndarray,
    model_name: str,
    task: str,
    cfg: TrainCfg,
    model_kwargs: Optional[dict] = None,
    pretrained_state: Optional[dict] = None,
    augment_fn=None,
    collect_snapshots: bool = False,
    train_ds=None,
    val_ds=None,
) -> dict:
    """训练一个模型，按验证集指标早停。

    返回 dict:
        best_metric   验证集上的最优指标（分类=acc，姿态=-mpjpe，越大越好）
        best_epoch    取得最优指标的 epoch
        best_state    最优权重（state_dict 的深拷贝）
        history       每 epoch 的 {epoch, train_loss, acc, macro_f1, ...}
        out_dim       输出维度
        cfg           训练配置（便于复盘）
        snapshots     （仅当 collect_snapshots=True）多个 cycle 末 checkpoint 列表
                      [{state_dict, epoch, val_acc}]，配合 snapshot ensemble
    """
    model_kwargs = dict(model_kwargs or {})
    set_perf_flags(cfg.tf32, cfg.matmul_precision)

    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)

    dev = _pick_device(cfg.device)
    is_cuda = (dev.type == "cuda")

    n = Xtr.shape[0]
    Xtr_ndim = Xtr.ndim
    if Xtr_ndim == 4:
        _, C, T, S = Xtr.shape
    elif Xtr_ndim == 5:
        # 多 Rx BVP 谱图 (B, A, S, F, T')
        _, A, S, F, Tp = Xtr.shape
    else:
        raise ValueError(f"Xtr.ndim={Xtr_ndim}，仅支持 4D 或 5D")

    if task == "classification":
        out_dim = int(np.max(ytr)) + 1
        ytr_t = _to_tensor(ytr, torch.long)
    else:
        J = ytr.shape[1]
        out_dim = J * 3
        ytr_t = _to_tensor(ytr.reshape(len(ytr), -1), torch.float32)

    if Xtr_ndim == 5:
        if model_name not in ("lenet_attn", "lenet_attn_mha", "lenet_cfc"):
            raise ValueError(
                f"5D 输入只支持 lenet_attn / lenet_attn_mha / lenet_cfc，当前 model_name={model_name}")
        from wfcslab.models.backbones import build_model_attn
        model = build_model_attn(model_name, (A, S, F, Tp), out_dim,
                                 task=task, dropout=cfg.dropout,
                                 **model_kwargs).to(dev)
    else:
        model = build_model(model_name, C, T, S, out_dim, task=task,
                            dropout=cfg.dropout, **model_kwargs).to(dev)

    # ---- 加载预训练权重（兼容 out_dim 不一致，仅加载 shape 匹配的参数）----
    if pretrained_state is not None:
        own_state = model.state_dict()
        loaded, skipped = 0, 0
        for k, v in pretrained_state.items():
            if k in own_state and own_state[k].shape == v.shape:
                own_state[k] = v
                loaded += 1
            else:
                skipped += 1
        model.load_state_dict(own_state)
        print(f"[pretrained] loaded {loaded} tensors, skipped {skipped} (shape mismatch)")
    model = _try_compile(model, cfg)

    opt = _make_optimizer(model, cfg)
    sched = None
    if cfg.scheduler == "cosine":
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)
    elif cfg.scheduler == "step":
        sched = torch.optim.lr_scheduler.StepLR(opt, step_size=max(1, cfg.epochs // 3),
                                                gamma=0.1)
    elif cfg.scheduler == "cyclic_snapshot":
        # Snapshot Ensemble [Huang et al. 2017]：cyclic cosine，每 cycle 末存一次
        # 默认 3 cycle × epochs/3。配合 train_one 内的 snapshot 逻辑
        sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            opt, T_0=max(1, cfg.epochs // 3), T_mult=1
        )
    # ---- warmup 线性预热（F-41 稳定性实验）----
    if cfg.warmup_epochs > 0 and sched is not None:
        wu = torch.optim.lr_scheduler.LinearLR(
            opt, start_factor=cfg.warmup_start_factor, end_factor=1.0,
            total_iters=cfg.warmup_epochs)
        sched = torch.optim.lr_scheduler.SequentialLR(
            opt, schedulers=[wu, sched], milestones=[cfg.warmup_epochs])

    if task == "classification":
        loss_fn = _make_loss(cfg, out_dim)
        # 若 loss 有 weight（CE / Focal），需要搬到 model 所在 device
        if hasattr(loss_fn, "weight") and loss_fn.weight is not None:
            loss_fn.weight = loss_fn.weight.to(dev)
    else:
        loss_fn = nn.MSELoss()

    # ---- DataLoader ----
    pin_mem = cfg.pin_memory and is_cuda
    if train_ds is not None:
        # 训练集以懒加载 Dataset 提供（mmap 共享内存，分批从磁盘读，避免大 fold OOM）
        tr_dl = DataLoader(
            train_ds, batch_size=cfg.batch_size, shuffle=True,
            num_workers=0, pin_memory=pin_mem, drop_last=False,
        )
    else:
        nw = max(0, cfg.num_workers) if Xtr.shape[0] > 64 else 0
        sampler = None
        if task == "classification" and cfg.balanced_sampler:
            sampler = _make_balanced_sampler(ytr, out_dim, cfg.class_weights)
        tr_dl = DataLoader(
            TensorDataset(_to_tensor(Xtr), ytr_t),
            batch_size=cfg.batch_size, shuffle=(sampler is None),
            sampler=sampler,
            num_workers=nw,
            pin_memory=pin_mem,
            prefetch_factor=cfg.prefetch_factor if nw > 0 else None,
            persistent_workers=(nw > 0),
            drop_last=False,
        )
    # 验证集：若提供 val_ds（懒加载，避免大 fold 一次性进内存）则从其分批推理，
    # 否则沿用原逻辑（Xval 整体搬上 GPU）。
    if val_ds is not None:
        Xval_t = None
    else:
        Xval_t = _to_tensor(Xval).to(dev, non_blocking=pin_mem)
    yval_eval = yval if task == "classification" else yval

    # ---- AMP ----
    use_amp = cfg.use_amp and is_cuda
    amp_dtype = torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # ---- 日志（TensorBoard / CSV）----
    tb_writer = None
    if cfg.tb_dir:
        try:
            from torch.utils.tensorboard import SummaryWriter
            # 若 tb_dir 存在但不是目录，先删（防御上次异常残留）
            if os.path.exists(cfg.tb_dir) and not os.path.isdir(cfg.tb_dir):
                os.remove(cfg.tb_dir)
            os.makedirs(cfg.tb_dir, exist_ok=True)
            tb_writer = SummaryWriter(log_dir=cfg.tb_dir,
                                      purge_step=cfg.clear_log_on_start)
        except Exception as e:
            print(f"  [warn] TensorBoard 初始化失败（{e!r}），跳过 TB 日志")
            tb_writer = None

    csv_fp = None
    csv_writer = None
    if cfg.log_csv:
        os.makedirs(os.path.dirname(cfg.log_csv) or ".", exist_ok=True)
        if cfg.clear_log_on_start and os.path.exists(cfg.log_csv):
            os.remove(cfg.log_csv)
        csv_fp = open(cfg.log_csv, "a", newline="", encoding="utf-8")
        csv_writer = csv.writer(csv_fp)
        csv_writer.writerow(["epoch", "train_loss", "val_acc",
                             "val_macro_f1", "lr", "epoch_time_s"])
        csv_fp.flush()

    # ---- tqdm 进度条 ----
    pbar = None
    if cfg.show_progress:
        try:
            from tqdm.auto import tqdm
            pbar = tqdm(range(cfg.epochs), desc=cfg.progress_desc or "train",
                        dynamic_ncols=True, leave=False, mininterval=0.5)
        except ImportError:
            pbar = None

    best = {"best_metric": -np.inf, "best_epoch": -1, "best_state": None,
            "history": [], "out_dim": out_dim, "cfg": cfg,
            "snapshots": [] if collect_snapshots else None}
    bad = 0
    train_t0 = time.time()
    # cyclic_snapshot: 收集每个 cycle 末的 snapshot（默认 3 cycle）
    snapshot_T0 = max(1, cfg.epochs // 3) if collect_snapshots else None

    for ep in range(cfg.epochs):
        model.train()
        tot, nb = 0.0, 0
        ep_t0 = time.time()
        use_mixup = (cfg.mixup_alpha > 0) and (task == "classification")
        for xb, yb in tr_dl:
            # 训练时数据增强（在搬到 GPU 前；augment_fn 期望 numpy 或 torch，返回同 shape）
            if augment_fn is not None:
                xb_np = xb.numpy() if isinstance(xb, torch.Tensor) else xb
                yb_np = yb.numpy() if isinstance(yb, torch.Tensor) else yb
                out = augment_fn(xb_np, yb_np)
                if isinstance(out, tuple):
                    xb_np, yb_np = out
                else:
                    xb_np = out
                xb = torch.from_numpy(np.ascontiguousarray(xb_np))
                yb = torch.as_tensor(yb_np, dtype=yb.dtype if isinstance(yb, torch.Tensor) else torch.long)
            xb = xb.to(dev, non_blocking=pin_mem)
            yb = yb.to(dev, non_blocking=pin_mem)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
                if use_mixup:
                    # Mixup： xb = lam * x_a + (1-lam) * x_b
                    # loss = lam * CE(out, ya) + (1-lam) * CE(out, yb)
                    lam = float(np.random.beta(cfg.mixup_alpha,
                                               cfg.mixup_alpha))
                    perm = torch.randperm(xb.size(0), device=xb.device)
                    xb_mix = lam * xb + (1.0 - lam) * xb[perm]
                    out = model(xb_mix)
                    loss = (lam * loss_fn(out, yb)
                            + (1.0 - lam) * loss_fn(out, yb[perm]))
                else:
                    out = model(xb)
                    loss = loss_fn(out, yb)
            scaler.scale(loss).backward()
            if cfg.grad_clip and cfg.grad_clip > 0:
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(opt)
            scaler.update()
            tot += float(loss.item()) * len(xb)
            nb += len(xb)
        if sched is not None:
            sched.step()

        # ---- 验证（2026-09-06: 改成分批推理，避免整集前向撑爆显存）----
        if val_ds is not None:
            ov = _batched_predict_dl(model, val_ds, cfg.eval_batch_size,
                                    use_amp, amp_dtype, dev)
        else:
            ov = _batched_predict(model, Xval_t, cfg.eval_batch_size,
                                 use_amp, amp_dtype)
        m = evaluate(task, ov, yval_eval)
        primary = m["acc"] if task == "classification" else -m["mpjpe"]
        ep_time = time.time() - ep_t0
        cur_lr = opt.param_groups[0]["lr"]

        # ---- Snapshot 收集（仅 collect_snapshots=True 时生效）----
        # CosineAnnealingWarmRestarts 的 T_0 边界：每次 LR 衰减到最低点
        if collect_snapshots and (ep + 1) % snapshot_T0 == 0 and ep + 1 < cfg.epochs:
            inner = getattr(model, "_orig_mod", model)
            best["snapshots"].append({
                "epoch": ep + 1,
                "val_acc": float(m["acc"]) if task == "classification" else float(primary),
                "state_dict": copy.deepcopy(inner.state_dict()),
            })
        rec = {"epoch": ep, "train_loss": tot / max(nb, 1), **m,
               "lr": float(cur_lr), "epoch_time_s": round(ep_time, 3)}
        best["history"].append(rec)

        # ---- 写日志 ----
        if csv_writer is not None:
            csv_writer.writerow([ep, f"{rec['train_loss']:.6f}",
                                 f"{m.get('acc', 0):.6f}",
                                 f"{m.get('macro_f1', 0):.6f}",
                                 f"{cur_lr:.2e}", f"{ep_time:.3f}"])
            csv_fp.flush()
        if tb_writer is not None:
            tb_writer.add_scalar("train/loss", rec["train_loss"], ep)
            tb_writer.add_scalar("val/acc", m.get("acc", 0), ep)
            if "macro_f1" in m:
                tb_writer.add_scalar("val/macro_f1", m["macro_f1"], ep)
            tb_writer.add_scalar("train/lr", cur_lr, ep)
            tb_writer.add_scalar("train/epoch_time_s", ep_time, ep)

        # ---- 早停 / 进度条 ----
        if primary > best["best_metric"]:
            best["best_metric"] = float(primary)
            best["best_epoch"] = ep
            # 注：torch.compile 包装下的 state_dict 拿到的还是原 model
            inner = getattr(model, "_orig_mod", model)
            best["best_state"] = copy.deepcopy(inner.state_dict())
            bad = 0
        else:
            bad += 1
            if bad >= cfg.patience:
                break

        if pbar is not None:
            done = ep + 1
            remain = (cfg.epochs - done) * (time.time() - train_t0) / done
            pbar.set_postfix(loss=f"{rec['train_loss']:.3f}",
                             val_acc=f"{m.get('acc', 0):.3f}",
                             best=f"{best['best_metric']:.3f}",
                             eta=_fmt_eta(remain))
            pbar.update(1)

        if cfg.verbose and ep % 10 == 0:
            print(f"    ep{ep:3d} loss={rec['train_loss']:.4f} val={m}")

    if pbar is not None:
        pbar.close()
    if csv_fp is not None:
        csv_fp.close()
    if tb_writer is not None:
        tb_writer.flush()
        tb_writer.close()

    # 把 best_state 装载回原 model（兼容 compile 包装）
    if best["best_state"] is not None:
        inner = getattr(model, "_orig_mod", model)
        inner.load_state_dict(best["best_state"])
    best["model"] = model
    best["train_time_s"] = round(time.time() - train_t0, 2)
    return best


@torch.no_grad()
def predict(model: nn.Module, X: np.ndarray, task: str, out_dim: int,
            device: str = "auto", batch_size: int = 256,
            use_amp: bool = True) -> np.ndarray:
    """批量推理。姿态任务返回 (n, J, 3)，分类任务返回 (n, n_classes) logits。"""
    dev = _pick_device(device)
    inner = getattr(model, "_orig_mod", model)
    inner = inner.to(dev).eval()
    is_cuda = (dev.type == "cuda")
    use_amp = use_amp and is_cuda
    outs = []
    Xt = _to_tensor(X)
    for i in range(0, len(X), batch_size):
        xb = Xt[i:i + batch_size].to(dev, non_blocking=is_cuda)
        with torch.amp.autocast("cuda", enabled=use_amp, dtype=torch.float16):
            o = inner(xb)
        outs.append(o.detach().cpu().numpy())
    o = np.concatenate(outs, axis=0)
    if task == "pose":
        return o.reshape(len(o), -1, 3)
    return o
