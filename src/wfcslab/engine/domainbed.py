"""
DomainBed 风格的留一域实验运行器。

这是整套代码里最"讲规矩"的部分，也是最容易被人偷工减料的部分。
Gulrajani & Lopez-Paz (2021) 用 45,900 个网络训练证明：**精心实现的 ERM
打平甚至超过了 8 种专门的域泛化算法**。本领域大量"我们的方法超过 baseline X%"
很可能是 baseline 没调好的假阳性。

所以这里强制执行四件事：
  1. 留一域划分，绝不用随机帧划分（相邻帧同时进训练和测试 = 时间泄漏）
  2. 验证集只从训练域出（除非显式指定 oracle 模式做上限分析）
  3. 每个方法用**相同的超参搜索预算**
  4. 多种子，报均值 ± 标准差

流程（对应 DomainBed 的 model selection 流程）
---------------------------------------------
    for 测试域 d:
        for 验证折 v（lodo 模式下 = 每个训练域轮流做验证）:
            for 超参试验 t（随机搜索）:
                在 train 上训练，按 val 指标早停 -> 记录 val 最优
            选 val 最优的超参
            用该超参在 train+val 上重训（固定 epoch 数）
            在测试域 d 上评估
        对 v 取平均
    对 d 取平均 -> 最终报告
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ..data.base import CSIDataset, lodo_splits_with_val
from ..pipeline import PipelineCfg, CSIPipeline
from ..engine.metrics import evaluate
from ..engine.trainer import TrainCfg, train_one, predict

__all__ = ["ExperimentCfg", "DEFAULT_HP_SPACE", "run_lodo_experiment",
           "aggregate", "sample_hparams"]


DEFAULT_HP_SPACE = {
    "lr": ("loguniform", 1e-4, 1e-2),
    "weight_decay": ("loguniform", 1e-6, 1e-2),
    "dropout": ("uniform", 0.0, 0.6),
    "batch_size": ("choice", [16, 32, 64]),
}


def sample_hparams(space: dict, rng: np.random.Generator) -> dict:
    """按分布采样一组超参。"""
    out = {}
    for k, spec in space.items():
        kind = spec[0]
        if kind == "loguniform":
            lo, hi = np.log(spec[1]), np.log(spec[2])
            out[k] = float(np.exp(rng.uniform(lo, hi)))
        elif kind == "uniform":
            out[k] = float(rng.uniform(spec[1], spec[2]))
        elif kind == "choice":
            out[k] = spec[1][int(rng.integers(len(spec[1])))]
        else:
            raise ValueError(f"未知分布: {kind}")
    return out


@dataclass
class ExperimentCfg:
    """一次完整留一域实验的配置。"""
    pipeline: PipelineCfg = field(default_factory=PipelineCfg)
    model: str = "cnn_gru"
    model_kwargs: dict = field(default_factory=dict)
    train: TrainCfg = field(default_factory=TrainCfg)
    val_mode: str = "lodo"              # lodo | random | oracle
    n_seeds: int = 3
    n_trials: int = 8
    hparam_space: dict = field(default_factory=lambda: copy.deepcopy(DEFAULT_HP_SPACE))
    retrain_on_full: bool = True        # DomainBed 流程：选完超参后在 train+val 上重训
    group_by_subject: bool = True
    pose_thresholds: tuple = (50.0,)

    def label(self) -> str:
        return f"{self.pipeline.tag()}__{self.model}"


def _concat(a, b):
    return np.concatenate([a, b], axis=0)


def run_lodo_experiment(
    ds: CSIDataset,
    cfg: ExperimentCfg,
    verbose: bool = True,
) -> list[dict]:
    """跑完整留一域实验。返回逐 (测试域, 种子) 的结果列表。"""
    task = ds.task
    results = []
    splits_all = lodo_splits_with_val(
        ds, val_mode=cfg.val_mode, seed=cfg.train.seed,
        group_by_subject=cfg.group_by_subject,
    )

    # 按测试域分组
    test_domains = sorted({s["test_domain"] for s in splits_all})

    for td in test_domains:
        fold_splits = [s for s in splits_all if s["test_domain"] == td]
        for seed in range(cfg.n_seeds):
            rng = np.random.default_rng(1000 * (td + 1) + seed)
            fold_metrics, fold_epochs = [], []

            for sp in fold_splits:
                tr_i, va_i, te_i = sp["train_idx"], sp["val_idx"], sp["test_idx"]
                Xtr_r, ytr = ds.X[tr_i], ds.y[tr_i]
                Xva_r, yva = ds.X[va_i], ds.y[va_i]
                Xte_r, yte = ds.X[te_i], ds.y[te_i]
                dtr, dva, dte = ds.domains[tr_i], ds.domains[va_i], ds.domains[te_i]

                # ---- 流水线：只在 train 上 fit ----
                pipe = CSIPipeline(cfg.pipeline)
                pipe.fit(Xtr_r, dtr, X_te=Xte_r, dom_te=dte)
                Xtr = pipe.transform(Xtr_r, dtr, unseen=False)
                Xva = pipe.transform(Xva_r, dva, unseen=True)
                Xte = pipe.transform(Xte_r, dte, unseen=True)

                # ---- 超参随机搜索 ----
                best_val, best_hp, best_res = -np.inf, None, None
                for t in range(cfg.n_trials):
                    hp = sample_hparams(cfg.hparam_space, rng)
                    tc = copy.deepcopy(cfg.train)
                    tc.seed = int(rng.integers(0, 10_000))
                    tc.lr = hp.get("lr", tc.lr)
                    tc.weight_decay = hp.get("weight_decay", tc.weight_decay)
                    tc.dropout = hp.get("dropout", tc.dropout)
                    tc.batch_size = int(hp.get("batch_size", tc.batch_size))

                    res = train_one(Xtr, ytr, Xva, yva, cfg.model, task, tc,
                                    model_kwargs=cfg.model_kwargs)
                    if res["best_metric"] > best_val:
                        best_val = res["best_metric"]
                        best_hp = hp
                        best_res = res

                best_epoch = max(1, (best_res["best_epoch"] + 1))
                fold_epochs.append(best_epoch)

                # ---- 用选中的超参评估测试域 ----
                if cfg.retrain_on_full:
                    Xfull = _concat(Xtr, Xva)
                    yfull = _concat(ytr, yva) if task == "classification" \
                        else np.concatenate([ytr, yva], axis=0)
                    tc = copy.deepcopy(cfg.train)
                    tc.seed = int(rng.integers(0, 10_000))
                    tc.lr = best_hp.get("lr", tc.lr)
                    tc.weight_decay = best_hp.get("weight_decay", tc.weight_decay)
                    tc.dropout = best_hp.get("dropout", tc.dropout)
                    tc.batch_size = int(best_hp.get("batch_size", tc.batch_size))
                    tc.epochs = best_epoch
                    tc.patience = 10 ** 6            # 固定 epoch 数，不再早停
                    res_full = train_one(Xfull, yfull, Xte, yte, cfg.model, task, tc,
                                         model_kwargs=cfg.model_kwargs)
                    logits = predict(res_full["model"], Xte, task,
                                     res_full["out_dim"], device=cfg.train.device)
                    m = evaluate(task, logits, yte,
                                 thresholds=cfg.pose_thresholds)
                else:
                    logits = predict(best_res["model"], Xte, task,
                                     best_res["out_dim"], device=cfg.train.device)
                    m = evaluate(task, logits, yte,
                                 thresholds=cfg.pose_thresholds)

                fold_metrics.append(m)
                if verbose:
                    vd = sp.get("val_domain")
                    print(f"  [test={td} val={vd} seed={seed}] val*={best_val:.4f} "
                          f"test={ {k: round(v, 4) for k, v in m.items()} }")

            # 对该测试域，跨验证折取平均
            keys = fold_metrics[0].keys()
            agg = {k: float(np.mean([fm[k] for fm in fold_metrics])) for k in keys}
            agg.update({
                "test_domain": int(td),
                "domain_name": ds.domain_names[td] if td < len(ds.domain_names) else str(td),
                "seed": seed,
                "n_folds": len(fold_metrics),
                "best_epochs": int(np.mean(fold_epochs)),
                "config": cfg.label(),
            })
            results.append(agg)

    return results


def aggregate(results: list[dict], primary: str | None = None) -> dict:
    """把逐 (测试域, 种子) 的结果汇总成 均值 ± 标准差。"""
    if not results:
        return {}
    keys = [k for k in results[0] if k not in (
        "test_domain", "domain_name", "seed", "n_folds", "best_epochs", "config")]
    if primary is None:
        primary = "acc" if "acc" in keys else "mpjpe"

    out = {"config": results[0].get("config", ""),
           "n_runs": len(results), "primary": primary}
    for k in keys:
        v = np.array([r[k] for r in results], dtype=float)
        out[f"{k}_mean"] = float(v.mean())
        out[f"{k}_std"] = float(v.std())

    # 逐测试域拆解
    per_dom = {}
    for r in results:
        per_dom.setdefault(r["domain_name"], []).append(r[primary])
    out["per_domain"] = {
        d: {"mean": float(np.mean(v)), "std": float(np.std(v))}
        for d, v in sorted(per_dom.items())
    }
    return out
