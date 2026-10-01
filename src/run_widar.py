"""Widar3.0 真实 CSI 跨域（LODO）消融诊断。

与 run_diag.py 同构，但数据来自真实 Widar3.0 CSI：
  1) build_widar_cache 把 .zip 里的 .dat 解析成 h5（支持 max_per_zip 均匀抽样、
     downsample 时间降采样，控制 h5 体量）。
  2) make_widar_dataset 读 h5 -> CSIDataset（domain=ENV）。
  3) 对每个预处理/协调配置，跑留一域（leave-one-domain-out）消融：
     外层：测试域是谁；内层：从训练域按被试分出 20% 验证集（不碰测试域）。

用法
----
    python run_widar.py --quick            # 小子集+少 epoch，先跑通整条链路
    python run_widar.py                    # 完整验证（默认子集）
    python run_widar.py --max-per-zip 1000 --epochs 25

可视化与产物
------------
- 每个 config 落盘 widar_run_<tag>.csv（每 epoch 一行 + 最后一行 test acc/F1）
- 实时 tqdm 进度条 + 总控进度（config × fold × epoch）
- 训练期间打印每折的 test acc / macroF1
- 全部跑完后画 3 张图（PNG）：
    widar_plot_acc.png    - 每配置平均 acc 柱状图（带 ±std 误差棒）
    widar_plot_f1.png     - 每配置平均 macroF1 柱状图
    widar_plot_loss.png   - 每个 config 第一折的 loss 曲线
- widar_results.json     - 全部结果（每个配置 + 总耗时 + 基线）
- tb_logs/               - TensorBoard 日志（`tensorboard --logdir tb_logs`）
"""
import argparse
import sys
import os
import time
import json
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, ".")

import numpy as np

from wfcslab.data.widar import build_widar_cache, make_widar_dataset
from wfcslab.pipeline import PipelineCfg, CSIPipeline
from wfcslab.engine.trainer import TrainCfg, train_one, predict
from wfcslab.engine.metrics import evaluate
from wfcslab.data.base import lodo_splits_with_val

CROSS = [
    ("raw/none/none", PipelineCfg(phase="none", clutter="none", domain="raw",
                                  normalize="global")),
    ("mti1/none/raw", PipelineCfg(phase="none", clutter="mti1", domain="raw",
                                  normalize="global")),
    ("csi_ratio/mpc/log", PipelineCfg(phase="csi_ratio", clutter="mpc", domain="log",
                                      normalize="global")),
    ("csi_ratio/mpc/log+combat", PipelineCfg(phase="csi_ratio", clutter="mpc", domain="log",
                                             harmonize="combat_meanonly", normalize="global")),
    ("csi_ratio/mpc/log+ea_trans", PipelineCfg(phase="csi_ratio", clutter="mpc", domain="log",
                                               harmonize="ea_trans", normalize="global")),
    ("csi_ratio/mpc/log+mixup0.4", PipelineCfg(phase="csi_ratio", clutter="mpc", domain="log",
                                                normalize="global"),
     {"mixup_alpha": 0.4}),
    ("csi_ratio/mpc/log+mixup0.2", PipelineCfg(phase="csi_ratio", clutter="mpc", domain="log",
                                                normalize="global"),
     {"mixup_alpha": 0.2}),
    # ---- P0-1（2026-09-04 推荐）：Body Velocity Profile + LeNet ----
    # STFT 谱图是动作的真正"指纹"（微多普勒频移）。BVP 内部已做
    # DC removal + log 压缩，所以 phase/clutter 都设为 none。
    # LeNetCSI 在 2D 频谱图上比 CNN_GRU 更合适（后者对频率维做
    # mean pooling 会丢动作信号）。
    ("none/none/bvp+lenet",
     PipelineCfg(phase="none", clutter="none", domain="bvp",
                 domain_kwargs={"n_fft": 64, "hop": 8},
                 normalize="global")),
]

# 与 CROSS 一一对应：每个 config 用什么 backbone
# 默认 cnn_gru；BVP 配置用 lenet
MODELS = ["cnn_gru"] * len(CROSS)
MODELS[-1] = "lenet"


def _safe_tag(s: str) -> str:
    return s.replace("/", "_").replace("+", "_p_")


def run_cfg(cfg: PipelineCfg, ds, splits: list, tag: str, epochs: int,
            out_dir: str, tb_root: str, batch_size: int = 128,
            mixup_alpha: float = 0.0,
            loss_type: str = "ce", focal_gamma: float = 2.0,
            balanced_sampler: bool = False,
            class_weights: list = None,
            model_name: str = "cnn_gru"):
    """对某个配置跑完整 LODO（所有折）。stage1-3 只算一次，按折只重算协调+归一化。"""
    from tqdm.auto import tqdm

    pipe = CSIPipeline(cfg)
    t_stage3 = time.time()
    X3_all = pipe.stage3(ds.X)
    stage3_time = time.time() - t_stage3

    cfg_log_dir = os.path.join(out_dir, "cfg_logs")
    cfg_tb_dir = os.path.join(tb_root, _safe_tag(tag))
    os.makedirs(cfg_log_dir, exist_ok=True)
    os.makedirs(cfg_tb_dir, exist_ok=True)

    accs, f1s, n_tr_list, n_te_list = [], [], [], []
    first_fold_losses = []  # 给画 loss 曲线用

    pbar = tqdm(splits, desc=tag, dynamic_ncols=True, position=0)
    for sp in pbar:
        tr, va, te = sp["train_idx"], sp["val_idx"], sp["test_idx"]
        dtr, dva, dte = ds.domains[tr], ds.domains[va], ds.domains[te]
        pipe.fit(ds.X[tr], dtr, X_te=ds.X[te], dom_te=dte,
                 X_tr3=X3_all[tr], X_te3=X3_all[te])
        Xtr_f = pipe.transform(ds.X[tr], dtr, unseen=False, X3=X3_all[tr])
        Xva_f = pipe.transform(ds.X[va], dva, unseen=False, X3=X3_all[va])
        Xte_f = pipe.transform(ds.X[te], dte, unseen=True, X3=X3_all[te])

        fold_csv = os.path.join(cfg_log_dir,
                                f"{_safe_tag(tag)}_d{sp['test_domain']}.csv")
        tcfg = TrainCfg(
            epochs=epochs, batch_size=batch_size, seed=0, patience=12,
            num_workers=2, pin_memory=True, use_amp=True, tf32=True,
            matmul_precision="high",
            show_progress=False,                # 外层已有 tqdm
            log_csv=fold_csv,
            tb_dir=None,                         # TB 写到 cfg 级目录（避免 fold 子目录爆炸）
            progress_desc=f"d{sp['test_domain']}",
            clear_log_on_start=True,
            mixup_alpha=mixup_alpha,
            loss_type=loss_type,
            focal_gamma=focal_gamma,
            balanced_sampler=balanced_sampler,
            class_weights=class_weights,
        )
        res = train_one(Xtr_f, ds.y[tr], Xva_f, ds.y[va],
                        model_name, "classification", tcfg)
        # 把折指标汇总到 cfg 级 TB（容错：若环境装坏就跳过）
        try:
            from torch.utils.tensorboard import SummaryWriter
            if os.path.exists(cfg_tb_dir) and not os.path.isdir(cfg_tb_dir):
                os.remove(cfg_tb_dir)
            os.makedirs(cfg_tb_dir, exist_ok=True)
            tbw = SummaryWriter(log_dir=cfg_tb_dir, purge_step=True)
            tbw.add_scalar(f"fold/test_acc", res["history"][-1].get("acc", 0), sp["test_domain"])
            for h in res["history"]:
                tbw.add_scalar("fold/val_acc", h["acc"], h["epoch"])
                tbw.add_scalar("fold/train_loss", h["train_loss"], h["epoch"])
            tbw.close()
        except Exception as e:
            print(f"  [warn] TB 写入失败（{e!r}），跳过")

        lg = predict(res["model"], Xte_f, "classification", res["out_dim"])
        m = evaluate("classification", lg, ds.y[te])
        accs.append(m["acc"]); f1s.append(m["macro_f1"])
        n_tr_list.append(len(tr)); n_te_list.append(len(te))
        if not first_fold_losses and splits.index(sp) == 0:
            first_fold_losses = [h["train_loss"] for h in res["history"]]

        # 折级实时进度
        done = len(accs); total = len(splits)
        pbar.set_postfix(
            fold=f"{done}/{total}",
            last_acc=f"{m['acc']:.3f}",
            mean_acc=f"{np.mean(accs):.3f}",
            mean_f1=f"{np.mean(f1s):.3f}",
            eta=f"{int((time.time() - pbar.start_t) / done * (total - done))}s"
            if done and pbar.start_t else "",
        )

    pbar.close()
    return (np.array(accs), np.array(f1s), np.array(n_tr_list),
            np.array(n_te_list), first_fold_losses, stage3_time)


def plot_results(results: dict, first_fold_loss: dict, out_dir: str,
                 baseline: float):
    """画 3 张图：acc / F1 柱状图 + loss 曲线。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tags = list(results.keys())
    acc_mean = [results[t]["acc_mean"] for t in tags]
    acc_std = [results[t]["acc_std"] for t in tags]
    f1_mean = [results[t]["f1_mean"] for t in tags]
    f1_std = [results[t]["f1_std"] for t in tags]

    short = [t.replace("/", "\n").replace("_p_", "+") for t in tags]
    x = np.arange(len(tags))

    # ---- acc 柱状图 ----
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x, acc_mean, yerr=acc_std, capsize=4, color="#4C8BF5",
           edgecolor="black", alpha=0.85)
    ax.axhline(baseline, color="gray", ls="--", lw=1.2,
               label=f"随机基线 {baseline:.3f}")
    ax.set_xticks(x); ax.set_xticklabels(short, fontsize=8)
    ax.set_ylabel("Accuracy")
    ax.set_title("LODO cross-domain accuracy (Widar3.0 real CSI)")
    ax.set_ylim(0, max(0.6, max(acc_mean) * 1.4))
    ax.legend()
    for i, v in enumerate(acc_mean):
        ax.text(i, v + max(acc_std) * 0.6 + 0.01, f"{v:.3f}",
                ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "widar_plot_acc.png"), dpi=150)
    plt.close(fig)

    # ---- F1 柱状图 ----
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x, f1_mean, yerr=f1_std, capsize=4, color="#FF9F40",
           edgecolor="black", alpha=0.85)
    ax.axhline(baseline, color="gray", ls="--", lw=1.2)
    ax.set_xticks(x); ax.set_xticklabels(short, fontsize=8)
    ax.set_ylabel("Macro-F1")
    ax.set_title("LODO cross-domain macro-F1 (Widar3.0 real CSI)")
    ax.set_ylim(0, max(0.6, max(f1_mean) * 1.4))
    for i, v in enumerate(f1_mean):
        ax.text(i, v + max(f1_std) * 0.6 + 0.01, f"{v:.3f}",
                ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "widar_plot_f1.png"), dpi=150)
    plt.close(fig)

    # ---- loss 曲线（每个 config 第一折的训练 loss）----
    fig, ax = plt.subplots(figsize=(9, 5))
    for t in tags:
        if t in first_fold_loss and first_fold_loss[t]:
            ax.plot(first_fold_loss[t], label=t, lw=1.2)
    ax.set_xlabel("epoch")
    ax.set_ylabel("train loss")
    ax.set_title("First-fold train loss per config")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "widar_plot_loss.png"), dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    here = os.path.dirname(os.path.abspath(__file__))
    # 项目根目录 = wifi-crossenv-lab 的父目录（即 wifi识别/）
    root = os.path.dirname(os.path.dirname(here))
    ap.add_argument("--raw-dir", default=os.path.join(root, "data", "raw", "widar3.0", "CSI"))
    ap.add_argument("--out-h5", default=os.path.join(root, "data", "processed", "widar3.0", "csi.h5"))
    ap.add_argument("--limit", type=int, default=6000,
                    help="全局最多取多少 .dat（先打乱再截取，保证跨域覆盖）")
    ap.add_argument("--downsample", type=int, default=2, help="时间降采样倍率")
    ap.add_argument("--max-time", type=int, default=256, help="每样本固定帧数")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--configs", default="all", help="逗号分隔的索引，如 0,3,4")
    ap.add_argument("--no-build", action="store_true", help="跳过建缓存（直接用已有 h5）")
    ap.add_argument("--force-build", action="store_true", help="强制重建缓存")
    ap.add_argument("--shuffle", action="store_true", default=True, help="解析前全局打乱（默认开）")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--quick", action="store_true", help="小子集+少 epoch 快速验证")
    ap.add_argument("--batch-size", type=int, default=128, help="训练 batch size（默认 128）")
    ap.add_argument("--workers", type=int, default=0,
                    help="build cache 多进程数（0=按 CPU 自动，1=单线程）")
    ap.add_argument("--focal-loss", action="store_true", help="用 Focal Loss 替代 CE")
    ap.add_argument("--focal-gamma", type=float, default=2.0, help="Focal Loss 的 γ 参数")
    ap.add_argument("--balanced-sampler", action="store_true",
                    help="用 WeightedRandomSampler 平衡各类采样频率")
    ap.add_argument("--class-weights", default=None,
                    help="逗号分隔的 8 个类别权重（如 1.0,1.2,0.9,1.1,1.0,1.3,0.8,1.2）")
    ap.add_argument("--out-dir", default=here, help="结果输出目录（默认脚本所在目录）")
    args = ap.parse_args()

    # --quick 是「preset」，只在命令行没显式覆盖时才生效
    # 否则用户传 `--epochs 25 --quick` 时会被静默改回 10
    import sys as _sys
    _argv_set = set(_sys.argv[1:])
    if args.quick:
        if "--limit" not in _argv_set:
            args.limit = 3000
        if "--epochs" not in _argv_set:
            args.epochs = 10

    os.makedirs(args.out_dir, exist_ok=True)
    tb_root = os.path.join(args.out_dir, "tb_logs")

    need_build = (not args.no_build) and (
        args.force_build or not os.path.exists(args.out_h5)
    )
    if need_build:
        n_workers = args.workers if args.workers > 0 \
            else max(1, (os.cpu_count() or 4) - 1)
        print(f"=== 构建缓存（n_workers={n_workers}）===")
        t0 = time.time()
        st = build_widar_cache(
            args.raw_dir, args.out_h5,
            max_time=args.max_time,
            limit=args.limit,
            downsample=args.downsample,
            shuffle=args.shuffle,
            seed=args.seed,
            force=True,
            n_workers=n_workers,
        )
        print("build 统计:", json.dumps(st, ensure_ascii=False))
        print(f"  构建耗时 {time.time() - t0:.0f}s")

    ds = make_widar_dataset(args.out_h5)
    print("\n" + ds.summary())
    # 用 unique 数量而非 max+1：Widar 标签 1-based，max+1 会算成 N+1
    n_cls = int(np.unique(ds.y).size)
    baseline = 1.0 / n_cls

    sel = list(range(len(CROSS))) if args.configs == "all" \
        else [int(x) for x in args.configs.split(",")]

    # 把 CLI 类别不平衡选项转成 train_kwargs
    cw = None
    if args.class_weights:
        cw = [float(x) for x in args.class_weights.split(",")]
        assert len(cw) == n_cls, (
            f"--class-weights 长度 {len(cw)} ≠ 类别数 {n_cls}")
    addon_kwargs = dict(
        loss_type=("focal" if args.focal_loss else "ce"),
        focal_gamma=args.focal_gamma,
        balanced_sampler=args.balanced_sampler,
        class_weights=cw,
    )

    results = {}
    first_fold_loss = {}
    total_t0 = time.time()
    for i in sel:
        # 兼容老 CROSS 条目（仅 2 元素）和带 train_kwargs 的新条目（3 元素）
        if len(CROSS[i]) == 2:
            tag, cfg = CROSS[i]
            train_kwargs = {}
        else:
            tag, cfg, train_kwargs = CROSS[i]
        # CLI 覆盖 CROSS 里的 train_kwargs（mixup_alpha 保留，加 focal/sampler）
        merged = {**train_kwargs, **addon_kwargs}
        t0 = time.time()
        try:
            accs, f1s, n_tr, n_te, loss_curve, stage3_t = run_cfg(
                cfg, ds,
                list(lodo_splits_with_val(ds, val_mode="random", val_frac=0.2,
                                           seed=0, group_by_subject=True)),
                tag=tag, epochs=args.epochs,
                out_dir=args.out_dir, tb_root=tb_root,
                batch_size=args.batch_size,
                model_name=MODELS[i],
                **merged,
            )
        except Exception as e:
            import traceback
            print(f"[{i}] {tag} 失败: {e!r}")
            traceback.print_exc()
            continue
        mean_a, std_a = float(np.mean(accs)), float(np.std(accs))
        mean_f, std_f = float(np.mean(f1s)), float(np.std(f1s))
        results[tag] = {
            "acc_mean": mean_a, "acc_std": std_a,
            "f1_mean": mean_f, "f1_std": std_f,
            "n_folds": len(accs),
            "mean_train_n": int(np.mean(n_tr)),
            "mean_test_n": int(np.mean(n_te)),
            "per_fold_acc": accs.tolist(),
            "per_fold_f1": f1s.tolist(),
            "stage3_time_s": round(stage3_t, 2),
            "cfg_time_s": round(time.time() - t0, 2),
        }
        if loss_curve:
            first_fold_loss[tag] = loss_curve
        print(f"[{i}] {tag:34s} acc={mean_a:.3f}±{std_a:.3f}  "
              f"macroF1={mean_f:.3f}±{std_f:.3f}  "
              f"({time.time() - t0:.0f}s, {len(accs)}/{len(list(lodo_splits_with_val(ds, val_mode='random', val_frac=0.2, seed=0, group_by_subject=True)))} folds)")

        # 增量落盘（防止后续 fold 失败丢失已得结果）
        out = {"baseline": baseline, "n_classes": n_cls,
               "total_elapsed_s": round(time.time() - total_t0, 2),
               "results": results}
        with open(os.path.join(args.out_dir, "widar_results.json"), "w",
                  encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)

    # ---- 总结 ----
    print(f"\n随机基线（{n_cls} 类）= {baseline:.3f}")
    print("\n=== 跨域 LODO 汇总 ===")
    print(f"{'config':<34s} {'acc (mean±std)':<18s} {'macroF1':<18s} {'fold':<6s} {'time':<8s}")
    for tag, r in results.items():
        print(f"{tag:<34s} "
              f"{r['acc_mean']:.3f}±{r['acc_std']:.3f}      "
              f"{r['f1_mean']:.3f}±{r['f1_std']:.3f}      "
              f"{r['n_folds']:<6d} {r['cfg_time_s']:<8.0f}")
    print(f"\n总耗时 {time.time() - total_t0:.0f}s")
    print("结果已写入 widar_results.json / cfg_logs/*.csv / widar_plot_*.png")
    print(f"TensorBoard: tensorboard --logdir {tb_root}")

    # ---- 画图 ----
    if results:
        try:
            plot_results(results, first_fold_loss, args.out_dir, baseline)
        except Exception as e:
            print(f"[warn] 画图失败：{e!r}")


if __name__ == "__main__":
    main()
