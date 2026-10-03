# P0 Widar v2 canonical 重测结果（2026-10-03 定案）

> Runner: `p0_widar_v2_runner.py`，2026-10-02 12:25 启动 → 10-03 11:47 完成，总时长 23h22m，RTX 2080 Ti 单卡。
> 产物：`src/bvp_test/f32_multiseed_v2.json`（30 runs）+ `f48_cbam_resnet18_v2.json`（30 runs）+ `p0_paired_stats_v2.json`（30 pairs）。

## 实验配置

| 项 | 值 |
|---|---|
| Cache | `csi_50k_v2.h5`（CANONICAL_5 13-zip，ges_filter={1..5}，**5 类**，158,500 池） |
| BVP cache | `_cache_bvp_ms64_128_50k_v2.npy` fp16，per-sample (9, 60, 33, 17)，50k = **31.55%** |
| 协议 | LODO，10 folds (d1..d10) × 3 seeds = **30 strict pairs**（v1 只有 6 folds × 3 seeds = 18） |
| Backbone | LeNetCSI_Attn (0.13M) vs CBAM+ResNet18 (11.42M, 89×) |

## 正式配对统计（step5 原始输出）

```
attn_mean = 20.67% ± 0.54
cbam_mean = 20.95% ± 0.83
delta_attn_minus_cbam = -0.28pp, sigma_delta = 1.10pp
CI95 = [-0.68, +0.11] pp      t = -1.41, p = 0.1587 (n.s.)
Cohen's d = -0.257, Hedges' g = -0.251
wins_attn = 14/30 (46.7%)
```

## 核心结论：v1 容量对比是错标签 artifact，修正标签后坍缩至随机

1. **两 backbone 在 v2 canonical 5-class LODO 下全部 ≈ 随机（20%）**：
   - Attn 30/30 runs ∈ [20.05%, 22.13%]，大量 run 在 ep 0–3 早停（val 不涨）
   - CBAM 30/30 runs ∈ [19.93%, 22.65%]
   - 随机基线 = 1/5 = 20% → **cross-domain 5-class Widar 任务在 BVP+LeNet/CBAM 上不可学**
2. **容量差异蒸发**：v1（8-index 错标签）Δ=−4.61pp (p=.028, 12k) / −5.75pp (p=.0609, 50k)；v2 **Δ=−0.28pp (p=.159)**，CI 极窄 [−0.68, +0.11] —— 是"精确的零"，不是欠功效（σΔ 从 13.0pp 坍缩到 1.10pp，30 pairs）。
3. **机制解释**：v1 的 per-zip gesture index 与 domain archive 强相关 → 模型走 domain 捷径，"38.48% vs 32.73%" 的相对差异是 spurious label-domain 耦合的可学习部分；干净全局标签切断捷径后，绝对精度与相对差异同时消失。
4. **对论文的意义**：labeling-crisis 的最强证据链闭合——
   - MMFi：v1 path-labeled cache → 双 backbone 近随机（−1.67pp）
   - Widar：v1 错标签 → 高精度但语义错（38%/33%）+ subset-fragile 显著性；v2 正标签 → 均匀 chance-level
   - 叙事升级：**"once labeling is fixed, both the absolute accuracy and the backbone contrast collapse to chance"**

## v1 vs v2 对照表

| 口径 | 标签 | 类数 | 池 | subset | Attn | CBAM | Δ | p | σΔ |
|---|---|---|---|---|---|---|---|---|---|
| C10 (12k v1) | per-zip 错配 | 8 index | 163,650 link-rec | 7.33% | 34.06 | 38.67? | −4.61 | 0.028 | 8.9 |
| C10v2 (50k v1) | per-zip 错配 | 8 index | 163,650 link-rec | 30.55% | 38.48 | 32.73 | −5.75 | 0.0609 | 13.0 |
| **C10v3 (50k v2)** | **canonical 全局** | **5** | **158,500** | **31.55%** | **20.67±0.54** | **20.95±0.83** | **−0.28** | **0.159** | **1.10** |

（C10 12k 行的 CBAM 数字见 v1 记录 P0_widar_50k_results.md；此处以论文 tab:main 为准。）

## 后续动作

- [ ] 论文更新方案（待用户确认）：tab:main 增补 C10v3 行 / §5 主段披露 / 摘要一句话 / 结论与 §9.2 子集政策联动
- [ ] WIDAR_TRUTH.md 增补 v2 重测定案
- [ ] v2 产物纳入 GitHub 仓库 results/（匿名许可）
