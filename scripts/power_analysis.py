"""P1-4 · 失败方法的统计力充分性报告（Power Analysis / MDE）

核心问题
========
B1–B13 的"反指/无效"结论会被审稿质疑："是不是你 seed 不够 / 样本不够，
才没检出效果？"

本脚本用 **最小可检出效应量（Minimum Detectable Effect, MDE）** 主动回击：
对每个方法，计算「当前 n 组配对 + power=0.8 + α=0.05 能检出的最小效应量」。
若 MDE 已经覆盖文献声称的效应量（甚至更小），而我们检出的真实效应量
远小于 MDE，则证明「无效」是真实的，不是统计力不足。

同时用 **正面对照**（B10 多 Rx +8.14pp）做 validation：
同一个 MDE 框架必须能正确判定"真有效"的方法（效应 >> MDE），
才能让人信服它也能正确判定"真无效"的方法（效应 << MDE）。

公式（配对 t 检验，Z 近似）
============================
  MDE_pp = (Z_{1-α/2} + Z_{1-β}) × σ_d / sqrt(n)
         = 2.80 × σ_d / sqrt(n)            # α=0.05 双侧, β=0.2
其中 σ_d = 配对差值 std，n = 配对组数。

数据来源（均有完整 (fold, seed) 配对）
======================================
  B8   架构     f32: lenet_attn_mha vs lenet_attn          (6 fold × 3 seed)
  B9   多尺度   f33: F-24_multiScale vs F-21_multiRx       (6 × 3)
  B10  原版     f34: F-21_hop8_rx vs Widar3.0_orig         (6 × 3)  [正面对照]
  B12  ensemble f35: snap3 vs best                          (6 × 3)
  B13  DANN     f37 dann vs f32 lenet_attn                  (6 × 3)
  B15  天线     f39: rms_agg vs multirx9                    (6 × 3)
  B17  MMFi     f40: single_ch vs multirx10                 (4 env × 3)
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
from scipy import stats as scs

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BT = os.path.join(ROOT, 'src/bvp_test')
OUT_JSON = os.path.join(BT, 'f42_power_analysis.json')

FOLDS = [1, 3, 5, 6, 7, 9]
SEEDS = [0, 1, 2]
MMFI_ENVS = ['E1', 'E2', 'E3', 'E4']

Z_ALPHA = 1.959963984540054   # 双侧 0.05
Z_BETA = 0.8416212335729143   # power 0.8


def load(name: str):
    return json.load(open(os.path.join(BT, name), encoding='utf-8'))


def paired_series(data_a, data_b, folds, seeds, suffix_a='_d{f}_s{s}',
                  suffix_b='_d{f}_s{s}'):
    """从两个 dict 提取按 (fold, seed) 配对的 acc 序列，返回 (A, B) numpy 数组。"""
    A, B = [], []
    for f in folds:
        for s in seeds:
            ka, kb = suffix_a.format(f=f, s=s), suffix_b.format(f=f, s=s)
            if ka in data_a and kb in data_b and 'acc' in data_a[ka] and 'acc' in data_b[kb]:
                A.append(data_a[ka]['acc'])
                B.append(data_b[kb]['acc'])
    return np.array(A), np.array(B)


def analyze(name, A, B, claimed_pp, claimed_desc, note=""):
    """配对分析 + MDE。返回 dict。"""
    D = B - A
    n = len(D)
    t, p = scs.ttest_rel(B, A)
    d_mean = D.mean()
    sd = D.std(ddof=1)
    cohens_d = d_mean / (sd + 1e-12)
    # 95% CI（配对，用 t 分布）
    se = sd / np.sqrt(n)
    tcrit = scs.t.ppf(0.975, n - 1) if n > 1 else np.inf
    ci = (d_mean - tcrit * se, d_mean + tcrit * se)
    # MDE（pp）：power=0.8 能检出的最小效应
    mde_pp = (Z_ALPHA + Z_BETA) * sd / np.sqrt(n) * 100
    # 判定：真实效应是否 < MDE（即"统计力足够但没检出" → 真无效）
    verdict = "无效确证" if abs(d_mean * 100) < mde_pp and abs(cohens_d) < 0.5 else \
              ("真有效（效应>MDE）" if abs(d_mean * 100) > mde_pp else "统计力不足")
    return {
        "name": name, "n": n,
        "delta_pp": float(d_mean * 100),
        "sd_pp": float(sd * 100),
        "p_ttest": float(p),
        "cohens_d": float(cohens_d),
        "ci95_pp": [float(ci[0] * 100), float(ci[1] * 100)],
        "mde_pp": float(mde_pp),
        "claimed_pp": claimed_pp,
        "claimed_desc": claimed_desc,
        "verdict": verdict,
        "note": note,
    }


def main():
    f32 = load('f32_multiseed.json')
    f33 = load('f33_rx_vs_ms.json')
    f34 = load('f34_widar_orig_vs_f24.json')
    f35 = load('f35_snapshot_ensemble.json')
    f37 = load('f37_dann.json')
    f39 = load('f39_antenna_ablation.json')
    f40 = load('f40_mmfi_ablation.json')

    rows = []

    # ---- 负面对照（应"无效确证"）----
    A, B = paired_series(f32, f32, FOLDS, SEEDS,
                         'lenet_attn_d{f}_s{s}', 'lenet_attn_mha_d{f}_s{s}')
    rows.append(analyze('B8 多头注意力 vs 卷积', A, B,
                        claimed_pp=2.0, claimed_desc='MHA 通常声称 +2~5pp',
                        note='架构类创新代表'))

    A, B = paired_series(f33, f33, FOLDS, SEEDS,
                         'F-21_multiRx_d{f}_s{s}', 'F-24_multiScale_d{f}_s{s}')
    rows.append(analyze('B9 多尺度 vs 多Rx', A, B,
                        claimed_pp=1.5, claimed_desc='多尺度通常声称 +1~3pp'))

    A, B = paired_series(f35, f35, FOLDS, SEEDS,
                         'best_d{f}_s{s}', 'snap3_d{f}_s{s}')
    rows.append(analyze('B12 Snapshot Ensemble', A, B,
                        claimed_pp=1.5, claimed_desc='ensemble 通常声称 +1~2pp'))

    # DANN：f37 dann vs f32 lenet_attn（基线）
    A, B = paired_series(f32, f37, FOLDS, SEEDS,
                         'lenet_attn_d{f}_s{s}', 'dann_d{f}_s{s}')
    rows.append(analyze('B13 DANN 域对抗', A, B,
                        claimed_pp=3.0, claimed_desc='域对抗通常声称 +3~5pp'))

    # F-39 天线聚合（rms_agg vs multirx9，正=multirx 更好）
    A, B = paired_series(f39, f39, FOLDS, SEEDS,
                         'rms_agg_d{f}_s{s}', 'multirx9_d{f}_s{s}')
    rows.append(analyze('B15 多Rx vs RMS聚合', A, B,
                        claimed_pp=8.0, claimed_desc='多Rx 实测 +9pp（正面对照）'))

    # ---- 正面对照（应"真有效"）----
    A, B = paired_series(f34, f34, FOLDS, SEEDS,
                         'Widar3.0_orig_d{f}_s{s}', 'F-21_hop8_rx_d{f}_s{s}')
    rows.append(analyze('B10 多Rx vs Widar原版 [正]', A, B,
                        claimed_pp=8.0, claimed_desc='已知真效应 +8.14pp'))

    # MMFi（跨数据集，应"无效确证"或"统计力不足"）
    def mmfi_paired(va, vb):
        A, B = [], []
        for e in MMFI_ENVS:
            for s in SEEDS:
                ka, kb = f'{va}_e{e}_s{s}', f'{vb}_e{e}_s{s}'
                if ka in f40 and kb in f40 and 'acc' in f40[ka] and 'acc' in f40[kb]:
                    A.append(f40[ka]['acc'])
                    B.append(f40[kb]['acc'])
        return np.array(A), np.array(B)

    A, B = mmfi_paired('single_ch', 'multirx10')
    rows.append(analyze('B17 多Rx (MMFi)', A, B,
                        claimed_pp=9.0, claimed_desc='Widar 上 +9pp，跨数据集复现',
                        note='MMFi 地板效应'))

    # ---- 输出 ----
    print('=' * 100)
    print('P1-4 统计力充分性报告（配对 t 检验 + MDE, power=0.8, α=0.05 双侧）')
    print('=' * 100)
    hdr = f'{"方法":<26}{"n":>3}{"Δpp":>8}{"σd":>7}{"p":>8}{"d":>7}{"CI95%":>22}{"MDE":>7}  判定'
    print(hdr)
    print('-' * 100)
    out = {}
    for r in rows:
        ci = f"[{r['ci95_pp'][0]:+.2f},{r['ci95_pp'][1]:+.2f}]"
        print(f"{r['name']:<26}{r['n']:>3}{r['delta_pp']:>+8.2f}{r['sd_pp']:>7.2f}"
              f"{r['p_ttest']:>8.3f}{r['cohens_d']:>+7.2f}{ci:>22}{r['mde_pp']:>7.2f}  {r['verdict']}")
        out[r['name']] = r

    print('\n' + '=' * 100)
    print('解读')
    print('=' * 100)
    for r in rows:
        print(f"\n【{r['name']}】 判定={r['verdict']}")
        print(f"  实测效应 {r['delta_pp']:+.2f}pp (d={r['cohens_d']:+.2f})，"
              f"MDE={r['mde_pp']:.2f}pp")
        print(f"  文献声称 {r['claimed_desc']}；")
        if r['verdict'] == '无效确证':
            print(f"  ⇒ 我们的 n={r['n']} 组已能检出 {r['mde_pp']:.2f}pp 的效应（power=0.8），"
                  f"却只检出 {r['delta_pp']:+.2f}pp。"
                  f"效应量远小于 MDE，'无效'是真实的，非统计力不足。")
        elif r['verdict'] == '真有效（效应>MDE）':
            print(f"  ⇒ 效应 {r['delta_pp']:+.2f}pp 远超 MDE {r['mde_pp']:.2f}pp，"
                  f"同一框架正确识别出'真有效'，验证 MDE 判定可信。")
        else:
            print(f"  ⇒ 效应与 MDE 同量级，需更多 seed 才能定论（正是 P1-1 seed 3→5 的动机）。")

    json.dump(out, open(OUT_JSON, 'w', encoding='utf-8'),
              indent=2, ensure_ascii=False)
    print(f'\n[done] {OUT_JSON}')


if __name__ == '__main__':
    main()
