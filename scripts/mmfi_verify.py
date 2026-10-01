# -*- coding: utf-8 -*-
"""
MMFi 解压完整性校验
====================
对比 MMFi_action_segments.csv 期望的 (env, subj, action) -> frame 集合，
与 data/raw/mmfi_unzipped 下实际落盘的文件，统计缺失率。

用法:
    python scripts/mmfi_verify.py                    # 全量
    python scripts/mmfi_verify.py --envs E01,E03     # 只查部分 env
    python scripts/mmfi_verify.py --quick            # 只统计文件数，不列明细
"""
import csv
import os
import sys
import argparse
from collections import defaultdict

ROOT = r"F:/python_workspace/wifi识别/data/raw/MMFi_Dataset"
CSV_PATH = os.path.join(ROOT, "MMFi_action_segments.csv")
UNZIP_DIR = r"F:/python_workspace/wifi识别/data/raw/mmfi_unzipped"


def parse_segments(seg: str):
    """'1-7; 8-15; 16-21' -> [1..7, 8..15, 16..21]
    MMFi CSV 用分号分隔多个区间（也兼容逗号）。"""
    out = []
    for ch in (",",):
        seg = seg.replace(ch, ";")
    for part in seg.split(";"):
        part = part.strip().strip("'\"").strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            try:
                out.extend(range(int(a.strip()), int(b.strip()) + 1))
            except ValueError:
                continue
        else:
            try:
                out.append(int(part))
            except ValueError:
                continue
    return out


def build_expect(env_filter=None):
    """返回 {(env, subj, action): set(frame_id)} 与统计。"""
    rows = list(csv.DictReader(open(CSV_PATH, encoding="utf-8")))
    expect = defaultdict(set)
    for r in rows:
        env, subj, act = r["Environment"], r["Student"], r["Action"]
        if env_filter and env not in env_filter:
            continue
        expect[(env, subj, act)].update(parse_segments(r["Segments"]))
    return expect


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--envs", default=None, help="逗号分隔，如 E01,E03")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    env_filter = set(args.envs.split(",")) if args.envs else None
    expect = build_expect(env_filter)

    total_exp = sum(len(v) for v in expect.values())
    print(f"CSV 期望: {len(expect)} 个 (env,subj,action) 三元组, 共 {total_exp} frames")
    if env_filter:
        print(f"  过滤 env: {sorted(env_filter)}")
    print()

    # 实际落盘
    actual = defaultdict(set)
    n_files = 0
    for env in sorted(set(k[0] for k in expect)):
        edir = os.path.join(UNZIP_DIR, env)
        if not os.path.isdir(edir):
            print(f"[{env}] 目录不存在 —— 完全未解压")
            continue
        for subj in os.listdir(edir):
            sdir = os.path.join(edir, subj)
            if not os.path.isdir(sdir):
                continue
            for act in os.listdir(sdir):
                adir = os.path.join(sdir, act, "wifi-csi")
                if not os.path.isdir(adir):
                    continue
                for fn in os.listdir(adir):
                    if not fn.endswith(".mat"):
                        continue
                    # frame000123.mat -> 123
                    stem = fn[:-4]
                    digits = "".join(ch for ch in stem if ch.isdigit())
                    if digits:
                        actual[(env, subj, act)].add(int(digits))
                        n_files += 1
        print(f"[{env}] 已扫, 累计文件 {n_files}")

    print(f"\n实际落盘: {len(actual)} 个三元组, 共 {n_files} frames")
    print()

    # 对比
    missing_total = 0
    complete = 0
    partial = []
    absent = 0
    for key, exp_set in sorted(expect.items()):
        act_set = actual.get(key, set())
        if not act_set:
            absent += 1
            missing_total += len(exp_set)
            continue
        miss = exp_set - act_set
        if not miss:
            complete += 1
        else:
            partial.append((key, len(exp_set), len(act_set), len(miss)))
            missing_total += len(miss)

    n_keys = len(expect)
    print("=" * 60)
    print(f"完整三元组      : {complete}/{n_keys}  ({complete/n_keys*100:.1f}%)")
    print(f"部分缺失        : {len(partial)}")
    print(f"完全缺失        : {absent}")
    print(f"缺失 frame 总数 : {missing_total}/{total_exp}  ({missing_total/total_exp*100:.2f}%)")
    print(f"覆盖率          : {(total_exp-missing_total)/total_exp*100:.2f}%")
    print("=" * 60)

    if not args.quick and partial:
        print("\n部分缺失 Top 20 (env, subj, act)  期望/实际/缺:")
        for key, ne, na, nm in sorted(partial, key=lambda x: -x[3])[:20]:
            print(f"  {key[0]}/{key[1]}/{key[2]}  {ne}/{na}/缺{nm}")

    # 按 env 汇总
    print("\n按 env 汇总覆盖率:")
    by_env = defaultdict(lambda: [0, 0])  # exp, miss
    for key, exp_set in expect.items():
        act_set = actual.get(key, set())
        by_env[key[0]][0] += len(exp_set)
        by_env[key[0]][1] += len(exp_set - act_set)
    for env in sorted(by_env):
        e, m = by_env[env]
        print(f"  {env}: {e-m}/{e} = {(e-m)/e*100:.2f}%")

    # 可跑 LODO 的域：覆盖率 >= 95%
    print("\n可用作 LODO 域 (覆盖率 >= 95%):")
    ok = [env for env, (e, m) in sorted(by_env.items()) if (e - m) / e >= 0.95]
    print("  ", ok if ok else "（无）")


if __name__ == "__main__":
    main()
