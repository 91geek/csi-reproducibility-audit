"""
MMFi 数据按 Env 重新组织
=========================

问题
----
MMFi 官方分发把 1 个 subject 打成 1 个 zip（S01.zip ~ 2GB），但 zip 内的目录是
``S01/AXX/wifi-csi/frame*.mat``，**不含 E 维度**。同一 subject 的多个 env 数据
混在同一个 zip 内，要靠 ``MMFi_action_segments.csv`` 的 Segments 字段（"1-7; 8-15; ..."）
把它们拆回 ``E01/S01/AXX/wifi-csi/frameXXX.mat``。

策略
----
1. 解析 CSV，得到 ``(env, student, action, [segments])`` 元组列表
2. 对每个 zip，**直接从内存 zip 读 frame 文件**，写到目标路径
   ``{out_root}/E{ee}/S{ss}/A{aa}/wifi-csi/frame{XXX}.mat``
3. **不复用重复 frame**：每个 env/action 的 segments 是 1 段连续 frame 区间，
   但不同 env 共享同一组 frame 编号区间，按 env 分别复制
4. 进度条 + 跳过已存在（断点续传）

用法
----
    # 解全部 40 subject（占空间 ≈ 78GB），输出到 E:/mmfi_unzipped/
    python scripts/prepare_mmfi_unzip.py --src "F:/python_workspace/wifi识别/data/raw/MMFi_Dataset/Zipfiles" \\
        --csv "F:/python_workspace/wifi识别/data/raw/MMFi_Dataset/MMFi_action_segments.csv" \\
        --out "F:/python_workspace/wifi识别/data/raw/mmfi_unzipped" \\
        --n-subjects 40

    # 调试：只解前 2 个 subject
    python scripts/prepare_mmfi_unzip.py --n-subjects 2

注意
----
- 中断后重跑会**跳过已存在目标文件**，但 zip 内 frame 不删；想要"重新解满"加 --force
- 写盘是主要瓶颈，预计 78GB 在 SSD 上约 20-40 分钟
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import re
import sys
import time
import zipfile
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("mmfi_unzip")


# ---------------------------------------------------------------------------
# CSV 解析
# ---------------------------------------------------------------------------
@dataclass
class VideoSeg:
    env: str       # 'E01'
    student: str   # 'S01'
    action: str    # 'A05'
    segments: List[Tuple[int, int]]  # [(1,7),(8,15),...] 1-based inclusive


def parse_segments(s: str) -> List[Tuple[int, int]]:
    """'1-7; 8-15; 16-21' -> [(1,7),(8,15),(16,21)]"""
    out = []
    for tok in s.split(";"):
        tok = tok.strip()
        if not tok:
            continue
        a, b = tok.split("-")
        out.append((int(a), int(b)))
    return out


def load_csv(csv_path: str) -> List[VideoSeg]:
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(
                VideoSeg(
                    env=r["Environment"].strip(),
                    student=r["Student"].strip(),
                    action=r["Action"].strip(),
                    segments=parse_segments(r["Segments"]),
                )
            )
    logger.info("CSV %s -> %d rows", csv_path, len(rows))
    return rows


# ---------------------------------------------------------------------------
# 单个 zip 内的 frame 名定位
# ---------------------------------------------------------------------------
_FRAMES_IN_ZIP_CACHE: Dict[str, Dict[str, str]] = {}


def build_zip_index(zip_path: str) -> Dict[str, str]:
    """返回 ``{action: {frame_no -> zip_member_name}}`` 索引。
    例：``{'A05': {'121': 'S01/A05/wifi-csi/frame121.mat', ...}}``
    """
    if zip_path in _FRAMES_IN_ZIP_CACHE:
        return _FRAMES_IN_ZIP_CACHE[zip_path]
    idx: Dict[str, Dict[str, str]] = defaultdict(dict)
    with zipfile.ZipFile(zip_path) as z:
        for name in z.namelist():
            # 匹配 .../AXX/wifi-csi/frameNNN.mat
            m = re.search(
                r"/(A\d+)/wifi-csi/frame(\d+)\.mat$",
                name.replace("\\", "/"),
            )
            if not m:
                continue
            action = m.group(1)
            fnum = m.group(2)
            idx[action][fnum] = name
    _FRAMES_IN_ZIP_CACHE[zip_path] = idx
    return idx


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def organize(
    csv_rows: List[VideoSeg],
    src_zip_dir: str,
    out_root: str,
    n_subjects: int | None = None,
    force: bool = False,
) -> None:
    """把 frame 按 (env, student, action) 重写到目标目录。"""
    # 只保留前 N 个 subject（如果指定）
    if n_subjects is not None:
        wanted = {f"S{str(i).zfill(2)}" for i in range(1, n_subjects + 1)}
        csv_rows = [r for r in csv_rows if r.student in wanted]
        logger.info("限制到前 %d 个 subject, CSV 剩 %d 行", n_subjects, len(csv_rows))

    # 按 subject 分组，方便并行打开 zip
    by_student: Dict[str, List[VideoSeg]] = defaultdict(list)
    for r in csv_rows:
        by_student[r.student].append(r)

    n_done = n_skip = n_err = 0
    t0 = time.time()
    for si, student in enumerate(sorted(by_student.keys()), 1):
        zip_path = os.path.join(src_zip_dir, f"{student}.zip")
        if not os.path.exists(zip_path):
            logger.warning("缺 zip: %s (跳过)", zip_path)
            continue
        idx = build_zip_index(zip_path)
        logger.info(
            "[%d/%d] %s: %d 动作组合, zip 索引 %d 类",
            si, len(by_student), student, len(by_student[student]), len(idx),
        )

        with zipfile.ZipFile(zip_path) as z:
            for vs in by_student[student]:
                target_dir = os.path.join(out_root, vs.env, vs.student, vs.action, "wifi-csi")
                os.makedirs(target_dir, exist_ok=True)
                members = idx.get(vs.action, {})
                if not members:
                    logger.warning("  %s: zip 内没有 %s/*/wifi-csi", student, vs.action)
                    n_err += 1
                    continue
                for seg_lo, seg_hi in vs.segments:
                    for fnum in range(seg_lo, seg_hi + 1):
                        fnum_s = str(fnum).zfill(3)
                        member = members.get(fnum_s)
                        if member is None:
                            continue  # 有些 frame 不存在
                        target = os.path.join(target_dir, f"frame{fnum_s}.mat")
                        if os.path.exists(target) and not force:
                            n_skip += 1
                            continue
                        try:
                            data = z.read(member)
                            with open(target, "wb") as fout:
                                fout.write(data)
                            n_done += 1
                        except Exception as e:
                            logger.warning("  写失败 %s: %s", target, e)
                            n_err += 1

        elapsed = time.time() - t0
        logger.info(
            "  [累计] 写=%d 跳=%d 错=%d 用时=%.1fs",
            n_done, n_skip, n_err, elapsed,
        )

    logger.info(
        "全部完成: 写=%d 跳=%d 错=%d 总用时=%.1fs -> %s",
        n_done, n_skip, n_err, time.time() - t0, out_root,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="MMFi 按 Env 重新组织 (CSV-driven)")
    ap.add_argument("--src", default=r"F:\python_workspace\wifi识别\data\raw\MMFi_Dataset\Zipfiles")
    ap.add_argument("--csv", default=r"F:\python_workspace\wifi识别\data\raw\MMFi_Dataset\MMFi_action_segments.csv")
    ap.add_argument("--out", default=r"F:\python_workspace\wifi识别\data\raw\mmfi_unzipped")
    ap.add_argument("--n-subjects", type=int, default=None,
                    help="只解前 N 个 subject（调试用），默认全部 40")
    ap.add_argument("--force", action="store_true", help="重写已存在的 frame 文件")
    args = ap.parse_args()

    if not os.path.isdir(args.src):
        sys.exit(f"src 不存在: {args.src}")
    if not os.path.isfile(args.csv):
        sys.exit(f"csv 不存在: {args.csv}")
    os.makedirs(args.out, exist_ok=True)

    rows = load_csv(args.csv)
    organize(rows, args.src, args.out, n_subjects=args.n_subjects, force=args.force)


if __name__ == "__main__":
    main()
