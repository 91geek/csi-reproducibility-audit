"""
MMFi 数据补全（单线程 + 重试版）
==================================

设计
----
- 单线程遍历所有 (env, subj, action, frame) 组合
- 文件已存在则跳过（断点续传）
- Permission denied 时指数退避重试，最多 5 次
- 单线程规避 AV 锁竞争（AV 同时只扫描少量文件，单线程写入不会触发锁）

预期
----
- 处理剩余 E04 (S31-S40) + E03 中 Permission denied 漏掉的 frame
- 单线程稳速 30-60 帧/秒 → 数十万个 frame 约 1-2 小时
"""
from __future__ import annotations

import argparse
import csv
import logging
import os
import re
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
logger = logging.getLogger("mmfi_fill")


@dataclass
class VideoSeg:
    env: str
    student: str
    action: str
    segments: List[Tuple[int, int]]


def parse_segments(s: str) -> List[Tuple[int, int]]:
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
            rows.append(VideoSeg(
                env=r["Environment"].strip(),
                student=r["Student"].strip(),
                action=r["Action"].strip(),
                segments=parse_segments(r["Segments"]),
            ))
    return rows


def write_with_retry(target: str, data: bytes, max_retries: int = 5) -> bool:
    """写一个文件，Permission denied 时指数退避重试。"""
    for attempt in range(max_retries):
        try:
            with open(target, "wb") as f:
                f.write(data)
            return True
        except PermissionError as e:
            if attempt < max_retries - 1:
                # 0.2, 0.5, 1.0, 2.0, 4.0s
                wait = 0.2 * (2 ** attempt)
                time.sleep(wait)
            else:
                logger.warning("永久失败 %s: %s", target, e)
                return False
        except Exception as e:
            logger.warning("非权限错误 %s: %s", target, e)
            return False
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=r"F:\python_workspace\wifi识别\data\raw\MMFi_Dataset\Zipfiles")
    ap.add_argument("--csv", default=r"F:\python_workspace\wifi识别\data\raw\MMFi_Dataset\MMFi_action_segments.csv")
    ap.add_argument("--out", default=r"F:\python_workspace\wifi识别\data\raw\mmfi_unzipped")
    ap.add_argument("--limit-envs", nargs="*", default=None,
                    help="只处理这些 env（如 E04），默认全部")
    ap.add_argument("--start-subj", type=int, default=1)
    ap.add_argument("--end-subj", type=int, default=40)
    args = ap.parse_args()

    csv_rows = load_csv(args.csv)
    if args.limit_envs:
        csv_rows = [r for r in csv_rows if r.env in args.limit_envs]

    # 按 student 分组
    by_student: Dict[str, List[VideoSeg]] = defaultdict(list)
    for r in csv_rows:
        by_student[r.student].append(r)

    students = [f"S{str(i).zfill(2)}" for i in range(args.start_subj, args.end_subj + 1)]

    # 进度统计
    total_written = total_skip = total_fail = 0
    t_start = time.time()

    for si, student in enumerate(students, 1):
        zip_path = os.path.join(args.src, f"{student}.zip")
        if not os.path.exists(zip_path):
            logger.warning("[%d/%d] 缺 zip: %s", si, len(students), zip_path)
            continue

        # 索引 zip 内 frame
        idx: Dict[str, Dict[str, str]] = defaultdict(dict)
        with zipfile.ZipFile(zip_path) as z:
            for name in z.namelist():
                m = re.search(
                    r"/(A\d+)/wifi-csi/frame(\d+)\.mat$",
                    name.replace("\\", "/"),
                )
                if m:
                    idx[m.group(1)][m.group(2)] = name

        student_segs = by_student.get(student, [])
        if not student_segs:
            continue

        logger.info("[%d/%d] %s 索引 %d 类, %d video segment",
                    si, len(students), student, len(idx), len(student_segs))

        with zipfile.ZipFile(zip_path) as z:
            for vs in student_segs:
                target_dir = os.path.join(args.out, vs.env, vs.student, vs.action, "wifi-csi")
                os.makedirs(target_dir, exist_ok=True)
                members = idx.get(vs.action, {})
                for seg_lo, seg_hi in vs.segments:
                    for fnum in range(seg_lo, seg_hi + 1):
                        fnum_s = str(fnum).zfill(3)
                        member = members.get(fnum_s)
                        if member is None:
                            continue
                        target = os.path.join(target_dir, f"frame{fnum_s}.mat")
                        if os.path.exists(target):
                            total_skip += 1
                            continue
                        try:
                            data = z.read(member)
                            if write_with_retry(target, data):
                                total_written += 1
                            else:
                                total_fail += 1
                        except Exception as e:
                            logger.warning("读 zip 失败 %s: %s", member, e)
                            total_fail += 1

        elapsed = time.time() - t_start
        rate = total_written / max(elapsed, 1)
        logger.info("[%d/%d] %s done. 写=%d 跳=%d 错=%d 用时=%.0fs 速率=%.1f帧/s",
                    si, len(students), student,
                    total_written, total_skip, total_fail, elapsed, rate)

    logger.info("完成: 写=%d 跳=%d 错=%d 总用时=%.1fs",
                total_written, total_skip, total_fail, time.time() - t_start)


if __name__ == "__main__":
    main()