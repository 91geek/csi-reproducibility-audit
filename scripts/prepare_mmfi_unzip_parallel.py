"""
MMFi 数据按 Env 重新组织 —— 多进程并行版（修复版）
==================================================

设计
----
- main 进程解析 CSV，按 student 分给 N 个 worker
- 每个 worker 进程独立打开自己的若干 zip（zip 是只读的，可以多进程并发打开）
- 断点续传：已存在 target 文件跳过
- 多进程并发加速 ~3-3.5x（NTFS 系统调用瓶颈）
"""

from __future__ import annotations

import argparse
import csv
import logging
import multiprocessing as mp
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
logger = logging.getLogger("mmfi_par")


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


def worker(args):
    """单 worker：处理自己分到的若干 student。"""
    (worker_id, students, csv_path, src_zip_dir, out_root) = args
    log = logging.getLogger(f"mmfi_w{worker_id}")
    log.info("[w%d] 启动, %d subject: %s..%s",
             worker_id, len(students), students[0], students[-1])

    # worker 自己读 CSV（避免主进程传大 list）
    csv_rows = [r for r in load_csv(csv_path) if r.student in set(students)]
    by_student: Dict[str, List[VideoSeg]] = defaultdict(list)
    for r in csv_rows:
        by_student[r.student].append(r)

    n_done = n_skip = n_err = 0
    t0 = time.time()

    for si, student in enumerate(students, 1):
        zip_path = os.path.join(src_zip_dir, f"{student}.zip")
        if not os.path.exists(zip_path):
            log.warning("[w%d] 缺 zip: %s", worker_id, zip_path)
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

            log.info("[w%d] %s (%d/%d) %d 动作, zip 索引 %d 类",
                     worker_id, student, si, len(students),
                     len(by_student.get(student, [])), len(idx))

        # 写文件（独立 zip 句柄用于实际 read）
        with zipfile.ZipFile(zip_path) as z:
            for vs in by_student.get(student, []):
                target_dir = os.path.join(out_root, vs.env, vs.student, vs.action, "wifi-csi")
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
                            n_skip += 1
                            continue
                        try:
                            data = z.read(member)
                            with open(target, "wb") as fout:
                                fout.write(data)
                            n_done += 1
                        except Exception as e:
                            log.warning("[w%d] 写失败 %s: %s", worker_id, target, e)
                            n_err += 1

        if si % 2 == 0 or si == len(students):
            elapsed = time.time() - t0
            log.info("[w%d] %s done (%d/%d). 写=%d 跳=%d 错=%d 用时=%.0fs",
                     worker_id, student, si, len(students),
                     n_done, n_skip, n_err, elapsed)

    return (worker_id, n_done, n_skip, n_err, time.time() - t0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=r"F:\python_workspace\wifi识别\data\raw\MMFi_Dataset\Zipfiles")
    ap.add_argument("--csv", default=r"F:\python_workspace\wifi识别\data\raw\MMFi_Dataset\MMFi_action_segments.csv")
    ap.add_argument("--out", default=r"F:\python_workspace\wifi识别\data\raw\mmfi_unzipped")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--start", type=int, default=1)
    ap.add_argument("--end", type=int, default=40)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # 切 subject range 给 workers
    all_students = [f"S{str(i).zfill(2)}" for i in range(args.start, args.end + 1)]
    chunks = [[] for _ in range(args.workers)]
    for i, s in enumerate(all_students):
        chunks[i % args.workers].append(s)

    logger.info("启动 %d workers, %d subject 分配: %s",
                args.workers, len(all_students), [len(c) for c in chunks])

    t0 = time.time()
    with mp.Pool(args.workers) as pool:
        results = pool.map(worker, [
            (i + 1, chunks[i], args.csv, args.src, args.out)
            for i in range(args.workers)
        ])

    total_done = sum(r[1] for r in results)
    total_skip = sum(r[2] for r in results)
    total_err = sum(r[3] for r in results)
    logger.info("全部完成: 写=%d 跳=%d 错=%d 总用时=%.1fs -> %s",
                total_done, total_skip, total_err, time.time() - t0, args.out)


if __name__ == "__main__":
    # Windows 上需要这个
    mp.freeze_support()
    main()
