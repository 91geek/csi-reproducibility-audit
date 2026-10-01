"""
MMFi 数据补全（4 worker 并发 + 单线程写 + 重试）
================================================

设计
----
- 4 个 worker 并发处理不同 subject（不同 zip 内 frame 无冲突）
- 每个 worker 单线程写入（避免 AV 锁竞争）
- 失败指数退避重试 5 次
- 已存在文件跳过（断点续传）

预期
----
- 4x 并发 + ~30 帧/秒（单线程实测 8 帧/秒 × 4 worker）
- E04 剩余 ~46K frame → 约 25 分钟
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
logger = logging.getLogger("mmfi_fill_par")


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
    for attempt in range(max_retries):
        try:
            with open(target, "wb") as f:
                f.write(data)
            return True
        except PermissionError as e:
            if attempt < max_retries - 1:
                wait = 0.2 * (2 ** attempt)
                time.sleep(wait)
            else:
                return False
        except Exception:
            return False
    return False


def worker(args):
    """单 worker：处理若干 student，每个 student 完整遍历所有 segment，单线程写入。"""
    (worker_id, students, csv_path, src_zip_dir, out_root) = args
    log = logging.getLogger(f"mmfi_w{worker_id}")
    log.info("[w%d] 启动, %d subject: %s..%s",
             worker_id, len(students), students[0], students[-1])

    csv_rows = [r for r in load_csv(csv_path) if r.student in set(students)]
    by_student: Dict[str, List[VideoSeg]] = defaultdict(list)
    for r in csv_rows:
        by_student[r.student].append(r)

    n_done = n_skip = n_fail = 0
    t0 = time.time()

    for si, student in enumerate(students, 1):
        zip_path = os.path.join(src_zip_dir, f"{student}.zip")
        if not os.path.exists(zip_path):
            log.warning("[w%d] 缺 zip: %s", worker_id, zip_path)
            continue

        # 索引 zip
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

        with zipfile.ZipFile(zip_path) as z:
            for vs in student_segs:
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
                            if write_with_retry(target, data):
                                n_done += 1
                            else:
                                n_fail += 1
                        except Exception:
                            n_fail += 1

        elapsed = time.time() - t0
        rate = n_done / max(elapsed, 1)
        log.info("[w%d] %s done (%d/%d). 写=%d 跳=%d 错=%d 用时=%.0fs 速率=%.1f帧/s",
                 worker_id, student, si, len(students),
                 n_done, n_skip, n_fail, elapsed, rate)

    return (worker_id, n_done, n_skip, n_fail, time.time() - t0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=r"F:\python_workspace\wifi识别\data\raw\MMFi_Dataset\Zipfiles")
    ap.add_argument("--csv", default=r"F:\python_workspace\wifi识别\data\raw\MMFi_Dataset\MMFi_action_segments.csv")
    ap.add_argument("--out", default=r"F:\python_workspace\wifi识别\data\raw\mmfi_unzipped")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit-envs", nargs="*", default=None)
    ap.add_argument("--start-subj", type=int, default=1)
    ap.add_argument("--end-subj", type=int, default=40)
    args = ap.parse_args()

    # 切 subject 给 workers
    all_students = [f"S{str(i).zfill(2)}" for i in range(args.start_subj, args.end_subj + 1)]
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
    total_fail = sum(r[3] for r in results)
    logger.info("全部完成: 写=%d 跳=%d 错=%d 总用时=%.1fs -> %s",
                total_done, total_skip, total_fail, time.time() - t0, args.out)


if __name__ == "__main__":
    mp.freeze_support()
    main()