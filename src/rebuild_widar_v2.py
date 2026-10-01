"""Rebuild Widar3.0 cache with proper labels (DFRF fix).

Original cache (csi_50k.h5) is BROKEN: ges integer is per-zip-index, not global class.
This script rebuilds using canonical 5-gesture set per Widar3.0 raw_dir **实测**
（DFRF Phase 1, 2026-09-30）—— raw_dir 实际不存在 9 类同质集，最大同质集为 5 类。

Output: data/processed/widar3.0/csi_50k_v2.h5
"""

from __future__ import annotations
import sys, os, time, shutil
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from wfcslab.data.widar import (
    build_widar_canonical_5_cache,
    CANONICAL_5_ZIPS,
)

RAW_DIR = r"F:/python_workspace/wifi识别/data/raw/widar3.0/CSI"
OLD_H5 = r"F:/python_workspace/wifi识别/data/processed/widar3.0/csi_50k.h5"
NEW_H5 = r"F:/python_workspace/wifi识别/data/processed/widar3.0/csi_50k_v2.h5"
BROKEN_H5 = r"F:/python_workspace/wifi识别/data/processed/widar3.0/_broken_csi_50k.h5"


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    log("=== Widar v2 rebuild driver (canonical 5-gesture set) ===")
    log(f"Canonical zips ({len(CANONICAL_5_ZIPS)}): {sorted(CANONICAL_5_ZIPS)}")
    log(f"Raw dir: {RAW_DIR}")
    log(f"Old h5 (broken): {OLD_H5}")
    log(f"New h5: {NEW_H5}")
    log(f"Broken archive: {BROKEN_H5}")

    if os.path.exists(OLD_H5) and not os.path.exists(BROKEN_H5):
        log(f"Step 1: renaming {OLD_H5} -> {BROKEN_H5}")
        shutil.move(OLD_H5, BROKEN_H5)
    elif os.path.exists(OLD_H5):
        log(f"WARNING: {OLD_H5} still exists but {BROKEN_H5} also exists; abort")
        return 1
    elif os.path.exists(BROKEN_H5):
        log(f"Step 1: {OLD_H5} already moved to {BROKEN_H5}")
    else:
        log("Step 1: no existing cache to rename")

    if os.path.exists(NEW_H5):
        log(f"Step 2: removing existing {NEW_H5} for fresh build")
        os.remove(NEW_H5)

    log("Step 3: building canonical 5-gesture cache (50k subset from 13 zip × 5 ges)...")
    stats = build_widar_canonical_5_cache(
        raw_dir=RAW_DIR,
        out_h5=NEW_H5,
        max_time=256,   # 匹配 broken cache 的 T=256，确保 fair A/B compare
        nrx=3, ntx=3,
        limit=50000,
        downsample=1,
        shuffle=True,
        seed=0,
        force=True,
        n_workers=1,
    )

    log("=== build_widar_canonical_5_cache done ===")
    for k, v in stats.items():
        log(f"  {k}: {v}")

    return 0


if __name__ == "__main__":
    sys.exit(main())