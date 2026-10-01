"""Build full Widar3.0 cache (163,650 samples) using all CPU cores.

策略：
- limit=None → 全部 271,050 dat（去重前）
- n_workers=12 (16 核 - 4 给系统/MMFi解压)
- 写到 csi_all.h5 (预估 ~90GB complex64)
- force=True 强制重建
"""
import os
import sys
import time
import json

# 路径
HERE = os.path.dirname(os.path.abspath(__file__))
LAB_ROOT = os.path.dirname(HERE)  # wifi-crossenv-lab
sys.path.insert(0, os.path.join(LAB_ROOT, "src"))


def main():
    from wfcslab.data.widar import build_widar_cache

    OUT_DIR = r"F:/python_workspace/wifi识别/data/processed/widar3.0"
    os.makedirs(OUT_DIR, exist_ok=True)
    OUT_H5 = os.path.join(OUT_DIR, "csi_all.h5")

    # 检查 csiread 是否可用
    try:
        import csiread
        print(f"[OK] csiread available: {csiread.__file__}")
    except ImportError:
        print("[FATAL] csiread not installed in current Python environment")
        print("  install: pip install csiread")
        sys.exit(1)

    # 不要 limit，强制 rebuild
    if os.path.exists(OUT_H5):
        print(f"[WARN] {OUT_H5} already exists, will be overwritten (force=True)")
        sz_gb = os.path.getsize(OUT_H5) / 1e9
        print(f"  current size: {sz_gb:.2f} GB")

    print(f"\n=== Building csi_all.h5 ===")
    print(f"  raw_dir: F:/python_workspace/wifi识别/data/raw/widar3.0/CSI")
    print(f"  out_h5:  {OUT_H5}")
    print(f"  limit:   None (full 271,050 dat, ~163650 unique)")
    print(f"  workers: 8 (out of 16 cores, leaving 8 for MMFi/7z background)")
    print(f"  max_time: 256")
    print(f"  downsample: 2")
    print()

    t0 = time.time()
    st = build_widar_cache(
        raw_dir=r"F:/python_workspace/wifi识别/data/raw/widar3.0/CSI",
        out_h5=OUT_H5,
        max_time=256,
        limit=None,
        downsample=2,
        shuffle=True,
        seed=0,
        force=True,
        n_workers=8,
        dedup=True,  # 关键：去掉 user1/2/3 跨日期的重复样本
    )
    dt = time.time() - t0
    print(f"\n=== Build done in {dt:.0f}s ({dt/60:.1f} min) ===")
    print(json.dumps(st, indent=2, ensure_ascii=False))
    print(f"  output h5: {OUT_H5} ({os.path.getsize(OUT_H5)/1e9:.2f} GB)")


if __name__ == "__main__":
    main()
