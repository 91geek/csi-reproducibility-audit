"""Build full Widar3.0 cache using chunked h5 writes (avoid 84 GB OOM).

关键问题：163,650 × 256 × 9 × 30 × complex64 = 84 GB → np.stack 会 OOM
解决：用 h5py.create_dataset(chunks=True) 边解析边写入，每次只 RAM 1 batch
"""
import os
import sys
import time
import json
import zipfile
import re
import tempfile
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import h5py
from tqdm.auto import tqdm

HERE = os.path.dirname(os.path.abspath(__file__))
LAB_ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(LAB_ROOT, "src"))

# 用 csiread
try:
    import csiread
except ImportError:
    print("[FATAL] csiread not installed: pip install csiread")
    sys.exit(1)

# 复用 widar.py 的核心解析函数
from wfcslab.data.widar import (
    parse_filename, _read_dat_from_zip, _canonicalize, _fix_time,
    DEFAULT_MAX_TIME, N_RX, N_TX, N_SC,
)

CSI_DIR = r"F:/python_workspace/wifi识别/data/raw/widar3.0/CSI"
OUT_DIR = r"F:/python_workspace/wifi识别/data/processed/widar3.0"
os.makedirs(OUT_DIR, exist_ok=True)
OUT_H5 = os.path.join(OUT_DIR, "csi_all.h5")

# 主进程模块级 worker 函数（必须能被 pickle）
_fn_re = re.compile(
    r"user(?P<user>[A-Za-z0-9]+)-(?P<env>\d+)-(?P<ges>\d+)-(?P<ori>\d+)-"
    r"(?P<rep>\d+)-r(?P<seg>\d+)\.dat$", re.IGNORECASE,
)


def parse_member(zp_member):
    """Worker: 解析一个 zip member，返回 (meta_dict, csi_array_or_None, error)."""
    zp, member = zp_member
    try:
        meta = parse_filename(member)
        if meta is None:
            return None, None, "bad filename"
        csi = _read_dat_from_zip(zp, member)
        if csi.ndim != 3:
            csi = _canonicalize(csi)
        return meta, csi, None
    except Exception as e:
        return None, None, str(e)[:200]


def main():
    print(f"[OK] csiread {csiread.__file__}")
    print(f"=== Building csi_all.h5 (chunked, dedup) ===")
    print(f"  raw_dir: {CSI_DIR}")
    print(f"  out_h5:  {OUT_H5}")
    print(f"  max_time: 256")

    if os.path.exists(OUT_H5):
        print(f"  removing existing {OUT_H5}")
        os.remove(OUT_H5)

    # 1) 列出所有 zip
    zips = sorted([os.path.join(r, f) for r, _, fs in os.walk(CSI_DIR)
                    for f in fs if f.lower().endswith('.zip')])
    print(f"  found {len(zips)} zips")

    # 2) 列出所有 (zp, member) 并 dedup
    print("  scanning members + dedup...")
    all_members = []
    for zp in zips:
        with zipfile.ZipFile(zp) as z:
            for name in z.namelist():
                if not name.lower().endswith('.dat'):
                    continue
                meta = parse_filename(name)
                if meta is None:
                    continue
                all_members.append((zp, name))
    print(f"  raw members: {len(all_members)}")
    seen = set()
    keep = []
    for zp, member in all_members:
        meta = parse_filename(member)
        key = (meta["user"], meta["env"], meta["ges"],
                meta["ori"], meta["rep"], meta["seg"])
        if key in seen:
            continue
        seen.add(key)
        keep.append((zp, member))
    print(f"  unique members: {len(keep)}")
    del all_members

    # 3) Shuffle
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(keep))
    keep = [keep[i] for i in perm]

    # 4) 创建 h5 文件 + 预创建 chunked dataset
    n_total = len(keep)
    sample_shape = (256, 9, 30)  # (T, A, S) - actual shape determined by first sample
    chunk_size = 1024  # h5 chunk for partial reads

    with h5py.File(OUT_H5, 'w') as f:
        # 用可扩展 dataset，从 0 开始，maxshape=(None, ...)
        # shape = (n_total, T, A, S) complex64
        ds_X = f.create_dataset(
            "X",
            shape=(n_total, 256, 9, 30),
            maxshape=(n_total, 256, 9, 30),  # 固定大小（不会超）
            dtype='complex64',
            chunks=(chunk_size, 256, 9, 30),
            compression='lzf',
            shuffle=True,
        )
        # labels / domains / subjects 创建后再 resize
        ds_y = f.create_dataset("y", shape=(n_total,), maxshape=(n_total,),
                                 dtype='int64', chunks=(chunk_size,))
        ds_dom = f.create_dataset("domains", shape=(n_total,), maxshape=(n_total,),
                                   dtype='int64', chunks=(chunk_size,))
        ds_sub = f.create_dataset("subjects", shape=(n_total,), maxshape=(n_total,),
                                   dtype='int64', chunks=(chunk_size,))

        # 5) ProcessPoolExecutor 解析
        n_workers = 8
        BATCH = 1024  # 每 batch 解析后写入 h5
        print(f"  starting {n_workers} workers, batch={BATCH}")

        # user2id dict（在外层主进程）
        user2id = {}
        written = 0
        t0 = time.time()
        # 用 imap_unordered 按完成顺序写入（保证 h5 index = 完成顺序）
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            # imap 不接受 chunksize in imap_unordered
            for i, (meta, csi, err) in enumerate(tqdm(
                ex.map(parse_member, keep, chunksize=8),
                total=n_total, desc="parse", dynamic_ncols=True
            )):
                if csi is None:
                    # 解析失败，丢弃
                    continue
                # 截断/补零到 256
                csi_fixed = _fix_time(csi, 256)
                # canonical shape 应该是 (T=256, A=9, S=30)
                assert csi_fixed.shape == (256, 9, 30), f"bad shape {csi_fixed.shape}"
                # 写 h5（单条）
                ds_X[written] = csi_fixed
                ds_y[written] = int(meta["ges"])
                ds_dom[written] = int(meta["env"])
                u = meta["user"]
                if u not in user2id:
                    user2id[u] = len(user2id)
                ds_sub[written] = user2id[u]
                written += 1
                if written % 5000 == 0:
                    dt = time.time() - t0
                    rate = written / dt
                    eta = (n_total - written) / rate / 60
                    print(f"  [{written}/{n_total}] {rate:.0f} samples/s, ETA {eta:.1f} min")

        dt_total = time.time() - t0
        print(f"  parsed {written}/{n_total} in {dt_total:.0f}s ({dt_total/60:.1f} min)")

        # 6) attrs
        f.attrs["max_time"] = 256
        f.attrs["n_subcarriers"] = 30
        f.attrs["n_antennas"] = 9
        f.attrs["n_classes"] = int(ds_y[:].max()) + 1
        f.attrs["n_domains"] = int(ds_dom[:].max()) + 1
        f.attrs["n_subjects"] = len(user2id)
        f.attrs["user_map"] = ",".join(
            f"{u}:{i}" for u, i in sorted(user2id.items(), key=lambda kv: kv[1])
        )
        f.attrs["n_parsed"] = written
        f.attrs["n_total_unique"] = n_total

    print(f"\n=== Build done in {(time.time()-t0)/60:.1f} min ===")
    print(f"  output: {OUT_H5} ({os.path.getsize(OUT_H5)/1e9:.2f} GB)")
    print(f"  parsed: {written} samples, {len(user2id)} users")


if __name__ == "__main__":
    main()
