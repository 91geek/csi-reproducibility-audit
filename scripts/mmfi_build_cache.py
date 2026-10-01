# -*- coding: utf-8 -*-
"""MMFi cache 构建（小批量冒烟 + 全量）

每帧 CSIamp=(3,114,10) -> canonicalize 后 (A=30, S=114)，无时间轴。
Widar3.0 的 BVP 算子对 MMFi 不适用（FFT 需要 4+ 帧）。
MMFi 实验设计需要调整，见 MMFI_EXPERIMENT_PLAN.md。
"""
import os, sys, time
import numpy as np
sys.path.insert(0, 'src')
from wfcslab.data.mmfi import build_mmfi_cache, make_mmfi_dataset

RAW = r"F:/python_workspace/wifi识别/data/raw/mmfi_unzipped"
OUT = r"F:/python_workspace/wifi识别/wifi-crossenv-lab/src/bvp_test/_mmfi_smoke.h5"
OUT_FULL = r"F:/python_workspace/wifi识别/wifi-crossenv-lab/src/bvp_test/_mmfi_full.h5"

LIMIT = int(os.environ.get("MMFI_LIMIT", "300"))
print(f"=" * 60)
print(f"MMFi cache 构建：limit={LIMIT}")
print(f"=" * 60)
t0 = time.time()
stats = build_mmfi_cache(RAW, OUT, max_time=512, limit=LIMIT, force=True)
print(f"\n[build_mmfi_cache] 用时 {time.time()-t0:.1f}s")
for k, v in stats.items():
    if isinstance(v, (list, tuple)):
        print(f"  {k}: {v[:3]}{'...' if len(v) > 3 else ''}")
    else:
        print(f"  {k}: {v}")

print(f"\n=== make_mmfi_dataset({OUT}) ===")
t1 = time.time()
ds = make_mmfi_dataset(OUT)
print(f"[make_mmfi_dataset] 用时 {time.time()-t1:.2f}s")
print(f"  X.shape: {ds.X.shape}  dtype: {ds.X.dtype}")
print(f"  y: min={ds.y.min()}, max={ds.y.max()}, n_class={len(set(ds.y.tolist()))}")
print(f"  domains: {sorted(set(ds.domains.tolist()))}  n_domain={len(set(ds.domains.tolist()))}")
print(f"  subjects: {sorted(set(ds.subjects.tolist()))[:10]}...  n_subject={len(set(ds.subjects.tolist()))}")
print(f"  domain_names: {ds.domain_names}")
print(f"  meta.max_time: {ds.meta.get('max_time')}")
print(f"  meta.n_antennas: {ds.meta.get('n_antennas')}")
print(f"  meta.n_subcarriers: {ds.meta.get('n_subcarriers')}")
print(f"  meta.first_raw_shape: {ds.meta.get('first_raw_shape')}")
print(f"  meta.raw_shape_variants: {ds.meta.get('raw_shape_variants')}")
