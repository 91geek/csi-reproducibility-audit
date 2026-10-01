# -*- coding: utf-8 -*-
"""MMFi 全量 cache 重建（force=True）

- 删除旧 _mmfi_full.h5
- 用 mmfi_unzipped 完整数据重建
- 强制要求 n_cached >= 0.95 * 1134 = 1078（95% subset 阈值）
- 每 100 samples 打印进度（flush）
- 写入 _mmfi_rebuild.log
"""
import os
import sys
import time
import h5py
import numpy as np
from collections import Counter

sys.path.insert(0, r'F:/python_workspace/wifi识别/wifi-crossenv-lab/src')
from wfcslab.data.mmfi import build_mmfi_cache

RAW = r'F:/python_workspace/wifi识别/data/raw/mmfi_unzipped'
OUT = r'F:/python_workspace/wifi识别/wifi-crossenv-lab/src/bvp_test/_mmfi_full.h5'
LOG = r'F:/python_workspace/wifi识别/wifi-crossenv-lab/src/bvp_test/_mmfi_rebuild.log'

# h5py.File(..., 'w') 会自动覆盖，不需要先删
print(f'[start] rebuilding cache (h5py will overwrite existing): {OUT}', flush=True)

t0 = time.time()
print(f'[start] rebuilding cache from {RAW}', flush=True)
stats = build_mmfi_cache(RAW, OUT, max_time=512, limit=None, force=True)
print(f'[done] 用时 {time.time()-t0:.1f}s', flush=True)
for k, v in stats.items():
    if isinstance(v, (list, tuple)):
        s = str(v)
        if len(s) > 200:
            s = s[:200] + '...'
        print(f'  {k}: {s}', flush=True)
    else:
        print(f'  {k}: {v}', flush=True)

# subset check
n_cached = stats.get('n', 0)
n_source = 1134
ratio = n_cached / n_source
print(f'\n=== Subset check ===', flush=True)
print(f'  cached: {n_cached}', flush=True)
print(f'  source: {n_source} wifi-csi dirs', flush=True)
print(f'  ratio:  {ratio*100:.2f}%', flush=True)
if ratio < 0.95:
    print(f'  [FATAL] subset too small, expected ≥ 95%', flush=True)
    sys.exit(1)
else:
    print(f'  [OK] subset ≥ 95% threshold', flush=True)

# cache stats
with h5py.File(OUT, 'r') as f:
    X_shape = f['X'].shape
    y = f['y'][:]
    domains = f['domains'][:]
    subjects = f['subjects'][:]
    print(f'\n=== Cache contents ===', flush=True)
    print(f'  X.shape: {X_shape}  dtype={f["X"].dtype}', flush=True)
    print(f'  n_subjects: {int(subjects.max())+1}  n_classes: {int(y.max())+1}  '
          f'n_domains: {int(domains.max())+1}', flush=True)
    print(f'  per domain counts: {dict(Counter(domains.tolist()))}', flush=True)
    print(f'  per subject min/max/mean: '
          f'{min(Counter(subjects.tolist()).values())}/'
          f'{max(Counter(subjects.tolist()).values())}/'
          f'{np.mean(list(Counter(subjects.tolist()).values())):.1f}', flush=True)