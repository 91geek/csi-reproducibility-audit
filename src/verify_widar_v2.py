"""Verify Widar3.0 v2 cache after rebuild.

基于 raw_dir 实测 (DFRF Phase 1, 2026-09-30)：
- 13 个 zip 共 158,500 样本，ges=1..5 (5 类) 跨 zip 语义一致
- 10 个 envs (loc × orient)，范围因 zip 而异 (1..6, 1..9, 1..10)，合并后 = 1..10
  20181112.zip 独有 env 1..10；其他 20181130_user* + 20181204 = 1..9；其他 zip = 1..6
- 用户来自 {0..16} (17 个 user_id)

Checks:
  1. h5 file exists and is parseable
  2. labels in {1..5} (5 类同质集)
  3. domains in {1..10}
  4. n_classes == 5
  5. per-class balance is reasonable (variance ratio < 2)
  6. no NaN/Inf in X

Prints a one-line PASS/FAIL plus a summary dict.
"""

from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent / "src"))

import h5py

V2_H5 = r"F:/python_workspace/wifi识别/data/processed/widar3.0/csi_50k_v2.h5"
EXPECTED_LABELS = list(range(1, 6))    # ges=1..5 (5 classes)
EXPECTED_DOMAINS = list(range(1, 11))  # env=1..10 (合并 13 zip 的 env 范围)


def main():
    p = Path(V2_H5)
    if not p.exists():
        print(f"FAIL: {V2_H5} not found")
        return 1

    f = h5py.File(V2_H5, "r")
    X_ds = f["X"]
    y = np.asarray(f["y"][...], dtype=np.int64)
    domains = np.asarray(f["domains"][...], dtype=np.int64)
    subjects = np.asarray(f["subjects"][...], dtype=np.int64)
    n = X_ds.shape[0]

    print(f"=== Widar v2 cache verify ===")
    print(f"path:    {V2_H5}")
    print(f"size:    {p.stat().st_size / 1e9:.2f} GB")
    print(f"X shape: {X_ds.shape}")
    print(f"n:       {n}")

    # 1. labels
    label_set = sorted(set(y.tolist()))
    print(f"y unique: {label_set} (expect {EXPECTED_LABELS})")
    label_ok = label_set == EXPECTED_LABELS

    # 2. domains
    dom_set = sorted(set(domains.tolist()))
    print(f"domains unique: {dom_set} (expect {EXPECTED_DOMAINS})")
    dom_ok = dom_set == EXPECTED_DOMAINS

    # 3. subjects
    sub_set = sorted(set(subjects.tolist()))
    n_users = len(sub_set)
    print(f"subjects unique: {sub_set} (n={n_users})")
    sub_ok = n_users >= 8   # 至少 8 个用户

    # 4. n_classes
    n_cls = len(label_set)
    print(f"n_classes: {n_cls} (expect 5)")
    n_cls_ok = n_cls == 5

    # 5. balance
    per_class = {int(c): int((y == c).sum()) for c in label_set}
    counts = list(per_class.values())
    ratio = max(counts) / max(min(counts), 1)
    print(f"per_class: {per_class}")
    print(f"class balance ratio: {ratio:.3f} (expect < 2.0 for balanced)")
    balance_ok = ratio < 2.0

    # 6. attrs
    n_classes_attr = int(f.attrs.get("n_classes", -1))
    n_domains_attr = int(f.attrs.get("n_domains", -1))
    print(f"attrs n_classes={n_classes_attr}, n_domains={n_domains_attr}")

    # 7. NaN check on sample
    sample_n = min(200, n)
    rng = np.random.default_rng(0)
    idx = rng.choice(n, sample_n, replace=False)
    nan_cnt = inf_cnt = 0
    for i in idx:
        x = np.asarray(X_ds[int(i)])
        nan_cnt += int(np.isnan(x).sum())
        inf_cnt += int(np.isinf(x).sum())
    print(f"NaN frac (sample {sample_n}): {nan_cnt / (sample_n * X_ds.shape[1] * X_ds.shape[2] * X_ds.shape[3]):.6f}")
    print(f"Inf frac (sample {sample_n}): {inf_cnt / (sample_n * X_ds.shape[1] * X_ds.shape[2] * X_ds.shape[3]):.6f}")
    nan_ok = nan_cnt == 0 and inf_cnt == 0

    f.close()

    all_ok = label_ok and dom_ok and sub_ok and n_cls_ok and balance_ok and nan_ok
    print()
    print(f"=== {'PASS' if all_ok else 'FAIL'} ===")
    print(f"  label_ok={label_ok}, dom_ok={dom_ok}, sub_ok={sub_ok}, "
          f"n_cls_ok={n_cls_ok}, balance_ok={balance_ok}, nan_ok={nan_ok}")

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())