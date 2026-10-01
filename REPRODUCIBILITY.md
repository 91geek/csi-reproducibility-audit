# WiFi CSI Cross-Domain Gesture Recognition: A Reproducibility Audit

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)

This repository contains the code, data cache construction scripts, statistical pipeline, and reproducibility scripts for the paper:

> **WiFi CSI Cross-Domain Gesture Recognition: A Reproducibility Audit Revealing Dataset-Conditioned Optimality of the Multi-Rx Representation**
>
> Anonymous Authors. Under review at IEEE Transactions on Mobile Computing (TMC), 2026.

## TL;DR

We re-evaluate ten CSI cross-domain improvements under a paired leave-one-domain-out × multi-seed protocol on **Widar3.0** and **MMFi**, with closed-form MDE-aware power pre-registration. The result: **seven of eleven comparisons fail the audit** (p > 0.28, |d| < 0.3, |Δ| < MDE). The single surviving representation improvement — Multi-Rx BVP expansion — yields **+8.14 pp (Cohen's d = 1.71) on Widar3.0** but **flips to multi-scale on MMFi** (+1.20 pp, p = 0.016). A 50k-subset reanalysis further reverses the original 12k backbone-capacity finding. The boundary is **dataset-conditioned optimality**: no single BVP representation is universally optimal; the choice must match the dataset's geometric separability.

## Quick Start (5 GPU-hours total)

```bash
# 1. Clone + install
git clone https://github.com/[redacted]/wifi-csi-reproducibility-audit.git
cd wifi-csi-reproducibility-audit
pip install -r requirements.txt

# 2. Download the public datasets (Widar3.0 + MMFi)
bash data/download_widar_mmfi.sh

# 3. Build the BVP caches (~15 min on a single CPU)
python src/cache/build_widar_cache.py --out data/widar_cache.npy
python src/cache/build_mmfi_cache_v1.py --out data/mmfi_cache_v1.npy
python src/cache/build_mmfi_cache_v2.py --out data/mmfi_cache_v2.npy

# 4. Run all 11 paired comparisons (~4 GPU-hours on RTX 3090)
bash reproduce.sh all

# 5. Generate all paper figures + tables
python src/figs/make_paper_figs.py --input results/ --output figs/
```

## Repository Layout

```
.
├── README.md                       # this file
├── LICENSE                         # MIT
├── requirements.txt
├── reproduce.sh                    # one-click reproduce all figures
│
├── data/
│   ├── README.md                   # download instructions for Widar3.0 + MMFi
│   ├── download_widar_mmfi.sh      # automated download (DOI-based)
│   └── widar_cache.npy             # (generated) ~50 GB raw BVP cache
│
├── src/
│   ├── models/
│   │   └── backbones.py            # LeNetCSI, LeNetCSI_Attn, ResNetSmall, CBAMResNet18
│   │
│   ├── cache/
│   │   ├── build_widar_cache.py    # Widar3.0 -> (N, 9, 30, 1, 256) BVP cache
│   │   ├── build_mmfi_cache_v1.py  # MMFi v1 path-labeled cache (1080 samples)
│   │   └── build_mmfi_cache_v2.py  # MMFi v2 segment-level cache (16447 samples)
│   │
│   ├── stats/
│   │   ├── paired_runner.py        # paired multi-seed evaluation harness
│   │   ├── mde_calculator.py       # closed-form MDE (Theorem 2)
│   │   └── stat_eval.py            # paired t-test + Hedges' g + 95% CI
│   │
│   ├── bench/                      # all 11 paired-comparison benches
│   │   ├── bench_f32_multiseed.py      # C1, C2, C5, C6, C10
│   │   ├── bench_f33_rx_vs_ms.py       # C5 (Multi-Rx vs Multi-scale)
│   │   ├── bench_f34_widar_orig.py     # C6 (Multi-Rx vs Widar-orig)
│   │   ├── bench_f35_snapshot.py       # C3 (Snapshot×3)
│   │   ├── bench_f36_mixup.py          # (mixup baseline)
│   │   ├── bench_f37_dann.py           # C4 (DANN)
│   │   ├── bench_f46_algo_methods.py   # Pretrain, Reweighting, ...
│   │   ├── bench_f48_cbam_resnet18.py  # C10 (CBAM+ResNet18 control)
│   │   └── bench_latency_table.py      # §10.4 System Overhead table
│   │
│   ├── figs/                       # figure generation (matplotlib)
│   │   ├── make_fig0_pipeline.py   # Fig 0: reproducibility audit pipeline
│   │   ├── make_fig1_boxplot.py    # Fig 1: paired forest plot
│   │   ├── make_fig2_variance.py   # Fig 2: variance decomposition
│   │   ├── make_fig3_mde.py        # Fig 3: MDE self-validation
│   │   ├── make_fig5_mechanism.py  # Fig 5: Multi-Rx mechanism
│   │   ├── make_fig6_tsne.py       # Fig 6: t-SNE domain separability
│   │   └── make_paper_figs.py      # master driver for all figures
│   │
│   └── analysis/
│       ├── four_data_points.py     # §9.4 4-data-points dataset-conditioned table
│       └── variance_decomp.py      # §6 variance decomposition
│
├── results/                        # per-bench JSON outputs (committed)
│   ├── f32_multiseed_50k.json
│   ├── f33_rx_vs_ms.json
│   ├── f34_widar_orig_vs_f24.json
│   ├── f35_snapshot_ensemble.json
│   ├── f37_dann.json
│   ├── f48_backbone_fairness.json
│   ├── latency_table.json
│   └── four_data_points.json
│
└── paper/
    ├── paper_arxiv.tex             # arXiv pre-print source
    ├── paper_arxiv.pdf
    ├── paper_arxiv.bbl
    ├── cover_letter_tmc.txt
    └── CHANGELOG.txt
```

## What the Audit Contains

### 11 Paired Comparisons (Table 1 in paper)

| ID    | Comparison                         | Family           | Status |
|-------|-------------------------------------|------------------|--------|
| C1    | LeNetCSI_Attn+MHA vs Conv-attn     | Architectural    | **fail** |
| C2    | Multi-scale vs Multi-Rx (Widar)    | Representational | fail   |
| C3    | Snapshot×3 vs single               | Algorithmic      | fail   |
| C4    | DANN vs source-only                | Algorithmic      | **fail** |
| C5    | Multi-Rx vs RMS-agg (Widar)        | Representational | **real** |
| C6    | Multi-Rx vs Widar-orig (Widar)     | Representational | **real** |
| C7a   | Multi-Rx vs RMS-agg (MMFi v1)      | Representational | fail   |
| C7b   | Multi-scale vs RMS-agg (MMFi v1)   | Representational | fail   |
| C7c   | Multi-scale vs Multi-Rx (MMFi v1)  | Representational | **real** (flips) |
| C10   | CBAM+ResNet18 vs LeNetCSI_Attn (12k) | Architectural | **real** (worse) |
| C10v2 | CBAM+ResNet18 vs LeNetCSI_Attn (50k) | Architectural | fail   |

### Statistical Foundation (§3.5)

The audit is governed by three closed-form theorems:

- **Theorem 1 (Paired Sample Size):** n ≥ (z_{1-α/2} + z_{1-β})² σ² / ε². At our σ ≈ 4.8 pp, ε = 2 pp requires n ≥ 56 — three times our budget.
- **Theorem 2 (MDE Closed Form):** MDE = (t_{α/2,n-1} + t_{β,n-1}) σ / √n.
- **Theorem 3 (Hedges' g Correction):** g = d · (1 - 3/(4n - 1)), SE_g = √(1/n + g²/(2n)).
- **Corollary 1 (Audit Criterion):** A report is "real" iff |Δ| > MDE **AND** p < α **AND** |g| ≥ 0.2.

### 4 Data-Point Boundary (§9.4)

The dataset-conditioned optimality claim rests on four data points:

| Setting                       | Δ (pp) | p     | Verdict                |
|-------------------------------|-------:|------:|------------------------|
| Widar 12k (C10, original)     | -4.61  | 0.028 | "larger backbone hurts"|
| Widar 50k (C10v2, reanalysis) | +5.75  | 0.061 | direction *reverses*   |
| MMFi v1 (C7c, path-labeled)   | +1.20  | 0.016 | "Multi-scale optimal"  |
| MMFi v2 (16,447 segments)     | -0.33  | 0.564 | "indistinguishable"    |

## Reproducibility Notes

### Hardware

- GPU: NVIDIA RTX 3090 (24 GB) recommended; runs on 12 GB GPUs with reduced batch size
- CPU: inference latency benchmark uses single thread (matches mobile/edge deployment)
- Total wall-clock time: ~5 GPU-hours for the full audit + ~30 minutes for all figures

### Random Seeds

- We pre-registered seeds {0, 1, 2, 3, 4} for the K=5 multi-seed protocol
- All `bench_*.py` scripts accept `--seeds` to override
- The 12k and 50k Widar subsets use the same seed pool for paired comparisons

### Data Availability

- **Widar3.0** is publicly available at [http://tns.thss.tsinghua.edu.cn/widar3.0/](http://tns.thss.tsinghua.edu.cn/widar3.0/) (Zheng et al., 2019)
- **MMFi** is publicly available at [https://github.com/ybhbingo/MMFi_dataset](https://github.com/ybhbingo/MMFi_dataset) (Yang et al., 2023)
- **CSIDA** is publicly available at [https://github.com/linteresa/CSIDA](https://github.com/linteresa/CSIDA) (Yang et al., 2023)
- **SignFi** is publicly available at [https://yongsen.github.io/SignFi/](https://yongsen.github.io/SignFi/) (Ma et al., 2018); [GitHub mirror](https://github.com/4three2one/SignFi)
- The BVP caches we construct (~50 GB combined) are released at Zenodo (DOI: 10.5281/zenodo.[redacted])

## License

This repository is released under the MIT License (see `LICENSE`). The released code is for research purposes only; commercial use requires permission from the original dataset authors (Zheng et al., Yang et al.).

## Citation

```bibtex
@article{anon2026csi_reproducibility,
  title={WiFi {CSI} Cross-Domain Gesture Recognition: A Reproducibility
         Audit Revealing Dataset-Conditioned Optimality of the
         Multi-{Rx} Representation},
  author={Anonymous Authors},
  journal={Under review at IEEE Transactions on Mobile Computing},
  year={2026}
}
```

## Acknowledgments

We thank the authors of Widar3.0 (Zheng et al., 2019) and MMFi (Yang et al., 2023) for publicly releasing the datasets that made this audit possible. We also thank the open-source PyTorch, NumPy, and SciPy communities.
