# csi-reproducibility-audit

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](requirements.txt)
[![arXiv:coming](https://img.shields.io/badge/arXiv-coming-red.svg)](https://arxiv.org/)

**Companion code for the reproducibility audit of WiFi CSI cross-environment gesture recognition.**

> The companion paper introduces a paired Leave-One-Domain-Out (LODO) + multi-seed evaluation protocol and applies it to audit ten state-of-the-art WiFi CSI gesture recognition methods across three datasets (Widar3.0, MMFi, UT-HAR). The audit reveals *dataset-conditioned optimality*: no single representation (Multi-Rx, RMS-aggregate, single-channel) dominates universally — a finding that contradicts the prevailing "Multi-Rx always wins" narrative in the literature.

## 📖 What this repo contains

| Path | Purpose |
|---|---|
| [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) | **Start here.** Full audit protocol, dataset manifest, randomness control, statistical tests, Zenodo links |
| [`reproduce.sh`](reproduce.sh) | One-click end-to-end reproduction (build cache → train → eval → stats → figure) |
| [`latex/`](latex/) | LaTeX source of the companion paper (`paper_arxiv.tex`) |
| [`figs/`](figs/) | All figures embedded in the paper |
| [`src/`](src/) | Training and evaluation code (model library in `wfcslab/`, runners in `bench_*.py`) |
| [`scripts/`](scripts/) | Data preprocessing (cache builders) and paper-figure generation |
| [`docs/`](docs/) | Cover letter, compression plan, revision guide, submission checklist |
| [`results/`](results/) | Pointer to Zenodo-stored numerical results (large files) |

## 🚀 Quick start (5 minutes)

```bash
git clone https://github.com/91geek/csi-reproducibility-audit.git
cd csi-reproducibility-audit
pip install -r requirements.txt
# Download dataset caches from Zenodo (see REPRODUCIBILITY.md §"Data Availability")
bash reproduce.sh --quick   # 30-min smoke test on 5% subset
```

A full reproduction takes ~14 hours on a single RTX 3090. See [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) for the detailed protocol, expected outputs, and ablation switches.

## 🎯 Key contributions of the audit

1. **A paired LODO + multi-seed evaluation protocol** with closed-form Minimum Detectable Effect (MDE), Hedges' *g* effect-size correction, and a three-criterion decision rule (|Δ| > MDE ∧ p < α ∧ |g| ≥ 0.2). See `latex/paper_arxiv.tex` §3.5 (Statistical Foundation).
2. **Three datasets audited**: Widar3.0 (50k canonical subset), MMFi (v2 cache, ~95% NaN-free), CSIDA (F-49 zarr, 6 classes × 2 env × 5 user).
3. **Ten methods × three representations × six ablations**: a 2,160-curve grid showing that the optimal representation is dataset-dependent.
4. **System overhead benchmark**: inference latency and parameter count for five backbones (Table §10.4 of the paper).

## 🔬 Audit result (TL;DR)

Across 18 paired (Multi-Rx vs RMS-aggregate) comparisons on the 50k Widar3.0 canonical subset:

- **Wins**: Multi-Rx 14 / 18 (77.8%)
- **Mean Δ**: +5.75 pp accuracy (95% CI: [−0.26, +11.77])
- **p-value**: 0.0609 (paired Wilcoxon, exact)
- **Cohen's d**: +0.442 (Hedges' g corrected)

⚠️ The p-value is slightly above α = 0.05; the CI crosses zero. We report it as **direction-stable but not strictly significant** and recommend the full audit protocol (with pre-registered MDE) over a single p-value cutoff. See `docs/cover_letter_tmc.txt` for the framing adopted in the TMC submission.

## 📦 Data availability

Raw CSI data is **not** redistributed here due to dataset licenses. Each dataset has its own download path:

| Dataset | Source | Size |
|---|---|---|
| Widar3.0 | [Personal Website](http://tns.thss.tsinghua.edu.cn/widar/) | ~22 GB |
| MMFi | [IEEE DataPort](https://ieee-dataport.org/open-access/mmfi) | ~37 GB |
| SignFi | [yongsen.github.io/SignFi](https://yongsen.github.io/SignFi/) ([GitHub mirror](https://github.com/4three2one/SignFi)) | ~6 GB (4× .mat) |

Pre-processed tensor caches (`.npy` / `.npz` complex64) are hosted on Zenodo under DOI `[to be assigned upon TMC acceptance]`. `reproduce.sh` will fetch and verify checksums automatically.

## 📝 Citation

```bibtex
@software{csi_reproducibility_audit_2026,
  title  = {Reproducibility Audit of WiFi CSI Cross-Environment Gesture Recognition},
  author = {[Author] and 91geek},
  year   = {2026},
  url    = {https://github.com/91geek/csi-reproducibility-audit},
  note   = {Companion code for the IEEE TMC submission}
}
```

See [`CITATION.cff`](CITATION.cff) for the structured citation.

## 📄 License

This codebase is released under the [MIT License](LICENSE). Paper figures and prose are © 2026 the authors, all rights reserved.

## 🤝 Contributing

Issues and PRs are welcome, but please note that this is a **frozen reproducibility artifact** for a specific paper submission. For experimental extensions or new methods, fork the repository.

## ✉️ Contact

- Open an issue for reproducible problems
- Email for submission-related correspondence: 56215141@qq.com
- ORCID: [0009-0003-3371-6056](https://orcid.org/0009-0003-3371-6056)
