# Results

This directory is **intentionally empty** in the repository.

## Why?

The numerical results of the audit (paired comparison tables, MDE tables, latency benchmarks, t-SNE projections) are too large to live alongside code (~150 MB total across the three datasets × ten methods × six ablations × five seeds).

## Where to find them

All numerical results are hosted on **Zenodo** under DOI `10.5281/zenodo.[to be assigned]` and referenced from `REPRODUCIBILITY.md` §"Data Availability".

After running `bash reproduce.sh`, intermediate results will be written to `src/bvp_test/cache/` and `src/cfg_logs/` (both gitignored). The final audit report (HTML) is written to `results/audit_report.html` once it has been generated.

## What `reproduce.sh` produces

| Output | Location | Format |
|---|---|---|
| Trained checkpoints | `src/cfg_logs/<config>/seed_<n>/` | `.pt` |
| Epoch-level metrics | `src/cfg_logs/<config>/seed_<n>/metrics.csv` | CSV |
| Final test accuracy | `src/cfg_logs/<config>/summary.json` | JSON |
| Paired comparison table | `src/bvp_test/paired_<dataset>.json` | JSON |
| MDE tables | `src/bvp_test/mde_<dataset>.json` | JSON |
| Latency benchmark | `src/bvp_test/latency_table.json` | JSON |
| Audit report (HTML) | `results/audit_report.html` | HTML |
