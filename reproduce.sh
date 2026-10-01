#!/bin/bash
# reproduce.sh — One-click reproduction of all 11 paired comparisons + 6 figures.
# Total wall-clock: ~5 GPU-hours on RTX 3090 + ~30 minutes for figures.
#
# Usage:
#   bash reproduce.sh all         # full pipeline
#   bash reproduce.sh widar       # Widar3.0 only (~2 GPU-hours)
#   bash reproduce.sh mmfi        # MMFi v1+v2 only (~2 GPU-hours)
#   bash reproduce.sh latency     # inference latency benchmark (~5 minutes)
#   bash reproduce.sh figs        # regenerate figures from committed results/
#
# Requires: CUDA-capable GPU, Python 3.10+, ~50 GB disk for caches.

set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

# Helper: pretty-print section header
section() {
    echo ""
    echo "============================================================"
    echo "  $1"
    echo "============================================================"
}

# Helper: run if results file doesn't exist
run_or_skip() {
    local script="$1"
    local output="$2"
    if [[ -f "results/$output" ]]; then
        echo "[skip] results/$output already exists"
    else
        echo "[run]  $script"
        python "$script"
    fi
}

# ============================================================
# 1. Cache construction (~15 minutes, CPU-only)
# ============================================================
build_caches() {
    section "Building BVP caches"
    if [[ ! -f "data/widar_cache.npy" ]]; then
        python src/cache/build_widar_cache.py --out data/widar_cache.npy
    fi
    if [[ ! -f "data/mmfi_cache_v1.npy" ]]; then
        python src/cache/build_mmfi_cache_v1.py --out data/mmfi_cache_v1.npy
    fi
    if [[ ! -f "data/mmfi_cache_v2.npy" ]]; then
        python src/cache/build_mmfi_cache_v2.py --out data/mmfi_cache_v2.npy
    fi
}

# ============================================================
# 2. Widar3.0 paired comparisons (~2 GPU-hours)
# ============================================================
bench_widar() {
    section "Widar3.0 paired comparisons"
    run_or_skip "src/bench/bench_f32_multiseed.py"     "f32_multiseed_50k.json"
    run_or_skip "src/bench/bench_f33_rx_vs_ms.py"     "f33_rx_vs_ms.json"
    run_or_skip "src/bench/bench_f34_widar_orig.py"   "f34_widar_orig_vs_f24.json"
    run_or_skip "src/bench/bench_f35_snapshot.py"     "f35_snapshot_ensemble.json"
    run_or_skip "src/bench/bench_f37_dann.py"         "f37_dann.json"
    run_or_skip "src/bench/bench_f46_algo_methods.py" "f46_algo_methods.json"
    run_or_skip "src/bench/bench_f48_cbam_resnet18.py" "f48_backbone_fairness.json"
}

# ============================================================
# 3. MMFi paired comparisons (~2 GPU-hours)
# ============================================================
bench_mmfi() {
    section "MMFi paired comparisons (v1 + v2 caches)"
    run_or_skip "src/bench/bench_f40_mmfi_ablation.py" "f40_mmfi_v1.json"
    run_or_skip "src/bench/bench_f48_mmfi_v2.py"     "f48_mmfi_v2.json"
    run_or_skip "src/bench/bench_f51_mmfi_ablation.py" "f51_mmfi_v2_extended.json"
}

# ============================================================
# 4. Inference latency benchmark (~5 minutes, CPU-only)
# ============================================================
bench_latency() {
    section "Inference latency benchmark (§10.4)"
    python src/bench/bench_latency_table.py
}

# ============================================================
# 5. Analysis scripts
# ============================================================
run_analysis() {
    section "Analysis: 4-data-points + variance decomposition"
    python src/analysis/four_data_points.py   --out results/four_data_points.json
    python src/analysis/variance_decomp.py    --out results/variance_decomposition.json
}

# ============================================================
# 6. Figure generation (~30 minutes)
# ============================================================
gen_figs() {
    section "Generating paper figures"
    python src/figs/make_paper_figs.py --input results/ --output figs/
    python src/figs/make_fig5_mechanism.py --input results/ --output figs/fig5_mechanism.png
    python src/figs/make_fig6_tsne.py      --output figs/fig_tsne_widar.png figs/fig_tsne_mmfi.png
}

# ============================================================
# Dispatch
# ============================================================
case "${1:-all}" in
    widar)
        build_caches
        bench_widar
        ;;
    mmfi)
        build_caches
        bench_mmfi
        ;;
    latency)
        bench_latency
        ;;
    figs)
        gen_figs
        ;;
    analysis)
        run_analysis
        ;;
    all)
        build_caches
        bench_widar
        bench_mmfi
        bench_latency
        run_analysis
        gen_figs
        section "Done — all results in results/, all figures in figs/"
        ;;
    *)
        echo "Usage: bash reproduce.sh {all|widar|mmfi|latency|figs|analysis}"
        exit 1
        ;;
esac
