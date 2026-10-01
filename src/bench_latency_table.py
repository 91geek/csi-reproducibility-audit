# bench_latency_table.py
# F-58 (System Overhead) — Inference latency benchmark for the 5 backbones
# in §3.2 / §8 of Paper A. Generates Table for §7.4 "System Overhead".
#
# Output:
#   - JSON to src/bvp_test/latency_table.json
#   - Stdout markdown table ready for paper inclusion

import json, time, sys, os
from pathlib import Path
import torch
import torch.nn as nn

# Add src to path so we can import backbones
sys.path.insert(0, str(Path(__file__).parent))

from wfcslab.models.backbones import (
    LeNetCSI, LeNetCSI_Attn, LeNetCSI_Attn_MHA, ResNetSmall, CBAMResNet18
)

# ============================================================
# Config: Widar3.0 input shape (N, A=9, S=30, F=1, T=256)
# BVP window = 256 time samples × 9 antennas × 30 subcarriers
# ============================================================

INPUT_SHAPES = {
    "Widar3.0":   dict(A=9,  S=30,  F=1, T=256),
    "MMFi":       dict(A=10, S=114, F=1, T=512),
}

BATCH_SIZES = [1, 32]
N_WARMUP = 3
N_TIMING = 20
SEED = 42

def build(model_name, A, S, F, T):
    out_dim = 8  # 8-class Widar / 27-class MMFi -- use 8 for cost comparison
    if model_name == "LeNetCSI":
        return LeNetCSI(in_channels=A, n_time=T, n_subc=S, out_dim=out_dim)
    elif model_name == "LeNetCSI_Attn":
        return LeNetCSI_Attn(in_antennas=A, in_subcarriers=S,
                             in_freq=F, in_time=T, out_dim=out_dim)
    elif model_name == "LeNetCSI_Attn_MHA":
        return LeNetCSI_Attn_MHA(in_antennas=A, in_subcarriers=S,
                                in_freq=F, in_time=T, out_dim=out_dim)
    elif model_name == "ResNetSmall":
        return ResNetSmall(in_channels=A, n_time=T, n_subc=S,
                           out_dim=out_dim)
    elif model_name == "CBAMResNet18":
        return CBAMResNet18(in_antennas=A, in_subcarriers=S,
                            in_freq=F, in_time=T, out_dim=out_dim,
                            width=64, hidden=256, dropout=0.4)
    else:
        raise ValueError(model_name)


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def time_model(model, A, S, F, T, batch_size, device='cpu'):
    model = model.to(device).eval()
    # LeNetCSI/ResNetSmall expect 4D (B, in_ch, T, S); Attn/CBAM expect 5D (B, A, S, F, T)
    x_5d = torch.randn(batch_size, A, S, F, T, device=device)
    x_4d = x_5d.view(batch_size, A, T, S)  # squeeze F=1, swap to (B, A, T, S)

    with torch.no_grad():
        # warmup -- try 5D first, fall back to 4D
        try:
            for _ in range(N_WARMUP):
                _ = model(x_5d)
            x = x_5d
        except (RuntimeError, TypeError):
            for _ in range(N_WARMUP):
                _ = model(x_4d)
            x = x_4d
        # timing — cap individual model to 30s
        t0 = time.perf_counter()
        for _ in range(N_TIMING):
            _ = model(x)
            if time.perf_counter() - t0 > 30:
                break
        t1 = time.perf_counter()
    latency_ms = (t1 - t0) / N_TIMING * 1000.0
    return latency_ms


def main():
    torch.manual_seed(SEED)
    results = {}
    models = ["LeNetCSI", "LeNetCSI_Attn", "LeNetCSI_Attn_MHA", "ResNetSmall", "CBAMResNet18"]

    print(f"\n{'Model':<22} {'A':>3} {'S':>4} {'T':>4} {'B':>4} {'Params(M)':>10} {'latency(ms)':>13}")
    print("-" * 70)

    for dataset, shape in INPUT_SHAPES.items():
        A, S, F, T = shape["A"], shape["S"], shape["F"], shape["T"]
        for model_name in models:
            try:
                model = build(model_name, A, S, F, T)
                n_params = count_params(model)
            except Exception as e:
                print(f"{model_name:<22} build failed: {e}")
                continue
            for B in BATCH_SIZES:
                try:
                    lat = time_model(model, A, S, F, T, B, device='cpu')
                except Exception as e:
                    print(f"{model_name:<22} B={B:<3} timing failed: {e}")
                    continue
                key = f"{model_name}|{dataset}|B={B}"
                results[key] = {
                    "model": model_name, "dataset": dataset,
                    "A": A, "S": S, "F": F, "T": T,
                    "batch_size": B, "params_M": n_params / 1e6,
                    "latency_ms": lat,
                    "latency_per_sample_ms": lat / B,
                }
                print(f"{model_name:<22} {A:>3} {S:>4} {T:>4} {B:>4} {n_params/1e6:>10.3f} {lat:>13.3f}")

    out_path = Path(__file__).parent / "bvp_test" / "latency_table.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved {len(results)} entries to {out_path}")

    # Print markdown table for paper
    print("\n\n=== LATENCY TABLE (paper-ready) ===\n")
    print("| Model | A | S | T | B=1 (ms) | B=32 (ms) | B=256 (ms) | Params (M) |")
    print("|---|---|---|---|---|---|---|---|")
    for model_name in models:
        rows = [k for k in results if k.startswith(f"{model_name}|Widar3.0")]
        if not rows:
            continue
        first = results[rows[0]]
        params = first["params_M"]
        b1  = next((results[r]["latency_ms"] for r in rows if results[r]["batch_size"] == 1),   float('nan'))
        b32 = next((results[r]["latency_ms"] for r in rows if results[r]["batch_size"] == 32),  float('nan'))
        b256 = next((results[r]["latency_ms"] for r in rows if results[r]["batch_size"] == 256), float('nan'))
        print(f"| {model_name} | {first['A']} | {first['S']} | {first['T']} | {b1:.2f} | {b32:.2f} | {b256:.2f} | {params:.3f} |")


if __name__ == "__main__":
    main()
