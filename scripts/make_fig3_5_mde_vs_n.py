"""
Fig. 3.5 (NEW): MDE vs n trade-off curve
========================================
Visualize the MDE formula (Eq. mde) as a function of sample size n,
for 3 representative σ_within values (RMS-agg=1.5pp, Single-Rx=2.1pp,
Multi-Rx=4.82pp). Mark our actual design points (n=18 / n=30).

Also annotate the 3 reported-effect thresholds:
  - 1pp  (most CSI papers)
  - 3pp  (some strong papers)
  - 5pp  (rare)
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'src', 'bvp_test')

# ============ MDE function ============
def mde(sigma, n, alpha=0.05, power=0.80):
    """MDE in pp given sigma_within and n."""
    if n - 1 < 1:
        return np.nan
    t_alpha = stats.t.ppf(1 - alpha/2, n - 1)
    t_beta = stats.t.ppf(power, n - 1)
    return (t_alpha + t_beta) * sigma / np.sqrt(n)


# ============ Compute MDE for n in [3, 100] and 3 sigmas ============
ns = np.array([3, 5, 7, 10, 15, 18, 20, 25, 30, 40, 50, 75, 100])
sigmas = [
    (1.50, 'RMS-agg',    '#4C78A8'),
    (2.10, 'Single-Rx',  '#54A24B'),
    (4.82, '★ Multi-Rx', '#E45756'),
]

fig = plt.figure(figsize=(14, 7))
gs = fig.add_gridspec(1, 2, width_ratios=[1.5, 1.0], wspace=0.30,
                      left=0.06, right=0.97, top=0.90, bottom=0.10)

# ============ (a) MDE vs n curve ============
ax = fig.add_subplot(gs[0, 0])

for sigma, label, color in sigmas:
    mdes = [mde(sigma, n) for n in ns]
    ax.plot(ns, mdes, marker='o', markersize=7, linewidth=2.0,
            color=color, label=f'{label} (σ={sigma:.2f}pp)')

# Mark our design points
ax.axvline(18, color='#888', linestyle=':', linewidth=1.0, alpha=0.6)
ax.axvline(30, color='#888', linestyle=':', linewidth=1.0, alpha=0.6)
ax.text(18.5, 6.5, 'n=18\n(MMFi LODO)', fontsize=8, color='#888', fontweight='bold')
ax.text(30.5, 6.5, 'n=30\n(Widar LODO)', fontsize=8, color='#888', fontweight='bold')

# Reference effect thresholds
ref_effects = [(1, '1pp'), (3, '3pp'), (5, '5pp')]
for ref, label in ref_effects:
    ax.axhline(ref, color='gray', linestyle='--', linewidth=0.7, alpha=0.5)
    ax.text(102, ref + 0.15, label, fontsize=8, color='gray', va='bottom', ha='left')

ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlim(2.5, 105)
ax.set_ylim(0.3, 10)
ax.set_xlabel('Sample size n (paired samples per comparison)', fontsize=10)
ax.set_ylabel('Minimum Detectable Effect MDE (pp)', fontsize=10)
ax.set_title('(a) MDE vs n at power=0.80, α=0.05\n'
             '(log-log; lower-left = easier to detect small effects)',
             fontsize=10.5)
ax.legend(fontsize=9, loc='upper right', framealpha=0.92)
ax.grid(True, which='both', linestyle=':', alpha=0.4)

# Annotate: n=5 design (CSI-Bench standard) vs n=30 (ours)
ax.annotate('CSI-Bench n=3\n(most papers)\nMDE ≈ 4-7pp',
            xy=(3, mde(4.82, 3)), xytext=(4.5, 4.0),
            fontsize=8.5, color='#444',
            arrowprops=dict(arrowstyle='->', color='#444', lw=0.8))
ax.annotate('Our n=30\nMDE ≈ 0.8-2.6pp',
            xy=(30, mde(4.82, 30)), xytext=(35, 0.55),
            fontsize=8.5, color='#E45756', fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='#E45756', lw=1.0))

# ============ (b) Power curve at our 3 design σ values ============
ax2 = fig.add_subplot(gs[0, 1])

def power_at_delta(sigma, n, delta, alpha=0.05):
    """Empirical power for detecting delta, with n paired samples."""
    if n - 1 < 1:
        return np.nan
    se = sigma / np.sqrt(n)
    df = n - 1
    t_crit = stats.t.ppf(1 - alpha/2, df)
    nc = delta / se  # noncentrality
    # Power = P(|T| > t_crit | H1)
    power = 1 - stats.nct.cdf(t_crit, df, nc) + stats.nct.cdf(-t_crit, df, nc)
    return power

deltas = np.linspace(0, 10, 200)
for sigma, label, color in sigmas:
    # Use n=30 (Widar LODO design)
    powers = [power_at_delta(sigma, 30, d) for d in deltas]
    ax2.plot(deltas, powers, linewidth=2.0, color=color,
             label=f'{label} (n=30, σ={sigma:.2f}pp)')
    # Mark the 80% power threshold
    idx_80 = np.argmin(np.abs(np.array(powers) - 0.80))
    ax2.plot(deltas[idx_80], 0.80, 'o', color=color, markersize=8)

ax2.axhline(0.80, color='black', linestyle='--', linewidth=1.0, alpha=0.6,
            label='power = 0.80 (target)')
ax2.axvline(1.0, color='gray', linestyle=':', linewidth=0.7, alpha=0.5)
ax2.axvline(3.0, color='gray', linestyle=':', linewidth=0.7, alpha=0.5)
ax2.axvline(5.0, color='gray', linestyle=':', linewidth=0.7, alpha=0.5)

ax2.set_xlabel('True effect Δ (pp)', fontsize=10)
ax2.set_ylabel('Statistical power', fontsize=10)
ax2.set_title('(b) Power vs Δ at n=30 (Widar LODO)\n'
              '(● marks 80% power threshold)',
              fontsize=10.5)
ax2.legend(fontsize=8.5, loc='lower right', framealpha=0.92)
ax2.grid(True, linestyle=':', alpha=0.4)
ax2.set_xlim(0, 10)
ax2.set_ylim(0, 1.05)

# Annotate key thresholds
ax2.text(0.5, 0.50,
         '1pp effects:\nonly RMS-agg\nhas >80% power',
         fontsize=8, color='#444', style='italic')
ax2.text(5.0, 0.55,
         '5pp+ effects:\nall 3 designs\nhave >80% power',
         fontsize=8, color='#444', style='italic')

fig.suptitle('Fig. 3.5: MDE trade-off — why we use n=30 × σ-within-aware design',
             fontsize=11.5, y=0.99)

out_png = os.path.join(OUT_DIR, 'fig3_5_mde_vs_n.png')
fig.savefig(out_png, dpi=150, bbox_inches='tight')
print(f'[Fig 3.5] -> {out_png}')
print()
print('=== MDE at our actual design points ===')
for sigma, label, color in sigmas:
    mde_18 = mde(sigma, 18)
    mde_30 = mde(sigma, 30)
    print(f'  {label:12s}: n=18 MDE={mde_18:.2f}pp, n=30 MDE={mde_30:.2f}pp')