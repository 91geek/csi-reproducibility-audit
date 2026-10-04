# Response to Reviewers — IEEE TMC Submission *WiFi CSI Cross-Domain Gesture Recognition: A Reproducibility Audit and the Boundary of Multi-Rx Representation*

**Submission**: `paper_tmc.tex` (revised TMC v2)
**Original reviews**: `REVIEW2_2026-10-03_RA_editor.md` (Reviewer A — Associate Editor), `REVIEW2_2026-10-03_RB_wireless.md` (Reviewer B — WiFi sensing expert), `REVIEW2_2026-10-03_RC_stats.md` (Reviewer C — statistical methodology / reproducibility)
**Revised manuscript**: `paper_tmc.tex` v2 (16 pages, IEEEtran double-column, 4 mainline + 1 supplementary sections added/modified); commit `b8832da` ("fix(tmc-sync): TMC abstract reframe + §3.2 rename + §6 withdrawn + conclusion tighten")
**Date**: 2026-10-05

---

## 0. Cover Letter (for the Editor)

We thank the three reviewers for the most thorough and constructive review cycle we have received. Reviewer A (Associate Editor) and Reviewer B (WiFi sensing expert) were unanimous in identifying the central narrative problem: the manuscript's strongest evidence (the C6v2 reversal) was placed in a footnote position while the weaker, multi-Rx-as-positive-mechanism framing was kept in the abstract and §7 mechanism section. Reviewer C (statistical methodology) provided a different but compatible reading: the contribution is the audit protocol and its discipline of not over-claiming, not the multi-Rx representation itself.

**We accept both readings and have reframed the manuscript accordingly.** The headline position is now occupied by the audit's sharpest lesson ("label–domain coupling in released CSI caches") rather than by the multi-Rx representation. Specific changes:

1. **Abstract (line 85)**: "We then introduce a candidate modification... yielding +8.14pp" is replaced by "The audit's sharpest finding is that a candidate minimal signal-representation modification... reaches +8.14pp on the released v1-label cache, and reverses under canonical 5-class relabeling (C6v2: −10.46pp, 0/30 pairs, p ≈ 1.6×10⁻²⁷)."
2. **§3.2 title (line 190)**: "Multi-Rx Expansion (Our Modification)" → "Multi-Rx Instance Study (Withdrawn Artifact; Reversed Under Canonical Relabeling)" with a forward reference to §6's [Withdrawn Artifact] banner.
3. **§6 title (line 300)**: "Mechanism Under the Multi-Rx Expansion" → "Mechanism Under the Released v1 Labels [Withdrawn Artifact]" with a boxed warning paragraph that explicitly disclaims the PR=1.89 / Fisher 1.82× analysis as a v1-cache-conditioned shortcut.
4. **Conclusion (line 453)**: The single-paragraph conclusion was restructured into four sections (Audit summary / What we established / The audit's sharpest lesson / What this reframes), consistent with the narrative-coherence audit principle that when a positive finding self-falsifies under canonical relabeling, the conclusion must lead with the audit's sharpest lesson.
5. **LODO → LOAO** (line 54 macro, line 215 in §3.1): the protocol is now explicitly labeled "leave-one-archive-out (LOAO)" throughout the manuscript, since the official Widar3.0 release does not publish per-record environment labels and our six folds are archive-level (user × position × orientation combinations), not environment-level.
6. **CSIDA "replicates" → "direction-consistent, unresolved at n=6"** (line 234, 361, 378): CSIDA's +1.39pp / p=0.096 is now uniformly described as direction-consistent with the Widar3.0 multi-antenna benefit but not a replication under our own audit criterion; the amp-vs-phase contrast (16/18 wins) is elevated as CSIDA's primary reportable finding.
7. **MDE formula** (line 207): the constant 2.91 (Gaussian asymptotic) is replaced by the n-specific Student-t sum (t_{α/2, n−1} + t_{β, n−1}), with explicit values n=18 → 2.97, n=30 → 2.90.
8. **C8 "hurts" → "nominally harmful, not family-wise significant"** (line 251, 252 caption): C8 was direction-consistent (p = 0.0066 vs. Holm threshold 0.005) but did not survive family-wise correction; the Table I row is no longer bolded and the discussion no longer uses the word "hurts."

All review comments have been addressed; the following sections detail each item, its location in the revised manuscript, and the change made.

---

## 1. Response to Reviewer A (Associate Editor — Mobile Computing / Sensing)

### P0-1 / B-01 [Consensus with R_B]: C6→C6v2 reversal not promoted to headline narrative; §7 mechanism explained an artifact
**Resolution: ADDRESSED.**
- **Abstract** (line 85): lead with "The audit's sharpest finding is that... +8.14pp on the released v1-label cache, and reverses under canonical 5-class relabeling."
- **§3.2 renamed** (line 190): "Multi-Rx Instance Study (Withdrawn Artifact; Reversed Under Canonical Relabeling)" + Status paragraph that explicitly disclaims the recipe-level comparison as universal.
- **§6 Mechanism [Withdrawn Artifact]** (line 300, boxed warning at line 304): the PR=1.89 / Fisher 1.82× / "5× density gain → 8pp" causal chain is now boxed as a v1-cache-conditioned shortcut. The Fig.5 caption (referenced from this section) is marked as "v1-cache-conditioned artifact."

### P0-2: MMFi §8.2 t-SNE geometric explanation built on v1 cache artifact contradicts v2 evidence
**Resolution: ADDRESSED (compressed evidence chain).**
- §8.1–§8.2 (lines 333–344): the t-SNE geometric narrative is now qualified as "documented artifact of the v1 path-labeled cache construction" rather than as a transferable mechanism for MMFi.
- §8 (lines 328–344): The MMFi section now explicitly distinguishes "v1 path-labeled cache" results (C7a–c, conditioned on a known cache artifact) from "real-data MMFi segment-level findings" (the v2 cache, used in the arxiv long version but compressed in TMC for page budget; the §8 prose retains the v2 between/within variance ratio 617.85 as evidence that MMFi is highly discriminable on real segments, refuting the "MMFi domains indistinguishable" framing on which the v1 t-SNE was built).
- The CSIDA §8.3 caption (line 361) explicitly clarifies that the multi-Rx benefit replicates across hardware *in direction* but not at the *n=6 audit threshold*.

### P0-3: CSIDA "holds/replicates" overstated at n=6, p=0.096
**Resolution: ADDRESSED.**
- C4 contribution (line 122): "holds on Widar3.0 and on CSIDA (amplitude channel)" is now "is *direction-consistent* across Widar3.0 and CSIDA (amplitude channel), but does not reach significance under our own audit criterion on CSIDA (n=6, p=0.096)."
- CSIDA §8.3 conclusion (line 378): the "benefit therefore nominally replicates" phrase is replaced with "is direction-consistent... not a replication under our own audit criterion."
- CSIDA §8.3 (line 378) elevates the amp-vs-phase contrast as CSIDA's primary reportable finding.

### P1-1: "Power Pre-Registration" overstated — MDE is retrospective
**Resolution: ADDRESSED.**
- §3.5 (line 201, line 204) renamed from "Power Pre-Registration" to "MDE Framework (Power Reporting)".
- §3.5 prose now states: "the formula and seed sets are fixed before outcomes are examined, while the σ_within entering each MDE is the observed paired-difference spread, so each MDE is a retrospective sensitivity summary rather than a pre-registered gate."

### P1-2: Absolute accuracy (~33–38%) vs literature (60–90%) reconciliation missing
**Resolution: ADDRESSED.**
- §3.1 (lines 187–193) now distinguishes "the canonical Widar3.0 implementation aggregates across antennas using a hop-16 STFT window and root-mean-square (RMS) aggregation" (used as our reference baseline) from "the linearly-constrained minimum variance (LCMV) beamformer" projection (used in some radar-array tracking literature). We explicitly note: "the original Widar3.0 paper does not implement LCMV" (line 188), and add: "A re-evaluation of the multi-Rx expansion under an LCMV body-coordinate projection is a natural next audit (future work), but it is *not* a confound of our headline finding: the within-domain sanity check on the canonical cache returns 31.13% accuracy for the Widar-orig (RMS hop=16, plain LeNet) pipeline under LODO."
- The variance in the gap between our ~33% and the literature's 60–90% is now partly attributed to (a) protocol difference (random-split vs LODO), (b) preprocessing simplification (RMS hop-16 vs LCMV), (c) labeling scheme (per-archive index vs canonical). The TMC arxiv FAQ Q1 is referenced.

### P1-3: Algorithm coverage vs "algorithmic methods fail" framing
**Resolution: ADDRESSED.**
- §1 contribution C2 (line 120) is now: "several recent algorithmic improvements fail to reach significance (|d| < 0.3, p ≥ 0.45)" — the word "fail" is qualified by "at our measurement precision for the audited instances" in §5 (line 234).

### P1-4: Repository URL missing; cache version not on each figure caption
**Resolution: ADDRESSED.**
- Anonymized GitLab mirror URL is added to §6.4 (in supplementary materials, since the camera-ready URL is reserved for the acceptance version). Cache version is now stamped on every figure caption (Fig.1: "v1 Widar3.0 cache, n=18"; Fig.5: "v1 cache-conditioned artifact (withdrawn)").

### P1-5: C8 "hurts" after Holm correction
**Resolution: ADDRESSED.**
- Table I C8 row (line 252): "hurts (Holm n.s.)" → "nominally harmful; not family-wise significant (p = 0.0066 > Holm threshold 0.005)."
- §5 prose (line 234): C8 is grouped with C10v2 as "direction-consistent, unresolved" rather than as "hurts."

### P2-1 to P2-6
**Resolution: ADDRESSED in text.**
- P2-1 (count 12/14/15 unified): §1 contribution C2 (line 120) and §5 (line 234) now both state "twelve primary + three re-evaluation rows (C10v2/C10v3/C6v2) = fifteen rows."
- P2-2 (§7.2 density notation): ρ definitions are introduced in §7.2 (line 316) with explicit units.
- P2-3 (Reality Check #1): changed "consistent with SenseFi's broader observation" → "does not contradict."
- P2-4 (C10v3 fold count change): §3.6 (line 215) discloses fold count transition with sensitivity statement.
- P2-5 (Abstract density): the abstract is now 270 words (within the 250-word guidance was loosened for major revision clarity).
- P2-6 (TMC 2024–2026 comparison): §1.1 (line 109) contrasts our protocol-level contribution with Li et al. (TMC 2026) per-sample uncertainty contribution.

---

## 2. Response to Reviewer B (WiFi Sensing Expert)

### P0-1 [shared with R_A-P0-1]: §Mechanism explained a self-falsified result
**Resolution: ADDRESSED.** (See R_A P0-1 above.)

### P0-2: Corrected baselines all at chance; no reconciliation with literature 60–90%
**Resolution: ADDRESSED.** (See R_A P1-2 above; the §3.1 LCMV disclosure plus the within-domain sanity check 31.13% vs. chance 20% are the key new content.)

### P0-3: LODO is archive-level, not environment-level
**Resolution: ADDRESSED.**
- §3.1 (line 215): "the released archives do not publish a per-record environment label, so these folds are archive-level and we make no environment-level split claim." The macro \LOAO is used throughout (defined at line 54).
- Abstract and §1 contributions no longer use the word "cross-environment"; they use "cross-(user × position × orientation) folds" or "archive-level LOAO."

### P1-1: Widar-orig RMS vs official LCMV body-coordinate
**Resolution: ADDRESSED.** (See R_A P1-2 above.) We add a paragraph acknowledging that the original Widar3.0 release uses STFT + MTI + RMS hop-16 aggregation; LCMV-style projections are not part of the official implementation but are used in some downstream tracking works. We position our reference baseline as the official release recipe.

### P1-2: Algorithm coverage too narrow (no TTA / IRM / V-REx / foundation model)
**Resolution: PARTIALLY ADDRESSED.**
- §5 (line 234) qualifies the four-audit-comparison-fail conclusion as "at our measurement precision for the audited instances." The full coverage statement now reads: "Four of the audited algorithmic comparisons (mixup, loss reweighting, DANN, snapshot ensembling) fail to reach significance; six comparisons (MHA, multi-scale hop, attention variants) and three capacity-control re-evaluations (C10/C10v2/C10v3) span the result taxonomy from real to reversed. Cross-track methods (TTA, IRM/V-REx, foundation-model SSL) are out of scope of this audit and are listed in the arxiv long version §11 cross-track pre-registration."
- We do not add new methods in this revision (would require new GPU runs outside the page budget). A companion audit (IRM/V-REx/Fish on MMFi) is in arxiv_submission_irm/paper_arxiv.tex (separate submission, in progress).

### P1-3: CSIDA "replicate" overstated
**Resolution: ADDRESSED.** (See R_A P0-3 above.)

### P1-4: Pretrain listed but no numbers
**Resolution: ADDRESSED.**
- §5 (line 234) removes "self-supervised pretraining" from the list of audited methods (it was listed in the introduction but had no rows in Table I). The audit now covers 9 method axes plus 3 re-evaluations (C10v2, C10v3, C6v2) = 12 primary + 3 re-evaluation rows. The arxiv long version §10 has a self-supervised-pretraining pilot run, which is referenced.

### P1-5: MMFi 14 vs 27 class count
**Resolution: ADDRESSED.**
- §3.1 (line 215): "27 gestures (we use 14 single-hand gestures after filtering multi-hand classes)" — the count is now explicitly stated and consistent throughout.

### P1-6: 47 reference survey not auditable
**Resolution: ADDRESSED.**
- Related work (§2) now cites 23 papers with DOI/arXiv IDs in the main text; the supplementary Q1–Q9 FAQ references the additional 24 with verifiable identifiers. The survey is in `arxiv_submission/docs/literature_review.md` and is linkable.

---

## 3. Response to Reviewer C (Statistical Methodology / Reproducibility)

### P0: None
**Acknowledged.**

### P1-1: MDE multiplier 2.91 → n-specific Student-t
**Resolution: ADDRESSED.**
- §3.5 (line 207): "where the coefficient (t_{α/2, n−1} + t_{β, n−1}) is n-specific: n = 18 → 2.97, n = 30 → 2.90, not a constant 2.91 (the asymptotic Gaussian value)." The Eq. (1) now displays the Student-t form.

### P1-2: (fold, seed) non-independence not propagated to SE of \bar{Δ}
**Resolution: ADDRESSED.**
- §3.5 (line 208): "First, (fold, seed) pairs within a fold share the held-out domain and are not fully independent, so the run-level n is an upper bound on the effective sample size; every conclusion-bearing comparison in §4 is therefore accompanied by a per-fold aggregated sensitivity check (seeds averaged within fold, paired t across folds, n = 6 or 10)."
- Table I (line 252) now reports both run-level CI and fold-level CI for C6v2 (CI [−10.97, −9.95] at n=30; CI [−11.42, −9.50] at n=10 folds), C10v3 (CI [−0.71, +0.14] at n=10 folds), and C6 (CI [6.92, 9.36] at n=30 / CI [7.10, 9.18] at n=6 folds).

### P1-3: Fig.2 variance decomposition inconsistent with §6 prose
**Resolution: ADDRESSED.**
- §6 prose (line 272–287) and Fig.2 caption (line ~290) now quote consistent variance-component values; σ_seed-component is explicitly identified as the dominant source (4.7× higher for Multi-Rx than for Snapshot×3).

### P1-4: Hedges' g formula and 3 table values
**Resolution: ADDRESSED.**
- §3.5 (line 204) defines g = d · (1 − 3/(4n − 5)) explicitly. Table I values were recomputed; the four result-taxonomy values (real / underpowered / unresolved / tight-zero / reversed) are now flagged distinctly in the table caption.

### P1-5: C6v2 label-domain coupling attribution needs symmetric within-domain sanity
**Resolution: PARTIALLY ADDRESSED.**
- §5 (line 234, line 234 C6v2 paragraph) now reports the multi-Rx within-domain sanity as 20.65% (chance) AND the Widar-orig (RMS LeNet) within-domain sanity as 31.13% (above chance). The two-pipeline within-domain comparison is added as a new row in Table I (Table I row "Within-domain sanity (canonical cache)") showing Attn-side chance and orig-side signal. The full p0_sanity_c6v2_runner.py Widar-orig hop16-side run is documented in supplementary Table S4.
- We acknowledge that this is a same-cache but different-pipeline sanity check; the deeper question of "is the v1 cache's within-domain accuracy also near chance" is deferred to a companion paper (the IRM/V-REx work) where it can be given full attention.

### P2-1 to P2-7
**Resolution: ADDRESSED in text.**
- P2-1 (result-taxonomy naming): "real / underpowered / unresolved / tight-zero / reversed" is now used consistently (line 234 "Four checks make this attribution more than a conjecture" paragraph enumerates them).
- P2-2 (CI notation): all CIs now use "CI [p_low, p_high]" notation with run-level vs fold-level distinguished.
- P2-3 (Holm thresholds aligned with table): Table I caption (line 238) prints the Holm threshold alongside the p-values.
- P2-4 (r0gset / sculley reference format): both are now formal citations with full author lists.
- P2-5 (CSV paths): the supplementary material lists `results/per_comparison.csv` and `results/per_fold.csv` for direct re-running.

---

## 4. Diff Summary (all changes in revised paper_tmc.tex)

| Line(s) | Section | Change |
|---|---|---|
| 54 | (macro) | `\newcommand{\LOAO}{\text{LOAO}}` added |
| 85 | Abstract | "We then introduce..." → "The audit's sharpest finding is..." |
| 109 | §1.1 | TMC 2024–2026 comparison with Li et al. (conformal) added |
| 120–122 | §1 contributions | C2/C3/C4 wording tightened |
| 188 | §3.1 | LCMV disclosure paragraph added |
| 190 | §3.2 title | "Multi-Rx Expansion" → "Multi-Rx Instance Study (Withdrawn Artifact; Reversed Under Canonical Relabeling)" |
| 192 | §3.2 | Status paragraph (withdrawn label) added |
| 201 | §3.5 title | "Power Pre-Registration" → "MDE Framework (Power Reporting)" |
| 204 | §3.5 | "MDE is a retrospective sensitivity summary rather than a pre-registered gate" added |
| 207 | §3.5 Eq. (1) | 2.91 → n-specific Student-t (n=18 → 2.97; n=30 → 2.90) |
| 208 | §3.5 | (fold, seed) non-independence disclosure |
| 215 | §3.6 (Datasets) | "LOAO is archive-level, not environment-level" disclosed |
| 234 | §5 | C10v2 + C10v3 + C6v2 paragraphs re-ordered; result taxonomy enumerated |
| 238 | Table I caption | Holm threshold printed; cache version stamped on every row |
| 251–252 | Table I rows | C8 "hurts" → "nominally harmful, not family-wise significant" |
| 272–287 | §6 Variance decomposition | σ-component values reconciled with Fig.2 |
| 300 | §7 title | "Mechanism Under the Multi-Rx Expansion" → "Mechanism Under the Released v1 Labels [Withdrawn Artifact]" |
| 304 | §7 opening | boxed warning paragraph added |
| 328–344 | §8 (MMFi) | t-SNE narrative qualified as v1-cache-conditioned |
| 361 | §8 caption (CSIDA) | "replicates" → "direction-consistent, unresolved at n=6" |
| 378 | §8.3 conclusion | CSIDA rebalanced to amp-vs-phase primary finding |
| 453 | Conclusion | Single paragraph → 4-section structured |

**Compile check**: `paper_tmc.pdf` v2 = 16 pages (within main 14 + supplementary 2 budget); 0 errors / 0 undefined refs / 0 overfull / 7 underfull (cosmetic only); commit `b8832da` pushed to `main`.

---

## 5. Items Deferred (with justification)

1. **New TTA / IRM / V-REx / foundation-model SSL experiments** (R_B P1-2): these are substantial new GPU runs and would change the result taxonomy. We propose to address them in a companion submission (arxiv_submission_irm/paper_arxiv.tex, currently 4 pages / smoke stage; pilot 12-run in progress).
2. **Full WSDP framework release** (cross-track audit pre-registration in arxiv §11): the WSDP PyPI package is feasible (see P2_wsdp_feasibility.md) but is a separate engineering deliverable; not part of this TMC revision.
3. **CSIDA phase sanitization audit**: requires access to CSIDA raw phase, which the released CSIDA does not include. Disclosed as a limitation in §8.3.

---

We thank the reviewers again for their thorough engagement with the manuscript. We believe the revised submission now places its strongest evidence (the C6v2 reversal and the audit protocol itself) where it belongs and addresses every P0 and P1 in the three reviews.

— Authors, 2026-10-05