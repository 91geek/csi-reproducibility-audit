# TMC 12-Page Double-Column Compression Plan

**From**: `arxiv_submission/paper_arxiv.tex` v12 (34 pages, single-column article)
**To**: `paper_arxiv_tmc.tex` (12 pages main + 6 pages appendix, double-column IEEEtran)

**Page budget**: TMC allows 12 pages for double-column submissions with mandatory $220/page overlength charges beyond 12. We aim for ≤ 12 pages main text (excluding references and appendix), with appendix content free.

---

## Section Reorganization (34 → 18 pages main text)

### STAY IN MAIN TEXT (12 pages total)

| § in v12 | Title | TMC destination | Compress to | Action |
|---|---|---|---|---|
| §1 (3pp) | Introduction | §1 (2.5pp) | cut 0.5pp | Drop §1.3 (method preview), merge §1.4 (contributions) into §1 intro |
| §3 (5pp) | Methodology | §2 (3pp) | cut 2pp | Drop §3.1 BVP background (cite literature instead), condense §3.2, keep §3.5 Statistical Foundation verbatim (1pp), fold §3.4+§3.5+§3.6 into §2.2 |
| §4 (1pp) | Experimental Setup | §3 (0.8pp) | cut 0.2pp | Compact table; drop §4.1 dataset story, only mention "Widar3.0 (6 rooms) + MMFi (4 envs)" |
| §5 (3.5pp) | Main Results (Forest plot) | §4 (2.5pp) | cut 1pp | Keep Table 1 + key bullets; move Fig 1 forest plot to a single column-width figure |
| §9 (5pp) | Boundary + 4 data points + Recommendation | §5 (3.2pp) | cut 1.8pp | Drop §9.1-9.2 (dataset-flip + t-SNE — move to appendix or keep t-SNE only); keep §9.4 (4 data points table) + §9.6 (recommendation) as centerpieces |

### MOVE TO APPENDIX (16 pages → 6 pages appendix)

| § in v12 | Title | TMC appendix | Compress to |
|---|---|---|---|
| §2 | Related Work | Appendix A | 1.5pp (drop long narrative) |
| §6 (2pp) | Variance Decomposition | Appendix B | 1pp |
| §7 (2pp) | MDE Analysis | Appendix C | 1pp |
| §8 (3pp) | Mechanism | Appendix D | 1.5pp (keep only Fig 5 + Table 5 mechanism summary) |
| §10 (3pp) | Discussion + Threats + Reproducibility + System Overhead | Appendix E (1.5pp) | Keep §10.4 latency table (0.5pp); drop §10.2 long threats narrative |
| Existing appendix (8pp) | Figure index, FAQ, proofs | Appendix F (1pp) | Keep proofs only |

---

## Specific Compression Actions

### §1 Introduction (3pp → 2.5pp)

```diff
- §1.3 Method Preview (0.5pp)
+ Bullet list of C1-C5 contributions inline
```

### §2 Methodology (5pp → 3pp)

```diff
- §2.1 BVP Operator Background (0.8pp)  # keep only formula (1)
+ Cite Zheng 2019 BVP paper for background

- §2.2 Multi-Rx Expansion (0.6pp)  # keep
- §2.3 LODO Protocol (0.2pp)       # compress to 1 paragraph
- §2.4 Paired Multi-Seed (0.4pp)   # keep theorem 1 + corollary inline
- §2.5 MDE Framework (0.4pp)       # keep theorem 2
- §2.6 Statistical Foundation (1.0pp)  # keep theorems 1-3 + corollary 1
+ Total ~3pp
```

### §3 Experimental Setup (1pp → 0.8pp)

```diff
- §3.1 Datasets (0.4pp)  # drop 1 sentence per dataset
- §3.2 Backbone (0.2pp)  # keep
- §3.3 Methods Compared (0.3pp)  # keep table only
- §3.4 Statistical Pipeline (0.1pp)  # keep
+ Total ~0.8pp
```

### §4 Main Results (3.5pp → 2.5pp)

```diff
- Full Table 1 (11 rows, 1.5pp)  # keep
- Forest plot Fig 1 (1pp)         # keep, single column
- Paragraph analysis (1pp)        # cut to 0.5pp bullet list
+ Total ~2.5pp
```

### §5 Boundary (5pp → 3.2pp)

```diff
- §5.1 Dataset-Flip Discovery (0.5pp)  # move to appendix
- §5.2 t-SNE Geometric Explanation (0.5pp)  # keep Fig 7, drop text
- §5.3 Honest Limitation Time (0.5pp)  # move to appendix
- §5.4 Honest Caveat v1 (0.5pp)  # move to appendix
- §5.5 Real-Segment Reanalysis (1pp)  # keep
- §5.6 4 Data Points (1pp)  # keep verbatim (centerpiece)
- §5.7 Recommendation (0.5pp)  # keep verbatim
+ Total ~3.2pp
```

---

## Figures Layout (Double-Column)

| Fig | Original | TMC layout | Width |
|---|---|---|---|
| Fig 0 Pipeline | 1 column-wide | 2-column wide (top of §1) | \textwidth |
| Fig 1 Forest plot | 1 column-wide | 1 column (left of Table 1) | \columnwidth |
| Fig 2 Variance | 1 column-wide | Move to Appendix B | \columnwidth |
| Fig 3 MDE | 1 column-wide | Move to Appendix C | \columnwidth |
| Fig 5 Mechanism | 2 column-wide | Move to Appendix D | \textwidth |
| Fig 6 t-SNE | 2 column-wide | 1 column (in §5.2) | \columnwidth |
| Fig 7 4 data points | 1 column-wide | 1 column | \columnwidth |
| Latency table | new in §10.4 | Move to Appendix E | \columnwidth |

---

## Tables Layout (Double-Column)

| Table | TMC layout |
|---|---|
| Table 1 (11 paired) | 2-column wide (\textwidth) in §4 |
| Table 2 (4 data points) | 1 column (\columnwidth) in §5.5 |
| Table 3 (Latency) | 1 column in Appendix E |
| Table 4 (MDE per comparison) | 1 column in Appendix C |

---

## References Strategy

Current references: 33 bibitems. TMC allows ~30 main-text refs + unlimited appendix refs.

**Main text**: 22-25 most-cited references (CSI sensing classics, methodology papers, TMC neighbor work)
**Appendix**: 8-11 supplementary references (algorithm-specific, dataset papers, statistical foundations)

---

## Appendix Structure (6 pages)

- **Appendix A: Related Work** (1.5pp)
- **Appendix B: Variance Decomposition Details** (1pp; Fig 2 + text)
- **Appendix C: MDE Per Comparison Table** (1pp; Fig 3 + Table 4)
- **Appendix D: Mechanism Details** (1.5pp; Fig 5 + extended text)
- **Appendix E: System Overhead + Threats** (1pp; Table 3 + condensed threats)
- **Appendix F: Theorem Proofs** (0.5pp; condensed from 4 proofs to 1 consolidated proof outline)

---

## Estimated Page Count

| Section | Pages |
|---|---|
| Title + abstract + keywords | 0.5 |
| §1 Introduction | 2.5 |
| §2 Methodology (with §3.5 Statistical Foundation) | 3.0 |
| §3 Experimental Setup | 0.8 |
| §4 Main Results | 2.5 |
| §5 Boundary | 3.2 |
| §6 Conclusion + Broader Impact (condensed) | 1.0 |
| References (30) | 1.5 |
| **Main text total** | **15.0** |

Oops, that's 15pp — over budget. Need to cut further.

### Additional cuts (15 → 12)

- Cut §1.2 (related work preview paragraph) — 0.3pp
- Compress §2 (drop §2.3 LODO + §2.4 paired; combine to 1 paragraph) — 0.5pp
- Move Fig 7 t-SNE to appendix (saves 0.5pp)
- Compress Table 1 to single column (saves 0.5pp)
- Combine §6 Conclusion + Broader Impact — 0.5pp
- Drop appendix A Related Work entirely — 1.5pp saved

After cuts: 12.2pp main text ✓

---

## Action Items (Sequential)

1. **Today**: Create `paper_arxiv_tmc.tex` skeleton with IEEEtran.cls + double-column settings
2. **Day 1-2**: Copy §1-§3 from v12, compress per above plan
3. **Day 3**: Compress §4 + §5 from v12
4. **Day 4**: Move §6-§10 to appendix, compress
5. **Day 5**: Validate compile, check page count, adjust
6. **Day 6**: Send to advisor for review
7. **Day 7**: Submit to TMC ScholarOne with cover letter + CoI forms

---

## Auto-Compression Hints

Some mechanical replacements (run once globally):

```bash
# Remove all "Appendix " prefix from section headings in the main text
# (they will become appendix sections after \appendix)
sed -i 's/\\section{/\\section{/g' paper_arxiv_tmc.tex

# Tighten spacing
sed -i 's/\\begin{figure}\[!h\]/\\begin{figure}[!t]/g' paper_arxiv_tmc.tex

# Reduce float specifiers
sed -i 's/\\begin{table}\[!h\]/\\begin{table}[!t]/g' paper_arxiv_tmc.tex
```

---

## Final Word Count Targets

| Section | Target words | Current v12 |
|---|---|---|
| Abstract | 250 | 410 (tighten to 280) |
| §1 Introduction | 800 | 1200 |
| §2 Methodology | 950 | 1500 |
| §3 Experimental Setup | 250 | 320 |
| §4 Main Results | 750 | 1100 |
| §5 Boundary | 950 | 1500 |
| §6 Conclusion | 300 | 450 |
| **Total main text** | **4250** | **6480** |

Compression ratio: ~35% reduction in word count, expected ~35% page reduction (34 → ~22pp). With above additional cuts: 22 → 12pp.
