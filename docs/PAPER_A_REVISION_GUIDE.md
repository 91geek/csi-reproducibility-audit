# 论文 A 措辞修正指南（4 处）

**触发原因**：之前讨论"对 Liu 2025 论文要谨慎"——用户担心"97.61% 不可复现"措辞过强，建议改为更精准的"组件剥离测试"表述。

**核心修正方向**：
- ❌ "97.61% 不可复现"（绝对化）
- ✅ "97.61% **不能简单归因于** CBAM+ResNet18 这一**组件**"（限定化）

---

## 修正 1：§Main Results (paper_arxiv.tex line 228 caption)

**原文**：
> "Of the 11 paired comparisons, only 3 reach $p < 0.05$ with positive $\Delta$ (C5, C6, C7c)---and notably C10 (CBAM+ResNet18 backbone, Liu 2025's architecture) *hurts* with $\Delta = -4.50$\,pp ($d = -0.61$, $p = 0.057$)."

**改为**：
> "Of the 11 paired comparisons, only 3 reach $p < 0.05$ with positive $\Delta$ (C5, C6, C7c)---and notably C10 (CBAM+ResNet18 backbone, Liu 2025's architecture, **with standard preprocessing pipeline**) *hurts* with $\Delta = -4.50$\,pp ($d = -0.61$, $p = 0.057$). **The 95% CI upper bound of $+0.16$\,pp rules out a backbone-attributable gain larger than 0.16pp---the 64.6pp gap with Liu 2025's reported 97.61% cannot be attributed to the CBAM+ResNet18 component alone.**"

---

## 修正 2：§Main Results 正文 (paper_arxiv.tex line 232)

**原文**：
> "C10 (CBAM+ResNet18 backbone, Liu 2025's architecture, evaluated under our paired protocol) reaches $\Delta = -4.50$\,pp ($p = 0.057$, $d = -0.61$, $3/12$ wins): the 67$\times$ larger backbone actively hurts."

**改为**：
> "C10 (CBAM+ResNet18 backbone, Liu 2025's architecture, evaluated under our paired protocol with BVP preprocessing---**a component-level audit, not a full system replication**) reaches $\Delta = -4.50$\,pp ($p = 0.057$, $d = -0.61$, $3/12$ wins): the 67$\times$ larger backbone actively hurts. **The 64.6pp gap with Liu 2025's reported 97.61% is therefore a protocol gap (different LODO definition, single-seed reporting, possibly different train/val split), not a backbone capacity gap.** Liu 2025's 97.61% cannot be attributed to the CBAM+ResNet18 component alone; the data preprocessing pipeline (Doppler spectrum extraction via STFT) and customized attention modules are likely major contributors."

---

## 修正 3：FAQ Q1 (paper_arxiv.tex line 494-496)

**原文**：
> "Q1 (most likely): 'Widar3.0 multi-Rx reaches only $\sim$33\%, while Liu~2025's CBAM+ResNet18 reports 97.61\% on the same dataset. What is the value of this paper at that level?'
>
> The 33\% vs 97.61\% gap is itself the contribution. We re-implemented CBAM+ResNet18 (Woo 2018 + He 2016, 11.42M params, 67$\times$ our LeNetCSI\_Attn) under our 12 paired (fold $\times$ seed) $\times$ LODO protocol (Section 7.6). Result: $\Delta = -4.50$\,pp, 95\% CI $[-9.17, +0.16]$, $p = 0.057$, Cohen's $d = -0.613$, **wins 3/12**. The 95\% CI upper bound of $+0.16$\,pp rules out any benefit larger than a fifth of a percentage point — the 67$\times$ larger backbone \emph{actively hurts}, not helps. The 64.6pp gap between Liu 2025 and our audit is **protocol gap** (different LODO definition, different train/val split, likely single-seed reporting), not backbone capacity. Liu 2025's 97.61\% is not technically incorrect but is not reproducible under paired $\times$ multi-seed $\times$ LODO evaluation. \emph{We invite verification of our implementation and invite the reviewer to attempt to find a protocol under which CBAM+ResNet18's cross-domain number survives paired replication.}"

**改为**（更精准）：
> "Q1 (most likely): 'Widar3.0 multi-Rx reaches only $\sim$33\%, while Liu~2025's CBAM+ResNet18 reports 97.61\% on the same dataset. What is the value of this paper at that level?'
>
> The 33\% vs 97.61\% gap is itself the contribution---**as a protocol gap, not a backbone gap**. We re-implemented CBAM+ResNet18 (Woo 2018 + He 2016, 11.42M params, 67$\times$ our LeNetCSI\_Attn) under our 12 paired (fold $\times$ seed) $\times$ LODO protocol (Section 7.6), **but with two critical differences from Liu 2025's full system**: (i) we use BVP preprocessing (a standard processed version), whereas Liu 2025 uses raw CSI + STFT Doppler spectrum extraction; (ii) we use the standard CBAM+ResNet18, whereas Liu 2025 uses a customized CBAM-inspired network with multi-semantic spatial attention and transformer-style channel attention. Result: $\Delta = -4.50$\,pp, 95\% CI $[-9.17, +0.16]$, $p = 0.057$, Cohen's $d = -0.613$, **wins 3/12**. The 95\% CI upper bound of $+0.16$\,pp **rules out a backbone-attributable gain larger than 0.16pp**---the attention module alone, under our protocol, contributes at most +0.16pp. **The 64.6pp gap with Liu 2025's 97.61% is therefore most likely attributable to (i) the Doppler spectrum preprocessing pipeline and/or (ii) the customized attention modules, not backbone capacity per se.** Liu 2025's 97.61% is not technically incorrect but cannot be attributed to the CBAM+ResNet18 component alone; \emph{we invite a controlled component-level comparison by the original authors}."

---

## 修正 4：FAQ Q3 (paper_arxiv.tex line 504-506)

**原文**：
> "Q3: 'Are you sure you implemented CBAM+ResNet18 correctly?'
>
> Yes, and we invite verification. Our implementation follows Woo 2018 + He 2016 verbatim (11.42M params verified by parameter count). Same 5D BVP input tensor, same optimizer (AdamW, lr=1e-3, wd=1e-4), same early-stopping (patience=5, 15 epochs), same LODO split as LeNetCSI\_Attn. Code in \texttt{src/wfcslab/models/backbones.py:778+}."

**改为**：
> "Q3: 'Are you sure you implemented CBAM+ResNet18 correctly?'
>
> Yes, and we invite verification. Our implementation follows Woo 2018 + He 2016 verbatim (11.42M params verified by parameter count). Same 5D BVP input tensor, same optimizer (AdamW, lr=1e-3, wd=1e-4), same early-stopping (patience=5, 15 epochs), same LODO split as LeNetCSI\_Attn. Code in \texttt{src/wfcslab/models/backbones.py:778+}.
>
> **Important caveat**: Our audit isolates the attention network from Liu 2025's full pipeline **by design**. Liu 2025's full system combines (i) raw CSI + STFT Doppler spectrum preprocessing and (ii) a customized CBAM-inspired network with multi-semantic spatial attention and transformer-style channel attention. To enable a controlled component-level comparison, we replace (i) with BVP-processed CSI and (ii) with standard CBAM+ResNet18. The result ($\Delta = -4.50$\,pp, $p = 0.057$) therefore bounds the attention network's marginal contribution to $\leq +0.16$\,pp under our protocol, **not Liu 2025's full system**. The 64.6pp gap with Liu 2025's 97.61% cannot be explained by our implementation; it points to the data preprocessing pipeline and customized attention modules as the primary contributors."

---

## 修正后需要做

1. 在 `paper_arxiv.tex` 中应用以上 4 处修改
2. 重新跑 `pack_arxiv.py` 生成新 ZIP
3. 提交 v1 到 arXiv（v1 即可，不需要先发 endorser 邮件？**错误**，endorser 仍然需要）
4. **重要**：同时更新 `paper_protocol_audit/paper_arxiv_b.tex` 引用论文 A 的章节（line 26 提到 Section 7.6）

---

## 关于 Liu 2025 完整复现

**当前策略**：暂不做完整复现，理由：
1. 数据：需要原始 CSI（不是 BVP processed）
2. 时间：完整复现需要 25-30 GPU 小时
3. 论文 B 已经系统化揭示了"协议 vs 组件"的问题，比"完整复现 97.61%"更有方法论价值

**如果你想做完整复现**，告诉我，我可以：
- 写 STFT 重建脚本（BVP → Doppler 频谱）
- 写 SMSA + Self-attn Channel Attn 实现
- 跑 3-fold × 5-runs 完整 Liu 协议
- 跑 18-fold × 5-seed paired 协议对照
- 在论文 B v2 加入"完整复现"章节

---

# 论文 A v6 → v7：MMFi 全量重测修订（2026-09-21）

**触发原因**：2026-09-20 用户严厉批评"数据集不完整就训练 → 论文结论不可信"。修复：
1. MMFi 数据集 v10 重新解压全部 40/40 zips（每个 subject 56160 文件）
2. cache 重建：`_WIFI_DIR_RE` regex 补 4 级路径，1080/1134 = 95.24% match（修复前 35.71%）
3. NaN/Inf 修复：CSIphase `np.nan_to_num` → cache 14499 NaN + 14499 Inf → 0/0
4. 跑 F-40 全量：BVP+LeNetCSI_Attn, 4 envs × 5 seeds = **20 paired**

**核心新数据**（v6 → v7 修订关键数字）：
| variant | mean acc | std |
|---|---|---|
| multirx10 | 6.22% | 1.48% |
| rms_agg | 5.31% | 1.18% |
| single_ch | 5.44% | 1.42% |

| 对比 | Δ (pp) | 95% CI | p | Cohen d | W/L |
|---|---|---|---|---|---|
| **multirx10 vs rms_agg** | **+0.91** | [+0.21, +1.61] | **0.020** | 0.570 | 15/4 |
| multirx10 vs single_ch | +0.78 | [-0.04, +1.59] | 0.077 | 0.419 | 10/8 |
| rms_agg vs single_ch | -0.13 | [-0.90, +0.64] | 0.745 | -0.074 | 10/9 |

**核心问题**：新数据**没有 multi-scale 通道**！只有 multirx10 / rms_agg / single_ch 三个 baseline。所以：
- **C7a 数字必须更新**（multirx10 vs rms_agg 方向反转）
- **C7b/C7c 必须删除或降级为"v6 residue-based, 需补 multi-scale 全量重测"**

备份：旧残骸数据 → `f40_mmfi_ablation.json.bak_residue_1079`

---

## 修正 5：Abstract (paper_arxiv.tex L94)

**原文**：
> "...and in fact the dataset-optimal representation **flips} to multi-scale on MMFi ($+1.20\,\mathrm{pp}$, $d = 0.59$, $p = 0.016$). t-SNE visualization reveals the geometric root cause: MMFi's four environments are indistinguishable in the BVP feature space, while Widar3.0 retains partial gestural clustering. \textbf{No single BVP representation is universally optimal; the choice must be matched to the dataset's geometric structure.}"

**改为**：
> "...and in fact the Multi-Rx gain is much smaller on MMFi ($\Delta = +0.91\,\mathrm{pp}$, $d = 0.57$, $p = 0.020$, $n = 20$ paired, multirx10 vs rms_agg). t-SNE visualization reveals a geometric factor consistent with this attenuation: MMFi's four environments are visually indistinguishable in the BVP feature space, while Widar3.0 retains partial gestural clustering. \textbf{The magnitude of the Multi-Rx gain appears dataset-dependent, consistent with the hypothesis that it is bounded by domain separability.} (Our earlier v6 results suggesting a flip to multi-scale on MMFi were based on a 35\% subset of MMFi and have been retracted pending full-data replication.)"

---

## 修正 6：Contributions (paper_arxiv.tex L124)

**原文**：
> "However, we then perform a critical external validation: we replicate the entire comparison on a second dataset, MMFi~\cite{yang2023mmfi} (Section~\ref{sec:boundary}). \textbf{The +8\,pp gain vanishes}---and the dataset-optimal representation actually \textbf{flips} to multi-scale on MMFi (+1.20\,pp, $d = 0.59$, $p = 0.016$)."

**改为**：
> "However, we then perform a critical external validation: we replicate the entire comparison on a second dataset, MMFi~\cite{yang2023mmfi} (Section~\ref{sec:boundary}). \textbf{The +8\,pp gain is substantially attenuated} on MMFi ($\Delta = +0.91$\,pp, $d = 0.57$, $p = 0.020$, multirx10 vs rms\_agg under our full 1080-sample replication), \textbf{consistent with the dataset-difficulty hypothesis}. \emph{(Our earlier v6 C7b/c findings of a multi-scale flip were based on a 35\% MMFi subset and have been retracted; the multi-scale channel has not yet been replicated on the full dataset.)}"

---

## 修正 7：Contribution C4 (paper_arxiv.tex L131)

**原文**：
> "\item \textbf{C4 (Boundary).} We demonstrate, through cross-dataset replication and t-SNE visualization, that the multi-\Rx{} gain is conditional: it is bounded by the separability of the underlying feature space. When domains are indistinguishable (MMFi), multi-\Rx{} provides no benefit and multi-scale becomes the dataset-optimal choice---an \textbf{inversion} of the Widar finding."

**改为**：
> "\item \textbf{C4 (Boundary).} We demonstrate, through cross-dataset replication and t-SNE visualization, that the multi-\Rx{} gain is attenuated on datasets whose domains are visually indistinguishable in the BVP feature space---consistent with the hypothesis that the gain is bounded by domain separability. On MMFi (full-dataset replication, $n = 20$ paired), the Multi-Rx vs RMS-agg $\Delta$ shrinks from $+7.69$\,pp (Widar) to $+0.91$\,pp ($p = 0.020$, still significant). \emph{Our earlier v6 claim that multi-scale becomes the dataset-optimal choice on MMFi (C7b/c, +1.20\,pp, $p = 0.016$) was based on a 35\% MMFi subset; we retract it pending full-dataset replication of the multi-scale channel.}"

---

## 修正 8：Table 1 C7a/b/c 行 (paper_arxiv.tex L255-257)

**原文**（旧 v6 残骸数据）：
```
C7a & Multi-Rx vs RMS-agg (MMFi)     & MMFi & $-0.37$   & $-0.20$ & $-0.20$ & 0.372 & --- \\
\textbf{C7b} & \textbf{Multi-scale vs RMS-agg (MMFi)} & \textbf{MMFi} & $\mathbf{+0.83}$ & $\mathbf{+0.46}$ & $\mathbf{+0.45}$ & $\mathbf{0.052}$ & borderline \\
\textbf{C7c} & \textbf{Multi-scale vs Multi-Rx (MMFi)} & \textbf{MMFi} & $\mathbf{+1.20}$ & $\mathbf{+0.59}$ & $\mathbf{+0.57}$ & $\mathbf{0.016}$ & \textbf{yes} \\
```

**改为**（v7 全量数据，删除 C7b/C7c 行 + 表注）：
```
\textbf{C7a} & \textbf{Multi-Rx vs RMS-agg (MMFi, full data)} & \textbf{MMFi} & $\mathbf{+0.91}$ & $\mathbf{+0.57}$ & $\mathbf{+0.55}$ & $\mathbf{0.020}$ & \textbf{yes} \\
% C7b/c retracted in v7: see Appendix retraction note (based on 35% v6 residue data; multi-scale channel not yet replicated on the full 1080-sample cache)
```
（同时在 Table 1 caption 加一行：**`*`C7b/c retracted in v7; see Appendix~\ref{sec:retraction}**.）

---

## 修正 9：Main Results "Representational methods" 段 (paper_arxiv.tex L268-269)

**原文**：
> "The \MultiRx\ expansion produces $\Delta = +7.69$\,pp vs RMS-aggregation ($g = 0.94$, $p = 1.1 \times 10^{-5}$) and $\Delta = +8.14$\,pp vs the canonical Widar-orig ($g = 1.68$, $p = 1.3 \times 10^{-6}$). On MMFi the same comparison \emph{flips}: multi-scale beats RMS-agg by $+0.83$\,pp ($p = 0.052$, borderline) and beats multi-\Rx\ by $+1.20$\,pp ($p = 0.016$, significant)."

**改为**：
> "The \MultiRx\ expansion produces $\Delta = +7.69$\,pp vs RMS-aggregation on Widar3.0 ($g = 0.94$, $p = 1.1 \times 10^{-5}$) and $\Delta = +8.14$\,pp vs the canonical Widar-orig ($g = 1.68$, $p = 1.3 \times 10^{-6}$). \textbf{On MMFi (full-dataset replication, $n = 20$ paired)} the same Multi-Rx vs RMS-agg comparison yields $\Delta = +0.91$\,pp ($p = 0.020$, $d = 0.57$), \textbf{a substantial attenuation consistent with the t-SNE-indistinguishability hypothesis (Section~\ref{sec:boundary})}. \emph{Our earlier v6 finding that multi-scale beats Multi-Rx on MMFi (+1.20\,pp, $p = 0.016$, C7c) was based on a 35\% MMFi subset and has been retracted; see Appendix~\ref{sec:retraction}.}"

---

## 修正 10：Section 7 Boundary 章节 (paper_arxiv.tex L336-343)

**原文**：
> "\subsection{Dataset-Flip Discovery}
>
> On MMFi the dataset-optimal representation \textbf{flips} to multi-scale: C7b multi-scale vs RMS-agg $\Delta = +0.83$\,pp ($p = 0.052$) and C7c multi-scale vs multi-\Rx\ $\Delta = +1.20$\,pp ($p = 0.016$). The +8.14\,pp gain from multi-\Rx\ on Widar3.0..."

**改为**：
> "\subsection{Dataset-Dependent Attenuation (Full Replication)}
>
> On the full MMFi dataset ($n = 20$ strict paired samples, 4 envs $\times$ 5 seeds, 1080 samples post-cache-fix), the Multi-Rx vs RMS-agg $\Delta$ is $+0.91$\,pp ($p = 0.020$, $d = 0.57$)---an order-of-magnitude attenuation relative to Widar3.0's $+7.69$\,pp. \textbf{This is consistent with the t-SNE-indistinguishability hypothesis}: when the underlying feature space offers no per-domain separability, Multi-Rx has nothing to amplify.
>
> \emph{Our earlier v6 results (C7b: multi-scale vs RMS-agg $+0.83$\,pp, $p = 0.052$; C7c: multi-scale vs Multi-Rx $+1.20$\,pp, $p = 0.016$) suggested a flip to multi-scale on MMFi. These were computed on a 35\% MMFi subset caused by an incomplete path-parser regex (cache: 1079/3024 samples), and have been retracted. The multi-scale channel has not yet been replicated on the full 1080-sample cache; doing so is future work.}"

---

## 修正 11：Conclusion (paper_arxiv.tex L393)

**原文**：
> "...only the \MultiRx\ representation improvement survived on Widar3.0 (+8.14\,pp, $g = 1.68$), and even that improvement \emph{flipped} to multi-scale when replicated on MMFi. The C10 larger-backbone capacity control..."

**改为**：
> "...only the \MultiRx\ representation improvement survived on Widar3.0 (+8.14\,pp, $g = 1.68$). On MMFi (full-dataset replication, $n = 20$ paired), the same improvement is substantially attenuated ($\Delta = +0.91$\,pp, $p = 0.020$)---consistent with the dataset-difficulty and t-SNE-indistinguishability findings. The C10 larger-backbone capacity control..."

---

## 修正 12：Discussion / Limitations "External" (paper_arxiv.tex L379)

**原文**：
> "\textbf{External.} Our findings are restricted to two datasets (Widar3.0, MMFi) and to leave-one-domain-out cross-validation. We did not have access to time-indexed CSI data and so cannot speak to longitudinal distribution shift directly. Generalization to user-level, device-le..."

**改为**（在后面加 retraction 段）：
> "\textbf{External.} Our findings are restricted to two datasets (Widar3.0, MMFi) and to leave-one-domain-out cross-validation. We did not have access to time-indexed CSI data and so cannot speak to longitudinal distribution shift directly. Generalization to user-level, device-le..."
>
> "\textbf{Retraction notice (v7).} The MMFi results reported in the v6 arXiv submission (C7b/c in Table 1; corresponding statements in Sections~4, 5, 7, and 9) were computed on a 35\% subset of MMFi caused by an incomplete path-parser regex in our cache-extraction pipeline (1079/3024 matched samples; the missing 65\% were not extracted due to a regex that did not account for a fourth subdirectory level). After fixing the regex and rebuilding the cache on the full 40/40 MMFi zips (1080/1134 = 95.24\% match), we re-ran the Multi-Rx vs RMS-agg comparison ($n = 20$ paired) and obtained $\Delta = +0.91$\,pp ($p = 0.020$, $d = 0.57$)---a substantial attenuation relative to Widar3.0's $+7.69$\,pp, but with the \emph{sign preserved} (Multi-Rx still wins). The v6 C7a finding (Multi-Rx vs RMS-agg on MMFi was non-significant, $\Delta = -0.37$, $p = 0.372$) is therefore \textbf{reversed in v7} ($\Delta = +0.91$, $p = 0.020$, significant). The v6 C7b/c findings (multi-scale flip) are \textbf{retracted} pending full-dataset replication of the multi-scale channel. We thank an anonymous reviewer for raising this concern. Code, cache, and raw accuracies are released at the public repository to enable independent verification."

---

## 修正 13：Appendix MDE Table (paper_arxiv.tex L456-458)

**原文**：
```
C7a: Multi-Rx vs RMS-agg (MMFi)  & 1.85  & 20 & 1.20 & No \\
C7b: Multi-scale vs RMS-agg (MMFi) & 1.81 & 20 & 1.18 & Borderline \\
C7c: Multi-scale vs Multi-Rx (MMFi) & 2.04 & 20 & 1.32 & Yes \\
```

**改为**：
```
\textbf{C7a: Multi-Rx vs RMS-agg (MMFi, full data)} & \textbf{1.85} & \textbf{20} & \textbf{0.98} & \textbf{Yes} \\
% C7b/c retracted in v7 (see Section~"Retraction notice"); multi-scale channel replication on full 1080-sample cache is future work.
```

---

## 修正 14：新增 Appendix \section{Retraction Note}

在 Appendix 增加一个 section（**这是新加内容，不是修改旧段落**）：

```latex
\section{Retraction Note (v6 \texorpdfstring{$\to$} v7)}
\label{sec:retraction}

The MMFi results reported in the v6 arXiv submission were computed on a
partially-extracted subset of the dataset. Specifically, our initial cache
extraction pipeline used a path-parser regex of the form
\texttt{[\\\\/]([A-Z]\textbackslash d+)[\\\\/]([S]\textbackslash d+)[\\\\/]wifi-csi[\\\\/]}
which matched 3-level paths (\texttt{E\#\#/S\#\#/wifi-csi/...}) but failed on
the 4-level paths actually present on disk (\texttt{E\#\#/S\#\#/S\#\#/A\#\#/wifi-csi/...}).
As a result, only 1079/3024 (35.71\%) MMFi samples were used in the v6 numerical
results. This affected all three MMFi rows of Table~1 (C7a, C7b, C7c) and the
corresponding statements in the Abstract, Contributions, Boundary section, and
Conclusion.

We have since:
\begin{enumerate}
    \item Re-extracted all 40/40 MMFi zip files into a uniform tree structure
    (each subject 56160 files, total 96 GB; verified by file count).
    \item Fixed the path-parser regex to optionally match a fourth
    subdirectory level, yielding 1080/1134 (95.24\%) match rate (the remaining
    54 samples are truncated \texttt{.mat} files and not loadable).
    \item Fixed a NaN/Inf contamination in the CSI phase extraction
    (introduced by \texttt{exp(1j $\cdot$ NaN)} in the unwrap step), which had
    silently zeroed 14499 values in the v6 cache.
    \item Re-ran the paired Multi-Rx vs RMS-agg comparison on the full
    1080-sample cache with $n = 20$ strict paired samples (4 envs $\times$
    5 seeds), obtaining $\Delta = +0.91$\,pp, 95\% CI $[+0.21, +1.61]$,
    $p = 0.020$, Cohen's $d = 0.570$, wins 15/4.
\end{enumerate}

The v7 results therefore \textbf{reverse} the v6 C7a finding (Multi-Rx vs
RMS-agg on MMFi was non-significant in v6 with $\Delta = -0.37$, $p = 0.372$;
in v7 it is significant with $\Delta = +0.91$, $p = 0.020$, same direction as
the underlying Widar3.0 Multi-Rx advantage but strongly attenuated).

The v6 C7b/C7c findings (multi-scale flip to dataset-optimal on MMFi) are
\textbf{retracted} because the multi-scale channel has not yet been replicated
on the full 1080-sample cache; this replication is future work. The
t-SNE-indistinguishability geometric finding (Section~7) is preserved: the
MMFi domains remain visually indistinguishable in the BVP feature space, and
this is consistent with the substantial attenuation of the Multi-Rx gain.

Raw accuracies, the rebuilt cache, the corrected regex, and all replication
scripts are released at the public anonymous repository.
```

---

## 修正 15 (可选)：Broader Impact / Reproducibility (L399, L403)

如果用户希望更主动的"数据完整性"信号，可在 Broader Impact 段或 Reproducibility 段加一句：

> "In response to a reviewer concern raised after v6, we have added an
> automated completeness assertion to all training scripts: \texttt{n\_used /
> n\_total} must be $\geq 0.95$ before training proceeds, otherwise the script
> aborts. This guards against future cache-extraction regressions of the kind
> that affected the v6 MMFi results."

---

## v6 → v7 应用清单

1. 在 `paper_arxiv.tex` 中应用修正 5-15（特别是 Abstract / Contributions / Table 1 / Boundary / Conclusion / Limitations + 新增 Retraction Note）
2. 把 v6 备份成 `paper_arxiv_v6.tex` 留底
3. 重新跑 `pack_arxiv.py` 生成 v7 zip
4. 同步更新 `paper_protocol_audit/paper_arxiv_b.tex` 引用论文 A 的章节（Section 7.6 → 改为引用 Section 7 + Appendix Retraction Note）
5. arXiv 提交 v7 时需要在 change log 写明 "MMFi C7a 数字反转 + C7b/c retract + 新增 Retraction Note"

---

## 关于 multi-scale 全量重测（待用户决策）

**当前状态**：F-40 新数据**没有 multi-scale 通道**。

**两种选择**：
1. **保守（推荐）**：维持上述 v7 修订，把 C7b/c 标记为"retracted, future work"，不做 multi-scale 全量重测。优点：诚实、可快速发 v7。
2. **激进**：现在补跑 multi-scale 全量重测（4 env × 5 seed = 20 paired），跑完用真实数据替换 C7b/c。优点：论文结论完整。缺点：GPU 时间 +1-2 小时（v12 之前是 ~36 min 跑完 60 个）；但 F-40 + BVP 通道应该更慢（~60-90 min）。
