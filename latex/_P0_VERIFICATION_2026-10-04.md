# P0 修改验证报告 (2026-10-04)

## 1. 应声断点修复对照表

| 断点 # | 严重性 | 修复内容 | 验证 |
|---|---|---|---|
| #1 Abstract ↔ 标题叙事相反 | 极严重 | Abstract 加 "The audit's sharpest lesson is dataset-conditioned optimality" + "labeled as a withdrawn v1-cache-conditioned artifact" | ✓ PDF page 1-2 |
| #2 §3.2 "Our Modification" | 极严重 | 改为 "Multi-Rx Instance Study (Reversed Under Canonical Relabeling)" + 段首 status note | ✓ 第 214 行 |
| #3 §4 Main Results 顺位 | 极严重 | 在 §4 第一段添加 "Headline preview: v1 → canonical reversal" 段，含 C6 vs C6v2 联合阅读提示 | ✓ 第 326 行前 |
| #4 §7 mechanism withdrawn 弱 | 高 | §7 标题加 "(Withdrawn Artifact)" + 段首 boxed 警告 | ✓ 第 397-403 行 |
| #5 §6 Discussion 顺序 | 高 | §6 重排：先 §6.1 v1→canonical reversal，再 §6.2 honest revisions，再 §6.3 implications | ✓ 第 575-588 行 |
| #6 §9 Conclusion 单段 | 中-高 | 重写为 3 个 paragraph：what established / sharpest lesson / what reframes | ✓ 第 632-650 行 |
| #7 §1 Intro 缺 roadmap | 中 | §1 末尾添加 "Narrative roadmap" 段描述 audit-and-revise arc | ✓ 第 144-152 行 |

## 2. P1 修改

| 修复 | 状态 |
|---|---|
| §6.4 + §6.5 合并为 "Honest Revisions" | ✓ §6.2 "Honest Revisions: v1 Cache → v2 Cache → Canonical Cache" |
| Fig. 5 caption 添加 withdrawn 标注 | ✓ 第 405 行 |

## 3. 编译验证

| Pass | 状态 | Errors | Undefined refs | Overfull/Underfull |
|---|---|---|---|---|
| Pass 1 | ✅ | 0 | n/a | n/a |
| Pass 2 | ✅ | 0 | n/a | n/a |
| Pass 3 | ✅ | 0 | 0 | 6 (all < 60pt, no content overflow) |

## 4. PDF 关键字出现次数（前/后对比）

| 关键字 | 修改前 | 修改后 | 变化 |
|---|---|---|---|
| "withdrawn" | 0 | 4 | +4 |
| "v1-cache" | 0 | 13 | +13 |
| "sharpest" | 1 | 6 | +5 |
| "instance study" | 0 | 3 | +3 |
| "audit's sharpest lesson" | 0 | 1 | +1 |

## 5. 关键叙事指标

修改后的论文结构在以下位置均出现 "v1-cache-conditioned artifact" 或 "withdrawn" 标识：

1. **Abstract** (page 1-2): "labeled as a withdrawn v1-cache-conditioned artifact"
2. **§1 Intro contributions** (line 141): C3 已包含 "best read as a v1-cache-conditioned artifact"
3. **§1 Narrative roadmap** (line 144-152): "marked withdrawn as a universal claim"
4. **§3.2** (line 214): "instance study... reverses under canonical relabeling"
5. **§4 Headline preview** (line 326 前): "Read C5/C6 jointly with C6v2"
6. **§4 Representational methods** (line 360): "Read C5/C6 jointly with C6v2"
7. **§6.1** (line 576): "v1 $\rightarrow$ Canonical Reversal (Sharpest Audit Finding)"
8. **§6.2** (line 583): "Honest Revisions: v1 Cache $\to$ v2 Cache $\to$ Canonical Cache"
9. **§7 标题** (line 397): "Withdrawn Artifact, Reversed Under Canonical Relabeling"
10. **§7 boxed warning** (line 400): "Status: Withdrawn as Universal Claim"
11. **§7 段首 restated caveat** (line 409): "Caveat (restated)"
12. **Fig. 5 caption** (line 405): "(4-panel, v1-cache-conditioned artifact, withdrawn as a universal claim)"
13. **§9 Conclusion** (line 632+): "the audit's sharpest lesson" + bullet list

## 6. 与 narrative_audit.md 对照

### 已修复（7/10）
- #1 Abstract ↔标题 ✓
- #2 §3.2 "Our Modification" ✓
- #3 §4 Main Results 顺位 ✓
- #4 §7 mechanism withdrawn 弱 ✓
- #5 §6 Discussion 顺序 ✓
- #6 §9 Conclusion 单段 ✓
- #7 §1 Intro 缺 roadmap ✓

### 已合并（1/10）
- #9 §6.4 + §6.5 合并为 §6.2 Honest Revisions ✓

### 标注改进（1/10）
- #10 Fig. 5 caption 添加 withdrawn 标注 ✓

### 未改（1/10）
- #8 §12 "Cross-Track Audit" 改名 — **保留原状**，因为这是 reproducibility audit paper 的 scope 声明，删改会引入新问题；保留但作为 honest acknowledgment 看待

## 7. 备份与回滚

- 备份位置：`arxiv_submission/.pre_p0_diagnostic.bak` (124 KB)
- 回滚命令：`cp arxiv_submission/.pre_p0_diagnostic.bak arxiv_submission/paper_arxiv.tex`

## 8. PDF 输出

- 文件：`arxiv_submission/paper_arxiv.pdf`
- 页数：50
- 大小：~2.5 MB
- 编译时间：< 30s/pass

## 9. 修改前后对照（段落级）

### Abstract 第二段（修改前 → 修改后）
**修改前**：
> "We then introduce a minimal signal-representation modification---expanding the BVP operator to retain the multi-Rx antenna dimension (comparison C6)---that yields +8.14 pp (Cohen's d = 1.71..."

**修改后**：
> "The audit's sharpest lesson is dataset-conditioned optimality: an apparent strong positive finding---the candidate multi-Rx representation modification of the BVP operator (comparison C6) reaching +8.14 pp (Cohen's d = 1.71..."

✓ **修复**：从 "introduce... yields" 的 contribution framing 改为 "candidate... reaching... reverses under" 的 audit-finding framing。

### §3.2 标题（修改前 → 修改后）
**修改前**：`\subsection{Multi-Rx Expansion (Our Modification)}`

**修改后**：`\subsection{Multi-Rx Instance Study (Reversed Under Canonical Relabeling)}` + `\label{sec:methodology-multirx}` + 段首 status note。

✓ **修复**：标题从 "Our Modification" 改为 "Instance Study"，并显式撤回声明。

### §4 顺位（修改前 → 修改后）
**修改前**：C6 (+8.14pp) 在 12 个 comparisons 列表中作为 headline bold row，C6v2 在表尾。

**修改后**：§4 第一段是 "Headline preview: v1 → canonical reversal" 段，明确告诉读者 "Read C6 jointly with C6v2"；然后才是原来的 systematic evaluation 段；最后 §4 的"Representational methods"段开头再次提醒 "Read C5/C6 jointly with C6v2"。

✓ **修复**：C6 与 C6v2 在读者认知中绑定，不再是 afterthought。

### §7 mechanism 标题（修改前 → 修改后）
**修改前**：`\section{Mechanism Under the Released v1 Labels (Reversed Under Canonical Relabeling)}`

**修改后**：`\section{Mechanism Under the Released v1 Labels (\textsc{Withdrawn Artifact, Reversed Under Canonical Relabeling})}` + boxed warning 在段首。

✓ **修复**：从括号注释变为 SC（小型大写）强调；段首 boxed warning 不可错过。

### §6 Discussion（修改前 → 修改后）
**修改前**：第一个 sub-section 是 §6.1 "Implications for the Field"，v1→v2 在 §6.5（最后一个）。

**修改后**：第一个 sub-section 是 §6.1 "The v1 $\rightarrow$ Canonical Reversal (Sharpest Audit Finding)"，第二个是 §6.2 "Honest Revisions: v1 Cache → v2 Cache → Canonical Cache"，第三个才是 §6.3 "Implications for the Field"。

✓ **修复**：v1→v2 reversal 作为 Discussion 开场，Implications 退居其后。

### §9 Conclusion（修改前 → 修改后）
**修改前**：单段 200+ 字，8 个独立事实堆叠。

**修改后**：3 个 paragraph：
1. "What the audit established" — 4 个 bullet
2. "The audit's sharpest lesson: dataset-conditioned optimality" — 1 段 5 句话
3. "What this reframes" — 1 段 3 句话

✓ **修复**：takeaway bullets 让读者快速抓到要点；"sharpest lesson" 独立成段作为科学记忆锚点。

## 10. SCI Q1 判断（保留并明确化）

论文未达 TMC Q1 水准（已在 `_NARRATIVE_AUDIT_2026-10-04.md` 中详细论证）。

**已明确化的 narrative 通过修改**：
- ✅ Abstract 现在不再把 multi-Rx 作为 method contribution 推销
- ✅ §3.2 标题与 §7 标题叙事方向一致（instance study + withdrawn artifact）
- ✅ §4 把 C6 与 C6v2 联合呈现
- ✅ §6 以 v1→v2 reversal 为讨论开篇
- ✅ §9 把 sharpest lesson 独立成段

**剩余 TMC scope 错位**（需用户决策）：
- 是否改投 arXiv + IMWUT reproducibility track？（根据 4 阶段评审统一意见）
- 是否保留 TMC 投稿但加 scope statement？
- 是否保留三份投稿包并明确分工（A=arXiv, B=spec, C 已合并 A）？

## 11. 下一步建议

1. 用户阅读修改后的 PDF，验证 narrative 一致性
2. 用户决策投稿策略（TMC / arXiv+IMWUT / 双轨）
4. 若用户满意当前修改，跑 git commit + push（不执行，等用户确认）