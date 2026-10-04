# 论文叙事诊断报告 (2026-10-04)

## 1. 诊断方法

逐节精读 `paper_arxiv.tex` (878 行)，重点检查：
- **应声（narrative echo）**：abstract ↔ §7 mechanism ↔ §4 results ↔ §9 conclusion 之间的内部一致性
- **承诺兑现度**：intro 承诺的 4 contributions (§1 C1-C4) 与 §3 method、§4 result、§6 boundary、§9 conclusion 的兑现程度
- **SCI Q1/TMC 标准**：novel contribution、methodology rigor、empirical validation、writing clarity、reproducibility
- **Findings vs Contributions 区分**：哪些是 audit discovery、哪些是 method contribution

## 2. 应声断点清单（10 处，按严重性排序）

### 断点 #1【极严重】Abstract ↔ 标题叙事相反
**位置**：
- Abstract 第 102-104 行：`"...we then introduce a minimal signal-representation modification---expanding the BVP operator to retain the multi-Rx antenna dimension---that achieves +8.14 pp"`
- 标题：`"Dataset-Conditioned Optimality of the Multi-Rx Representation"`

**问题**：Abstract 把 +8.14pp 描述为 method **contribution**；标题说 optimality 是 **dataset-conditioned**（即 not universal）。
读者先看 abstract 会得出"multi-Rx is a good fix"；看标题会得出"multi-Rx 取决于 dataset"。
**两种叙事方向相反。**

### 断点 #2【极严重】§7 mechanism 已撤回，但 §3.2 仍称 "Our Modification"
**位置**：
- §3.2 (L211) 标题：`Multi-Rx Expansion (Our Modification)`
- §7 (L394) 已声明：`"this mechanism is conditional on the released label-domain coupling... v1-cache-conditioned artifact"`
- C6v2 行 (Table 1)：`-10.46 pp`，反转

**问题**：方法学章节用 "Our Modification" 的贡献性语言；机制分析章节用 "withdrawn artifact" 的撤回性语言。
读者无法把同一项 multi-Rx 改造在 §3 和 §7 之间一致理解。

### 断点 #3【极严重】§4 Main Results 顺位错置
**位置**：§4 (L323+)，bold 行依次为 `C5 (+7.69)`、`C6 (+8.14)`、`C7b (+1.84)`、`C8 (-7.90)`、`C9 (-10.17)`、`C10 (-4.61)`、`C10v2 (-5.75)`、`C10v3 (-0.28)`、`C6v2 (-10.46)`。

**问题**：C6 的 +8.14pp 是 v1-cache-conditioned，**紧接着的 C6v2**才是它的"真值"反转 (-10.46pp)。
但 §4 把 C6 当 headline 结论，C6v2 放在表尾。
读者扫到 C6 = +8.14pp 会以为论文找到了 universal fix；要读到表末才发现它 self-falsify。

### 断点 #4【高】§6 Discussion 的 "Implications" 子节不先谈 v1→v2
**位置**：§6.1 Implications (L558-580) 三个 implications 直接列出，没提 v1→v2 reversal；
v1→v2 逆转作为 §6.5 最后一个 sub-paragraph 才出现。

**问题**：论文最核心的 audit discovery 是 v1-cache-conditioned。
但 §6 用三个"implications"开场，v1→v2 作为 afterthought 出现。
读者顺着 §6 读下去会以为 v1 cache 没问题。

### 断点 #5【高】§9 Conclusion 单段落堆叠
**位置**：§9 (L611)，**单一段落 200+ 字**，包含 8 个独立事实：
- 12 paired comparisons + 3 re-evaluations
- C10v2 50k 不显著
- C10v3 chance-level
- C6v2 反转
- 5 seed × 6 fold 协议
- LODO scope caveat
- MDE 框架
- Holm correction

**问题**：没有 bullet / 没有加粗 takeaway / 没有明确的 "the audit's sharpest lesson"。
SCI Q1 结论要求 *"what should the reader remember"* — 此处无。

### 断点 #6【中-高】§1 Intro 缺少 narrative roadmap
**位置**：§1 (L143) Intro 结束在 contributions list，**没有 roadmap** 提示后面会发生什么。

**问题**：论文实际叙事弧是 audit-and-revise：
`v1 cache → C6 +8.14pp → §7 mechanism → §6 boundary discovers v1 was path-labeled → §6.5 v2 reanalysis → C6v2 reverses → Conclusion`

但 intro 没有这个 arc preview，读者不知道后面会自我证伪。

### 断点 #7【中】§3 Methodology 缺少 "what will be falsified" 警告
**位置**：§3 (L195-289)，五个 sub-section 依次为 BVP / Multi-Rx / LODO / Paired / MDE。
**问题**：§3 包含 "Multi-Rx Expansion (Our Modification)" 但没提此 modification 会在 §7 被撤回。
读者按 §3 训练模型会以为 multi-Rx 是论文的 deliverable。

### 断点 #8【中】§12 "Cross-Track Audit" 主动声明无 cross-track 数字
**位置**：§12 (L552+) "No cross-track numbers are claimed in this version"。
**问题**：作为 reproducibility audit paper，主动声明"不报告 cross-track"很奇怪。
Section title 与内容矛盾。

### 断点 #9【中】§6.4 (Threats to Validity) 与 §6.5 (v2 Reanalysis) 分裂
**位置**：§6.4 谈 v1 cache 是 path-labeled（threat）；§6.5 谈 v2 reanalysis（remedy）。
**问题**：同一议题的两半应在同一 subsection；现在分裂为两节。

### 断点 #10【低】Figure 5 caption 没明确 "withdrawn"
**位置**：Fig. 5 (L399) caption 描述 mechanism decomposition 但未注明此 figure 是 for a withdrawn claim。

## 3. SCI Q1 判断

### 期刊定位
论文目标：TMC Q1 (IEEE Transactions on Mobile Computing)。
TMC scope：**novel mobile computing systems + empirical evaluation + reproducibility**。

### 判断：**未达 TMC Q1 水准**

理由：

| 维度 | 当前论文 | TMC Q1 标准 | 评价 |
|---|---|---|---|
| **Novel system contribution** | BVP multi-Rx (已被 §7 撤回) | 必须 | ❌ 缺失 |
| **Novel methodology** | Paired-LODO + MDE 协议 | 高价值 | ✅ 有，但仅 audit protocol |
| **Empirical validation** | 12+3 paired comparisons, 5 seeds | 必须 | ✅ 充分 |
| **Statistical rigor** | Holm + MDE + Hedges | 期望 | ✅ 良好 |
| **Reproducibility** | Code + cache + per-seed accuracies | 期望 | ✅ 完整 |
| **Writing clarity** | Multi-cache confusion, v1↔v2↔canonical | 必须 | ❌ 多重缓存破坏论证 |
| **Narrative coherence** | 10 处断点 | 必须 | ❌ Abstract ↔ §7 ↔ §4 互冲 |
| **Findings vs contributions** | Finding 被包装成 contribution | 必须 | ❌ +8.14pp 是 finding 不是 contribution |
| **Constructive 价值** | 主要是 negative finding | 期望 | ⚠️ 对 reproducibility 价值高，但 constructive 弱 |
| **Scope 匹配** | TMC 偏 system，论文偏 audit | 期望 | ❌ 错位 |

### 替代 venue 建议（基于 4 阶段评审统一意见）
- **首选**：arXiv（preprint + community feedback） + IMWUT Reproducibility-in-Sensing Track
- **次选**：IEEE TSP Reproducibility Track
- **不推荐**：TMC Q1 main track（scope mismatch）

## 4. 改进方案（按优先级）

### P0（必须改 — 应声断裂）
- **P0-1** Abstract 重写为 audit-finding-not-method framing
- **P0-2** §3.2 改名 "Multi-Rx Expansion (Our Modification)" → "Multi-Rx Instance Study"
- **P0-3** §4 Main Results 在 C6 后立刻接 C6v2 reversal
- **P0-4** §7 mechanism 段首添加 prominent withdrawn warning
- **P0-5** §6 Discussion 把 v1→v2 提到 Implications 之前
- **P0-6** §9 Conclusion 改为 takeaway bullets + sharpest-lesson 段

### P1（应该改 — 叙事强化）
- **P1-1** §1 Intro 末尾添加 narrative roadmap（audit-and-revise arc）
- **P1-2** §6.4 + §6.5 合并为 "Honest Revisions: v1 → v2 → canonical"

### P3（可保留 — 标注改进）
- **P3-1** Fig. 5 caption 添加 "Withdrawn: v1-cache-conditioned" 标注
- **P3-2** §12 改名 "Honest Scope Acknowledgment"

## 5. 不改的（不修改即可）

- §3.1 BVP Background、§3.3 LODO Protocol、§3.4 Paired Statistical Protocol、§3.5 Statistical Foundation、§3.6 MDE Framework — 这些都是 methodology 章节，与 findings 无矛盾
- §4 Paired Forest Plot + Algorithmic methods / Representational methods — 数字准确
- §5 Variance Decomposition — 独立 finding
- §6.1 Implications（数字正确，只调顺序）
- §6.2 §6.3 latency / engineering — 与 audit 无冲突
- §8 Discussion — 待 Latex话题
- §10 Broader Impact — 与 audit 无冲突
- 附录、figures、tables — 保留

## 6. 修改后预期效果

- **应声一致性**：Abstract 诚实承诺 audit findings；§3 中性描述 instance study；§4 C6 与 C6v2 同时呈现；§7 明确 withdrawn；§9 突出 sharpest lesson
- **SCI Q1 定位**：放弃 TMC Q1 主张，转向 arXiv + reproducibility venue
- **Constructive 价值**：审计协议 (C1) + dataset-conditioned optimality (C4) 成为论文主线