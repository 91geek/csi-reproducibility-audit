# 47 引用核查报告（P2-7）

**生成日期**：2026-10-04
**范围**：arxiv_submission/paper_arxiv.tex (38 bibitems) + arxiv_submission/paper_tmc.tex (32) + arxiv_submission_b/paper_arxiv_b.tex (16) → 去重后 **48 个唯一引用 key**。
**方法**：提取每条 `\bibitem` 的元数据（key / 作者 / 标题 / venue / 年 / arXiv ID 或 DOI），对每个 key 做联网核验。

## 1. 核查结果汇总

| 类别 | 数量 | 说明 |
|---|---|---|
| ✅ 完全核实（48 个中） | 44 | 元数据与公开文献一致 |
| ⚠️ 需要修正（venue 错） | 2 | roy2007 (ICASSP→EUSIPCO), mcdermott2021 (Nature MI→Sci Transl Med) |
| ⚠️ 需要修正（venue 错） | 1 | yang2022mmfi (MobiSys→NeurIPS D&B) |
| ⚠️ 标题微调（不影响接受） | 2 | wang2022caution（多了"gait"）, yang2022efficientfi（"towards"→"Toward"） |
| ❌ 幻觉引用 | 0 | 48 个全部可在 arXiv / DOI / 会议页面查到 |
| ❌ 完全错误的关键事实 | 0 | — |

## 2. 已确认存在但需修正的具体问题

### ❌ roy2007（TMC 唯一引用）—— 双错（venue + title）

**当前（错）**：
```latex
\bibitem{roy2007} O. Roy and M. Vetterli, ``The effective rank of a noisy broadband mode,'' \emph{IEEE ICASSP}, 2007.
```

**核实**：实际为 EUSIPCO（European Signal Processing Conference）而非 ICASSP；标题应为 "The effective rank: A measure of effective dimensionality"。

**证据**：
- EPFL 官方 publication page: https://graphsearch.epfl.ch/publication/110188 — title="The Effective Rank: A Measure of Effective Dimensionality"，venue="2007 15th European Signal Processing Conference, 606-610, IEEE"
- Roy/Vetterli Google Scholar profile: 标题与 venue 一致

**修正为**：
```latex
\bibitem{roy2007} O. Roy and M. Vetterli, ``The effective rank: A measure of effective dimensionality,'' in \emph{Proc.\ 15th Eur.\ Signal Process.\ Conf.\ (EUSIPCO)}, 2007, pp.~606--610.
```

### ❌ mcdermott2021（仅 paper_b）—— venue 错（最严重）

**当前（错）**：
```latex
\bibitem{mcdermott2021}
M.~B.~A.~McDermott, S.~Wang, N.~Marin, et al.
\newblock Reproducibility in machine learning for health research: still a long way to go.
\newblock {\em Nature MI}, 3(4):e461--e467, 2021.
```

**核实**：发表在 **Science Translational Medicine**，不是 Nature MI。完整卷期页码也对不上。

**证据**：
- PubMed: PMID 33762434, DOI 10.1126/scitranslmed.abb1655, journal Sci Transl Med 13(586)
- Google Scholar: "MBA McDermott, S Wang, N Marin, ... - Science translational medicine, 13 (586), eabb1655, 2021"（404 引用）

**修正为**：
```latex
\bibitem{mcdermott2021}
M.~B.~A.~McDermott, S.~Wang, N.~Marin, et al.
\newblock Reproducibility in machine learning for health research: still a ways to go.
\newblock {\em Science Translational Medicine}, 13(586):eabb1655, 2021.
```

### ⚠️ yang2022mmfi（仅 paper_b）—— venue 错（MobiSys → NeurIPS D&B）

**当前（错）**：
```latex
\bibitem{yang2022mmfi}
J.~Yang, H.~Huang, Y.~Zhou, et al.
\newblock {mm-Fi}: Multi-modal non-intrusive 4D human sensing over commodity WiFi.
\newblock In {\em MobiSys}, 2022.
```

**核实**：MobiSys 2022 上没有 mm-Fi 这篇。实际发表于 NeurIPS 2023 Datasets & Benchmarks Track。

**证据**：
- arXiv: 2305.10345（v1 May 2023 提交）
- Google Scholar (Jianfei Yang profile): "MM-Fi: Multi-Modal Non-Intrusive 4D Human Dataset for Versatile Wireless Sensing J Yang, H Huang, Y Zhou, X Chen, Y Xu, S Yuan, H Zou, CX Lu, L Xie NeurIPS-23 Datasets and Benchmarks Track, 2023"

**修正为**：
```latex
\bibitem{yang2022mmfi}
J.~Yang, H.~Huang, Y.~Zhou, et al.
\newblock {MM-Fi}: Multi-modal non-intrusive 4D human dataset for versatile wireless sensing.
\newblock In {\em NeurIPS Datasets \& Benchmarks Track}, 2023.
```

## 3. 标题微调（不致命，建议改）

### ⚠️ wang2022caution（仅 arxiv）—— title 多了 "gait"

**当前**：``CAUTION: a robust WiFi-based human authentication system via few-shot open-set **gait** recognition''
**实际**：``CAUTION: A Robust WiFi-Based Human Authentication System via Few-Shot Open-Set **Recognition**''
（DOI 10.1109/JIOT.2022.3156099；论文本身是"人类认证"系统，"gait"不是 title 词——是认证的子任务）

### ⚠️ yang2022efficientfi（仅 arxiv）—— "towards" 应为 "Toward"

**当前**：``EfficientFi: **towards** large-scale lightweight WiFi sensing via CSI compression''
**实际**：``EfficientFi: **Toward** Large-Scale Lightweight WiFi Sensing via CSI Compression''
（IEEE Xplore + Google Scholar 6 个版本均用 "Toward"）

**建议改为**：
```latex
\bibitem{wang2022caution} D. Wang, J. Yang, W. Cui, L. Xie, and S. Sun, ``CAUTION: a robust WiFi-based human authentication system via few-shot open-set recognition,'' \emph{IEEE Internet of Things Journal}, 9(18):17323--17333, 2022.

\bibitem{yang2022efficientfi} J. Yang, X. Chen, H. Zou, D. Wang, Q. Xu, and L. Xie, ``EfficientFi: toward large-scale lightweight WiFi sensing via CSI compression,'' \emph{IEEE Internet of Things Journal}, 9(15):13086--13095, 2022.
```

## 4. arxiv preprint 类的 4 条（无正式 venue）—— **建议保留 arXiv ID，venue 改为 "arXiv preprint"**

| Key | arXiv ID | 当前 venue | 实际状态 | 建议 |
|---|---|---|---|---|
| kim2026wifijepa | 2607.11064 | `\emph{ECCV}` | **ECCV 2026 已接收**（eccv.ecva.net/virtual/2026/poster/5874） | 保留 ECCV 2026 |
| zhu2026amfm | 2602.11200 | 无（仅 arXiv ID） | arXiv preprint，无 conference 接收证据 | venue 改为 "arXiv preprint" |
| phuc2026cmambapose | 2606.13700 | 无 | arXiv preprint，无 conference 接收证据 | venue 改为 "arXiv preprint" |
| hou2026repos | 2607.02986 | 无 | arXiv preprint，无 conference 接收证据 | venue 改为 "arXiv preprint" |

**注**：zhu2026amfm / phuc2026cmambapose / hou2026repos 当前的 bibitem 写法没问题——venue 字段缺失只列 arXiv ID 是合理的（reviewer 可查 arXiv）。

## 5. 全部 48 个引用核查状态（key → status）

### A. arxiv/tmc 共有的 27 个（都核过）
| Key | Venue 声明 | 核实结果 |
|---|---|---|
| zheng2019widar | IEEE TPAMI 2019 | ✅ Real (TPAMI 2019 BVP 原文) |
| wang2022csi | IEEEXplore 2022 | ✅ Real (CSI 综述) |
| sheng2020 | IEEE INFOCOM 2020 | ✅ Real (WiFi-CSI INFOCOM) |
| zhang2021 | IEEE TMC 2021 | ✅ Real (AttnSense TMC) |
| ma2022 | IEEE IoT-J 2022 | ✅ Real |
| xie2023 | IEEE TMC 2023 | ✅ Real |
| sculley2018 | ICLR Workshop 2018 | ✅ Real (ICLR Workshop, not NeurIPS Workshop) |
| pineau2021 | JMLR 2021 | ✅ Real |
| henderson2018 | AAAI 2018 | ✅ Real |
| colas2018 | ICML Workshop 2018 | ✅ Real |
| dror2018 | ACL 2018 | ✅ Real (Hitchhiker's Guide) |
| arjovsky2019irm | arXiv:1907.02893 2019 | ✅ Real (IRM paper) |
| bouthillier2021 | MLSys 2021 | ✅ Real (MLSys 2021, not NeurIPS) |
| zhang2026sdp | arXiv:2601.08463 2026 | ✅ Real |
| guarino2026 | Computer Communications 249:108431 2026 | ✅ Real (作者 I. Guarino) |
| wang2026survey | IEEE COMST 28:5227-5266 2026 | ✅ Real |
| kim2026wifijepa | arXiv:2607.11064, ECCV 2026 | ✅ Real + ECCV 2026 confirmed |
| zhu2026amfm | arXiv:2602.11200 2026 | ✅ Real (Origin Research + HKU) |
| woo2018cbam | ECCV 2018 | ✅ Real |
| he2016resnet | IEEE CVPR 2016 | ✅ Real |
| zhang2017rexart | ICLR 2017 | ✅ Real (Zhang et al. "Rethinking generalization") |
| arpit2017 | ICML 2017 | ✅ Real (memorization) |
| yang2023sensefi | Patterns 4(3):100703 2023 | ✅ Real |
| sterzinger2026datta | WACV 2026 | ✅ Real (J. Strohmayer et al., DATTA) |
| cohen2026ruview | GitHub 2026 | ✅ Real (R. Cohen, not S. Cohen) |
| zhu2025csibench | NeurIPS D&B 2025 | ✅ Real (G. Zhu et al., CSI-Bench) |
| wu2021 | IEEE TMC 22(5):3062-3078, 2023 | ✅ Real (WiTraj, year is 2023) |
| yang2023mmfi | IEEE TMC 2023 | ✅ Real (MMFi) — 注：与 yang2022mmfi (NeurIPS D&B) 不同 |
| vassallo2025time | Internet of Things 32:101634 2025 | ✅ Real (A. Brunello et al.) |
| csida2022 | Mendeley Data V1 2022 | ✅ Real (DOI 10.17632/gyr6c4nbsc) |
| dasilva2026pss | arXiv:2609.26288 2026 | ✅ Real |
| li2026conformal | IEEE TMC 2026 | ✅ Real (DOI 10.1109/TMC.2026.3676932) |
| chen2026perceptalign | arXiv:2601.12252 2026 | ✅ Real (S. Jia et al.) |
| phuc2026cmambapose | arXiv:2606.13700 2026 | ✅ Real |
| hou2026repos | arXiv:2607.02986 2026 | ✅ Real |
| yousefi2017survey | IEEE Comm Mag 55(10):98-104, 2017 | ✅ Real (arXiv:1708.07129) |
| yang2022efficientfi | IEEE IoT-J 2022 | ✅ Real, 标题微调 "towards"→"Toward" |
| wang2022caution | IEEE IoT-J 2022 | ✅ Real, 标题多了 "gait" 词 |

### B. 仅 TMC（1 个）
| Key | Venue | 核实 |
|---|---|---|
| roy2007 | IEEE ICASSP 2007 | ❌ **错** — 实际是 EUSIPCO 2007, 标题也对错 |

### C. 仅 paper_b（16 个）
| Key | Venue | 核实 |
|---|---|---|
| wang2018glue | EMNLP BlackboxNLP 2018 | ✅ Real (A. Wang et al., GLUE) |
| mattson2020mlperf | MLSys 2020 | ✅ Real (MLPerf) |
| gulrajani2021domainbed | ICLR 2021 | ✅ Real (In Search of Lost Domain Generalization) |
| zheng2019widar3 | IEEE TPAMI 41(10):2471-2485, 2019 | ✅ Real (Widar3.0 TPAMI 2019) |
| yang2022mmfi | MobiSys 2022 | ❌ **错** — 实际 NeurIPS D&B 2023 |
| liu2025widar | arXiv:2512.04521 2025 | ✅ Real (R. Liu et al.) |
| cohen1988 | Lawrence Erlbaum 1988 | ✅ Real (Statistical Power Analysis 2nd ed.) |
| demsar2006 | JMLR 7:1-30 2006 | ✅ Real (Statistical comparisons of classifiers) |
| mcdermott2021 | Nature MI 3(4):e461-e467 2021 | ❌ **错** — 实际 Sci Transl Med 13(586):eabb1655 |
| woo2018cbam | ECCV 2018 | ✅ Real |
| he2016resnet | CVPR 2016 | ✅ Real |
| pineau2021 | JMLR 22:1-36 2021 | ✅ Real |
| zhang2026sdp | arXiv:2601.08463 2026 | ✅ Real |
| dasilva2026pss | arXiv:2609.26288 2026 | ✅ Real |
| li2026conformal | IEEE TMC 2026 | ✅ Real |
| chen2026perceptalign | arXiv:2601.12252 2026 | ✅ Real |

## 6. 修复优先级

| 优先级 | 引用 | 文件 | 修改 |
|---|---|---|---|
| **P0** | roy2007 | paper_tmc.tex L510 | 标题 + venue 修正 |
| **P0** | mcdermott2021 | paper_arxiv_b.tex L513-516 | venue + 卷期页修正 |
| **P0** | yang2022mmfi | paper_arxiv_b.tex L483-486 | venue + title (mm-Fi → MM-Fi) 修正 |
| P1 | wang2022caution | paper_arxiv.tex L875 | 去掉 "gait" 词 |
| P1 | yang2022efficientfi | paper_arxiv.tex L873 | "towards" → "Toward" |
| P2 | zhu2026amfm | paper_arxiv.tex L837 | venue 字段明确为 "arXiv preprint"（当前隐含，可不改） |
| P2 | phuc2026cmambapose | paper_arxiv.tex L866 | 同上 |
| P2 | hou2026repos | paper_arxiv.tex L867 | 同上 |

## 7. 教训

1. **AI 改写 venue 时会用错邻居会议**——例如 "ICASSP" 与 "EUSIPCO" 都是 IEEE 旗下 signal processing 旗下载体，AI 容易写错。
2. **AI 改写 title 时容易塞"关键词"**——例如把 "gait recognition" 这种论文 subject 词塞进 title，或 "towards/toward" 这种英美写法混用。
3. **AI 改写 venue 时常记错卷期页码**——例如把 Science Translational Medicine 13(586):eabb1655 错记成 Nature MI 3(4):e461-e467（两者都是医学 + 顶刊 + 临床相关，容易混淆）。
4. **MM-Fi vs MMFi**：同名/相近名数据集极易混淆（MM-Fi 由 Yang et al. 于 NeurIPS 2023 发布，MMFi 是 Widar3.0 团队的 TMC 数据集）。
