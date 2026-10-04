# P1/P2 修改验证报告（4 阶段评审统一意见落地）

**生成时间**：2026-10-04 17:15

## 1. 统一意见落地对照表

| 优先级 | 内容 | 实施位置 | 状态 |
|---|---|---|---|
| **🟥 P0-1** | 双轨投稿说明（arXiv + IMWUT） | §13 Broader Impact — 新增 "Dual-track submission plan" + "Community-feedback protocol" 两段 | ✅ |
| **🟥 P0-4** | Holm 分类（3 mutually exclusive） | §3.5 MDE Framework — 新增 "Holm classification: three mutually exclusive categories" paragraph | ✅ |
| **🟥 P0-5** | LODO → LOAO 命名统一 | §3.3 LODO Evaluation Protocol — 标题改为 "(Track dual sub-section)"，新增 \LOAO 宏 + 段首 terminology paragraph | ✅ |
| **🟧 P1-1** | hardware fingerprint audit | §10.7 新增 "Hardware Fingerprint Audit (NIC × CFO/SFO × SNR)"，含 Table 8（SNR/CFO/SFO/NIC）+ 3 段文字 | ✅ |
| **🟧 P1-3** | deployment readiness | §10.6 System Overhead — 新增 "Deployment readiness audit" paragraph，含 (a) calibration (b) power (c) memory (d) real-time | ✅ |
| **🟧 P1-5** | CRLB on cross-domain accuracy | §3.6 新增 "Cramér-Rao Lower Bound on Cross-Domain Accuracy"，含 Eq. (CRLB) + Wald interval 计算 | ✅ |
| **🟧 P1-6** | DDI summary | §10.5 新增 "Domain-Domain Independence (DDI) Summary"，含 Table 7 (5-row DDI proxy) + interpretation | ✅ |
| **🟧 P1-7** | user-LODO experiment | §11.7.5 新增 "User-LODO Experiment (Future Work)"，含 16-user / 80-run 样本量 + 0.5·σ 推导 | ✅ |
| **🟨 P2-1** | privacy section 加强 | §13 Broader Impact — 新增 "Demographic generalization limitation" 段（disability / age / BMI / assistive device） | ✅ |
| **🟨 P2-2** | software stack version pinning | §13 Broader Impact — 新增 Table 9 (Python/PyTorch/NumPy 等 12 项版本 pinning) | ✅ |
| **🟨 P2-4** | community feedback window | §13 Broader Impact — 新增 "Community-feedback protocol"（30-day window 三桶分类） | ✅ |

## 2. 编译验证

| Pass | 状态 | Errors | Undefined refs | Overfull/Underfull |
|---|---|---|---|---|
| Run 1 (首次) | ❌ | 1 (`\CSBI` typo) | n/a | n/a |
| Run 1b (typo 修复) | ✅ | 0 | n/a | n/a |
| Run 2 | ✅ | 0 | 1 (`app:crlb` 未定义) | 9 |
| Run 3 | ✅ | 0 | 1 (`app:crlb` 未定义) | 9 |
| Run 4 (app:crlb 修复) | ✅ | 0 | n/a | 9 |
| Run 5 (final) | ✅ | **0** | **0** | **9** (all < 60pt) |

## 3. PDF 关键字出现次数（前/后对比）

| 关键字 | 修改前 | 修改后 | 变化 |
|---|---|---|---|
| "Holm classification" | 0 | 1 | +1 |
| "archive-level" | 1 | 4 | +3 |
| "LOAO" | 0 | 3 | +3 |
| "Hardware Fingerprint" | 0 | 1 | +1 |
| "DDI" | 0 | 8 | +8 |
| "deployment readiness" | 0 | 1 | +1 |
| "CRLB" | 0 | 7 | +7 |
| "user-LODO" | 0 | 4 | +4 |
| "software stack" | 0 | 1 | +1 |
| "community-feedback" | 0 | 1 | +1 |
| "demographic generalization" | 0 | 1 | +1 |
| "fingerprint" | 0 | 3 | +3 |
| "30-day" | 0 | 3 | +3 |
| "Reproducibility-in-Sensing" | 0 | 1 | +1 |

## 4. 新增结构

论文主体结构（修改后）：
- §1 Introduction（+ Narrative roadmap paragraph）
- §2 Related Work
- §3 Methodology
  - §3.1 BVP Operator
  - §3.2 Multi-Rx **Instance Study (Reversed)** + status note
  - §3.3 LODO **Evaluation Protocol (Archive-Level on Widar3.0, Environment-Level on MMFi/CSIDA)** + \LOAO
  - §3.4 Paired Multi-Seed
  - §3.5 Statistical Foundation + **Holm classification** paragraph
  - §3.6 **CRLB on Cross-Domain Accuracy**（新）
  - §3.7 MDE Framework
- §4-§8（§4 Main Results, §5 Variance, §6 MDE, §7 Mechanism withdrawn, §8 Boundary）
- §9 Boundary
  - §9.1-§9.5（原 §10.1-§10.5）
  - §9.6 Updated Dataset-Dependent Optimality
  - §9.7 **DDI Summary**（新）+ Table 7
  - §9.8 Fifth Data Point (CSIDA)
  - §9.9 Dataset Breadth
  - §9.10 Recommendation
  - §9.11 Cross-Track Audit
- §10 Discussion
  - §10.1 v1 → Canonical Reversal
  - §10.2 Honest Revisions
  - §10.3 Implications
  - §10.4 Threats to Validity
  - §10.5 Reproducibility
  - §10.6 System Overhead + **Deployment readiness audit**（扩充）
  - §10.7 **Hardware Fingerprint Audit**（新）+ Table 8
- §11 Conclusion（已重写为 3 paragraph + bullets）
- §12 Broader Impact
  - Positive applications
  - Privacy concerns
  - **Demographic generalization limitation**（新）
  - Reproducibility + **Software stack pinning**（新）+ Table 9
  - **Dual-track submission plan**（新）
  - **Community-feedback protocol**（新）
  - + §11.7.5 **User-LODO Experiment (Future Work)**（新，注：此段在 §10 Discussion 末）
- Appendix（unchanged）

## 5. 关键叙事指标

| 指标 | 修改前 | 修改后 |
|---|---|---|
| 论文页数 | 50 | 58 |
| 引用 Holm 分类 | 隐含 | 显式 3 mutually exclusive categories |
| LODO vs LOAO 区分 | 仅 §3.3 段尾隐含 | §3.3 标题 + 段首 + 全文使用 |
| 双轨投稿声明 | 无 | §13 显式 arXiv + IMWUT |
| Community feedback | 无 | 30-day window + 三桶分类 |
| Hardware fingerprint | 无 | Table 8 + 3 段叙述 |
| Deployment readiness | latency only | latency + calibration + power + memory + real-time |
| CRLB | 无 | §3.6 完整短段 + Eq. (CRLB) |
| DDI | 1 个数字（MMFi v2: 617.85） | Table 7 完整 5 行 + interpretation |
| User-LODO future work | 无 | §11.7.5 显式规划 + 样本量计算 |
| Demographic generalization | 无 | 整段 limitation |
| Software pinning | 无 | Table 9 + 12 项版本 |

## 6. 备份

- 备份：`arxiv_submission/.pre_p0_diagnostic.bak`（修改前）
- 回滚命令：`cp arxiv_submission/.pre_p0_diagnostic.bak arxiv_submission/paper_arxiv.tex`

## 7. 待用户决策

1. **是否 git commit + push？** 修改尚未提交
2. **是否同步修改 paper_tmc.tex / paper_arxiv_b.tex？** 当前只改了 arxiv 主稿
3. **是否现在生成 v9.3 submission zip？** arXiv 上线前需要 anonymize + zip
4. **是否现在写 endorser 邮件？** arxiv endorser 邮件需要更新版本号 + abstract snippet

## 8. PDF 输出

- 文件：`arxiv_submission/paper_arxiv.pdf`
- 页数：58
- 大小：~2.6 MB
- 编译时间：< 30s/pass