# TMC 投稿操作指南 (ScholarOne Manuscripts)

**目标期刊**：IEEE Transactions on Mobile Computing (TMC)
**投稿系统**：https://mc.manuscriptcentral.com/tmc-ieee
**当前状态**：paper_tmc.tex 编译成功，10 页 PDF 生成（page budget 14 页内）

---

## Step 1：注册 ORCID（如果还没有）

1. 访问 https://orcid.org/register
2. 用邮箱注册（推荐个人学术邮箱，非机构邮箱）
3. ORCID 是 IEEE 作者必需项

## Step 2：注册 ScholarOne TMC 账号

1. 访问 https://mc.manuscriptcentral.com/tmc-ieee
2. 点 "Create an Account"（右上角）
3. 填：
   - First Name / Last Name
   - Email (建议和 ORCID 一致)
   - Username / Password
   - Country / Affiliation
4. 系统会发验证邮件，**点击邮件里的链接激活**

## Step 3：登录并提交

登录后首页 → "Author Dashboard" → "Start New Submission"

按 7 个步骤填：

### 3.1 Manuscript Type
- **Type**: Regular Paper
- 不选 Special Issue（CFP 没截止前不选）

### 3.2 Title
填论文标题（必须**完全匹配** paper_tmc.tex 中的 title）：
```
WiFi CSI Cross-Domain Gesture Recognition: A Reproducibility Audit and the Boundary of Multi-Rx Representation
```

### 3.3 Abstract
直接从 paper_tmc.tex 的 abstract 复制（去掉 \begin{abstract} 标签）

### 3.4 Keywords
从 paper_tmc.tex 的 IEEEkeywords 复制（5-7 个）：
```
WiFi sensing, channel state information (CSI), cross-domain gesture recognition, reproducibility, paired multi-seed evaluation, minimum detectable effect (MDE), antenna diversity, dataset-dependent optimality.
```

### 3.5 Authors and Affiliations
- 对每个 author 填：first name, last name, email, affiliation, ORCID
- 标记 **Corresponding Author**（一般是通讯作者）
- 注意：IEEEtran 的 `\IEEEauthorblockA` 对应 ScholarOne 的 affiliation 字段

### 3.6 Cover Letter
上传 `cover_letter_tmc.txt` 作为 PDF（用 Word 打开 → 另存为 PDF）
- 命名格式：`cover_letter_tmc.pdf`
- 内容见同目录文件

### 3.7 Classifications
填 IEEE TMC 的分类：
- **Primary**: Mobile and Ubiquitous Computing
- **Secondary**: Wireless Communications / Signal Processing

### 3.8 Suggested Reviewers
从 cover letter 里复制 5 位（Zheng Yang, Bin Guo, Daqing Zhang, Xiaolong Zheng, Jie Xiong）：
- 姓名 / 机构 / 邮箱（必须真实，否则 reviewer 邀请失败）

### 3.9 Opposed Reviewers（可选）
如果有利益冲突的人（例如合作者、竞争对手），填入避免被分配

### 3.10 Upload Files
按顺序上传：

| # | File | Description |
|---|---|---|
| 1 | `paper_tmc.pdf` | 主论文 PDF（10 页）|
| 2 | `paper_tmc.tex` | LaTeX 源文件（IEEE 要求）|
| 3 | `cover_letter_tmc.pdf` | 1 页 cover letter（转 PDF 后上传）|
| 4 | `fig0_pipeline.pdf` | Figure 1 |
| 5 | `fig1_main_boxplot.pdf` | Figure 2 |
| 6 | `fig2_variance_decomp.pdf` | Figure 3 |
| 7 | `fig3_mde_validation.pdf` | Figure 4 |
| 8 | `fig5_mechanism.pdf` | Figure 5 |
| 9 | `fig_tsne_widar.pdf` | Figure 6a |
| 10 | `fig_tsne_mmfi.pdf` | Figure 6b |
| 11 | `fig_b10_E4_PR_vs_gain.pdf` | Figure (Appendix B, optional) |

**注意**：IEEE ScholarOne 通常要求 PDF 单文件上传（10 MB 内）。如果超 10 MB，压缩 figs 或删除 fig_b10。

### 3.11 Supplementary Material（可选）
Q1-Q9 FAQ、Per-domain PR 图、Algorithm pseudocode、MDE per-comparison table —— 这些可以作为 supp.pdf 上传（不要超过 4 页）。

如果要做 supp.pdf，可以新建一个 supp.tex，结构：
```latex
\documentclass[journal,11pt]{IEEEtran}
\begin{document}
\title{Supplementary Material: ...}
\maketitle
% FAQ Q1-Q9
% Per-domain PR figure
% MDE per-comparison table
% Algorithm pseudocode
\end{document}
```

### 3.12 Confirmation & Submit
- 仔细检查所有字段
- 点 "Submit"
- 系统会显示 confirmation 邮件
- 截图保存 submission ID（TMC-YYYY-NNNN 格式）

## Step 4：版权协议

提交后 24 小时内，IEEE 会通过邮件发送 **IEEE Copyright Form**。
- 用 ORCID 登录 IEEE eCF system
- 选择 "Traditional Publication Agreement"（CC-BY 也可，但传统更稳）
- 签字

## Step 5：等待审稿

- **审稿周期**：3-6 个月
- **状态**：
  - Awaiting Reviewer Assignment
  - Under Review
  - Required Reviews Complete
  - Decision in Process
- **AE 决定**：
  - Accept
  - Minor Revision（直接同意，1-2 月内交修改版）
  - Major Revision（重写部分，重投）
  - Reject and Resubmit（多给一次机会）
  - Reject（结束，转投）

---

## 重要事项

1. **DOI 注册**：论文被接受后 IEEE 自动注册 DOI
2. **Open Access 选项**：TMC 提供 Open Access（APC \$2,450），一般不选（除非导师要求）
3. **Preprint Policy**：TMC 允许 arXiv preprint，**不冲突**
4. **ORCID 强制**：所有 author 必须有 ORCID
5. **Author Year**：每篇论文所有 author 的 ORCID 必须独立

---

## 故障排查

- **PDF 超 10 MB**：删除 fig_b10_E4_PR_vs_gain.pdf（这是 appendix 用的，可选）
- **编译失败**：参考 compile.log，在 paper_tmc.tex 检查错误
- **审稿周期超 6 个月**：写信给 EIC 询问（rare）

---

## 投稿包文件清单

完成所有步骤后，`arxiv_submission/` 目录应该包含：

```
paper_tmc.tex            # LaTeX 源文件
paper_tmc.pdf            # 编译后 PDF（10 页）
paper_tmc.aux/.log/.out  # 编译中间文件
cover_letter_tmc.txt     # cover letter 源文本
cover_letter_tmc.pdf     # cover letter（提交时上传）
figs/                    # 8 张 PDF figures
supp.tex (TODO)          # supplementary material（可选）
```

---

**预计时间线**：
- D+0：提交
- D+7：进入 review queue
- D+30：reviewer 邀请
- D+90：reviewer 完成（3 of 3）
- D+120：AE 决定（typical major revision）
- D+150：交修改版
- D+210：accept
- D+240：DOI 注册 + 在线发表

总周期约 **6-8 个月**。
