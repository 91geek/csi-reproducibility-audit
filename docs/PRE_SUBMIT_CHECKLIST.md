# Pre-Submission Checklist — 24-Hour Hold Period

> **arXiv 在你提交后给你 24 小时 "preview 期"**，这段时间只有你自己能看到。
> **公示后任何错误都至少需要走一遍 replace 流程（24 小时 hold + 重新审核）。**
> **所以，提交前自己先过一遍这个 checklist，再点 "Submit"。**

---

## 第一阶段：提交前（10 分钟）

### Metadata
- [ ] Title 完全正确（拼写、空格、大小写）
- [ ] Authors 列表顺序正确，姓/名顺序符合 arXiv 习惯（First Last）
- [ ] Abstract 完全正确，字符数 ≤ 1920
- [ ] Comments 字段信息无误（目标期刊、repo URL、页数）
- [ ] Subjects 选对（主分类 cs.LG + cross-list eess.SP）
- [ ] License 选 CC BY 4.0
- [ ] ORCID 已添加（可选）

### LaTeX 编译
- [ ] 本地 `pdflatex paper_arxiv.tex` 编译成功（**无 error，无 fatal warning**）
- [ ] PDF 页数符合预期（≈ 18 页正文 + 4 附录 = 22 页）
- [ ] 所有 figure 都能正确显示（**没有 "Figure X not found" 或 missing image**）
- [ ] 所有 reference 都正确渲染（**没有 [?] 占位符**）
- [ ] 所有 cross-reference 都正确（Fig.~\ref{...}、Sec.~\ref{...}、Tab.~\ref{...}）

### 文件清单
- [ ] 主文件：paper_arxiv.tex
- [ ] 参考文献：paper_arxiv.bbl（如使用 inline \begin{thebibliography}，可省略）
- [ ] 图片：figs/ 目录下所有 .png
- [ ] 总文件数 < 100（arXiv 限制）
- [ ] 单个文件 < 50 MB（arXiv 限制）
- [ ] 总上传大小 < 100 MB（实际一般 < 10 MB）

### Endorsement
- [ ] endorser 已确认（邮件/链接确认）
- [ ] 或自动 endorser 系统已分配（投稿界面显示 "Pending endorsement"）

---

## 第二阶段：提交后 24h hold（30 分钟）

arXiv 会在 hold 期显示你的私有预览页。在公示前最后过一遍：

### 元数据（点击 "View" 后显示的）
- [ ] Title 拼写/标点完全正确
- [ ] Abstract 完整无截断（特别注意 \n 转 LaTeX 换行是否丢失）
- [ ] Authors 顺序正确
- [ ] Comments 不超过 600 字符

### PDF 渲染（点击 "PDF" 后看到的）
- [ ] 首页 Title 正确
- [ ] 首页 Authors 完整
- [ ] 首页 Abstract 完整（不是被截断的）
- [ ] 首页关键词（如有）正确

### 整篇 PDF 抽查（按页扫描）
- [ ] 第 1 页：Title + abstract + intro 段开头 OK
- [ ] 第 3 页：Fig. 0 pipeline 图正确显示（v4 多 Rx violin inset）
- [ ] 第 5 页：Fig. 1 paired forest plot 正确显示（C1-Q10 + C10 hurts 注解）
- [ ] 第 8 页：Table 1 12 paired comparisons 完整
- [ ] 第 12 页：Fig. 5 mechanism 4-panel 正确
- [ ] 第 16 页：Fig. 11 Backbone Fairness 正确（f48_backbone_fairness.png）
- [ ] 第 18-22 页：Appendix C/D/E 完整
- [ ] 第 22 页：reference list 完整（不应该是 [?] 占位符）

### 关键数字抽检（防止抄错）
- [ ] Abstract 中 +8.14pp 出现
- [ ] Abstract 中 4.82pp 出现（seed std）
- [ ] Abstract 中 CBAM+ResNet18 Δ=-4.50pp 出现
- [ ] §7.6 中 95% CI [-9.17, +0.16] 出现
- [ ] §7.6 中 d=-0.613 出现

### Endorser 状态
- [ ] "Endorsement" 状态显示为 "Approved"（不是 Pending）

---

## 第三阶段：公示前最后决策（30 分钟）

确认 hold 期检查无误后，决定：

### 决策 A：立即公示
- 点 "Finalize submission" → 论文进入公示队列
- 通常 24 小时内（多数次日）正式公开

### 决策 B：撤回修改
- 点 "Withdraw submission" → 论文**完全删除**
- arXiv ID 被回收，**作者提交记录不被公开**
- 可以从头开始（包括换 endorser、换标题、改 abstract）
- ⚠️ **Withdraw 是干净的**——不留痕迹

### 决策 C：请求延期
- arXiv 不提供延期机制
- 唯一方法是**先 withdraw，再用同一 ID 重投**

**建议**：对第一次投稿，**24 小时够用就立即公示**，不要拖延。

---

## 第四阶段：公示后立即执行

拿到 arXiv ID 后 1 小时内：

### 立刻发邮件
- [ ] 导师 / 合作者：告知 arXiv ID，请他们分享
- [ ] endorser：感谢 + 提供 arXiv ID 让他们 reference
- [ ] 实验室同学：请他们浏览/讨论

### 立刻更新外部信息
- [ ] GitHub README：加 arXiv badge
  ```
  [![arXiv](https://img.shields.io/badge/arXiv-2609.XXXXX-b31b1b.svg)](https://arxiv.org/abs/2609.XXXXX)
  ```
- [ ] 个人主页 / Google Scholar：更新 publication list
- [ ] Twitter/X：发第一条学术 thread（见 TWITTER_THREAD.txt）

### 立刻准备 TMC 投稿
- [ ] 把 arXiv ID 加到 TMC 论文 \thanks{} 中
- [ ] TMC 投稿 cover letter 引用 arXiv ID
- [ ] TMC 投稿"prior publication" 字段写上 arXiv URL

### 立刻开始监控
- [ ] 设置 arXiv 邮件通知（默认就有）
- [ ] 设置 Google Scholar Alert："WiFi CSI" / "BVP" / "multi-Rx"
- [ ] 设置 Semantic Scholar Alert（同上）
- [ ] 设置 Twitter/X 关注 arXiv 推送 bot（@arXiv_cs_LG 等）

---

## 紧急情况处理

### 公示后发现 typo 或 minor error
- 立即走 replace 流程（论文更新）
- 在 replacement reason 写"Minor typo correction in Section X"
- ⚠️ Replace 也是 24h hold + 审核，所以**公示后能改但要快**

### 公示后发现重大错误（数据错了、方法错了）
- 立即发邮件给 arXiv admin (help@arxiv.org)
- 申请 withdraw + replace
- 学术界对"主动撤稿"通常不会苛责，反而认为学术诚信

### 公示后发现版权问题（用了未授权的图/数据）
- 立即 withdraw + 联系 arXiv admin
- 重新打包后再投
- ⚠️ **这点必须提交前 100% 确认**：我们的 BVP cache 来自论文 A 自有实验，无版权问题
- ⚠️ Fig. 0/1/5/10 全部来自 `scripts/make_*` 自生成，无版权问题
- ⚠️ t-SNE 用的 Widar3.0/MMFi 数据集已说明是公开数据集引用，OK

---

## 提交后 30 天检查表

### 第 7 天检查
- [ ] Google Scholar 上能搜到自己的论文
- [ ] Semantic Scholar 已收录
- [ ] 下载量 > 50（基本阅读量）
- [ ] 没有收到 "Your paper was withdrawn" 警告

### 第 14 天检查
- [ ] 在 cs.LG / eess.SP 类目内出现
- [ ] 收到第一封同行反馈邮件（如果有）
- [ ] Google Scholar 显示 "Cited by X" 数字 > 0

### 第 30 天检查
- [ ] 下载量趋势（如果有几百下载，说明论文有影响力）
- [ ] 引用情况（期待开始被同领域工作 reference）
- [ ] 评论区 / Twitter 讨论（如有）
- [ ] GitHub repo 是否被 star（如果放了链接）

---

## FAQ（提交后最常见的 5 个问题）

### Q1: "我投了 arXiv 还能投 TMC 吗？"
A: 可以。arXiv 是 preprint，TMC 允许同期 arXiv。TMC 投稿时在 cover letter 写明 "A preprint of this manuscript is available at arXiv:XXXXX"。

### Q2: "什么时候 replace v2？"
A: 三种情况建议 replace：
1. 公示后发现 typo / 引用错误 / 图丢失
2. TMC R1 review 回来需要修改（建议改完先 replace arXiv，再投 TMC R1 revision）
3. TMC 接收后，把 IEEE Xplore 链接写进 comment，并补充 final 内容

### Q3: "arXiv 上显示的日期是哪个？"
A: arXiv 显示 v1 的公示日期。后续 v2/v3 替换不影响日期。这是"priority claim 的法律证据"。

### Q4: "我能在 arXiv 上加新作者吗？"
A: **不能**。arXiv 作者列表一旦提交永久固定。如需加作者必须走 "withdraw + resubmit with new authors"，会获得新 ID。⚠️ **强烈建议提交前 100% 确认作者列表**。

### Q5: "我能在 arXiv 上撤稿吗？"
A: arXiv 允许 withdraw，但**撤稿记录保留**（admin 可见但不公开）。学术界对"主动撤稿"通常 OK，但对"撤稿后不重投"的情况会怀疑。⚠️ **建议不撤**：如有错误，走 replace v2。

---

## 关键时间点预估

| 节点 | 估计时间 | 操作 |
|---|---|---|
| 提交到首次公示 | 24-48 小时（周末延后） | 仅作者可见 hold 期 |
| 公示到 Google Scholar 收录 | 1-3 天 | 自动 |
| 公示到 Semantic Scholar 收录 | 1-7 天 | 自动 |
| 公示到首次引用 | 1-14 天 | 同行主动引用或 pre-review 引用 |
| TMC 投稿窗口 | 公示后 1-2 周内 | 在 arXiv 优先 claim 后立即投 |
| TMC R1 review | 公示后 3-4 个月 | 期间 arXiv 可保持 v1 |
| TMC R1 revision 提交 | 公示后 3-4 个月 | 同步 arXiv v2 |

---

## 最后的最后

arXiv 投稿一旦完成，是**永久不可撤回**（除了 withdraw 留记录）。所以：

1. **作者列表**反复确认（找导师/合作者核对）
2. **标题/abstract**反复检查（让人帮读一遍）
3. **数据/图**反复验证（数对了吗？图对了吗？）
4. **代码/data 可用性**确认（GitHub repo 公开了吗？DOI 给了吗？）

**记住：投出去的 arXiv v1 是你学术身份的一部分。慢一点比错一点好。**

---

*Generated for Paper A (WiFi CSI Cross-Domain Gesture Recognition: A Reproducibility Audit and the Boundary of Multi-Rx Representation), 2026-09-08.*