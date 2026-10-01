# arXiv 第一次投稿 — 完整材料包

> 本目录包含论文 A（WiFi CSI 跨域手势识别可复现性审计）第一次投 arXiv 所需的全部材料。

---

## 🚀 快速开始（5 步流程）

### Step 1: 准备账号 + Endorser（先做！）
1. 注册 arxiv.org 账号（用机构邮箱）
2. 找 endorser：用 `ENDORSER_REQUEST.txt` 中的邮件模板
3. 等 endorser 确认（24-72h）

### Step 2: 检查 metadata
打开 `ARXIV_METADATA.txt`，复制字段到 arXiv 投稿表单

### Step 3: 检查 LaTeX 包
```
python arxiv_submission/pack_arxiv.py   # 重新打包（如有改动）
ls -la arxiv_submission/paper_arxiv.zip  # 应为 3-4 MB
```

### Step 4: 提交到 arXiv
- 在 arxiv.org/submit 上传 `paper_arxiv.zip`
- 填 metadata（title/authors/abstract/comments/subjects/license）
- 提交

### Step 5: 24h hold 期检查
- 按 `PRE_SUBMIT_CHECKLIST.md` 走一遍
- 点 "Finalize submission" → 公示

---

## 📁 文件清单

| 文件 | 大小 | 用途 |
|---|---|---|
| `README.md` | ~3 KB | 本文件，导航用 |
| `ARXIV_METADATA.txt` | ~7 KB | 投稿表单所有字段值 |
| `ENDORSER_REQUEST.txt` | ~8 KB | 3 封 endorser 请求邮件模板（正式/半正式/随意）+ 找人策略 |
| `pack_arxiv.py` | ~4 KB | 重新打包脚本（如正文改动需重跑） |
| `PACKAGING.md` | ~3 KB | 打包脚本使用说明 + arXiv 兼容性细节 |
| `paper_arxiv.tex` | ~52 KB | arXiv 兼容 LaTeX 主文件 |
| `paper_arxiv.bbl` | ~3 KB | 内联 bibliography |
| `figs/` | 21 张 PNG | 所有 figure（fig0/1/2/3/4/5/6/10/11 + B10 + t-SNE） |
| `paper_arxiv.zip` | ~3.5 MB | 完整打包，可直接上传 |
| `PRE_SUBMIT_CHECKLIST.md` | ~10 KB | 24h hold 期 + 公示后 + 30 天检查表 |
| `TWITTER_THREAD.txt` | ~10 KB | 中英双语 Twitter thread + 其它传播渠道策略 |

---

## 📋 投稿流程时间线（参考）

```
Day 0 (今天)
├── 10:00 收到这个 README
├── 10:30 注册 arxiv.org 账号
├── 11:00 发 endorser 请求邮件（同时发给 3-5 位备选）
└── 18:00 等 endorser 回复（如果没人回，发自动 endorser 申请）

Day 1 (明天)
├── 09:00 endorser 应已回复（大概率），如未回准备后补
├── 10:00 重新检查 ARXIV_METADATA.txt
├── 11:00 跑 pack_arxiv.py 生成 zip
├── 12:00 提交到 arxiv.org/submit
└── 14:00 (美东时间前) 提交后立即进入 hold 期

Day 2-3 (公示)
├── 美东时间 20:00 收到 arXiv 邮件：公示成功 + arXiv ID
├── 公示后 1h 内：发 endorser 感谢邮件 + 告知导师/同事
├── 公示后 1h 内：发第一条 Twitter thread（用 TWITTER_THREAD.txt）
└── 公示后 24h 内：准备 TMC 投稿材料（见下方）

Day 7-14
├── TMC 投稿包准备完毕
├── 投 TMC（cover letter 引用 arXiv ID）
└── 在 arXiv comment 中如有更新，replace v2

Day 90-180
├── TMC R1 review 回来
├── arXiv replace v2 同步更新
└── 投 TMC R1 revision
```

---

## ❓ 常见问题（FAQ）

### Q: endorser 没人回怎么办？
A: 等 3-5 个工作日，没回就换人。同时 arXiv 有 backup 自动 endorser 系统——投稿时填"没有 endorser"，系统会转给同分类活跃 contributor（24-72h 内回复，通过率 ~80%）。

### Q: 提交后发现 typo 怎么办？
A: hold 期可以 withdraw 修改。公示后走 replace 流程（也 24h hold）。**关键 typo 优先修，typo 不影响结论的可以等 replace**。

### Q: arXiv ID 怎么写到 TMC 论文里？
A: 在论文首页 \thanks{} 加 "A preprint of this work is available at arXiv:2609.XXXXX [YYYY]"。TMC 投稿 cover letter 也引用。

### Q: TMC 投稿前要等 arXiv 公示吗？
A: **不需要**。可以同时投：arXiv 提交后立刻投 TMC（投 TMC 时不写 arXiv ID，公示后用 replace 加进 TMC 论文 \thanks{}）。

### Q: 中文版 / 英文版 thread 都要发吗？
A: 英文发 Twitter/X（最大曝光）；中文发知乎/小红书/微信（中国学术圈）。建议**先发英文**，看反响后再决定中文。

### Q: 怎么验证投稿成功？
A: 三种方式：
1. arXiv 邮件通知（24h 内）
2. arxiv.org/abs/2609.XXXXX 页面可访问
3. Google Scholar 搜索论文标题（1-3 天后）

---

## 📞 紧急联系

| 问题类型 | 联系 |
|---|---|
| arXiv 系统问题 | help@arxiv.org |
| endorser 流程 | arxiv.org/help/endorsement |
| 版权问题 | arxiv.org/help/license |
| 投稿后撤稿 | help@arxiv.org |

---

## ⚖️ 版权声明

- 论文正文：CC BY 4.0（最大传播）
- 代码：MIT 或 Apache 2.0（建议）
- 数据：Widar3.0 原始数据由 Zheng 2019 提供，引用即可

---

## 📊 投稿后追踪指标

### 关键 KPI（Day 1 / Day 7 / Day 30）

| 指标 | Day 1 | Day 7 | Day 30 |
|---|---|---|---|
| arXiv ID 已分配 | ✓ | ✓ | ✓ |
| Google Scholar 收录 | — | ✓ | ✓ |
| 下载量 | ≥30 | ≥100 | ≥300 |
| 引用数 | 0 | 1-3 | 5-15 |
| Twitter 浏览量 | 500-2000 | — | — |

### 失败指标（需要补救）
- Day 7 下载量 < 30 → 检查 SEO / 标题 / Twitter 推广
- Day 30 引用数 = 0 → 检查 abstract 表述 / 找合作者推荐
- 收到"被 withdraw"邮件 → 立即处理

---

## 🎯 完成度 checklist

### 投稿前（建议今天完成）
- [ ] arxiv.org 账号注册
- [ ] endorser 邮件发送
- [ ] ARXIV_METADATA.txt 检查（特别是 abstract 字符数 ≤ 1920）
- [ ] pack_arxiv.py 跑一遍，确认 zip 大小 3-5 MB

### 投稿中（明天）
- [ ] 提交到 arxiv.org/submit
- [ ] 24h hold 期检查（按 PRE_SUBMIT_CHECKLIST.md）
- [ ] 公示后 1h 内：发感谢邮件 + Twitter thread

### 投稿后（公示后 7 天）
- [ ] Google Scholar 收录确认
- [ ] Twitter thread 互动回复
- [ ] 知乎/微信中文传播（如适用）
- [ ] 准备 TMC 投稿包

### 投稿后（公示后 30 天）
- [ ] 引用数追踪
- [ ] 下载量趋势分析
- [ ] GitHub repo 更新（如有）
- [ ] 准备 v2 replace（如果有 typo 或 reviewer 反馈）

---

*Generated for Paper A (WiFi CSI Cross-Domain Gesture Recognition: A Reproducibility Audit), 2026-09-08.*
*Companion docs: ../docs/叙事A_章节草稿_v2.md (full draft), ../docs/paper_ieee_tmc.tex (TMC version).*