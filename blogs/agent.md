# agent.md

## 协作文档

本项目的主要协作文档为 Google Docs：

- [OPHIS 协作文档](https://docs.google.com/document/d/1qLSEY2FaVF9uqbyZNLjBCIp8cHN4bC_Q_O3xNBVBFuI/edit?tab=t.0)

文档涵盖项目定位、核心想法（Better closed loop、Scaling Observables）、
建议 benchmark、相关 ChatGPT 会话链接等。

## 作者

- Shaoyang Guo（本人，github: guoshaoyang）

## 文档编写原则

- **完整性**：任何提到的内容都要给出足够细节，能直接照做（如任务/基准要包含：
  怎么跑、怎么限制、单次 test 的 compute/时间消耗、可编辑范围、官方预算）。
- **可追溯**：所有数据、数字、配置要标注来源（论文原文 / config.json / 官方
  leaderboard / 代码实现），区分「官方原话」与「本文作者解读」。
- **raw 细节优先**：宁可直接引用原始接口、原始命令、原始配置，也不要用
  二手转述；raw 细节能让人判断是否可用。
- **避免误解**：bench 论文没提出的概念（如 observable/causal/闭环）不能写成
  是它们的主张，要明确标注「OPHIS 解读」。
- **单一产物**：每个主题只维护一个产物文件（如 `docs/survey_task.html`），
  不散落多个草稿；修改即同步线上。

## 工作树纪律

- 保持工作树干净：不要堆积临时文件、草稿、未使用的研究笔记。
- 新内容优先并入已有文件，而不是新建文件。
- 线上产物（GitHub Pages）与本地 `docs/` 保持同步，改完即推送。

