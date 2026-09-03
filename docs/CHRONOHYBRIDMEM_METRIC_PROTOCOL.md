# ChronoHybridMem 指标协议（Metric Protocol）

本文档定义本项目（证据检索系统）的指标口径、已实现/可重算/未实现状态，
并明确检索指标与端到端指标的边界。

## 0. 范围声明（重要）

ChronoHybridMem 当前是**证据检索（evidence retrieval）**系统：给定查询返回
排序的原始记忆（message）。因此：
- 报告的指标是**检索级指标**（Hit@K、Recall@K、MRR、nDCG、Evidence
  Recall@K）；
- **不**将检索指标冒充为端到端问答指标（F1 / BLEU / ROUGE /
  LLM-as-a-Judge / 对话质量）——本项目无端到端答案生成评估；
- 任何关于"最新状态 / stale / 时序"的论断必须有对应标注与协议，否则不报。

## 1. 定义

- **查询集**：1976 个可完成查询（offset 758 统一排除，理由见
  `EVALUATION_NEW_METHOD_1976.md`）。
- **gold evidence**：数据集 `qa.evidence` 对应消息。
- **命中**：查询返回列表 Top-K 中出现任一 gold evidence。
- **first_gold_rank**：最早命中的 gold evidence 的位置（1-based）；无则
  None。

### 指标公式

- Hit@K = 有任一 gold 出现在 Top-K 的查询比例
- Recall@K（查询级）= 同 Hit@K（本项目单查询检索，query-level）
- MRR = mean(1/first_gold_rank)，未命中记 0
- nDCG@10 = mean(1/log2(first_gold_rank+1))，gold 为单个二元相关
  （仅当 first_gold_rank ≤ 10 计分；多 gold 取最早）
- Evidence Recall@K = 所有 gold evidence 中被排在 Top-K 的比例
  （evidence-level，来自评估器逐条计数）

## 2. 实现状态

| 指标 | 状态 | 来源/备注 |
|---|---|---|
| Hit@1/3/5/10 | ✅ 已实现，可重算 | 每题 first_gold_rank 重算 |
| Recall@1/5/10（Q-level） | ✅ 已实现，可重算 | 同上 |
| MRR | ✅ 已实现，可重算 | 同上 |
| nDCG@10 | ✅ 已实现，可重算 | 同上 |
| Evidence Recall@1/3/10 | ✅ 已实现（各自证据池口径） | 评估器 chunk 聚合 |
| Evidence Recall@5 | ⚠️ 未实现 | 评估器只导出 1/3/10 深度；需改评估器重跑 |
| 同口径 evidence 逐条重算 | ⚠️ 未实现 | 需 evidence 级 rank；758 排除造成 ~6 条口径差 |
| Latest-state accuracy | ⛔ 未实现 | 缺状态-时间戳 gold 标注 |
| Stale evidence rate | ⛔ 未实现 | 同上 |
| 时间排序一致性 | ⛔ 未实现 | 同上 |
| per-query 延迟 / P50 / P95 | ⛔ 未实现 | 评估器未计时 |
| 内存/磁盘/吞吐/失败率 | ⛔ 未实现 | 需资源采样工具 |
| 端到端 F1/BLEU/LLM-Judge | ⛔ 不适用 | 本项目为证据检索 |

- ✅ = 已实现且可从逐题明细重算
- ⚠️ = 未实现（技术原因见备注），不填入数字
- ⛔ = 无标注/无接口，明确 NOT_IMPLEMENTED

## 3. 统计检验协议

- 方法：paired bootstrap（10,000 次重采样，固定 seed 20260826），
  NEW vs P1 同一 1976 题集。
- 报告：点估计 Δ 与 95% 分位区间。
- 声明规则：区间含 0 → 不声称统计显著（如 Hit@1 Δ +0.0071
  CI [-0.0051, +0.0197]）；区间不含 0 → 显著（MRR、Hit@10）。
- 无区间 → 不得声称"统计显著"。

## 4. 子集口径

- 多跳：gold evidence ≥ 2 的查询（422 问）
- 时间类：**仅** LoCoMo temporal category（category 2，321 问）Hit@1，
  因数据无独立时序状态标注

## 5. 禁止项

- 不得为解释失败而改变主实验设置或筛除样本；
- 不得在无 gold 标注处伪造指标；
- 不得以本地代理结果冒充官方 leaderboard 成绩；
- 不得提交数据集/密钥/本地模型/大型 chunk 到版本库。
