# ChronoHybridMem 最新改进方法全量评测（1976 个可完成查询，本地 Qwen3-4B 代理）

- 日期：2026-09-03（修订 2026-09-04）
- 评估对象：最新改进方法 = 分支 `research/p3-evidence-graph` @ `0e3ecba`，
  配置 `--structured-query-plan --evidence-need-retrieval --evidence-need-quota 2
  --need-select-by-bm25`（下称 **NEW**）
- 评测入口：`scripts/evaluate_locomo_retrieval.py`
- 范围：locomo10.json 全部 eligible 查询 **1977** 中的 **1976 个可完成查询**；
  **offset 758 统一排除**（conv-43 超长对话必然挂起），所有方法一致排除
- 说明：本地代理实验，非官方 leaderboard 成绩

## 方法口径

- 同一题集：P1 / P4-A q2 / NEW 三方法均只统计 **1976 个共同可完成查询**
  （三方题级交集再剔除 758），逐题配对可比
- P1 题级数据来自配对诊断重跑
  （`p5-diagnostics` 的 `p1-local-proxy-structured-diagnostics-chunk-*`），
  非历史聚合文件
- 每 query 明细：`.locomo/newmethod-per-query-details.json`（1976 条）

## 主结果（同一 1976 题集）

| 指标 | P1 | P4-A q2 | **NEW (P4-A+bm25)** |
|---|---:|---:|---:|
| Hit@1 | 0.5779 | 0.5779 | **0.5850** |
| Hit@3 | 0.7176 | 0.7201 | **0.7323** |
| Hit@10 | 0.7601 | 0.7642 | **0.7809** |
| MRR | 0.6497 | 0.6504 | **0.6618** |
| nDCG@10 | 0.6773 | 0.6787 | **0.6914** |
| Recall@1 | 0.5779 | 0.5779 | **0.5850** |
| Recall@5 | 0.7439 | 0.7485 | **0.7606** |
| Recall@10 | 0.7601 | 0.7642 | **0.7809** |
| Evidence Recall@10* | 0.5976 | 0.5998 | **0.6086** |

\* Evidence Recall 口径说明：P1/P4-A 的历史证据计数为 1977 题全量
（2806/2804 evidence items），NEW 为 1976 题（2800 items）；758 排除造成
约 6 条 evidence 口径差。同口径的 evidence-level 逐条重算未实现
（评估器只导出每题 best rank 与 1/3/10 深度聚合），见 NOT_IMPLEMENTED 节。

**相对 P1 的变化**（同一 1976 题集）：Hit@1 +0.0071、Hit@3 +0.0147、
Hit@10 +0.0208、MRR +0.0121、nDCG@10 +0.0141。

## Paired bootstrap 95% CI（NEW vs P1，同一 1976 题集）

10,000 次 paired 重采样，seed 20260826：

| 指标 | 点估计 Δ | 95% CI | 结论 |
|---|---:|---|---|
| Hit@1 | +0.0071 | [-0.0051, +0.0197] | **含 0，未达统计显著** |
| MRR | +0.0120 | [+0.0036, +0.0207] | 排除 0，显著 |
| Hit@10 | +0.0207 | [+0.0116, +0.0299] | 排除 0，显著 |

**如实声明**：Hit@1 的提升在 95% bootstrap 下包含 0，**不能声称 Hit@1
统计显著**；MRR 与 Hit@10 的提升显著（排除 0）。

## 子集

- 多跳子集（gold evidence ≥ 2，422 问）：Hit@1 **0.4289**
- LoCoMo temporal category（category 2，321 问）：Hit@1 **0.5919**
  （见论文材料：时间类当前只报告该 category Hit@1）

## 排除与完整性

- 唯一缺失 offset：758（conv-43，29 session / 680 消息；rank 请求在
  llama-server 端多服务器多次重试均挂起 → 统一排除，不伪造该题指标）
- 合并核验：唯一 offsets 1976、无重复、758 不在列
- 0600 区间分段（600-757 / 759-800）确保覆盖无重复

## NOT_IMPLEMENTED（未伪造，需补充）

1. **Evidence Recall@5**：评估器只导出 evidence 深度 1/3/10；需改评估器
   导出 5 后重跑全部方法。
2. **evidence-level 逐条同口径重算**（排除 758 后 P1/P4-A 的精确 evR）：
   需 evidence 级 rank 数据。
3. **时间推理全套**（Temporal Hit@1、Latest-state accuracy、Stale evidence
   rate、时间排序一致性）：数据无"状态-时间戳"gold 标注。
4. **效率指标**（per-query 延迟、P50/P95、内存/磁盘、吞吐、失败率）：
   评估器未计时，需加计时与资源采样。
5. **每 query 延迟 / 错误类型逐条**：同上，需评估器导出。

## 产物

- `.locomo/newmethod-chunk-*.json`（11 段，gitignored）
- `.locomo/newmethod-full1977-merged.json`（n=1976 修正版，gitignored）
- `.locomo/newmethod-per-query-details.json`（1976 条明细，gitignored）
- `.locomo/unified-metrics-1976.json`（三方法同题集指标 + bootstrap，gitignored）
- 本文件（docs/，提交）
