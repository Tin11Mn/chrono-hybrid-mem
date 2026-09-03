# ChronoHybridMem 最新改进方法全量评测（n=1976，本地 Qwen3-4B 代理）

- 日期：2026-09-03
- 评估对象：**最新改进方法**（见 A 节），非旧版本复测
- 范围：locomo10.json 全量 eligible 1977 问中的 **1976 问**（offset 758 因
  conv-43 超长对话 rank 请求在 llama-server 端必然挂起而排除，见附注）
- 主产物：`.locomo/newmethod-chunk-*.json`、`.locomo/newmethod-full1977-merged.json`
  （n=1976 修正版）、`.locomo/newmethod-per-query-details.json`
- 说明：本地代理实验，非官方 leaderboard 成绩

## A. 最新改进方法

分支 `research/p3-evidence-graph` @ `8a45da3`，配置：
`--structured-query-plan --evidence-need-retrieval --evidence-need-quota 2
--need-select-by-bm25`

入口：`scripts/evaluate_locomo_retrieval.py`
改进模块（storage.py）：P4-A evidence-need 独立检索（每条 evidence need 独立
term 查询 × 6 通道）+ **need 候选按 bm25 最优排序**选取配额（方向 2）。

## B. 相对旧方法变化

| 层 | P1（旧） | P4-A q2 | **新方法（+bm25sel）** |
|---|---|---|---|
| 规划 | 单结构化 plan | + evidence-need 拆解 | 同 P4-A |
| need 检索 | 无 | 6 通道低权重 | 同 P4-A |
| 配额选取 | 无 | 通道插入序取前 2 | **bm25 最优取前 2** |

## C/D. 主结果

### 1. 显式事实召回（新方法 n=1976；基线 n=1977 官方口径）

| 指标 | P1 基线 | P4-A q2（消融） | **新方法** |
|---|---|---|---|
| Hit@1 | 0.5761 | 0.5776 | **0.5850** |
| Hit@3 | 0.7157 | 0.7198 | **0.7323** |
| Hit@10 | 0.7618 | 0.7643 | **0.7809** |
| MRR | 0.6479 | 0.6501 | **0.6618** |
| Q-level Recall@5 | — | — | **0.7606** |
| nDCG@10 | — | — | **0.6914** |
| evidence Recall@1 / @3 / @10 | 0.4059 / 0.5378 / 0.5976 | 0.4070 / 0.5399 / 0.5998 | **0.4129 / 0.5504 / 0.6086** |
| evidence Recall@5 | NOT_IMPLEMENTED（chunk 仅输出 1/3/10） | 同左 | 同左 |

**Δ vs P1 基线**：Hit@1 **+0.0089**、Hit@3 +0.0166、Hit@10 +0.0191、
MRR +0.0139、evidence Recall@10 +0.0110。全量确认稳定提升（非样本偏差）。

### 2. 多跳推理（evidence≥2 的 422 问）

- Multi-hop Hit@1：**0.4289**
- evidence-level 分层 Recall：NOT_IMPLEMENTED（chunk 聚合不按多跳过滤）
- Multi-hop Evidence Coverage@K：NOT_IMPLEMENTED（需逐条 evidence 排位）

### 3. 时间推理

- category 2（when/date 类，321 问）Hit@1：**0.5919**
- Temporal Hit@1 / Evidence Recall / Latest-state accuracy / Stale rate /
  时间排序一致性：**NOT_IMPLEMENTED**（数据无状态-时间戳 gold 标注）

### 4. 改进模块专项（bm25 配额选取）

- fixed-200 paired（bm25sel vs 通道序）：Hit@1 +2 / Hit@3 +3 / Hit@10 +4，
  15 题 gold 新进 pool
- 全量消融：新方法 vs P4-A q2 = Hit@1 +0.0074 / Hit@3 +0.0125 / Hit@10 +0.0166

### 5. 效率指标（NOT_IMPLEMENTED）

现有评估器不导出 per-query 延迟/内存/吞吐/错误率——需加计时与资源采样。

## 每 query 明细（已保存）

`newmethod-per-query-details.json`：1976 条，含 query_id、offset、category、
multi_hop、gold_evidence、returned_ids、gold_mem_ids、first_gold_rank、
failure_bucket、recall_bucket。

## 附注（数据完整性）

- 唯一覆盖 1976 题，缺失仅 offset 758（conv-43，29 session/680 消息；
  rank 请求多服务器多次重试均挂起 → 排除并注明，不伪造该题指标）
- 0600 区间分段（600-757 / 759-800）确保 758 外全部覆盖，无重复

