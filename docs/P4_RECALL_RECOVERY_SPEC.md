# P4-0：471 Recall Failure Audit 规格（召回恢复主线）

> 状态：2026-08-25 已实现 `--include-question-diagnostics` 导出与四桶分类器（`test_locomo_evaluation.py` + `test_locomo_p4_audit.py` 合计 12 passed）。下一步为固定 100 题本地 Qwen 代理审计运行。

> 本文档取代 `P4_EVIDENCE_ROLE_STATE_AUDIT_SPEC.md` 中"排序优先"的方向假设：P4 主线从"把 Top-10 内的正确证据排到第一"切换为"把 Top-10 外的 471 个 gold evidence 找回来"。角色/状态排序（Evidence Role / Current-Previous / Correction-Retraction / Authority）降为 P5 支线，在召回空间建立之后再进入。

## 前提与假设

1. P1 的本地 Qwen3-4B 完整 LoCoMo 代理结果是可用于**假设筛选**的开发信号，不是官方 `gpt-4o-mini` 或平台成绩。
2. P4-0 不修改 Add/Search 默认行为、不改变数据库模式、不生成答案，也不以隐藏数据集身份或答案选择策略。
3. 471 个 `recall_miss_top10` 失败未必全部是"检索器没找到"；在动手改检索之前，先用 trace 把失败原因拆开。
4. 任何带模型的运行均使用本地 loopback 代理并明确标注；本地数据不入库。

## 目标

把 471 个 Top-10 外的失败拆成四种互斥的召回失败原因，为 P4-A/B/C/D 的选择提供证据基础：

```text
① Channel Miss       任何 P1 通道（raw / raw_porter / fact / fact_porter /
                     context / context_porter / support_raw / support_fact /
                     entity_* / dense）都没找到 gold
② Fusion Miss        某通道找到 gold，但 RRF 把它挤出 P1 反事实 Top-30
③ Quota Displacement gold 已进 P1 反事实 Top-30，但被 graph/anchor/adjacent
                     配额挤出了实际 rerank pool（仅 P3 类配额通道开启时出现）
④ Reranker Drop      gold 已进 rerank pool（≤30），但被 rerank 排出最终 Top-10
```

判定顺序（互斥，取第一个命中的分支）：

```text
gold 在任一 p1 通道？           否 → channel_miss
gold 在 p1_counterfactual_top30？ 否 → fusion_miss
gold 在 rerank_pool？             否 → quota_displacement
                                      → reranker_drop
```

关键事实：`rerank_pool_ids ≠ p1_counterfactual_top30_ids`（配额通道开启时）。二者必须分开记录，否则会把"配额挤掉"误判成"reranker drop"，P4-A/B/C 的增益也会被错算。

## 已实现（本会话，代码 + 测试完成）

`scripts/evaluate_locomo_retrieval.py` 新增 `--include-question-diagnostics`（默认关闭；关闭时聚合输出、调用数、排序完全不变）：

- 逐题记录字段：`question_offset`、`category`、`first_gold_rank`、`failure_bucket`（`top1_hit` / `ranking_top3` / `ranking_top10` / `recall_miss_top10`）、`result_ids`、`gold_mem_ids`、`gold_channel_presence`（gold 在每个 P1 通道内的 1-based 位置）、`gold_counterfactual_positions`、`gold_pool_positions`、`recovered_by_sidecar`（gold 是否仅由 graph/anchor/adjacent 通道带来）、`plan`、`p1_counterfactual_top30_ids`、`rerank_pool_ids`、`final_ids`。
- `recall_bucket`：仅当 `max(top_ks) >= 10` 且 gold 不在最终 Top-10 时按上面的判定顺序计算。
- content↔mem 映射：`_load_mem_by_content` 从样本临时 DB 只读 `raw_messages`（纯派生数据，不开新检索、不改库）。
- 分类器：`_classify_recall_failure(gold_mem_ids, retrieval_trace)`；trace 缺失（`p1_channels` 为空）时返回 `None`（原因未知，不冒充 channel_miss）。
- 测试：`tests/test_locomo_p4_audit.py`（默认关闭回归、逐题记录、端到端 `channel_miss` / `reranker_drop`、纯函数四桶 + 空 trace）。**12 passed**（与既有 `test_locomo_evaluation.py` 合计）。

已知边界：

- 每个通道有 `candidate_limit = min(max(top_k*4, 50), 200)` 截断：gold 可能在通道内但排在截断线下，审计只看到"该通道记录列表内没有"。若审计发现大量 `channel_miss` 集中在个别通道，再考虑加诊断级扩展通道上限（默认关闭）。
- 无 Search 模型时 `reranker_drop` 是"pool→Top-K 截断"的结构语义（无重排发生）；带模型运行时才是真正的 reranker 行为。

## 命令

```powershell
# 固定 100 题逐题 P1 审计（本地代理）；本地数据不提交
D:\ANACONDA3\python.exe -m scripts.evaluate_locomo_retrieval `
  --dataset ..\chrono-hybrid-mem\.locomo\locomo10.json `
  --local-search-model-url http://127.0.0.1:8081/v1 `
  --local-search-model-name local `
  --structured-query-plan `
  --max-questions 100 `
  --include-question-diagnostics `
  --output .locomo\p4-p1-audit-fixed100.json

# 测试
D:\ANACONDA3\python.exe -m pytest tests\test_locomo_evaluation.py tests\test_locomo_p4_audit.py -q
```

## 后续 P4 路线（组件独立验证后再组合）

- **P4-A Evidence-Need Independent Retrieval**：**已实现（默认关闭）**。`evidence_need_retrieval / evidence_need_quota（默认 2）/ evidence_need_rrf_weight（默认 0.01）`。开启时每条 `evidence_needs` 独立 term 化查询，走 raw / raw_porter / fact / fact_porter / context / context_porter 六个通道；need 候选以低 RRF 权重参与融合（不进 `p1_counterfactual_top30_ids`，保持 P1 基线可对照），并通过 `reserved_need_ids` 配额保证进入 rerank pool（`MODEL_RERANK_LIMIT=30` 硬上限，base_budget = 30 − special_quota − need_quota）。trace 新增 `evidence_need_diagnostics / evidence_need_channels / evidence_need_union_ids / reserved_need_ids / promoted_need_ids / displaced_p1_for_need_ids`；diagnostics 新增 `gold_need_channel_presence`。测试：`tests/test_p4a_evidence_need.py`（5 passed：默认关闭回归、need 通道运行、配额保留、无 need 跳过、requires structured plan）。配额 2/4/6 小范围扫描待跑。
- **P4-B Neighbor Recovery**：机制已存在（`adjacent_turn_expansion`，默认关，same-session ±1，seed 4 / quota 4，`_adjacent_turn_candidates`）；工作 = 开启 + 配对验证，验证"Context 窗口命中但 source ID 错位"类失败。
- **P4-C Bridge Second-Pass**：仅 `intent == multi_hop` 或 `evidence_needs >= 2` 触发；从第一跳原始证据确定性提取 1–3 个 bridge terms（大写 span / speaker / Fact 文本），最大两轮；不引入 Add-time 关系抽取。
- **P4-D Query Relaxation**：仅当仍存在明显纯 lexical miss 时；有限变体（entity+relation / entity+expansion / 单 evidence need），不做大范围 query rewriting。

## 门控指标

- Pre-rerank Gold Recall@30（`p1_counterfactual_top30_ids` 直接可算，不用重跑 rerank）。
- Final Hit@10；Gold outside Top10 count；Newly Recovered Gold。
- **Recall Recovery Rate** = P1 原 Top10 miss 中被 P4 拉回 Top10 的比例。
- **New Gold Introduced** = P4 pool 出现但 `p1_union_ids` 从未出现的 gold（对应 `promoted_*_ids` 与 `p1_union_ids` 的差集）。
- Hit@1 不得下降超过 0.005（按 paired 逐题统计，不只全局差值）；Search LLM 调用增量 = 0；user leakage = 0；unsupported evidence = 0。

## 边界

- 始终：保持 P1 默认行为、数据集不入库、不增加 Search 调用、明确本地代理性质、审计结果可复算。
- 先询问：数据库模式、Add-time 提取结构、Prompt/API 合约、依赖、CI、默认排序或任何平台提交的改动。
- 禁止：根据金答案、数据集身份或人工标签在同一评测题上改变排序；把审计桶直接当作 P4 的有效性证明；把多个 P4 组件一起打开却无法归因。
