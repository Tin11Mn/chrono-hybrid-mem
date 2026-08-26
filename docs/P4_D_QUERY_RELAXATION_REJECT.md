# P4-D Query Relaxation：实现 + 审计否决记录

- 日期：2026-08-26
- 状态：**已实现（默认关闭）但基于数据否决（REJECT）**
- 产物：`.locomo/p4d-relax-fixed200.json`；分析脚本
  `.locomo/_p4d_analysis.py`、`.locomo/_p4d_trigger_check.py`、
  `.locomo/_p4d_diff.py`。

## 设计（spec P4-D）

"仅当仍存在明显纯 lexical miss 时；有限变体（entity+relation / entity+expansion
/ 单 evidence need），不做大范围 query rewriting。"

实现：`--query-relaxation`（默认 off，需 `--evidence-need-retrieval`）。触发
条件 = **所有 need 通道返回零命中**（纯 lexical miss 信号）；触发后跑一次有界
FTS5 前缀查询（`word*`，词源 = plan core_terms + entities + expansion_terms +
query 原文词，去停用词、长度≥3），走 raw / raw_porter 两个通道，低权重 0.01、
保留配额 2（可并入 `--sidecar-shared-quota` 共享池）。零 Search-LLM 调用增量。

## 审计证据（P4-D 无靶）

fixed-200 + 全量 1977 诊断分析：

1. **channel_miss 112 题中 76 题连 need 通道也 miss**，抽样 12 题全部是
   **推理型问题**：query 词与 gold 证据文本零词汇重叠——
   - "Caroline's identity" → gold "transgender stories…support"
   - "national park or theme park" → gold "camping trip…campfire"
   - "political leaning" → gold "religious conservatives…LGBTQ rights"
   - "How many children" → gold "my son…their brother"
2. P1 主查询已用 query 原文词 + core_terms + entities 做 OR 组合；任何词汇
   松弛变体都是 P1 查询的子集或重复，gold 与查询零重叠时 FTS 无法匹配。
3. porter 通道已覆盖大部分词形变化，前缀匹配的增量空间极小。

## fixed-200 实验结果

| run | Hit@1 | Hit@3 | Hit@10 | MRR | evR@10 |
|---|---|---|---|---|---|
| P4-A q2 | 0.5750 | 0.7100 | 0.7400 | 0.6411 | 0.5992 |
| P4-D relax | 0.5700 | 0.7000 | 0.7450 | 0.6377 | 0.6031 |

**relax 触发次数 = 0**（200 题 `relax_union_ids` 全空）：need 通道的 OR 宽查询
几乎从不为空，触发条件（全部 need 通道零命中）在数据上几乎不存在。真正的失败
模式是"need 通道有结果但 gold 不在其中"（推理型），不是"查询无结果"。

4 题的 rank 差异（offset 34/44/69/79）逐题核对 `relax_union_ids` 均为空 →
差异全部来自 LLM plan/rerank 运行随机性，与 relax 无关。P4-D 相对 P4-A 的
聚合差值是噪声，不是组件效果。

## 结论：REJECT

- 触发条件在真实数据上不触发（OR 宽查询从不为空）；
- 即使强行触发，推理型 miss 也无法被词汇松弛救回（零词汇重叠）；
- 保留实现（默认关闭）+ 测试作为可控消融，文档记录否决依据。

## 实现

- `app/storage.py`：`RELAX_RRF_WEIGHT_DEFAULT / RELAX_QUOTA_DEFAULT`；构造参数
  `query_relaxation / relax_rrf_weight / relax_quota`（校验：需 evidence_need_retrieval）；
  need union 为空时生成一次前缀查询（raw / raw_porter），trace 新增
  `relax_diagnostics / relax_channels / relax_union_ids / reserved_relax_ids /
  promoted_relax_ids / displaced_p1_for_relax_ids`；融合与配额分支接入（支持共享池）。
- `scripts/evaluate_locomo_retrieval.py`：`--query-relaxation / --relax-rrf-weight /
  --relax-quota` CLI + 校验 + diagnostics 字段。
- 测试：`tests/test_p4a_evidence_need.py` 新增 3 例（default-off 回归、前缀恢复、
  requires evidence_need_retrieval）。P4 套件 19 passed。
