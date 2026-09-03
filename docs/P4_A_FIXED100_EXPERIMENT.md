# P4-A / P4-B / P4-C 固定样本配对实验（2026-08-25，本地 Qwen3-4B 代理）

## 组合实验结论（fixed-200，P4-A q2 + P4-C bridge，不叠加）

运行：`--evidence-need-retrieval --evidence-need-quota 2 --bridge-retrieval --bridge-quota 2`；产物 `.locomo/p4ac-combo-fixed200.json`。

| 配置 | Hit@1 | Hit@3 | Hit@10 | MRR | NewGold | OffMiss→Top10 | Hit1 +/− |
|---|---|---|---|---|---|---|---|
| off | 0.565 | 0.685 | 0.730 | 0.6292 | — | — | — |
| **P4-A q2 单独** | **0.575** | **0.710** | **0.740** | **0.6411** | 8 | **4** | **+2** |
| P4-C bridge 单独 | 0.565 | 0.685 | 0.740 | 0.6296 | 3 | 3 | 0 |
| combo (q2+bridge) | 0.560 | 0.705 | 0.735 | 0.6326 | **10** | 3 | **−1** |

- combo 的 recall 空间确实叠加（NewGold 10 = need 8 + bridge 5 去重），但 **Hit@1 反降**（0.560 < off 0.565）：双配额把 rerank pool 压缩到 26 个 P1 候选（30−2−2），更多 P1 正确候选被 displaced，reranker 在更紧的池中表现更差；OffMiss→Top10 从 q2 的 4 降到 3。
- **最终配置：P4-A q2 单独启用（默认关）**；P4-B REJECT、P4-C CONDITIONAL 但组合负收益，均不作为默认。三组件独立验证完毕。

## P4-C 结论（fixed-200，CONDITIONAL）

运行：`--structured-query-plan --max-questions 200 --include-question-diagnostics --bridge-retrieval --bridge-max-terms 3 --bridge-quota 2`（P4-A/B 关）；产物 `.locomo/p4c-bridge-fixed200.json`。实现：`app/storage.py`（`extract_bridge_terms` + second-pass 查询 + quota）、`scripts/evaluate_locomo_retrieval.py`（`--bridge-retrieval` 等）、`tests/test_p4c_bridge.py`（5 passed）。

| 配置 | Hit@1 | Hit@3 | Hit@10 | MRR | NewGoldInPool | BridgePromoted | OffMiss→Top10 | Hit1 +/− |
|---|---|---|---|---|---|---|---|---|
| off | 0.565 | 0.685 | 0.730 | 0.6292 | — | — | — | — |
| bridge | 0.565 | 0.685 | **0.740** | 0.6296 | **3** | **3** | **3** | +1/−1 |

- **Bridge second-pass 确实找回 P1 没有的 gold**：3 题 NewGold（由 bridge 通道带入 pool），3 个 off-miss 被拉回 Top-10；Hit@10 +0.010。
- 但 Hit@1 中性（+1 gained / −1 lost），增益小于 P4-A q2（NewGold 8 / OffMiss→Top10 4 / Hit@1 +0.01）。
- 触发面小（仅 `intent==multi_hop` 或 `needs>=2`），且确定性提取会带入对话常用大写词（"Shared"/"Thanks"）噪音；第二查询以 bridge OR + need AND 约束，噪音候选影响有限。
- 判定：**CONDITIONAL**——recall 假设成立、Hit@1 不降，作为可选组件保留（与 P4-A 互补：P4-A 打 fusion_miss，P4-C 打跨消息第二跳）；与 P4-A 的组合待测。

## P4-B 结论（fixed-200，REJECT）

运行：`--structured-query-plan --max-questions 200 --include-question-diagnostics --adjacent-turn-expansion`（P4-A 关）；产物 `.locomo/p4b-adj-fixed200.json`。

| 配置 | Hit@1 | Hit@3 | Hit@10 | MRR | miss | NewGoldInPool | AdjPromotedGold | Hit1 +/− |
|---|---|---|---|---|---|---|---|---|
| off | 0.565 | 0.685 | 0.730 | 0.6292 | 54 | — | — | — |
| adj | 0.565 | 0.685 | 0.730 | 0.6292 | 54 | 0 | 0 | +0/−0 |

- **P4-B（±1 same-session 邻居，seed 4 / quota 4）零增益**：所有指标与 off 完全一致，`AdjPromotedGold=0`——邻居通道没有把任何 gold 带进 pool，Hit@1 无翻转。
- 与 P4-0 审计一致：context 族通道找到 gold 的 case 是"窗口所属者 ID 正确"型（gold 自己的 prev+current+next 窗口匹配），不是"gold 出现在邻居窗口、返回邻居 ID"型——所以 ±1 邻居没有可补的洞。
- 判定：REJECT（本 fixed-200 样本）；无副作用（与 P4-A 可安全共存），但单独不提供 recall。

## Fixed-200 P4-A（配对，off vs q2 vs q6）

运行：`--structured-query-plan --max-questions 200 --include-question-diagnostics`；产物 `.locomo/p4a-{off,q2,q6}-fixed200.json`。

| 配置 | Hit@1 | Hit@3 | Hit@10 | MRR | Top10 miss | 桶分布 |
|---|---|---|---|---|---|---|
| off | 0.565 | 0.685 | 0.730 | 0.6292 | 54 | fusion 41 / channel 9 / reranker 4 |
| q2 | **0.575** | **0.710** | 0.740 | **0.6411** | 52 | fusion 39 / channel 9 / reranker 3 / quota_disp 1 |
| q6 | 0.570 | 0.705 | 0.745 | 0.6391 | 51 | fusion 39 / channel 9 / quota_disp 2 / reranker 1 |

| 配置 | NewGoldInPool | NeedPromotedGold | OffMiss→Pool | OffMiss→Top10 | cfTop30 recall | actualPool recall | Hit1 +/− |
|---|---|---|---|---|---|---|---|
| q2 | **8** | 8 | 7 | **4** | 150/200 | 153/200 | +6/−4（净+2） |
| q6 | 8 | 8 | 7 | 4 | 150/200 | 152/200 | +7/−6（净+1） |

Fixed-200 结论：
- **q2 是选定配置**：Hit@1 +0.010、Hit@3 +0.025、Hit@10 +0.010、MRR +0.012，miss −2；8 题 New Gold Introduced，其中 4 题 off-miss 被拉回 Top-10（占 54 个 off-miss 的 7.4%）。
- q6 的 lost 更多（10/154/167 为 q6 特有），配额副作用（quota_displacement 2 + 更多 displaced P1 候选）压过额外收益。
- Pre-rerank recall：cfTop30 不变（need 不进 P1 反事实，基线可对照），actualPool recall 0.75→0.765（+0.015）——未达 +0.05 的门槛，但 New Gold（8 题）+ OffMiss→Top10（4 题）是 recall 假设成立的直接证据。Hit@1 未降（+0.010），Search LLM 调用增量 = 0。
- lost（18/69/104/107）机制同 fixed-100 深挖：配额挤掉 P1 末尾候选（q10 类）或 reranker 对池内容变化的排序波动（q69 类）。

## Fixed-100（初筛，off vs q2/q4/q6）

运行：`--structured-query-plan --max-questions 100 --include-question-diagnostics`，P4-A off 基线复用 `.locomo/p4-p1-audit-fixed100.json`；三个 on 配置 `p4a-q{2,4,6}-fixed100.json`（`--evidence-need-retrieval --evidence-need-quota N`）。

## 聚合

| 配置 | Hit@1 | Hit@3 | Hit@10 | Top10 miss | 桶分布 |
|---|---|---|---|---|---|
| off | 0.54 | 0.68 | 0.70 | 30 | fusion 20 / channel 8 / reranker 2 |
| q2 | 0.55 | 0.69 | 0.70 | 30 | fusion 19 / channel 8 / reranker 2 / quota_disp 1 |
| q4 | 0.53 | 0.69 | 0.70 | 30 | fusion 19 / channel 8 / quota_disp 2 / reranker 1 |
| q6 | 0.56 | 0.67 | 0.71 | **29** | fusion 19 / channel 8 / quota_disp 2 |

## 核心指标（配对数，vs off）

| 配置 | NewGoldInPool | NeedPromotedGold | OffMiss→Pool | OffMiss→Top10 | cfTop30 recall | actualPool recall |
|---|---|---|---|---|---|---|
| q2 | 5 | 5 | 4 | 2 | 72/100 | 74/100 |
| q4 | 5 | 5 | 4 | 1 | 72/100 | 73/100 |
| q6 | 5 | 5 | 4 | 2 | 72/100 | 73/100 |

- **New Gold Introduced = 5 道题**：gold 进入 P4-A 的 rerank pool 但从未进过 P1 pool——**P4-A 真正增加了召回，不是重排已有候选**。
- `cfTop30 recall` 不变（need 不进 `p1_counterfactual`，P1 基线可对照）；`actualPool recall` +1~2。
- 三个配置的 NewGoldInPool 相同（5），配额只影响"哪些 need 候选进池"，不影响 need 通道本身的发现能力。

## Hit@1 翻转（配对）

| 配置 | gained | lost | 净 |
|---|---|---|---|
| q2 | 3（44,62,80） | 2（18,69） | +1 |
| q4 | 1（62） | 2（10,69） | -1 |
| q6 | 4（44,61,62,79） | 2（10,69） | +2 |

## 深挖（q6）

**Saved（P4-A 真实收益）：**
- q61（cat3 fusion_miss）：off 时 gold 完全没进 pool；q6 时 **need 通道找到 gold（need_3 命中）→ promoted 进 pool → reranker 排第 1**。这是最干净的"New Gold Introduced → 拉回"案例（关系/多跳题的 need 化查询直接命中原文）。
- q48（reranker_drop）：off 时 gold 在 pool 位置 18 被 rerank 排出；q6 时 need 通道重申后 reranker 排到第 9，拉回 Top-10。

**Lost（副作用机制）：**
- q10（cat2）：off 时 gold（mem_48）在 pool 位置 28 且被 rerank 排第 1；q6 时 **mem_48 被 need 配额直接挤出 pool**（出现在 `displaced_p1_for_need_ids`）→ miss。**配额挤掉 P1 末尾候选的直接伤害**。
- q69（cat2）：off 时 gold 在 pool 位置 3 排第 1；q6 时 gold 仍在 pool 位置 3，但 reranker 把它排到第 5——**非配额问题，是 pool 内容变化引起的 reranker 排序波动**。

## 结论

1. **P4-A recall hypothesis 成立**：evidence-need 独立通道确实找回 P1 从未召回的 gold（5 题 NewGoldInPool，其中 2 题 off-miss 被拉回 Top-10，q61 直接从 fusion_miss 变 hit@1）。
2. **配额有真实代价**：q6 的 6 个配额挤掉了 P1 Top30 末尾的候选，q10 的 gold（P1 第 28 名）被挤掉；q2 副作用最小（只 displaced 1 个）。q69 类波动提示 reranker 对 pool 内容敏感，收益与副作用都经"池内容变化"传导。
3. **配额选择**：100 题样本噪声大，q2（净 +1）与 q6（净 +2）都为正；q4 为负。需 fixed-200 配对确认；当前倾向 q2 作为安全默认（副作用最小），或按"need 候选仅在 gold 无其他通道覆盖时启用"设计降副作用。
