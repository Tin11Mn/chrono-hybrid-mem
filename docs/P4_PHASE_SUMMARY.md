# P4 Recall Recovery Cascade：阶段总结（2026-08-26）

> 状态：P4-0/A/B/C/D + 共享池配额修复全部落地，组件独立验证完毕，决策已定。
> 本文档为阶段收尾，汇总决策链、指标、教训与后续入口。

## 一、目标回顾

P4 主线（`research/p3-evidence-graph`，提交 `9b4e95f..e46e8d6`）：
把 Top-10 外的召回失败找回来——"先让正确证据进来，再讨论怎么把它排第一"。
基线 = P1 全量 1977 问（本地 Qwen3-4B 代理）：Hit@1 0.5761 / Hit@3 0.7157 /
Hit@10 0.7618 / MRR 0.6479 / evidence Recall@10 0.5976。

门控：各组件默认关闭、独立验证；零 Search-LLM 调用增量；Hit@1 不降 >0.005。

## 二、决策链

| 组件 | 机制 | 结论 | 关键证据 |
|---|---|---|---|
| **P4-0 审计** | trace 导出 + 四桶分类器 | 工具，长期保留 | 471 Top-10 miss 分解：fusion_miss / channel_miss / reranker_drop / quota_displacement |
| **P4-A** evidence-need 独立检索 | 每条 evidence need 独立 term 查询，6 通道低权重 + 保留配额 | **KEEP → 新基线** | fixed-200 Hit@1 +0.010；**全量 1977 Hit@1 +0.0015**（0.5776）、Hit@3 +0.0041、Hit@10 +0.0025、MRR +0.0022；8 题 off-miss→Top10、4 题→Top1 |
| **P4-B** ±1 邻居恢复 | same-session 相邻消息 | **REJECT** | fixed-200 零增益，且与 P4-A 组合稀释（combo Hit@1 0.560 < off 0.565） |
| **P4-C** bridge 第二遍检索 | 多跳计划确定性提取 bridge terms 二次查询 | **CONDITIONAL**（需共享池） | fixed-200 Hit@10 +0.010、NewGold 3、OffMiss→Top10 3；单独 Hit@1 中性 |
| **P4-D** 查询松弛 | 纯 lexical miss 时 FTS5 前缀查询 | **REJECT** | 全量审计：channel_miss 无词汇靶（推理型零词汇重叠）；fixed-200 触发率 0 |
| **共享池配额** | need/bridge/relax 共享总保留池 | **KEEP（修复组合非加性）** | combo shared=2：Hit@1 0.570（回到基线上方）、Hit@10 0.745 / evR@10 0.6031 全配置最高 |

## 三、最终配置

```powershell
# 新基线（P4-A q2，一行启用）
python -m scripts.evaluate_locomo_retrieval ... --baseline-mode

# 等价展开
--structured-query-plan --evidence-need-retrieval --evidence-need-quota 2 \
--evidence-need-rrf-weight 0.01

# 可选叠加（P4-C bridge，需共享池避免非加性）
--bridge-retrieval --bridge-max-terms 3 --bridge-quota 2 --sidecar-shared-quota 2
```

组件全部默认关闭；启用组合时的正确姿势：sidecar 组件（need/bridge）共享
`--sidecar-shared-quota`，避免各自配额叠加压缩 P1 池。

## 四、全量指标（P4-A q2, n=1977）

| 指标 | P4-A q2 | P1 基线 | Δ |
|---|---|---|---|
| Hit@1 | 0.5776 | 0.5761 | **+0.0015** |
| Hit@3 | 0.7198 | 0.7157 | **+0.0041** |
| Hit@10 | 0.7643 | 0.7618 | **+0.0025** |
| MRR | 0.6501 | 0.6479 | **+0.0022** |
| evR@10 | 0.5998 | 0.5976 | +0.0022 |

全量 bucket（1977）：命中 1518 / fusion_miss 272 / channel_miss 112 /
reranker_drop 63 / quota_displacement 12。
P4-A 恢复统计：off-miss 且具 need 证据 299 题中救回 pool 14 → Top10 8 → Top1 4。

## 五、关键教训

1. **配额非加性是组合失败的主因**（非组件本身无效）：P4-A q2 + bridge q2
   无共享时把 P1 池压到 26，Hit@1 跌破基线；共享池（合计 ≤2）修复。任何
   新 sidecar 组件必须接入共享池，禁止各自固定配额叠加。
2. **channel_miss 多为推理型而非词汇型**：112 题 channel_miss 中 76 题连
   need 通道也 miss，抽样全为 query 与 gold 零词汇重叠（"political leaning"
   → "religious conservatives…LGBTQ rights"）。词汇松弛（P4-D）无靶；
   此类问题需要语义/实体扩展或 rerank 层解决，不在词汇检索空间内。
3. **审计桶 ≠ 有效性证明**：P4-C 的 NewGold 真实存在但 Hit@1 中性——召回
   增益必须配对 Hit@1 检验，只看 recall 会误判。
4. **counterfactual 只覆盖检索层**：`p1_counterfactual_top30_ids` 可用于
   pool 层配对对比，但最终 Hit@1 是 rerank 产物，严格配对仍需跑对照。

## 六、交付物

- 代码：`app/storage.py`（P4-A/C/D 通道 + 共享池 + trace）、
  `scripts/evaluate_locomo_retrieval.py`（CLI + diagnostics）
- 测试：`tests/test_p4a_evidence_need.py`（10）、`tests/test_p4c_bridge.py`（9）、
  `tests/test_locomo_p4_audit.py`（6）——P4 套件 25 passed，全量 674 passed
  （1 个历史 readme 一致性失败，与 P4 无关）
- 文档：`P4_RECALL_RECOVERY_SPEC.md`（spec）、`P4_0_FIXED100_AUDIT_RESULTS.md`、
  `P4_A_FIXED100_EXPERIMENT.md`、`P4_A_FULL1977_VALIDATION.md`、
  `P4_SIDECAR_SHARED_QUOTA.md`、`P4_D_QUERY_RELAXATION_REJECT.md`
- 提交链（均推送 origin/research/p3-evidence-graph）：
  `dfe769f` P4-0 → `13df08d` P4-A → `726baef` P4-B → `3151b0b` P4-C →
  `52350e3` combo 记录 → `5dab0d4` 全量验证 → `d867fb9` baseline-mode →
  `940f3f6` 共享池 → `e46e8d6` P4-D 否决
- 产物（gitignored `.locomo/`）：fixed-100/200 配对、全量 10 chunk、分析脚本

## 七、后续入口（未做）

1. **官方评测迁移验证**：P4-A 在官方 gpt-4o-mini 口径确认增益迁移（当前全部
   为本地 Qwen3-4B 代理信号）。
2. **P5 角色/状态排序支线**：Evidence Role / Current-Previous /
   Correction-Retraction / Authority（spec 中降级的排序支线），在召回空间
   建立后进入。
3. **推理型 channel_miss 攻关**：76 题零词汇重叠——需实体链接/语义检索或
   训练侧方案，词汇检索层无解（P4-D 已证）。
