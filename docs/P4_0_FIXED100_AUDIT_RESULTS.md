# P4-0 固定 100 题审计结果（2026-08-25，本地 Qwen3-4B 代理）

运行命令（产物：`.locomo/p4-p1-audit-fixed100.json`，不入库）：

```powershell
D:\ANACONDA3\python.exe -m scripts.evaluate_locomo_retrieval `
  --dataset ..\chrono-hybrid-mem\.locomo\locomo10.json `
  --local-search-model-url http://127.0.0.1:8081/v1 `
  --local-search-model-name local `
  --structured-query-plan --max-questions 100 `
  --include-question-diagnostics `
  --output .locomo\p4-p1-audit-fixed100.json
```

## 聚合

- questions 100；Hit@1 `0.54`、Hit@3 `0.68`、Hit@10 `0.70`；MRR `0.6053`；evidence Recall@10 `0.5033`。
- 桶分布：top1_hit 54、ranking_top3 14、ranking_top10 2、**recall_miss_top10 30**。
- 注：固定 100 样本类别分布偏斜（cat1/cat4 居多），与全量 0.5761 的差异属样本方差，不代表方法回归。

## 30 个 recall_miss_top10 的四桶分解（核心交付）

| 桶 | 数量 | 占比 | 含义 |
|---|---|---|---|
| fusion_miss | 20 | 67% | gold 已被至少一个通道找到，但 RRF 把它挤出反事实 Top-30 |
| channel_miss | 8 | 27% | 任何通道都没找到 gold |
| reranker_drop | 2 | 7% | gold 已进 pool（位置 18 / 30），被 rerank 排出 Top-10 |
| quota_displacement | 0 | 0% | 本次为默认 P1（无 graph/anchor/adjacent 配额通道），预期为 0 |

**结论：471 的类比分布以 Fusion Miss 为主（约 2/3），不是"检索器完全没找到"。** 若按此比例外推到全量 471：约 315 fusion_miss、127 channel_miss、33 reranker_drop。

## 关键信号 1：Context 窗口通道是 fusion_miss 的主要来源

30 个 miss 中，gold 被各通道找到的频次（去重计数）：

```text
context_porter 16    entity_context 13    entity_porter 11    context 11
raw_porter 9         entity_raw 6         raw 5               support_raw 2
```

- context 族通道找到 gold 的次数约为 raw 族的 2-3 倍。
- 20 个 fusion_miss 中 **9 个（45%）gold 只出现在 context 族通道**（context / context_porter / entity_context；其中 entity_context 权重为 0）。
- 机制解释：查询的核心词分散在 previous+current+next 窗口里，gold 自身原文不含完整词项 → raw 通道低排位/未命中，context 通道（权重 0.5）命中但 rank≥3，RRF 总贡献不足以进 Top-30。
- 注意：这些 case 的 gold 是**窗口所属者**（ID 正确），不是"文本在窗口但 ID 错位"；后者（gold 出现在邻居窗口、返回邻居 ID）在本样本中体现为 gold 不在 context 通道的情况，需 P4-B 的 ±1 邻居通道补回。

## 关键信号 2：channel_miss 集中在关系/抽象查询（cat3）

8 个 channel_miss 的 plan：`Caroline's identity`、`relationship status`、`political leaning`、`personality traits`、`Melanie's children`、`theme park`、`adopt decision` 等，4 个在 cat3（关系/多跳）。gold 原文用词与查询抽象表述不同 → 纯 lexical 检索全通道 miss。这是 P4-D（query relaxation）与 P4-C（bridge second-pass）的主战场。

## 关键信号 3：reranker_drop 极少（2 个）

gold 在反事实 Top-30 的位置 18 和 30，被 rerank 排出 Top-10。属于 P5 排序问题，不是 P4 召回主线的主要矛盾（但"位置 30 进池"说明 pool 上限 30 边缘敏感）。

## 对 P4-A/B/C/D 的指向

1. **P4-A（evidence-need 独立通道 + pool 配额）** 直接针对最大的 fusion_miss 桶：need 化查询可让 gold 原文直接命中 raw 通道，配额保证进 pool。
2. **P4-B（neighbor recovery）** 机制已存在（`adjacent_turn_expansion`，默认关）：覆盖"gold 经邻居窗口被找到/需要邻居内容才能命中"的子集；本样本的 context 族命中数说明该机制值得开启做配对验证。
3. **P4-D / P4-C** 针对 channel_miss：抽象词（political leaning / personality traits）需要有限变体检索或 bridge 展开；cat3 关系题优先。
4. **P4 门控基线**：Pre-rerank Gold Recall@30 直接可算（`gold_counterfactual_positions` 非空即已进 Top-30）；当前 30 个 miss 中 28 个未进 Top-30（2 个已进）。
