# P4-A 全量 1977 题验证结果（evidence_need_retrieval, quota 2）

- 日期：2026-08-26
- 配置：`--evidence-need-retrieval --evidence-need-quota 2 --evidence-need-rrf-weight 0.01`（默认关闭，本验证显式开启）
- 数据集：locomo10 全量 1977 问（10 × 200 chunk 并行，2 路上限，`--include-question-diagnostics`）
- 基线：`p1-local-proxy-structured-full.json`（P1 全量 off，同数据集同模型代理）
- chunk 产物：`.locomo/p4a-q2-full-chunk-{0000..1800}.json`（10 个，全部 exit 0）
- 合并脚本：`.locomo/_p4a_full_merge.py`

## 结论

**P4-A（evidence-need 独立检索 + quota 2）全量验证通过，KEEP。**

- Hit@1 不降反升（+0.0015），满足"Hit@1 不降 > 0.005"门槛。
- Hit@3 / Hit@10 / MRR / evidence Recall 全线小幅上涨。
- 零 Search-LLM 调用增量（复用同一结构化查询计划的 evidence_needs）。
- 全量恢复收益：8 题 off-miss → Top10（其中 4 题直接 Top1）。

## 全量指标对比

| 指标 | P4-A q2 (n=1977) | P1 基线 (n=1977) | Δ |
|---|---|---|---|
| Hit@1 | 0.5776 | 0.5761 | **+0.0015** |
| Hit@3 | 0.7198 | 0.7157 | **+0.0041** |
| Hit@10 | 0.7643 | 0.7618 | **+0.0025** |
| MRR | 0.6501 | 0.6479 | **+0.0022** |
| evidence Recall@1 | 0.4070 | 0.4059 | +0.0011 |
| evidence Recall@3 | 0.5399 | 0.5378 | +0.0021 |
| evidence Recall@10 | 0.5998 | 0.5976 | +0.0022 |

## 全量 recall bucket 分布（1977 问）

| bucket | 数量 | 占比 |
|---|---|---|
| 命中（无失败） | 1518 | 76.8% |
| fusion_miss | 272 | 13.8% |
| channel_miss | 112 | 5.7% |
| reranker_drop | 63 | 3.2% |
| quota_displacement | 12 | 0.6% |

failure_bucket：top1_hit 1142，recall_miss_top10 466，ranking_top3 281，ranking_top10 88。

## 全量恢复统计（evidence-need off-miss）

- 具 need 证据但 P1 反事实池外（off-miss）的题：**299**
- 其中被 P4-A 救回 pool：**14**（4.7%）
- 救回 Top10：**8**（question offsets: 61, 143, 336, 1362, 1389, 1605, 1608, 1676）
- 救回 Top1：**4**（offsets: 143, 336, 1362, 1676）

注：救回数量占 off-miss 比例不高，但 Hit@1 净变化为正（+0.0015），说明配额置换的副作用未抵消收益；固定-200 实验（NewGold 8 / OffMiss→Top10 4 / Hit1 +6−4）与此一致。

## 运行说明

- 中途 1400/1600/1800 首跑 4 路并行触发 llama-server KV cache 溢出（`failed to find free space in the KV cache` → `Context size has been exceeded`），2 路并行重跑全部成功。并行度上限：**≤2**（每请求 4–6K tokens，4 请求 × ~5.3K > 16384 总 KV）。
- llama-server：`--ctx-size 16384`（8192 会在长对话 rank 时溢出）。

## 复现命令（单 chunk 示例）

```powershell
& '...\chrono-hybrid-mem\.venv-local\Scripts\python.exe' -m scripts.evaluate_locomo_retrieval `
  --dataset '..\chrono-hybrid-mem\.locomo\locomo10.json' `
  --local-search-model-url 'http://127.0.0.1:8081/v1' --local-search-model-name local `
  --structured-query-plan --max-questions 200 --question-offset 0 `
  --include-question-diagnostics --evidence-need-retrieval --evidence-need-quota 2 `
  --output '.locomo\p4a-q2-full-chunk-0000.json'
```

合并：`python .locomo/_p4a_full_merge.py`
