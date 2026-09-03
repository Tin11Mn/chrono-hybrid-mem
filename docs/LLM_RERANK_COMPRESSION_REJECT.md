# LLM Rerank Candidate Compression（方向 3）：fixed-200 实验记录

- 日期：2026-09-03
- 状态：**实现完成（默认 off）但 fixed-200 净负（REJECT）**
- 假设：435 题 gold 在 pool 11-30 位被本地 Qwen 排错；把 LLM rank 输入压缩到
  前 15 个（融合序）可聚焦注意力、提升顶部排序。

## 实现（--llm-rerank-top-n，默认 0）

`llm_rerank_top_n > 0` 时只把 rerank_pool 前 N 个候选（融合序）送给
Search 模型排序；未入选候选保持融合序排在 LLM 输出之后。零 Search-LLM 调用
增量（同一次 rank，输入变短）。

## fixed-200 结果（配对）

| run | Hit@1 | Hit@3 | Hit@10 | MRR |
|---|---|---|---|---|
| P4-A bm25sel（无压缩） | **0.5850** | 0.7250 | 0.7600 | 0.6546 |
| P4-A bm25 + top-15 | 0.5650 | 0.6950 | 0.7250 | 0.6313 |

paired（top-15 vs bm25sel）：Hit@1 gained 6 / lost 10（净 -4）、Hit@3 -6、
Hit@10 -7。**18 题 gold 在 pool ≥15 位被压缩切出 LLM 视野**。

## 结论：REJECT

压缩候选输入让本地 Qwen **看不到深处的正确证据**——与假设相反，模型需要
看全部候选才能找到 gold（435 题 gold 在 11-30 位正是被切掉的对象）。
聚焦收益 < 视野损失。

## 交付

- `llm_rerank_top_n` 参数（storage + CLI + trace `llm_rank_candidate_count`）
- 测试 3 例（默认全量、top_n=2 截断、范围校验）；P4-A 套件 16 passed
- 代码默认 off 保留为可控消融

## 最终改进配置（方向 2 单独）

`--evidence-need-retrieval --evidence-need-quota 2 --need-select-by-bm25`
（fixed-200 Hit@1 0.585，+0.020 vs P1 off）
