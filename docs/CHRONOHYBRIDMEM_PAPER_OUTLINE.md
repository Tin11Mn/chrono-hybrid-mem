# ChronoHybridMem 论文大纲（PAPER OUTLINE）

> 状态：主方法已确认并完成本地全量评测（1976 可完成查询）。所有引用数字
> 必须来自 `CHRONOHYBRIDMEM_RESULTS_FOR_PAPER.md`；未实现项不得写数值。

## 1. 主方法（2026-09 确认）

**P4-A evidence-need 独立检索 + need 候选 bm25 最优选取**
（`--need-select-by-bm25`，分支 research/p3-evidence-graph @ 0e3ecba）

流程：
1. 结构化查询规划（intent / core terms / entities / evidence_needs）
2. P1 多通道 RRF 融合检索（raw/porter/fact/context/entity 等 11 通道）
3. **P4-A**：每条 evidence need 独立词项化查询，走 6 个补充通道，低 RRF
   权重（0.01）参与融合，need 候选通过保留配额（2）进入 rerank pool
4. **need 候选按 best-bm25 排序**再被配额选取（而非通道插入序）——
   使配额总能选到质量最高的 need 候选
5. Search 模型（本地 Qwen3-4B 代理）对 pool 重排序输出 Top-K

与 P1 的关系：P1 是基础检索层；P4-A 提供每条证据需求的独立召回通道；
bm25 选取是该组件的关键微调（fixed-200 Hit@1 +0.020；全量 Hit@1 +0.0071）。

## 2. 实验结构

- **主实验**：NEW（P4-A + bm25）在 1976 可完成查询上的指标
- **基线对照**：P1（结构化规划，无 P4 组件）
- **消融**：P4-A q2（同 NEW 但配额按通道序，无 bm25 排序）
- 同一题集逐题配对；offset 758 统一排除（复现见 REPRO 文档）
- 统计：paired bootstrap 10,000（seed 20260826），如实报告含 0 的区间

## 3. 主指标

论文主指标（都已实现，见 RESULTS_FOR_PAPER）：
- **Hit@1、Hit@3、Hit@10**
- **MRR**
- **Evidence Recall@10**
- 补充：Recall@5（Q-level）、nDCG@10、Hit@5

多跳：gold ≥ 2 子集 Hit@1 = 0.4289（422 问）

## 4. 时间类结果（当前口径与边界）

- **只报告**：LoCoMo temporal category（category 2，321 问）Hit@1 = 0.5919
- **待完成（不得填入数字）**：
  - Latest-state accuracy
  - Stale evidence rate
  - 时间排序一致性（temporal ordering consistency）
  - 依赖：每条 gold evidence 的"状态-时间戳/最新态"标注（数据补充）

## 5. 效率指标（待完成）

- per-query 延迟 / P50 / P95
- 内存与磁盘占用
- 写入/查询吞吐
- 失败率与超时率
- 依赖：评估器计时与资源采样（未实现，不写数字）

## 6. 其余未实现/边界

- 官方（gpt-4o-mini）口径迁移：未验证
- 证据检索 ≠ 端到端问答：不报告 F1/BLEU/LLM-as-a-Judge
- 已知失败模式与案例：见 `CHRONOHYBRIDMEM_FAILURE_ANALYSIS.md`
