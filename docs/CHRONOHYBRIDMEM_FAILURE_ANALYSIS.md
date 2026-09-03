# ChronoHybridMem 失败样本分析：offset 13（可复现，证据驱动）

## 1. 样本选择

- query_id：`q_0013`（offset 13）
- LoCoMo 类别：1（多跳语义类）
- 问题：*"What career path has Caroline decided to persue?"*
- gold evidence：D4:13（mem_71）、D1:11（mem_11），两跳证据
- 新方法结果：first_gold_rank = **None**（两条 gold 均不在 Top-10），
  recall_bucket = **fusion_miss**
- 选择理由：多跳问题、gold 完全落出 Top-10、三方法同败（非新方法回归）、
  单例稳定复现

## 2. 复现记录

- 配置与全量一致：`--structured-query-plan --evidence-need-retrieval
  --evidence-need-quota 2 --need-select-by-bm25`，本地 llama-server
  （Qwen3-4B, 16384 ctx, 已运行实例）
- 复跑产物：`.locomo/_repro-offset13.json`
- 复现一致：rank None / fusion_miss / top3 = mem_69, mem_70, mem_10 ——
  与全量运行完全相同（稳定复现）

## 3. 证据与诊断

### 3.1 词汇关系（已由逐题明细与文本核验）

| 证据 | 内容摘要 | 与查询共词（去停用词） |
|---|---|---|
| D4:13 (mem_71) | "…thinking of working with trans people, helping them accept themselves…" | 1（仅 "caroline"） |
| D1:11 (mem_11) | "I'm keen on counseling or working in mental health…" | 1（仅 "caroline"） |

查询核心词 {career, path, decided, pursue} 与两条 gold **零语义词重叠**——
gold 内容用 "counseling / mental health / trans people" 表达"职业方向"，
查询用 "career path / pursue"，是**同义/上位词错配（推理型）**，非字面命中。

### 3.2 检索通道痕迹（trace）

- mem_71：仅被低权重通道捕获 —— context rank 10、context_porter rank 15、
  entity_context rank 15；**未进 P1 counterfactual top-30**（gold_counterfactual_positions 空）
- mem_11：**任何 P1 通道均未捕获**（gold_channel_presence 无 mem_11）
- evidence-need 通道（4 条 need 查询）对两条 gold **均未命中**
  （gold_need_channel_presence 空）
- 结果：两条 gold 都未进入 rerank pool（gold_pool_positions 空）→ 排序无从谈起

### 3.3 失败原因归类（区分证据等级）

**A. 已由证据证明**
1. **词汇错配（推理型）**：query 与 gold 无语义词重叠，need 通道（基于
   evidence_needs 的 term 查询）与 P1 通道均无法命中 mem_11 —— 该证据在
   词面层不可检索（与 P4-D 审计结论一致：channel_miss/fusion_miss 中大量
   此类推理型案例）。
2. **低权重通道融合不足**：mem_71 被 context/entity_context 通道捕获
   （rank 10-15），但 RRF 融合（context 权重 0.5、entity 权重 0.0）后未进
   counterfactual top-30 —— fusion_miss 机制确认。

**B. 基于证据的推测**
- mem_71 若能进 pool（例如 need 通道命中），rerank 可能将其排入 Top-10；
  但 mem_11 即便进 pool 也需要语义理解才能识别 —— 修复 mem_71 的单边
  收益有限。

**C. 尚无法判断**
- 无法离线确认若两条 gold 都进 pool 后 LLM rerank 的具体排序（需改融合
  后重跑）；"如果修复融合能否整体解决该类"需实验验证。

## 4. 同类规模（全量 1976，证据驱动）

- fusion_miss 共 **233** 题
- 其中 **151 题（65%）gold 仅出现在低权重通道**（context/entity/support），
  未被 raw/fact 主通道捕获 —— 与 offset 13 的 mem_71 同构
- 215 题（92%）在 need 通道有 gold 却仍 fusion_miss（need 候选虽进 pool，
  但 gold 未在 need 候选前 N 内被保留/提升）
- 分布：context_porter 171、entity_context 167、context 127 居前

## 5. 结论

offset 13 是**检索层（词面错配 + 低权重融合不足）的既有失败**，P1/P4-A/NEW
三方法一致复现，非新方法（bm25 选取）引入。修复方向（均未实现、不改变本
实验设置）：
1. 同义/概念级扩展（词表或语义检索）以覆盖推理型错配；
2. 提升 context/entity 低权重通道候选在融合中的保留（但不做，属后续工作）。

本分析只诊断该样本，未为"解释失败"改变主实验配置或筛除样本。
