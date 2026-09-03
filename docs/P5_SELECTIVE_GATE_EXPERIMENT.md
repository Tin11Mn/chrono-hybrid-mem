# P5 Selective Rerank Gate：实现与 fixed-200 实验记录

- 日期：2026-08-26
- 状态：**实现完成（默认关闭）但 fixed-200 净收益为负（REJECT as-is）**
- 背景：P5 全量配对诊断（`research/p5-paired-diagnostics` 分支）显示
  P1 有 361 题 gold 在 rank 2-10（276 题在 2-3）；P4-A 转正 56 题 rank-2/3
  但也 displaces 71 题 Top-1 → 需要"带证据保护的选择性门控"而非无条件提升。

## 设计（--p5-gate，默认 off）

确定性、零 Search-LLM 调用增量的 Top-1/Top-2 置换门控，仅在同时满足：

1. **near-tie**：Top-1 与 Top-2 的融合分差 < `--p5-near-tie-epsilon`（默认 0.0005；
   分数源 = `MemoryResult.score`，LLM rerank 不返回分数）；
2. **严格更强**：runner-up 的 P1 通道命中数 > Top-1 且 ≥ `--p5-min-evidence-channels`
   （默认 2），或有 fact 注解。

替换后原 Top-1 保持 rank 2（不丢失可见证据）。

## fixed-200 实验结果（配对，同一 200 题）

| 指标 | gate off | gate on | Δ |
|---|---|---|---|
| Hit@1 | 0.5650 | 0.5600 | **-0.0050** |
| Hit@3 | 0.6850 | 0.6900 | +0.0050 |
| Hit@10 | 0.7300 | 0.7300 | 0.0000 |
| MRR | 0.6287 | 0.6267 | -0.0020 |

paired：Hit@1 gained 3 / lost 4 / unchanged 193（净 -1）；Hit@3 +1；Hit@10 0。
rank-2/3 slice（24 题）中 3 题转成 Top1（12%）。

## 迭代过程（关键教训）

1. **初版门控（绝对阈值 ≥2 通道）在 fixed-50 上崩 Hit@1（0.54→0.42）**：
   P1 通道是同一内容的多视角投影（raw/porter/fact/context/entity），同一消息
   几乎必然命中 ≥2 通道 → 绝对阈值无区分度，门控退化为"凡是 near-tie 就 swap"。
   修正为**相对比较**（runner-up 通道数 > Top-1）。
2. **修正后（通道计数版）fixed-50 安全但 fixed-200 净 -1**：3 gains / 4 displaces。
   通道数更多≠更正确（多视角计数不是正确性信号），near-tie 下仍会误换。
3. **信号升级（query token 重叠版）fixed-200 更差：Hit@1 净 -7**（0.565→0.530），
   swap 23 次 / displaces 8 题。query 词（人名/地名/普通名词）复现于多条消息，
   重叠数偏向"复述问题词更多的长消息"，同样不是正确性信号；`min_evidence=2`
   时 ov2=ov1+1 的微小优势即触发，误换率上升。
4. **strata 收窄版（temporal/correction）fixed-200：swap 4 / gains 0 / displaces 1**，
   Hit@1 净 -2（0.565→0.555）。strata 拦截 158/200 题（42 匹配）→ 触发面缩小后
   gains 同步归零（strata 匹配的题里 rank-2/3 gold 本就不多，且 runner-up
   overlap 更高仍 ≠ gold 在 runner-up，如 offset 129: ov1=3 ov2=5 却 displaces）。
   **三种信号（通道计数 / query 重叠 / strata 收窄）全部失败。**
5. **LLM confidence 信号（strata=all）：门控 zero swap，且 confidence prompt
   本身把 Hit@1 打崩 -0.045**（0.565→0.520，control 与 gate 完全一致）。
   Qwen 的 confidence 输出里 Top-1 恒为 1.0、Top-2 < 1.0，margin 0.05 永不
   满足 → 门控永不触发；同时要求模型额外输出 confidence 会干扰其排序判断。
   关键实验设计教训：off（旧 prompt）与 on（confidence prompt）prompt 不一致，
   排序差异不能归因门控——必须用 control（同 prompt、margin=1.0 不 swap）归因。
   **四种信号（通道计数 / query 重叠 / strata / LLM confidence）全部失败。**

## 结论

- **P5 门控四次信号尝试均不通过筛选**：通道计数 -0.005、query 重叠 -0.035、
  strata 收窄 -0.010、LLM confidence 0（不触发）但 prompt 副作用 -0.045，
  全部在"不降 >0.005"门槛之外。
- **根本约束（四次收敛）**：无 gold 泄漏下，检索层信号（融合分差 + 通道计数 /
  query token 重叠 / strata）与模型自身 confidence 均无法可靠区分"该换"与
  "不该换"；且让本地 Qwen 自报置信度会破坏其排序能力（-0.045）。本地代理的
  检索/重排信号已到天花板，P5 门控式置换在该口径下彻底关闭。
- 代码保留（默认 off，`--p5-strata`/`--p5-confidence-margin` 可配）+ 测试（10 例）
  作为可控消融。

## 后续方向（未做）

1. **转官方口径（唯一剩余路径）**：本地 Qwen 代理的检索与重排信号均已到天花板
   （P4-A 71 换 71 平；P5 四次信号失败），且让本地模型自报置信度会破坏排序。
   官方 gpt-4o-mini 的 rerank 质量不同，门控/排序优化可能在那里才有净收益；
   这是影响 leaderboard 排名的唯一剩余路径。

## 复现命令

```powershell
python -m scripts.evaluate_locomo_retrieval `
  --dataset '..\chrono-hybrid-mem\.locomo\locomo10.json' `
  --local-search-model-url 'http://127.0.0.1:8081/v1' --local-search-model-name local `
  --structured-query-plan --max-questions 200 --include-question-diagnostics `
  [--p5-gate] --output '.locomo\p5-gate-{off,on}-fixed200.json'
```
