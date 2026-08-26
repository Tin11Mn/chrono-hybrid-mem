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
2. **修正后 fixed-50 安全（0 swap 损伤）但 fixed-200 净 -1**：3 gains / 4 displaces。
   通道数更多≠更正确（多视角计数不是正确性信号），near-tie 下仍会误换。

## 结论

- **P5 门控 as-is 不通过筛选**：fixed-200 上 Hit@1 净 -1（-0.005），且 Hit@1 门控
  为"不降 >0.005"，此配置边缘失败。
- **根本约束**：无 gold 泄漏下，纯检索层信号（融合分差 + 通道计数）无法可靠区分
  "该换"与"不该换"；P4-A（71 换 71 平）与 P5 门控（3 换 4）同源。
- 代码保留（默认 off）+ 测试（5 例）作为可控消融；后续方向见下。

## 后续方向（未做）

1. **信号升级**：用 rerank 输出本身的可信度信号（如 Top-1/Top-2 的语义距离、
   与 query 的 token 重叠差）替代通道计数；
2. **strata 预注册**：仅在 plan intent ∈ {temporal, correction} 或 query 含
   状态语言时启用门控（缩小触发面）；
3. **接受 Hit@1 中性**：若目标改为 Hit@3/10 提升，门控 Hit@3 +0.005 虽小但非负，
   可在更大样本（全量）上验证是否稳定。

## 复现命令

```powershell
python -m scripts.evaluate_locomo_retrieval `
  --dataset '..\chrono-hybrid-mem\.locomo\locomo10.json' `
  --local-search-model-url 'http://127.0.0.1:8081/v1' --local-search-model-name local `
  --structured-query-plan --max-questions 200 --include-question-diagnostics `
  [--p5-gate] --output '.locomo\p5-gate-{off,on}-fixed200.json'
```
