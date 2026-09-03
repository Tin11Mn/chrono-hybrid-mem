# Selective Evidence-Graph Gate：fixed-200 实验记录

- 日期：2026-08-26
- 状态：**实现完成（默认 off）但数据上无靶（REJECT as-is）**
- 背景：P3-A 历史实验（20 题切片）显示图谱全局默认退化（Hit@1 0.40→0.35、
  419 条消息仅 0.72% 关系覆盖率）。选择性门控假设"只在 multi_hop/实体密集时
  触发"能避开无关系场景。

## 设计（--graph-selective，默认 off）

- 触发条件：plan intent == multi_hop，或 plan entities ≥ 2（实体密集）
- 触发后走既有 `_one_hop_graph_candidates`（图谱机制零改动）
- trace 记录 `selective_enabled / selective_triggered / selective_reason`

## fixed-200 实验结果（配对，同 200 题）

| 指标 | off | graph-selective | Δ |
|---|---|---|---|
| Hit@1 | 0.5650 | 0.5650 | 0.0000 |
| Hit@3 | 0.6850 | 0.6850 | 0.0000 |
| Hit@10 | 0.7300 | 0.7300 | 0.0000 |
| MRR | 0.6287 | 0.6292 | +0.0005（LLM 噪声） |

- 门控触发 **104 题**（entity_dense 100 + multi_hop 4）
- **graph candidates = 0（全部空）**：触发后图谱通道一个候选都抽不出来
- gold 经图谱通道命中：0 题

## 结论：REJECT（数据无靶，非设计问题）

选择性门控精确化了触发面，但改变不了根本事实：**LoCoMo 对话是闲聊式，显式
关系边极稀缺**（与 P3-A 的 0.72% 覆盖率一致）。图谱通道无论全局还是选择性
触发，都抽不出候选 → 空通道无害但零收益。

**图谱方向在 LoCoMo 上关闭**；其适用场景是关系密集数据（ScriptMem、显式
时序事件），不在当前数据集内。

## 交付

- 代码：`graph_selective` 参数 + 触发条件 + trace（`app/storage.py`）、
  CLI `--evidence-graph/--graph-selective/--graph-rrf-weight/--graph-max-candidates/
  --graph-quota`（evaluate 脚本）、confidence rank 截断降级（`app/model.py`）
- 测试：`tests/test_graph_selective.py` 7 例；完整 graph 套件 473 passed
- 已推送 `f8e0b68`；本实验记录随代码保留

## 复现

```powershell
python -m scripts.evaluate_locomo_retrieval `
  --dataset '..\chrono-hybrid-mem\.locomo\locomo10.json' `
  --local-search-model-url 'http://127.0.0.1:8081/v1' --local-search-model-name local `
  --structured-query-plan --max-questions 200 --include-question-diagnostics `
  --evidence-graph --graph-selective `
  --output '.locomo\graph-selective-fixed200.json'
```
