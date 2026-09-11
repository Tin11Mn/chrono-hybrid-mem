# LoCoMo 端到端 QA 评测：实验设计与实施清单

- 日期：2026-09-10
- 状态：**设计稿（待用户确认）**；对应的运行前成本见 [`API_COST_ESTIMATE.md`](API_COST_ESTIMATE.md)
- 前置：[`LOCOMO_E2E_PROTOCOL_AUDIT.md`](LOCOMO_E2E_PROTOCOL_AUDIT.md)（**已完成**）
- 硬约束：本文件批准前不执行任何付费 API 调用；不修改任何已有结果文件

---

## 1. 研究问题（本实验唯一要回答的问题）

> SF v2 通过 **provenance-preserving semantic fact layer** 提升了原始长期证据的可访问性之后，
> 这种 **evidence-level improvement 是否真正转化为标准 LoCoMo 端到端长期记忆问答性能提升**？

需要建立的证据链：

```text
P4-A + BM25
  → add SF v2
  → Evidence Recall@10 / Hit@K 提升        （已证实：0.6127 → 0.6555，Hit@1 0.5850 → 0.6108）
  → end-to-end F1 / BLEU-1 / LLM-Judge 提升 （本实验要测）
  → category-level improvement             （本实验要测）
  → statistical significance               （paired bootstrap）
  → acceptable efficiency cost             （本实验要测）
```

**若 evidence 提升但 QA 未提升，必须如实报告**，并按 §13 的预设路径归因。

---

## 2. 设计原则（不可协商）

### 2.1 Raw evidence remains the source of truth

- Add 只收原始消息（role / content / timestamp）。
- Fact / Context / Porter / Dense / Session Fact 均为**辅助检索表示**。
- `/search` 返回的**必须**是原始历史消息。
- **Search 绝不生成 benchmark final answer。**
- 每条 session fact 必须可追溯回 `source_message_id`。

### 2.2 Answer generation 是 harness 的独立步骤

```text
ChronoHybridMem.Search  →  retrieved original evidence
evaluation harness       →  answer model  →  generated answer
evaluation harness       →  scorer（F1 / BLEU-1）+ judge
```

Answer/judge 代码**不得**放进 `app/`（那是记忆服务），必须放在 `scripts/` 的 harness 中。

### 2.3 泄漏控制（对应审计 §B.6）

| 禁止 | 说明 |
|---|---|
| gold answer 进检索模型 | `qa.answer` / `qa.adversarial_answer` 不得出现在任何 Search 输入 |
| gold evidence 进检索模型 | `qa.evidence` 仅作评分标签 |
| **LoCoMo `observation` 作正式语义记忆** | 官方 LLM 标注，测试时不可见；仅可用于 offline oracle 上限分析 |
| `event_summary` / `session_summary` 进管线 | 同上 |
| `category` 用于 query routing 或调参 | 类别标签只能用于**报告分组**，不能改变检索行为 |
| 测试标注做 query-specific tuning | 不允许 |
| 因结果不好改题集 | 不允许 |

**事实层来源**：`.locomo/session-facts-full.json`，由本地 Qwen3-4B 从
**原始 role/content/timestamp** 生成（`.locomo/_gen_session_facts.py`，temperature 0），
**不是**数据集自带的 `observation`。每条 fact 携带 `source_message_id`、`session_id`、
speaker grounding。

### 2.4 两套实验并行保留，不混淆

| 实验 | 题集 | 指标 | 状态 |
|---|---|---|---|
| **Evidence-Level Diagnostic** | 1976（**含** cat5 446 题） | Hit@1/3/10、MRR、nDCG@10、Evidence Recall@10 | 已完成，**保留不动** |
| **End-to-End LoCoMo QA** | **1540** 非对抗（cat 1-4） | F1 / BLEU-1 / LLM-Judge + evidence-level 子集 | 本任务新增 |

---

## 3. 流水线

```text
locomo10.json（原始会话）
  │  仅 conversation + qa.question + qa.category + qa.answer/evidence（标签）
  ▼
ChronoHybridMem.Add            （逐 session 写入；本地，无 LLM 答案生成）
  ├─ raw_messages（唯一事实来源）
  └─ FTS5 / facts / Porter / adjacent 索引
  ▼
SF v2 memory construction      （离线事实层注入：session_facts + source_message_id）
  ▼
ChronoHybridMem.Search         （P1 结构化规划 → 多路检索 → RRF → evidence-ID rerank）
  │  返回：原始证据（content + mem_id），top-k=10
  ▼
evaluation harness
  ├─ evidence 拼接（speaker/timestamp 渲染，per selected answer prompt）
  ├─ answer model（gpt-4o-mini，T=0）→ generated answer
  ├─ scorer：F1 / BLEU-1
  └─ judge（gpt-4o-mini，T=0）→ CORRECT / WRONG
  ▼
results/locomo_e2e/<run-id>/per_question.jsonl  （每题独立落盘，可断点续跑）
```

---

## 4. 复用既有资产（**不重新发明**）

| 资产 | 位置 | 复用方式 |
|---|---|---|
| 统一评测框架（canonical records / metrics / adapters / workload 导出） | `.p4release-bak/evaluation/unified/`（**未提交**） | 直接复用其 `records.py` / `metrics.py` / `datasets/locomo.py`；**首次提交时纳入版本控制** |
| LoCoMo 会话解析 + gold evidence 映射 | `scripts/evaluate_locomo_retrieval.py::sessions_and_evidence` | 直接 import，保证与已发布 evidence-level 口径一致 |
| SF v2 事实层注入 | `scripts/evaluate_locomo_retrieval.py::_inject_session_facts` | 直接复用 |
| SF v2 检索实现 | `app/storage.py`（`session_facts` 表 + RRF 通道 + reserved quota + rerank fact 注入） | 直接调用 `MemoryStore`，**不修改** |
| Mem0/LightMem/AML 官方 prompt | 见审计 §C | **逐字复制**到 `prompts/`，不重写 |
| 事务事实层缓存 | `.locomo/session-facts-full.json`（272 session / 4250 fact） | 直接复用 |

> **0 处修改 `app/`**；**0 处修改 `scripts/evaluate_locomo_retrieval.py`**。
> 这保证已有 1976 evidence-level 结果不受本任务影响。

---

## 5. 题集与分母（必须逐处声明）

| 集合 | 数量 | 用途 |
|---|---:|---|
| LoCoMo 全部 QA | 1986 | 参考 |
| **非对抗（`category != 5`）** | **1540** | **E2E 主题集** |
| 其中 evidence 可解析 | 1531 | 可算 evidence-level 指标 |
| 减 offset 758 | **1530** | evidence-level 在 E2E 内的可比子集 |
| E2E 实跑（758 记 timeout） | **1539** | answer/judge 有值 |

规则：

- **不得**用"含 `answer` 键"过滤（会得到 1542，含 2 条 cat5）——见审计 §B.4。
- 9 条无 gold evidence 的题：**保留**在 E2E，evidence-level 字段记 `null` 并计入
  `evidence_metrics_excluded` 计数；**不得**为了好看而删除。
- offset 758：`status="timeout"`，保留在分母，单独统计失败率。
- 失败/超时题：**保留在分母**，不得静默丢弃。

---

## 6. 指标定义

### 6.1 F1 —— 四种谱系全报（审计 §C.6）

| id | 实现 | 复刻对象 |
|---|---|---|
| `f1_official` | 去逗号 → lower → 去 `string.punctuation` → 去 `a/an/the/and` → 空白规整 → **Porter stem** → **Counter 多重集** F1；cat1 逗号拆分取 `mean(max)`；cat3 取 `;` 前段 | 官方 LoCoMo `task_eval/evaluation.py` |
| `f1_mem0` | lower → 把 `. , ! ?` 换成空格 → split → **集合** F1 | Mem0 / A-Mem |
| `f1_memoryart` | lower → `word_tokenize` → **集合** F1，无归一化 | MemoryART |
| `f1_memoryos` | `re.findall(r'\b\w+\b', lower)` → **集合** F1 | MemoryOS |

**主表使用 `f1_official`**（因为它是 LoCoMo 的官方定义）；其余三种在同表附加列或附表中报告，
用于与各论文"同谱系"对齐。**禁止**把不同谱系混进同一列。

### 6.2 BLEU-1

| id | 实现 |
|---|---|
| `bleu1_m1` | nltk `sentence_bleu(ref, hyp, weights=(1,0,0,0), SmoothingFunction().method1)`（Mem0 / A-Mem） |
| `bleu1_m4` | 同上但 `method4`（MemoryART） |

官方 LoCoMo **没有** BLEU-1，因此 BLEU 列必须标注"非官方指标"。

### 6.3 LLM-as-a-Judge

- 主口径：Mem0/LightMem 式**二值** CORRECT/WRONG，JSON `{"label": "..."}`
- `temperature=0`，`max_tokens` 记录实际值
- 每题只跑 1 次（与 released 代码一致）；文本记录论文声称的 10 次运行差异
- 输出落盘：`judge_result`、`judge_raw_response`、`judge_input/output_tokens`、`judge_latency_ms`

### 6.4 Evidence-level（继续保留，§12）

每题同时记录：`hit1`、`hit3`、`hit10`、`mrr`、`evidence_recall10`、`ndcg10`（若已实现）。
这样**同一次运行**即可回答"evidence 提升是否转化为 QA 提升"。

### 6.5 类别与总体

- 分类：Single-hop(cat4) / Multi-hop(cat1) / Temporal(cat2) / Open-domain(cat3)
- 总体：**micro（题级）平均**，并同时给出分子/分母
- 每类报告 n

---

## 7. 每题 JSONL schema（§16 精确实现）

`results/locomo_e2e/<run-id>/per_question.jsonl`，一行一题，**追加写 + flush + fsync**：

```json
{
  "question_id": "conv-26:qa:7",
  "conversation_id": "conv-26",
  "qa_index": 7,
  "category_id": 4,
  "category_name": "single_hop",
  "question": "...",
  "reference_answer": "...",
  "retrieved_evidence_ids": ["mem_12", "mem_88"],
  "retrieved_evidence": [{"mem_id": "mem_12", "dia_id": "D1:3", "speaker": "Caroline",
                          "timestamp": 1683551760, "content": "Caroline: ..."}],
  "generated_answer": "...",
  "answer_model": "gpt-4o-mini",
  "answer_model_returned": "<response.model>",
  "answer_prompt_version": "mem0.v1",
  "answer_prompt_hash": "<sha256>",
  "answer_temperature": 0.0,
  "answer_max_tokens": null,
  "top_k_evidence": 10,
  "judge_model": "gpt-4o-mini",
  "judge_model_returned": "<response.model>",
  "judge_prompt_version": "mem0.v1",
  "judge_prompt_hash": "<sha256>",
  "judge_result": "CORRECT",
  "judge_raw_response": "...",
  "f1": 0.0,
  "f1_official": 0.0,
  "f1_mem0": 0.0,
  "f1_memoryart": 0.0,
  "f1_memoryos": 0.0,
  "bleu1_m1": 0.0,
  "bleu1_m4": 0.0,
  "hit1": 0,
  "hit3": 0,
  "hit10": 0,
  "mrr": 0.0,
  "ndcg10": 0.0,
  "evidence_recall10": 0.0,
  "answer_input_tokens": 0,
  "answer_output_tokens": 0,
  "judge_input_tokens": 0,
  "judge_output_tokens": 0,
  "search_latency_ms": 0,
  "answer_latency_ms": 0,
  "judge_latency_ms": 0,
  "search_model_calls": 0,
  "search_input_tokens": 0,
  "search_output_tokens": 0,
  "status": "ok",
  "attempts": 1,
  "error": null,
  "run_id": "<run-id>",
  "method": "sfv2_qwen3_4b",
  "config_digest": "<sha256>"
}
```

字段纪律：

- **保留**用户 §16 列出的全部键名。
- `status` ∈ `ok | timeout | error | unsupported_provenance`。
- `error` 为 `null` 或**不含凭证**的简短诊断。
- 不写入任何 API key、endpoint 凭证。

---

## 8. 断点续跑（checkpoint / resume）

要求（§16）：

1. 每题**独立落盘**；进程崩溃不丢已完成的题。
2. 重启后：`--resume` 读取已有 JSONL，**已成功完成的题不再调用 API**。
3. retry **只针对失败题**（`status != "ok"`）。
4. checkpoint 机制：
   - `<run-id>/per_question.jsonl` —— 逐题结果（唯一真相）
   - `<run-id>/checkpoint.json` —— `{run_id, config_digest, completed_question_ids, failed_question_ids, cumulative_cost_usd, updated_at}`
   - `<run-id>/run_config.json` —— 冻结配置快照（含 prompt hash、digest）
5. **配置变更即拒绝续跑**：若 `config_digest` 与 checkpoint 不一致，harness 必须
   **fail closed**，要求换新 `--run-id`。这样避免"部分用旧配置、部分用新配置"的脏结果。
6. 原子写：先写 `.tmp` 再 `os.replace`。
7. **不得因为中途失败浪费已花费的 API 成本。**

---

## 9. Token / 延迟 / 成本记账

每题逐条记录（见 §7 schema），并在 run 级汇总：

```text
per-stage: calls, input_tokens, output_tokens, wall_clock
overall:   总 tokens、总成本、Avg tokens/query、Avg LLM calls/query,
           P50/P95 latency、failure rate、retry rate、memory construction cost
```

落盘：`results/locomo_e2e/<run-id>/cost_summary.json`

- 成本按**实测 usage** 计算；费率写在 `run_config.json` 的 `pricing` 字段。
- 本地模型的 turn 记为 `input_tokens`/`output_tokens` 且 `billed=false`。
- 响应中真实返回的 model id 必须记录；若与请求不符需在结果文档中声明。

---

## 10. 统计显著性（§18）

- **paired bootstrap**，**10000** 次重采样，**seed = 20260826**，**95% CI**
- 配对对象：**同一题集、逐题配对**（缺题按失败处理，不静默丢弃）
- 指标：F1（各谱系）、BLEU-1、Judge Accuracy、Hit@1、MRR、Hit@10、
  Evidence Recall@10
- 同时报 wins / losses / net
- **命名纪律**：若计算 `Pr(Δ > 0)`，必须称为
  **bootstrap probability**，**不得**写成 p-value。

---

## 11. 分阶段运行（§15，严格顺序）

### Stage 1 — Smoke Test（固定 20 题，覆盖多个 category）

只验证**流水线正确性**，不看分数：

- [ ] 数据读取（含 9 条无 evidence 题的容错）
- [ ] memory 构建（Add + SF v2 注入）
- [ ] search
- [ ] evidence 拼接
- [ ] answer generation
- [ ] judge parsing
- [ ] metric calculation（四种 F1 + 两种 BLEU 与手算样例一致）
- [ ] checkpoint
- [ ] resume（人为中断后重启，验证不重复调用）
- [ ] token accounting

**通过标准**：全部 9 项通过 + 手算校验一致。**不通过不得进入 Stage 2。**

### Stage 2 — Fixed 100（按 category 分层抽样）

验证：API 稳定性、timeout、retry、cost、latency、category 统计、result persistence。

**通过标准**：失败率 < 5% 且失败均有明确 `status`；成本与 §API_COST_ESTIMATE 同量级；
category 统计齐全。

### Stage 3 — Full LoCoMo QA（1539 题 × 配置）

**只有 Stage 1 与 Stage 2 全通过后才执行。**

额外要求：先跑 `chrono-main`（SF v2 + 4B）单配置，确认与已发布 evidence-level
Hit@1/MRR **逐题一致**（parity test，见 §12），再跑其余配置。

---

## 12. 有效性校验（防止 harness 自身出错）

**Parity test**：在 `chrono-main` 配置下，E2E harness 对同一 1976 题集
（含 cat5，仅作校验）算出的 Hit@1 / Hit@3 / Hit@10 / MRR 必须与已发布数字一致：

| 方法 | 已发布 Hit@1 | 已发布 MRR |
|---|---:|---:|
| P4-A + BM25 | 0.5850 | 0.6618 |
| SF v2 + Qwen3-4B | 0.6108 | 0.6929 |

允许的差异仅为 LLM 非确定性导致的极小抖动（已确认 temperature=0 下基本可复现）；
**任何系统性偏差都必须先修好再跑付费 API**。这是本任务最重要的自检。

---

## 13. 失败情形的预设分析路径（§23）

若 evidence 提升但 QA 未提升，**按以下顺序**归因，不得跳步：

1. **Answer Model 是否没有利用 evidence** —— 检查 gold evidence 已在 top-10 但答案仍错的题；
   若占比高，是 reader 瓶颈。
2. **Top-K 是否过大** —— 对比 top-5 / 10 / 20 的 QA 变化；过大时噪声证据稀释。
3. **是否存在 stale / conflicting evidence** —— 检查检索结果内部矛盾率。
4. **是 ranking 问题还是 recall 问题** —— 用 Hit@1 与 Hit@10 的差值分离。
5. **不同类别转化效率是否不同** —— 分类别算 `ΔQA / ΔEvidenceRecall`。

同时报告：`judge` 与 `F1` 的一致率（避免单一 judge 的偏置）。

---

## 14. 文件清单

### 14.1 新增

| 路径 | 内容 | 状态 |
|---|---|---|
| `docs/LOCOMO_E2E_PROTOCOL_AUDIT.md` | 协议审计（A/B/C） | ✅ 本次已建 |
| `docs/API_COST_ESTIMATE.md` | 成本估算 | ✅ 本次已建 |
| `docs/LOCOMO_E2E_EXPERIMENT.md` | 本文件（设计 + 清单） | ✅ 本次已建 |
| `docs/LOCOMO_BASELINE_ALIGNMENT.md` | 基线对齐表（as-reported vs reproduced，分表） | 待建 |
| `docs/LOCOMO_E2E_RESULTS.md` | 最终结果与论文表格 A–D | 待建（跑完后） |
| `scripts/evaluate_locomo_e2e.py` | 主 harness：reuse Search、拼 evidence、answer、checkpoint/resume、记账 | 待建 |
| `scripts/score_locomo_answers.py` | 四种 F1 + 两种 BLEU-1（纯离线，可单测） | 待建 |
| `scripts/judge_locomo_answers.py` | LLM-as-a-Judge（可独立重跑，不重跑 answer） | 待建 |
| `scripts/summarize_locomo_e2e.py` | 汇总：分类别、总体、paired bootstrap、表 A–D | 待建 |
| `scripts/audit_locomo_dataset.py` | 数据集审计（category 映射/题数/泄漏字段），把本次探针固化为可复现工具 | 待建 |
| `prompts/locomo_answer_mem0.txt` | Mem0 `ANSWER_PROMPT` 逐字副本 | 待建 |
| `prompts/locomo_judge_mem0.txt` | Mem0 `ACCURACY_PROMPT` 逐字副本 | 待建 |
| `prompts/locomo_answer_lightmem.txt` | LightMem `ANSWER_PROMPT` 逐字副本 | 待建 |
| `prompts/locomo_judge_lightmem.txt` | LightMem `ACCURACY_PROMPT` 逐字副本 | 待建 |
| `prompts/locomo_answer_memoryart.txt` | MemoryART `build_prompt` 模板逐字副本 | 待建 |
| `prompts/locomo_answer_memoryos.txt` | MemoryOS system+user 模板逐字副本 | 待建 |
| `prompts/locomo_judge_memoryos.txt` | **`NOT_APPLICABLE`**（MemoryOS 无 judge），写文件说明原因 | 待建 |
| `prompts/locomo_answer_aml.txt` / `locomo_judge_aml.txt` | AML `OPEN_ENDED_ANSWER_TEMPLATE` / `ACCURACY_PROMPT` | 待建 |
| `prompts/README.md` | prompt 来源、hash、对应 profile | 待建 |
| `results/locomo_e2e/<run-id>/per_question.jsonl` | 逐题结果 | 运行产出 |
| `results/locomo_e2e/<run-id>/checkpoint.json` | 断点 | 运行产出 |
| `results/locomo_e2e/<run-id>/run_config.json` | 冻结配置 + prompt hash + digest | 运行产出 |
| `results/locomo_e2e/<run-id>/raw_model_outputs.jsonl` | 原始模型响应（answer + judge） | 运行产出 |
| `results/locomo_e2e/<run-id>/metric_summary.json` | 总体指标 | 运行产出 |
| `results/locomo_e2e/<run-id>/category_summary.json` | 分类别指标 | 运行产出 |
| `results/locomo_e2e/<run-id>/cost_summary.json` | token / 延迟 / 成本 | 运行产出 |
| `results/locomo_e2e/<run-id>/paired_bootstrap.json` | 配对统计 | 运行产出 |
| `results/locomo_e2e/README.md` | 运行索引与命令 | 待建 |
| `tests/test_locomo_e2e_scoring.py` | F1/BLEU 与手算样例一致 | 待建 |
| `tests/test_locomo_e2e_checkpoint.py` | 断点/续跑/配置变更 fail-closed | 待建 |
| `tests/test_locomo_e2e_metrics.py` | 指标聚合与分母正确性 | 待建 |

### 14.2 修改（**最小化，且不触碰已有结果**）

| 路径 | 修改内容 | 风险 |
|---|---|---|
| `docs/CHRONOHYBRIDMEM_METRIC_PROTOCOL.md` | 把"端到端 F1/BLEU/LLM-Judge ⛔ 不适用"一行改为指向本 E2E 协议；补 E2E 指标定义 | 低（纯文档） |
| `docs/CHRONOHYBRIDMEM_RESULTS_FOR_PAPER.md` | 补 SF v2 行 + 新增 E2E 表 | 低 |
| `progress.md` / `findings.md` | **追加**本次工作与发现；**不删**任何历史条目 | 低 |
| `README.md` + 5 语言 README | 结果产出后补 E2E 小节 | 中（需同步 6 语言，`scripts/check_readme_consistency.py` 会校验） |
| `.gitignore` | 允许 `results/locomo_e2e/` 的**汇总**入库，逐题大文件可选忽略 | 低 |

### 14.3 **不动**（保护既有实现与历史）

- `app/**`（含 SF v2 实现）—— 0 修改
- `scripts/evaluate_locomo_retrieval.py` —— 0 修改（只 import）
- 任何已完成实验的结果/文档（`docs/EVALUATION_NEW_METHOD_1976.md`、
  `docs/SESSION_FACT_SEMANTIC_LAYER_EXPERIMENT.md`、`.locomo/*`、P3/P5 全部记录）
- `research/p3-evidence-graph` 的 4 个 REJECT 提交 —— 保留
- `.p4release-bak/evaluation/unified/` —— **保留并纳入版本控制**（见 §15）

---

## 15. 需要一并处理的仓库风险

审计 §A.2 发现：`.p4release-bak/evaluation/unified/`（约 500KB、18 个测试文件）
**0 个文件被跟踪**，且该目录是**已从 `git worktree list` 注销的孤立 worktree 备份**。
本任务建议在首次提交时：

1. 把 `evaluation/unified/` 及其文档、锁文件复制到主仓库并提交；
2. 在 `docs/` 记录其来源（`competition/2026-cycle-2-p0` @ `be2cc40` 之后的未提交工作）；
3. **不删除** `.p4release-bak` 原目录。

---

## 16. 论文表格模板（§20）

> `—` 表示"该论文未报告该指标"，**禁止**推算填充。协议不兼容者**分表**。

### Table A — LoCoMo End-to-End QA

**A-1 同协议重跑组**（本 harness 统一 answer/judge/metric，可直接比较）

| Method | Single-hop | Multi-hop | Temporal | Open-domain | Overall | F1(official) | BLEU-1(m1) | LLM-Judge |
|---|---|---|---|---|---|---|---|---|
| P1 | | | | | | | | |
| P4-A + BM25 | | | | | | | | |
| **SF v2 + Qwen3-4B** | | | | | | | | |
| SF v2 + Qwen3-8B *(Reranker Scaling)* | | | | | | | | |

**A-2 as-reported 组**（文献原报数字，**非同一协议**，仅作参考）

| Method | Available models | Single | Multi | Temporal | Open | Overall | F1 谱系 | BLEU 谱系 | Judge | 可比性说明 |
|---|---|---|---|---|---|---|---|---|---|---|
| Mem0 | gpt-4o-mini | 38.72/27.13/67.13 | 28.64/21.58/51.15 | 48.93/40.51/55.51 | 47.65/38.72/72.93 | 66.88±0.15 J | ② | m1 | 二值 | 论文 Table1/2 算术不一致；judge 10 次 vs 代码 1 次 |
| MemoryOS | gpt-4o-mini | 35.27/25.22 | 41.15/30.76 | 20.02/16.52 | 48.62/42.99 | — | ④ | — | — | 含 cat5；prompt 含硬编码例答 |
| A-Mem | gpt-4o-mini | 27.02/20.09 | 45.85/36.67 | 12.14/12.00 | 44.65/37.06 | — | ② | m1 | — | 含 cat5；官方 runner cat5 存在 gold 泄漏 |
| MemoryART | GPT-4o-mini（论文） | 31.43/36.05 | 47.04/36.85 | 27.46/22.07 | 48.86/43.33 | — | ③ | m4 | — | 脚本用 deepseek-chat；memory builder 未公开 |
| LightMem | gpt-4o-mini / qwen3-30b | — | 62.41 | 74.14 | 44.79 | 71.95 J | 未公开 | 未公开 | 二值 | F1/BLEU 无公开实现；token 口径不含 embedding/judge |
| ChronoHybridMem + SF v2 | gpt-4o-mini | | | | | | ①② | m1,m4 | 二值 | 本工作 |

> Mem0 行的 `F1/BLEU/J` 三值同格，保持论文 Table 1 原顺序。

### Table B — ChronoHybridMem Evidence Diagnostics（已可填，来自审计 §A.3）

| Method | Hit@1 | Hit@3 | Hit@10 | MRR | Evidence Recall@10 | nDCG@10 |
|---|---:|---:|---:|---:|---:|---:|
| P1 | 0.5779 | 0.7176 | 0.7601 | 0.6497 | 0.5976\* | 0.6773 |
| P4-A q2 | 0.5779 | 0.7201 | 0.7642 | 0.6504 | 0.6045 | 0.6787 |
| P4-A + BM25 | 0.5850 | 0.7323 | 0.7809 | 0.6618 | 0.6127 | 0.6914 |
| SF v2 + Qwen3-4B | 0.6108 | 0.7677 | 0.8219 | 0.6929 | 0.6555 | —（未导出） |
| SF v2 + Qwen3-8B | 0.6336 | 0.7905 | 0.8340 | 0.7119 | 0.6703 | —（未导出） |

\* P1 的历史证据池口径（1977 题）；其余为 1976 题同口径重算。n=1976，offset 758 统一排除。

### Table C — SF v2 Ablation

| Variant | Hit@1 | Hit@10 | MRR | EvRecall@10 | QA F1 | QA Judge |
|---|---:|---:|---:|---:|---:|---:|
| P1 | 0.5779 | 0.7601 | 0.6497 | 0.5976\* | | |
| P4-A | 0.5779 | 0.7642 | 0.6504 | 0.6045 | | |
| P4-A + BM25 | 0.5850 | 0.7809 | 0.6618 | 0.6127 | | |
| + SF v2 | 0.6108 | 0.8219 | 0.6929 | 0.6555 | | |

组件级拆解（`semantic fact generation` / `dense fact retrieval` / `source-message mapping` /
`reserved quota` / `reranker fact augmentation`）**必须先确认可通过真实开关分别关闭**，
否则该子表标注为 `NOT_IMPLEMENTED`，**不得**编造。

### Table D — Reranker Scaling

| Method | Hit@1 | Hit@3 | Hit@10 | MRR | EvRecall@10 | QA F1 | QA Judge |
|---|---:|---:|---:|---:|---:|---:|---:|
| SF v2 + Qwen3-4B | 0.6108 | 0.7677 | 0.8219 | 0.6929 | 0.6555 | | |
| SF v2 + Qwen3-8B | 0.6336 | 0.7905 | 0.8340 | 0.7119 | 0.6703 | | |

配对（4B→8B，n=1976）：Hit@1 Δ **+0.0228** CI95 [+0.0051, +0.0405]，p(>0)=0.993；
MRR Δ **+0.0191** CI95 [+0.0073, +0.0311]，p(>0)=0.999。

> **表述纪律**：本表只能归因于 **reranker scaling / stronger discriminator**，
> **不得**归因于 SF v2 方法创新。

---

## 17. 验收标准

- [ ] Stage 1 九项全通过，F1/BLEU 与手算样例逐位一致
- [ ] Stage 2 失败率 < 5%，成本/延迟/分类统计齐全
- [ ] Parity test：`chrono-main` 复现已发布 Hit@1 / MRR
- [ ] 每题 JSONL 含 §7 全部字段；`status` 分布被如实报告
- [ ] 断点续跑验证：中断后重启不重复调用 API
- [ ] 成本闸门生效（超限停止且不写出部分结果）
- [ ] 泄漏自检：检索输入中不含 answer/evidence/observation/category
- [ ] 论文表格 A–D 生成，`—` 与"不可比"标注齐全
- [ ] `progress.md` / `findings.md` 已追加（历史条目未删）
- [ ] 已完成实验的结果文件**未被修改**（`git status` 可证）
