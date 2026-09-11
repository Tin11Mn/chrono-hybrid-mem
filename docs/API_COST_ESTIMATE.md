# LoCoMo 端到端 QA 评测：API 成本估算

- 日期：2026-09-10
- 状态：**运行前估算（pre-run estimate）**。所有数字在真实运行前均为估算；
  真实运行后必须用每题的 `usage` 实测值替换本文件的估算值。
- 硬规则：**本文件批准前不得进行任何付费 API 调用。**
- 关联：[`LOCOMO_E2E_PROTOCOL_AUDIT.md`](LOCOMO_E2E_PROTOCOL_AUDIT.md)、
  [`LOCOMO_E2E_EXPERIMENT.md`](LOCOMO_E2E_EXPERIMENT.md)

---

## 1. 成本结构总览

| 阶段 | 执行位置 | 是否付费 API | 成本量级 |
|---|---|---|---|
| Memory construction（SF v2 session facts） | 本地 Qwen3-4B（llama.cpp） | **否** | 一次性，已完成 |
| Search（P1 规划 + rerank） | 本地 Qwen3-4B / 8B | **否** | GPU 墙钟时间为主 |
| Answer generation | gpt-4o-mini | **是** | 主要 API 成本 |
| Judge（LLM-as-a-Judge） | gpt-4o-mini | **是** | 次要 API 成本 |

**核心结论**：由于 ChronoHybridMem 的 Add/Search 走本地模型，
**API 成本几乎全部来自 Answer + Judge**，量级为 **< $1.5**；
真正的稀缺资源是**本地 GPU 墙钟时间**（估算 16–30 小时，见 §6）。

---

## 2. 实测输入参数（不是假设）

全部由本会话直接测量，命令与产物见 `.locomo/_probe_*.py`（审计探针，gitignored）。

| 参数 | 值 | 来源 |
|---|---|---|
| 非对抗题数 | **1540** | `locomo10.json`，`category != 5` |
| 端到端可跑题数 | **1539** | 1540 − offset 758（conv-43 超长会话挂起） |
| 平均对话轮 content 长度 | **145.2 字符**（p50 127 / p90 258 / max 492） | 5882 轮实测 |
| 平均问题长度 | **55.4 字符** | 1540 题实测 |
| 平均参考答案长度 | **29.6 字符**（p95 82 / max 388） | 1540 题实测 |
| 每题检索返回条数 | **10.0**（固定 top-10，无一题不足） | `sfv2-full-*.json` / `v8b-full-1976.json` |
| rerank pool 大小 | **30**（每题恒定） | 同上，`rerank_pool_ids` |
| 每题 Search LLM 调用 | **2**（structured plan + rank） | `app/model.py` + `app/storage.py` |
| 字符→token 换算 | **4.0 字符/token** | 保守英文近似；真实值以 `usage` 为准 |

---

## 3. Memory Construction

### 3.1 主实验（本地 Qwen3-4B）→ **$0 API**

| 项 | 值 |
|---|---|
| LLM calls | **272**（每 session 1 次；272 sessions） |
| 生成产物 | 4250 条 fact，status 全 `ok` |
| 平均 fact 长度 | 61.3 字符（p50 60 / p95 92 / max 154） |
| 输入 tokens（估） | 272 × (平均会话 668 tok + prompt 模板 ~250 tok) ≈ **250,000** |
| 输出 tokens（估） | 4250 × 61.3 / 4 ≈ **65,000** |
| API 成本 | **$0.00** |
| 状态 | **已完成**（`.locomo/session-facts-full.json`，2026-09-04） |

### 3.2 若改用 gpt-4o-mini 构建（仅用于 `aml-official` 对齐 profile）

| 项 | 值 |
|---|---|
| calls | 272 |
| 输入 tokens | ~250,000 |
| 输出 tokens | ~65,000 |
| API 成本 | ~$0.077（按 §5 费率） |

> ⚠️ 正式论文主实验**不采用**此路径：会改变与已发布 evidence-level 结果的
> 会话事实层一致性（见 `LOCOMO_E2E_EXPERIMENT.md` §2.3）。

---

## 4. Search

### 4.1 主实验（本地 Qwen3 reranker）→ **$0 API**

每次 Search 的 LLM 调用构成：

| 调用 | 输入估算 | 输出估算 |
|---|---|---|
| `plan_query_structured`（P1 结构化规划） | 问题 14 tok + system ~150 tok ≈ **200** | ≈ **200**（intent/core_terms/expansion_terms/entities/temporal_cues/evidence_needs；实测 `evidence_needs` 均值 3.94，max 4） |
| `rank_candidates`（证据 ID 重排） | 30 候选 × 36.3 tok = 1089 + rubric ~450 + 问题/选项 ~60 ≈ **1,600** | ≤ **400**（`max_tokens=400`，正常题远小于此） |
| **每题合计** | **≈ 1,800** | **≈ 300–600** |

每题 ÷ 每配置总量：

| 配置 | 题数 | Search calls | 输入 tokens | 输出 tokens |
|---|---:|---:|---:|---:|
| P1 | 1539 | 3,078 | ~2,770,000 | ~620,000 |
| P4-A + BM25 | 1539 | 3,078 | ~2,770,000 | ~620,000 |
| SF v2 + Qwen3-4B | 1539 | 3,078 | ~2,770,000 | ~620,000 |
| SF v2 + Qwen3-8B（scaling） | 1539 | 3,078 | ~2,770,000 | ~620,000 |
| **4 配置合计** | 6,156 | **12,312** | **~11,080,000** | **~2,480,000** |

**API 成本 = $0.00**（本地端点，loopback only）。

### 4.2 若改用 gpt-4o-mini 作 Search 模型

| 配置 | calls | 输入 tokens | 输出 tokens | API 成本 |
|---|---:|---:|---:|---:|
| 单配置 | 3,078 | ~2,770,000 | ~620,000 | **~$0.79** |
| 4 配置 | 12,312 | ~11,080,000 | ~2,480,000 | **~$3.15** |

---

## 5. Answer 与 Judge（唯一的实际付费项）

### 5.1 费率假设 ⚠️

| 项 | 值 |
|---|---|
| 模型 | `gpt-4o-mini` |
| 输入费率（假设） | **$0.150 / 1M tokens** |
| 输出费率（假设） | **$0.600 / 1M tokens** |

> **必须声明**：审计环境无法访问 OpenAI 官方定价页（DNS 受限），上表为业内广泛引用的
> gpt-4o-mini 费率 **假设**，**不是**已验证事实。执行前必须：
> 1. 从提供方当前价目表确认费率并记录到本文件；
> 2. 运行后以每题的 `usage.{prompt_tokens,completion_tokens}` 计算**实测**成本；
> 3. 若实际返回的 model id 与假设不符（例如指向新 snapshot），成本与可比性均需重新声明。
>
> 本文件所有美元数字都是**在费率假设下的估算**，不得当作实际账单。

### 5.2 Answer 调用

模板：Mem0 `ANSWER_PROMPT`（system，1427 字符 ≈ **357 tok**）+ memory 块 + 问题。

| top-k | 输入 tok/call | 总输入 tok (×1539) | 输出 tok/call（上限） | 总输出 tok |
|---:|---:|---:|---:|---:|
| 5 | 567 | 872,768 | ≤ 30 | 46,170 |
| **10（主）** | **749** | **1,152,105** | ≤ 30 | **46,170** |
| 20 | 1,112 | 1,710,777 | ≤ 30 | 46,170 |
| 30 | 1,475 | 2,269,449 | ≤ 30 | 46,170 |
| 60 | 2,564 | 3,945,467 | ≤ 30 | 46,170 |

### 5.3 Judge 调用

模板：Mem0 `ACCURACY_PROMPT`（1188 字符 ≈ **297 tok**）+ 问题 + gold + 生成答案。

| 项 | 值 |
|---|---|
| 输入 tok/call | **328** |
| 总输入 tok (×1539) | **505,169** |
| 输出 tok/call | ~37（1 句理由 + `{"label":...}`） |
| 总输出 tok | **56,943** |

### 5.4 单配置（`chrono-main`, top-k=10）总成本

| 阶段 | calls | 输入 tokens | 输出 tokens | 成本 |
|---|---:|---:|---:|---:|
| Memory construction | 272 | 250,000 | 65,000 | $0.00（本地） |
| Search | 3,078 | 2,770,000 | 620,000 | $0.00（本地） |
| Answer | 1,539 | 1,152,105 | 46,170 | $0.200 |
| Judge | 1,539 | 505,169 | 56,943 | $0.110 |
| **合计** | 6,428 | **4,677,274** | **788,113** | **≈ $0.31** |

### 5.5 全部配置总成本

| 配置 | top-k | Answer+Judge 输入 tok | Answer+Judge 输出 tok | 成本 |
|---|---:|---:|---:|---:|
| P1 | 10 | 1,657,274 | 103,113 | $0.31 |
| P4-A + BM25 | 10 | 1,657,274 | 103,113 | $0.31 |
| SF v2 + Qwen3-4B | 10 | 1,657,274 | 103,113 | $0.31 |
| SF v2 + Qwen3-8B | 10 | 1,657,274 | 103,113 | $0.31 |
| **合计（4 配置）** | | **6,629,096** | **412,452** | **≈ $1.24** |

含敏感性分析（top-k 20 复跑一次 SF v2 4B）：+$0.39 → **≈ $1.63**

---

## 6. 效率指标预算（§17 要求）

### 6.1 本地 GPU 墙钟时间（真正的瓶颈）

由已完成的 full-1976 运行外推：

| 运行 | 实测 | 外推到 1539 题 |
|---|---|---|
| Qwen3-4B（SF v2，200 题/约 28 分钟） | ~28 min / 200 q | **≈ 3.6 h / 配置** |
| Qwen3-8B（`_v8b_full.log`：41 窗口，约 2000–2600 s / 50 q） | ~37 min / 50 q | **≈ 19 h / 配置** |

| 计划 | 内容 | GPU 墙钟 |
|---|---|---|
| Stage 1 smoke | 20 题 × 4 配置 | ~1.5 h |
| Stage 2 fixed-100 | 100 题 × 4 配置 | ~6 h |
| Stage 3 full | 3 配置（P1 / P4A+BM25 / SF v2 4B） | **~11 h** |
| Stage 3 full + scaling | 追加 SF v2 8B | +19 h |
| **合计（含 scaling）** | | **≈ 37 h** |

> ⚠️ 这是**串行**估算。已有运行经验表明 conv-43（offset 758）在 llama-server 端会挂起，
> 必须配置 `--model-timeout` 并记录为 timeout，不得让它拖垮整段。

### 6.2 需要从第一次运行就记录的量（§17）

Memory construction / Search / Answer / Judge 各自：
`calls`、`input tokens`、`output tokens`、`wall-clock`、`latency`；
最终汇总：Avg tokens/query、Avg LLM calls/query、total tokens、total API cost、
**P50 / P95 latency**、failure rate、retry rate、memory construction cost。

> 现状：`docs/CHRONOHYBRIDMEM_METRIC_PROTOCOL.md` 已如实记录
> "per-query 延迟 / P50 / P95 ⛔ 未实现"。本任务的 harness 必须补齐，
> 并在 `results/locomo_e2e/` 落盘（§17 + §19）。

---

## 7. 分阶段成本闸门

| 阶段 | 题数 | API calls | 估算成本 | 准入条件 |
|---|---:|---:|---:|---|
| 离线单元测试 | 0 | 0 | $0.00 | 全绿 |
| Stage 1 smoke | 20 | 80 | **$0.02** | 覆盖多 category；验证 checkpoint/resume/token 记账 |
| Stage 2 fixed-100 | 100 | 400 | **$0.09** | 分层抽样；验证超时/重试/成本/延迟/持久化 |
| Stage 3 full | 1539 | 12,312 | **$1.24** | Stage 1+2 全通过 |
| 敏感性（top-k 20） | 1539 | 3,078 | **$0.39** | 可选 |

**累计上限（全部执行）= 约 $1.75**。建议将**硬上限设为 $5.00**，并在 harness 中实现
**运行中累计成本闸门**（超过上限即停止且不写出部分结果）。

---

## 8. 运行前必须具备的记账字段（§9 要求）

每次真实 API 返回必须落盘：

- `model`（请求值）与**响应中真实返回的 model id / snapshot**
- 请求参数：`temperature`、`max_tokens`、prompt 版本、**prompt hash**
- `usage.prompt_tokens` / `usage.completion_tokens` / `usage.total_tokens`
- 单次 `latency_ms`
- 失败时的错误类型与是否重试

**文档必须写明**：若原论文使用的旧 snapshot 已不可访问，则标注
`reproduced with currently available GPT-4o-mini API version`，**不得**假装完全等价。

---

## 9. 批准清单（需用户确认）

- [ ] 接受"本地 Add/Search + gpt-4o-mini Answer/Judge"的成本结构（≈$0.31/配置）
- [ ] 接受全部 4 配置的估算总额 ≈$1.24（含 scaling），硬上限 $5.00
- [ ] 接受 ≈37 h 本地 GPU 墙钟（含 8B scaling）
- [ ] 确认 §5.1 费率假设，或提供实际费率
- [ ] 授权 Stage 1（20 题，≈$0.02）作为首次付费调用
