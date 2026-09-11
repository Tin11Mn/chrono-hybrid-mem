# prompts/ — LoCoMo E2E 逐字 prompt 副本

每个 prompt 文件**逐字**来自对应方法的一手来源（不改写、不重排、不增删字符），
并附 SHA256。`prompts/locomo_*.txt` 是本 harness 唯一读取的 prompt；
若改动了 prompt 文本，hash 会变，运行配置里的 `config_digest` 随之失效（fail-closed）。

## 索引

| 文件 | profile | 用途 | 来源 | SHA256 |
|---|---|---|---|---|
| `locomo_answer_mem0.txt` | `mem0` | Answer | Mem0 P1 harness `ANSWER_PROMPT`（`mem0ai/mem0@2274b5ac/evaluation/`，论文附录 A 同文） | `748EB924…CCA520D` |
| `locomo_judge_mem0.txt` | `mem0` | Judge（二值） | Mem0 P1 `ACCURACY_PROMPT`（`evaluation/metrics/llm_judge.py`，论文自述改编自 MemGPT / Packer et al. 2023） | `6D33FF20…B37E853` |

## Mem0 prompt 关键事实（核对点）

- **Answer**：Jinja 占位符 `{{speaker_1_user_id}}` / `{{speaker_1_memories}}` 等。
  `speaker_N_user_id` = speaker **名字**；`speaker_N_memories` =
  `json.dumps(["<timestamp>: <memory>", …], indent=4)`（按时间升序）。
  末行为裸 `Answer:`，模型续写答案。system message 发送。
- **Judge**：user message，`response_format={"type":"json_object"}`，`temperature=0.0`，
  解析 `json.loads(extract_json(...))["label"]` ∈ `CORRECT|WRONG`。
  **保留** typographic quote `’`（U+2019）。

## 当前已实现的 profile

按用户裁决（item 5）：**只实现 `chrono-main` + `mem0` 两个主 profile**。
其余（LightMem / MemoryART / MemoryOS / AML）本轮**只做 as-reported 对齐，不运行**，
因此**不复制它们的 prompt**（避免假装对齐）。如需扩展，在新增 profile 时
以同样方式（逐字 + 来源 + hash）加入。

## 占位符约定

| 占位符 | 含义 |
|---|---|
| `{{speaker_1_user_id}}` / `{{speaker_2_user_id}}` | 对话两位 speaker 的名字 |
| `{{speaker_1_memories}}` / `{{speaker_2_memories}}` | 该 speaker 名下检索到的 evidence，JSON 数组 `"<timestamp>: <memory>"` |
| `{{question}}` | benchmark question |
| `{question}` / `{gold_answer}` / `{generated_answer}` | judge 模板（`.format`） |

## 泄漏约束（强制执行）

- Answer / judge prompt **绝不**包含 `qa.answer` / `qa.adversarial_answer` / `qa.evidence` /
  `observation` / `session_summary` / `event_summary`。
- Answer prompt 只含**检索返回的原始 evidence**（`result_ids` → 消息文本）+ 时间戳 + 问题。
- gold answer 只在 **judge** 中出现，judge 不参与检索或答案生成。
