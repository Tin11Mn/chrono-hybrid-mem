# ChronoHybridMem

ChronoHybridMem is a Docker-deployable textual-memory system for the Agent Memory Challenge academic-methods track. It stores conversation evidence and returns ranked source records. It does not generate benchmark answers.

## 中文说明

### 项目定位

ChronoHybridMem 面向“文本记忆”场景，目标是从长期对话中检索能够直接支持用户问题的原始证据。系统的输出是记忆记录及其排序，而不是由模型编造的最终答案。

项目元数据：

- 作者：孟昊轩（Haoxuan Meng）
- 机构：郑州大学（Zhengzhou University）
- 联系邮箱：hxmeng@gs.zzu.edu.cn
- 赛道：文本记忆 / 学术方法榜
- 部署方式：公开 GitHub 仓库，由平台构建 Docker 并部署
- 许可证：[MIT](LICENSE)

### 核心流程

```text
写入 Add：请求校验 -> SQLite 持久化原始消息 -> 可选 gpt-4o-mini 事实抽取 -> FTS5 索引

检索 Search：查询规划 -> 精确 user_id 隔离 -> 多路 FTS5 候选召回
             -> RRF/可选 dense 融合 -> gpt-4o-mini 对已有证据 ID 排序
             -> 返回原始证据记录
```

模型只负责事实抽取、查询规划和候选排序，不负责生成评测答案。所有返回内容都来自已经写入的原始记忆。

### 线上竞赛路径

主分支使用平台提供的 `gpt-4o-mini`。当前线上候选版本强化了证据排序提示，要求模型：

1. 优先选择原始消息中直接陈述问题所需事实的证据；
2. 遵守 `latest`、`previous`、`before`、`current` 等明确时间约束；
3. 多跳问题优先选择包含决定性关系的消息；
4. 避免只与主题相关、但不能直接回答问题的候选；
5. 不使用世界知识进行补充推断；
6. 只返回输入候选中的原始 evidence ID。

平台不会提供 gpt-4o-mini 预评测环境，因此本仓库不宣称本地代理实验结果等于官方排行榜成绩。最终成绩以平台部署和评测结果为准。

#### P1 结构化查询规划

P1 将现有 `gpt-4o-mini` 查询规划调用改为有界的结构化结果，把核心词、实体和
时间线索送入主要检索通道，把扩展词和证据需求送入低权重辅助通道。它不增加模型
调用次数，不改变 Add/Search API，也不生成评测答案。竞赛路径默认启用；设置
`MEMORY_STRUCTURED_QUERY_PLAN=false` 可回退到扁平规划器进行消融。

在固定 20 题 LoCoMo 开发切片上，以 Qwen3-4B 作为 `gpt-4o-mini` 本地代理时，
Hit@1 从 0.55 提升至 0.60，MRR 从 0.5917 提升至 0.6125，Hit@10 保持 0.65；
exact evidence recall@10 从 0.4839 降至 0.4194。该结果仅用于方法筛选，不是官方成绩。

### 本地研究方法

仓库还保留可选的本地研究后端，用于离线方法筛选：

- 多路 BM25/FTS5 检索；
- Porter 词干化检索；
- 邻近消息上下文检索；
- BGE dense embedding 检索；
- lexical+dense z-score 融合；
- 时间感知 dense key；
- Qwen3-Reranker 官方 yes/no logprob 重排；
- 固定 20 题、200 题和全量 LoCoMo 分段门控。

本地方法不会改变 `/add` 和 `/search` API。模型权重、LoCoMo 数据集和 API 密钥均不提交到 Git。

当前记录的最佳本地代理结果为：

| 方法 | Hit@1 | Hit@3 | Hit@10 | MRR |
|---|---:|---:|---:|---:|
| v0.2 基线 | 0.2671 | 0.4355 | 0.5695 | 0.3567 |
| v0.3 本地混合检索 | 0.4355 | 0.6186 | 0.7577 | 0.5183 |
| Qwen3-Reranker | 0.5195 | 0.6763 | 0.7577 | 0.5812 |
| Qwen + 时间感知 dense key | **0.5225** | **0.6808** | **0.7653** | **0.5856** |
| P1 本地 Qwen 结构化规划 | **0.5761** | **0.7157** | **0.7618** | **0.6479** |

以上是公开 LoCoMo 数据上的离线检索结果，不是官方平台成绩。评测使用严格的 exact evidence-turn matching。

### Docker 运行

竞赛兼容镜像：

```bash
docker build -t chrono-hybrid-mem:latest .
docker run --rm -p 8000:8000 \
  -v chrono-memory-data:/data \
  -e MEMORY_REQUIRE_MODEL=true \
  -e OPENAI_API_KEY=your_runtime_secret \
  chrono-hybrid-mem:latest
```

密钥只通过运行时环境变量提供，不写入镜像、代码或 Git 历史。

本地研究镜像：

```bash
docker build -f Dockerfile.local -t chrono-hybrid-mem:local .
docker run --rm -p 8000:8000 \
  -v chrono-memory-data:/data \
  -v chrono-local-models:/models \
  chrono-hybrid-mem:local
```

健康检查：`GET http://localhost:8000/health`

### API

#### `POST /add`

写入一批消息。相同 `request_id` 可以安全重试，系统保持幂等。

```json
{
  "request_id": "run:1:chunk:0",
  "user_id": "run:1:conversation:0",
  "session_id": "run:1:session:0",
  "messages": [
    {"role": "user", "content": "Alice prefers tea."}
  ]
}
```

#### `POST /search`

只搜索指定 `user_id` 的记忆，并返回最多 `top_k` 条原始证据。

```json
{
  "query": "What does Alice prefer?",
  "user_id": "run:1:conversation:0",
  "top_k": 10
}
```

返回示例：

```json
{
  "data": [
    {
      "id": "mem_1",
      "content": "Alice prefers tea.",
      "score": 1.0,
      "created_at": "2026-08-07T00:00:00Z"
    }
  ]
}
```

### 本地开发与测试

```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows PowerShell
.venv\Scripts\Activate.ps1

pip install -r requirements.txt
python -m pytest -q
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

本地 dense/reranker 实验使用 `requirements-local.txt`。模型缓存应放在 Git 忽略目录中。

### 离线评测

虚构示例可直接运行：

```bash
python scripts/evaluate_retrieval.py --cases examples/demo_eval.json
```

LoCoMo 评测数据通过本地路径传入，不复制到仓库：

```bash
python scripts/evaluate_locomo_retrieval.py \
  --dataset C:/path/to/locomo10.json \
  --top-k 1,3,10
```

完整协议和注意事项见 [docs/EXTERNAL_EVALUATION.md](docs/EXTERNAL_EVALUATION.md)。评测数据不得用于训练或提交，必须遵守数据集许可证和竞赛规则。

### 安全与运行注意事项

- 所有搜索都执行精确 `user_id` 隔离；
- SQLite WAL 支持并发读取和串行写入；
- `request_id` 保证 Add 重试幂等；
- 模型排序只返回已有候选 ID，不生成答案；
- `MEMORY_REQUIRE_MODEL=true` 且没有 `OPENAI_API_KEY` 时会快速失败；
- 不要把 API 密钥、模型权重或评测数据提交到仓库；
- Docker 默认数据库路径为 `/data/chrono_hybrid_mem.db`；
- 平台官方成绩只能由平台自己的 gpt-4o-mini 环境产生。

## English Documentation

### Project scope

ChronoHybridMem is a textual-memory retrieval service for the Agent Memory Challenge academic-methods track. It stores long-term conversation evidence and returns ranked source records. It does not generate benchmark answers.

Metadata:

- Author: Haoxuan Meng (孟昊轩)
- Institution: Zhengzhou University (郑州大学)
- Contact: hxmeng@gs.zzu.edu.cn
- Track: Textual Memory / Academic Methods
- Deployment: Public GitHub repository with platform-managed Docker deployment
- License: [MIT](LICENSE)

### Core pipeline

```text
Add: validate request -> persist raw messages in SQLite -> optional gpt-4o-mini fact extraction -> FTS5 indexing

Search: query planning -> exact user_id isolation -> multi-channel FTS5 retrieval
        -> RRF/optional dense fusion -> gpt-4o-mini ordering of supplied evidence IDs
        -> return original evidence records
```

The model is used for fact extraction, query planning, and candidate ordering. It is never used to generate benchmark answers. Returned content always comes from stored memory records.

### Competition-facing path

The main deployment path uses the platform-provided `gpt-4o-mini`. The current candidate strengthens the evidence-ordering rubric so the model should:

1. prefer the original message that directly states the requested fact;
2. obey explicit temporal constraints such as `latest`, `previous`, `before`, and `current`;
3. select the decisive relation for multi-step questions;
4. avoid candidates that are topically related but not directly evidential;
5. avoid world-knowledge inference; and
6. return only original evidence IDs supplied in the request.

The platform does not expose a gpt-4o-mini evaluation environment to participants. Local proxy metrics must therefore not be presented as official leaderboard results. The platform deployment is the source of truth for the final score.

#### P1 structured query planning

P1 changes the existing `gpt-4o-mini` query-planning call to return a bounded structured
plan. Core terms, entities, and temporal cues use the primary retrieval channels, while
expansion terms and evidence needs use low-weight support channels. It adds no model call,
does not change the Add/Search API, and never generates benchmark answers. The competition
path enables P1 by default; set `MEMORY_STRUCTURED_QUERY_PLAN=false` for a flat-planner
ablation.

On a fixed 20-question LoCoMo development slice using Qwen3-4B only as a local
`gpt-4o-mini` proxy, Hit@1 improved from 0.55 to 0.60 and MRR from 0.5917 to 0.6125,
while Hit@10 remained 0.65. Exact evidence recall@10 decreased from 0.4839 to 0.4194.
These measurements are method-selection evidence, not an official leaderboard result.

### Optional local research path

The repository also contains opt-in local components for offline research:

- multi-channel BM25/FTS5 retrieval;
- Porter-stemmed retrieval;
- neighboring-message context retrieval;
- BGE dense embeddings;
- lexical+dense z-score fusion;
- timestamp-aware dense keys;
- the official Qwen3-Reranker yes/no logprob protocol; and
- fixed 20-question, fixed 200-question, and segmented full-set gates.

These components preserve the public `/add` and `/search` API. Model weights, LoCoMo data, and credentials are excluded from Git.

The best recorded local proxy result is:

| Method | Hit@1 | Hit@3 | Hit@10 | MRR |
|---|---:|---:|---:|---:|
| v0.2 baseline | 0.2671 | 0.4355 | 0.5695 | 0.3567 |
| v0.3 local hybrid | 0.4355 | 0.6186 | 0.7577 | 0.5183 |
| Qwen3-Reranker | 0.5195 | 0.6763 | 0.7577 | 0.5812 |
| Qwen + timestamp-aware dense key | **0.5225** | **0.6808** | **0.7653** | **0.5856** |
| P1 local Qwen structured planner | **0.5761** | **0.7157** | **0.7618** | **0.6479** |

These are offline LoCoMo retrieval-only results using strict exact evidence-turn matching, not official platform scores.

### Docker

Competition-compatible image:

```bash
docker build -t chrono-hybrid-mem:latest .
docker run --rm -p 8000:8000 \
  -v chrono-memory-data:/data \
  -e MEMORY_REQUIRE_MODEL=true \
  -e OPENAI_API_KEY=your_runtime_secret \
  chrono-hybrid-mem:latest
```

The API key is supplied only at runtime. It is not stored in the image, source tree, or Git history.

Optional local research image:

```bash
docker build -f Dockerfile.local -t chrono-hybrid-mem:local .
docker run --rm -p 8000:8000 \
  -v chrono-memory-data:/data \
  -v chrono-local-models:/models \
  chrono-hybrid-mem:local
```

Health check: `GET http://localhost:8000/health`

### API reference

#### `POST /add`

Adds a batch of messages. Retrying the same `request_id` is idempotent.

```json
{
  "request_id": "run:1:chunk:0",
  "user_id": "run:1:conversation:0",
  "session_id": "run:1:session:0",
  "messages": [{"role": "user", "content": "Alice prefers tea."}]
}
```

#### `POST /search`

Searches only the requested `user_id` and returns up to `top_k` original evidence records.

```json
{
  "query": "What does Alice prefer?",
  "user_id": "run:1:conversation:0",
  "top_k": 10
}
```

Example response:

```json
{
  "data": [{
    "id": "mem_1",
    "content": "Alice prefers tea.",
    "score": 1.0,
    "created_at": "2026-08-07T00:00:00Z"
  }]
}
```

### Development and tests

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m pytest -q
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Install `requirements-local.txt` for optional dense and local-reranker experiments. Keep model caches outside version control.

### Offline evaluation

Run the fictional built-in example:

```bash
python scripts/evaluate_retrieval.py --cases examples/demo_eval.json
```

Run the public LoCoMo evaluator with a local dataset path:

```bash
python scripts/evaluate_locomo_retrieval.py \
  --dataset C:/path/to/locomo10.json \
  --top-k 1,3,10
```

See [docs/EXTERNAL_EVALUATION.md](docs/EXTERNAL_EVALUATION.md) for the complete protocol. Evaluation data must not be used for training or submitted to Git.

### Security and operations

- Every search enforces exact `user_id` isolation.
- SQLite WAL supports concurrent reads and serialized writes.
- `request_id` makes Add retries idempotent.
- Model ranking returns existing candidate IDs and never generates answers.
- `MEMORY_REQUIRE_MODEL=true` fails fast without `OPENAI_API_KEY`.
- Never commit API keys, model weights, or evaluation data.
- Docker stores the default database at `/data/chrono_hybrid_mem.db`.
- Only the platform's gpt-4o-mini deployment can produce the official score.

## License

Released under the [MIT License](LICENSE).
