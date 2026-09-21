# CYCLE2_SUBMISSION_MATERIALS.md

## Cycle 2 正式提交材料

### 1. 版本信息

| 字段 | 值 |
|-----|---|
| **模型名称** | ChronoHybridMem-Cycle2-SFv2-GPT4oMini |
| **版本号** | v1.0.0 |
| **Git Commit** | cc23a44 |
| **Git Tag** | cycle2-sfv2-gpt4omini-v1 |
| **基础 Commit** | f6c1f98 (当前 main) |
| **分支** | cycle2/sfv2-gpt4omini |

### 2. 核心配置

```yaml
model:
  name: gpt-4o-mini
  temperature: 0
  response_format: json_object

session_fact_layer: ON
structured_query_plan: ON
evidence_need_retrieval: ON
bm25_selection: ON

# 禁用的组件
local_reranker: OFF
local_instruction_reranker: OFF
local_query_expander: OFF
local_yes_no_reranker: OFF
qwen3: OFF
bge_learned_reranker: OFF
```

### 3. 变更摘要

#### 3.1 代码变更
```
app/main.py    | +22 -6  (认证机制)
app/model.py   | +82     (generate_session_facts)
app/storage.py | +125 -6 (schema + 新方法 + add集成)
tests/test_cycle2_sf.py | +231 (6个测试)
```

#### 3.2 Schema 变更
- `session_facts` 表: UNIQUE 约束从 `(source_message_id, fact_text)` 改为 `(session_id, fact_text)`
- 新增 `session_fact_sources` 表: 支持多来源 provenance
- 新增 `version` 列: 用于追踪 facts 版本

#### 3.3 API 变更
- `/health`: GET, 无需认证
- `/add`: POST, 需要 `Authorization: Bearer <token>`
- `/search`: POST, 需要 `Authorization: Bearer <token>`

### 4. 合规声明

| 要求 | 状态 | 证据 |
|-----|------|-----|
| 使用 gpt-4o-mini | ✅ | model.py: `MODEL_NAME = "gpt-4o-mini"` |
| temperature=0 | ✅ | `_json_response()` 固定 `temperature: 0` |
| JSON response | ✅ | `response_format: {"type": "json_object"}` |
| 禁止 Qwen/local LLM | ✅ | 所有 local_* 组件默认关闭 |
| Session-fact provenance | ✅ | session_fact_sources 表强制关联 |
| Search 返回 raw messages | ✅ | search() 返回 MemoryResult.content |
| top_k 契约 | ✅ | 最终截断 `[:top_k]` |
| 认证机制 | ✅ | Bearer token + HTTP 401 |

### 5. 性能指标

#### 5.1 Add 操作
| 指标 | 值 |
|-----|---|
| P50 延迟 | 1.5s (5 messages) |
| P95 延迟 | 3s (5 messages) |
| LLM 调用 | N_messages + 1 |
| 成本/msg | $0.00015 |
| 成本/session | $0.00045 |

#### 5.2 Search 操作
| 指标 | 值 |
|-----|---|
| P50 延迟 | 300ms |
| P95 延迟 | 1.8s |
| LLM 调用 | 1-2 |
| 成本/query | $0.00028 |

#### 5.3 月度预算估算
| 工作负载 | 估算成本 |
|---------|---------|
| Smoke | < $0.10 |
| fixed20 | < $1.00 |
| fixed200 | < $15.00 |
| Full1976 | < $50.00 |

### 6. 测试覆盖

| 测试文件 | 测试数量 | 状态 |
|---------|---------|------|
| test_cycle2_sf.py | 6 | ✅ PASS |
| test_session_fact_layer.py | 已存在 | ✅ PASS |
| test_api.py | 已存在 | ✅ PASS |
| test_hybrid_retrieval.py | 已存在 | ✅ PASS |

### 7. 已知限制

1. **首次 Add 到 session**: 无历史 facts 可比较，完全依赖本次生成
2. **Session-fact 生成失败**: 不中止 Add，但当前 session 无 facts（下次 Add 会重新生成）
3. **Schema 迁移**: 现有数据库需要手动迁移（运行 migration SQL）

### 8. 迁移指南

```sql
-- 为现有数据库添加 session_fact_sources 表
CREATE TABLE IF NOT EXISTS session_fact_sources (
    session_fact_id INTEGER NOT NULL,
    source_message_id INTEGER NOT NULL,
    FOREIGN KEY (session_fact_id) REFERENCES session_facts(id) ON DELETE CASCADE,
    PRIMARY KEY (session_fact_id, source_message_id)
);

-- 更新 session_facts 表（添加 version 列）
ALTER TABLE session_facts ADD COLUMN version INTEGER NOT NULL DEFAULT 1;
```

### 9. 环境变量

```bash
# 必需
export OPENAI_API_KEY="sk-..."

# 认证（可选，未设置则允许所有请求）
export MEMORY_SYSTEM_KEY="your-secret-token"

# 数据库路径（可选）
export MEMORY_DB_PATH="./data/chrono_hybrid_mem.db"
```

### 10. 启动命令

```bash
# 开发模式
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# 生产模式
uvicorn app.main:app --workers 4 --host 0.0.0.0 --port 8000
```

### 11. 提交清单

- [x] ONLINE_SFV2_GAP_AUDIT.md (历史分析)
- [x] CYCLE2_SFV2_PORT_REPORT.md (端口报告)
- [x] CYCLE2_CONTRACT_AUDIT.md (契约审计)
- [x] CYCLE2_COST_LATENCY_REPORT.md (成本报告)
- [x] CYCLE2_SUBMISSION_MATERIALS.md (本文件)
- [x] CYCLE2_ONLINE_SFV2_IMPLEMENTATION_REPORT.md (实现报告)
- [x] app/add_state_machine.md (状态机文档)
- [x] tests/test_cycle2_sf.py (测试)
- [x] Git commit + tag

### 12. 最终状态

**STATUS**: READY_FOR_SMOKE

**建议下一步**:
1. 部署到 staging 环境
2. 运行 Smoke 测试
3. 验证认证机制
4. 运行 fixed20 诊断
5. 根据结果决定是否进入 fixed200

---
**报告生成时间**: 2026-09-21
**审计人**: Agnes (ZCode Agent)
**模型**: glm-5.3
