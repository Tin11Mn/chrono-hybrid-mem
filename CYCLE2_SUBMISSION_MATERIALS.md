# CYCLE2_SUBMISSION_MATERIALS.md

## Cycle 2 正式提交材料

### 1. 版本信息

| 字段 | 值 |
|-----|---|
| **模型名称** | ChronoHybridMem-Cycle2-SFv2-GPT4oMini |
| **版本号** | v1.0.0 |
| **Git Commit** | 8338376 |
| **Git Tag** | cycle2-sfv2-gpt4omini-v2 |
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

### 3. API 使用说明

#### 3.1 环境变量

```bash
export OPENAI_API_KEY="sk-24pAWCUG6dOKRdvlHyKwWExhibsbMWRJN6lluuCIPGCGDrjd"
export OPENAI_BASE_URL="https://api.chatanywhere.tech/v1"
# 可选：设置后启用认证
export MEMORY_SYSTEM_KEY="your-secret-token"
```

#### 3.2 端点

| 端点 | 方法 | 认证 |
|-----|------|-----|
| `/health` | GET | 无需 |
| `/add` | POST | Bearer token 必需 |
| `/search` | POST | Bearer token 必需 |

#### 3.3 示例请求

```bash
# Health check
curl http://localhost:8000/health

# Add
curl -X POST http://localhost:8000/add \
  -H "Authorization: Bearer your-token" \
  -H "Content-Type: application/json" \
  -d '{"request_id":"1","user_id":"u1","session_id":"s1","messages":[{"role":"user","content":"Hi"}]}'

# Search
curl -X POST http://localhost:8000/search \
  -H "Authorization: Bearer your-token" \
  -H "Content-Type: application/json" \
  -d '{"query":"What did Alice say?","user_id":"u1","top_k":10}'
```

### 4. 合规声明

| 要求 | 状态 | 证据 |
|-----|------|-----|
| 使用 gpt-4o-mini | ✅ | `model.py` - `MODEL_NAME = "gpt-4o-mini"` |
| temperature=0 | ✅ | `_json_response()` 固定 `temperature: 0` |
| JSON response | ✅ | `response_format: {"type": "json_object"}` |
| 禁止 Qwen/local LLM | ✅ | 所有 local_* 组件默认关闭 |
| Session-fact provenance | ✅ | `session_fact_sources` 表强制关联 |
| Search 返回 raw messages | ✅ | `search()` 返回 `MemoryResult.content` |
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

### 6. 变更历史

| Commit | Tag | 变更 |
|--------|-----|-----|
| cc23a44 | v1 | 基础实现 + 认证 |
| 3034acd | - | 文档材料 |
| 8338376 | v2 | 修复事务可见性 + API 端点配置 |

### 7. 提交清单

- ✅ CYCLE2_SUBMISSION_MATERIALS.md (本文件)
- ✅ CYCLE2_SFV2_PORT_REPORT.md (端口报告)
- ✅ CYCLE2_CONTRACT_AUDIT.md (契约审计)
- ✅ CYCLE2_COST_LATENCY_REPORT.md (成本/延迟报告)
- ✅ CYCLE2_ONLINE_SFV2_IMPLEMENTATION_REPORT.md (实现报告)
- ✅ app/add_state_machine.md (状态机文档)
- ✅ tests/test_cycle2_sf.py (6 个合规测试)

### 8. 最终状态

**STATUS**: READY_FOR_SMOKE

**建议下一步**:
1. 部署到平台
2. 运行 Smoke 测试
3. 验证认证机制 (Authorization: Bearer <token>)
4. 运行 fixed20 诊断
5. 根据结果决定是否进入 fixed200

---
**报告生成时间**: 2026-09-21
**分支**: cycle2/sfv2-gpt4omini
**Commit**: 8338376
**Tag**: cycle2-sfv2-gpt4omini-v2
