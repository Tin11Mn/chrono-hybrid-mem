# CYCLE2_SUBMISSION_MATERIALS.md

## Cycle 2 正式提交材料

### 1. 版本信息

| 字段 | 值 |
|-----|---|
| **系统名称** | ChronoHybridMem-Cycle2-P4A-GPT4oMini |
| **版本号** | v0.4.0-cycle2-api |
| **Git Commit** | FINAL_COMMIT_TO_BE_FILLED_AFTER_FREEZE |
| **Git Tag** | FINAL_TAG_TO_BE_CREATED_AFTER_FREEZE |
| **基础分支** | cycle2/sfv2-gpt4omini |
| **分支** | cycle2/sfv2-gpt4omini |

### 2. 核心配置

```yaml
model:
  name: gpt-4o-mini
  temperature: 0
  response_format: json_object

session_fact_layer: OFF
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
export OPENAI_API_KEY="<runtime-secret-from-secret-manager>"
export OPENAI_BASE_URL="<optional-compatible-https-endpoint>"
export MEMORY_REQUIRE_MODEL="true"
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
| Session-fact provenance | 不作为当前提交能力 | 当前标准镜像不启用 SFv2；不要在申请中声称已开启 |
| BM25 evidence-need selection | ✅ | `MEMORY_NEED_SELECT_BY_BM25=true` |
| Search 返回 raw messages | ✅ | `search()` 返回 `MemoryResult.content` |
| top_k 契约 | ✅ | 最终截断 `[:top_k]` |
| 认证机制 | ✅ | Bearer token + HTTP 401 |

### 5. 性能指标

#### 5.1 Add 操作
| 指标 | 值 |
|-----|---|
| P50/P95 延迟 | 部署后压测填写 |
| LLM 调用 | 由实际配置和消息数量决定 |
| 成本 | 部署后按实际模型供应商填写 |

#### 5.2 Search 操作
| 指标 | 值 |
|-----|---|
| P50/P95 延迟 | 部署后压测填写 |
| LLM 调用 | 由实际配置决定 |
| 成本/query | 部署后按实际模型供应商填写 |

### 6. 变更历史

| Commit | Tag | 变更 |
|--------|-----|-----|
| cc23a44 | v1 | 基础实现 + 认证 |
| 3034acd | - | 文档材料 |
| 0b5163b | current | 文档和 Cycle 2 提交材料基线（提交前仍需最终冻结） |

### 7. 提交清单

- ✅ CYCLE2_SUBMISSION_MATERIALS.md (本文件)
- ⚠️ CYCLE2_SFV2_PORT_REPORT.md (历史研究报告，不代表当前提交能力)
- ✅ CYCLE2_CONTRACT_AUDIT.md (契约审计)
- ⚠️ CYCLE2_COST_LATENCY_REPORT.md (SFv2 历史估算，不是当前部署实测)
- ⚠️ CYCLE2_ONLINE_SFV2_IMPLEMENTATION_REPORT.md (历史研究报告，不代表当前提交能力)
- ⚠️ app/add_state_machine.md (SFv2 条件路径文档)
- ✅ tests/test_cycle2_sf.py (6 个合规测试)

### 8. 最终状态

**STATUS**: READY_FOR_DEPLOYMENT_REVIEW

**建议下一步**:
1. 撤销并轮换曾经暴露的旧 Key
2. 固定最终 commit/tag，并更新本文件
3. 部署持久化数据库和 HTTPS API
4. 运行 Smoke 测试并验证认证机制
5. 通过 Smoke 后再提交 Full

---
**报告生成时间**: 2026-09-21
**分支**: cycle2/sfv2-gpt4omini
**Commit**: FINAL_COMMIT_TO_BE_FILLED_AFTER_FREEZE
**Tag**: FINAL_TAG_TO_BE_CREATED_AFTER_FREEZE
