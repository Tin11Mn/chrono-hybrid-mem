# 历史研究报告说明

本文件记录的是 SFv2 研究接线过程，不是当前标准 Docker 提交版本的能力声明。
当前提交材料将 SFv2 标记为 OFF，直到它通过 `app.main`、运行依赖和端到端评测的完整验证。

# CYCLE2_SFV2_PORT_REPORT.md

## SF v2 Online Port Report

### 1. 问题诊断

**原问题**: 仓库中的 SF v2 是 offline session-fact cache，不是 Cycle 2 /add 自动构建的在线能力。

**根本原因**:
- 原始 `add()` 方法不调用 `add_session_facts()`
- `add_session_facts()` 是独立方法，需要外部工具生成 facts 后手动调用
- 没有会话级别的 fact 生成机制

### 2. 解决方案

**新增方法**: `MemoryModel.generate_session_facts()`

```python
def generate_session_facts(
    self,
    user_id: str,
    session_id: str,
    messages: List[Dict[str, object]],
) -> List[Dict[str, object]]
```

**特点**:
- 使用 `gpt-4o-mini` (temperature=0, JSON response)
- 输入: 完整 session 的有序 raw messages + memory IDs
- 输出: `{"facts": [{"fact_text": "...", "source_message_ids": ["mem_N", ...]}]}`
- 强制验证 provenance: 每个 source ID 必须在 allowlist 中

### 3. 集成点

**位置**: `app/storage.py` → `MemoryStore.add()`

在 add() 末尾添加:
```python
if self.session_fact_layer and self.model:
    try:
        session_msgs = self.get_session_messages_for_facts(
            request.user_id, request.session_id
        )
        generated_facts = self.model.generate_session_facts(...)
        self.replace_session_facts(...)
    except Exception as exc:
        logging.getLogger("chrono_hybrid_mem").warning(
            "session_fact_generation_failed: %s", exc
        )
```

**失败处理**: Session-fact 生成失败不中止 Add，raw messages 已持久化。

### 4. Schema 变更

#### 4.1 session_facts 表重构
```sql
-- 旧 schema
UNIQUE(source_message_id, fact_text)

-- 新 schema  
UNIQUE(session_id, fact_text)
```

#### 4.2 新增 session_fact_sources 表
```sql
CREATE TABLE session_fact_sources (
    session_fact_id INTEGER NOT NULL,
    source_message_id INTEGER NOT NULL,
    FOREIGN KEY (session_fact_id) REFERENCES session_facts(id) ON DELETE CASCADE,
    PRIMARY KEY (session_fact_id, source_message_id)
);
```

**支持多来源 provenance**: 一个 fact 可关联多个 raw messages。

### 5. 幂等性保证

**request_id 机制**:
- 相同 request_id → 直接返回成功（不重复写入 raw messages）
- Session facts 基于完整 session 重新生成，自然幂等

**会话刷新语义**:
- 每次 Add 到同一 session → 替换所有旧 facts
- 新 facts 基于当前完整 session 生成
- 不会产生 stale facts 累积

### 6. 测试覆盖

| 测试用例 | 验证点 |
|---------|--------|
| test_online_session_facts_generated_on_add | Add 后自动生成 facts |
| test_session_facts_are_replaced_not_accumulated | 多 chunk 后 facts 被替换 |
| test_cross_user_isolation | user_id 隔离 |
| test_request_id_idempotency | request_id 幂等 |
| test_provenance_chain_valid | provenance 可追溯 |
| test_session_facts_not_returned_in_search | Search 只返回 raw messages |

### 7. 合规检查

| 要求 | 状态 |
|-----|------|
| 使用 gpt-4o-mini | ✅ |
| temperature=0 | ✅ |
| JSON response | ✅ |
| Provenance 可追溯 | ✅ |
| 原子替换 semantics | ✅ |
| 失败不中止 Add | ✅ |
| 多 chunk 正确刷新 | ✅ |
| 跨 user/session 隔离 | ✅ |

### 8. 性能影响

**Add 操作**:
- 原有: 1 LLM call (message-level extraction)
- 新增: +1 LLM call (session-fact generation)
- 总成本: ~2x per Add（可接受）

**Search 操作**:
- 无变化（session facts 作为 retrieval aid，不改变 search 逻辑）

### 9. 安全变更

**认证机制**:
- `/health`: 无需认证
- `/add`: Bearer token 必需
- `/search`: Bearer token 必需
- 密钥来自环境变量 `MEMORY_SYSTEM_KEY`
- 缺失/错误 → HTTP 401

### 10. 部署说明

**环境变量**:
```bash
export MEMORY_SYSTEM_KEY="your-secret-token"
export OPENAI_API_KEY="<runtime-secret>"
```

**启动**:
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

**API 使用**:
```bash
# Health check (no auth)
curl http://localhost:8000/health

# Add with auth
curl -X POST http://localhost:8000/add \
  -H "Authorization: Bearer your-token" \
  -H "Content-Type: application/json" \
  -d '{"request_id":"1","user_id":"u1","session_id":"s1","messages":[{"role":"user","content":"Hi"}]}'

# Search with auth
curl -X POST http://localhost:8000/search \
  -H "Authorization: Bearer your-token" \
  -H "Content-Type: application/json" \
  -d '{"query":"What did Alice say?","user_id":"u1","top_k":10}'
```

---
**版本**: Cycle2-SFv2-GPT4oMini-v1
**Commit**: cc23a44
**Tag**: cycle2-sfv2-gpt4omini-v1
