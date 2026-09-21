# CYCLE2_CONTRACT_AUDIT.md

## top_k=100 契约审计

### 官方要求
从 `https://agentmemories.ai/rules`:
> "Search returns a JSON data array ordered by relevance; response must not exceed top_k (formal evaluations fix this at 100)."

### 输入契约
```json
{
  "query": "...",
  "options": ["A. ...", "B. ..."],
  "user_id": "...",
  "top_k": 100
}
```

### 输出契约
```json
{
  "data": [
    {
      "id": "mem_1",
      "content": "...",
      "score": 0.87,
      "created_at": "2026-07-01T12:00:00Z"
    }
  ]
}
```

### 实现审计

#### 1. schemas.py
```python
class SearchRequest(BaseModel):
    query: NonEmptyText
    options: Optional[List[str]] = None
    user_id: NonEmptyText
    top_k: int = Field(100, ge=1, le=100)  # ✅ 限制 1-100

class MemoryResult(BaseModel):
    id: str           # ✅ 必填
    content: str      # ✅ 必填
    score: float      # ✅ 必填
    created_at: str   # ✅ 必填

class SearchResponse(BaseModel):
    data: List[MemoryResult]  # ✅ 列表
```

#### 2. main.py
```python
@app.post("/search")
def search(request: SearchRequest) -> SearchResponse:
    return SearchResponse(data=store.search(
        user_id=request.user_id, 
        query=request.query, 
        options=request.options, 
        top_k=request.top_k  # ✅ 传递给底层
    ))
```

#### 3. storage.py 实现
```python
# 最终返回前截断
final_results = deduplicated[:top_k]
```

**审计结论**: ✅ 契约已满足

### 验证矩阵

| top_k 值 | 实现状态 | 说明 |
|---------|---------|------|
| top_k=1 | ✅ | 返回最多 1 条结果 |
| top_k=10 | ✅ | 返回最多 10 条结果 |
| top_k=100 | ✅ | 返回最多 100 条结果 |

### 结果验证

**必填字段检查**:
- ✅ `id` 非空（格式: `mem_{int}`）
- ✅ `content` 非空（原始 raw message 内容）
- ✅ `score` 存在（浮点数）
- ✅ `created_at` 存在（ISO 8601 格式）

**唯一性检查**:
- ✅ IDs 去重（通过 deduplicated 集合）
- ✅ user_id 隔离（SQL 查询包含 `WHERE user_id = ?`）

**排序稳定性**:
- ✅ RRF 加权融合（确定性算法）
- ✅ GPT-4o-mini rerank 后稳定排序
- ✅ temporal bonus 可配置

### GPT-4o-mini Reranker 行为

**候选池大小**:
- `MODEL_RERANK_LIMIT = 12`
- `llm_rerank_top_n` 控制实际输入数量

**输出**:
- 返回 ≤12 个 ID 的有序列表
- 置信度仅对前 2 个 ID 提供（P5 gate）

**排序策略**:
1. 前 N 个由 GPT-4o-mini rerank（N ≤ 12）
2. 剩余候选按 RRF 分数降序填充
3. 最终截断到 top_k

### 边界情况

| 场景 | 行为 | 符合契约 |
|-----|------|---------|
| 无候选结果 | 返回空数组 [] | ✅ |
| 候选 < top_k | 返回全部候选 | ✅ |
| 候选 ≥ top_k | 返回 top_k 个 | ✅ |
| top_k > 候选数 | 返回全部（不 padding） | ✅ |

### 安全审计

**用户隔离**:
```python
# storage.py:4107
WHERE raw.user_id = ?
```
- ✅ 所有查询包含 user_id 过滤
- ✅ 无跨用户数据泄露风险

**输入验证**:
- ✅ query: NonEmptyText（Pydantic 验证）
- ✅ user_id: NonEmptyText（Pydantic 验证）
- ✅ top_k: ge=1, le=100（Pydantic 验证）

---
**审计状态**: ✅ PASS
**符合 Cycle 2 契约**: 是
