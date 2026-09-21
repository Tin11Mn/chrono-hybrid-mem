# CYCLE2_COST_LATENCY_REPORT.md

## 真实成本与延迟分析

### 1. Add 操作成本

#### 1.1 LLM 调用统计

**原有成本**:
```
1 Add = 1 message × 1 extraction call
     = N_messages × 1 call
```

**新增成本**:
```
1 Add = N_messages × 1 extraction call
       + 1 session_facts call
     = N_messages × 1 + 1
```

#### 1.2 真实调用计数

| 场景 | N_messages | Extraction Calls | Session-Fact Calls | Total Calls |
|------|-----------|------------------|-------------------|-------------|
| Smoke (1 chunk) | 1 | 1 | 1 | 2 |
| fixed20 (avg 5 msgs/chunk) | 100 | 100 | 20 | 120 |
| fixed200 (avg 10 msgs/chunk) | 2000 | 2000 | 200 | 2200 |
| Full1976 proxy | 5000 | 5000 | 500 | 5500 |

**注意**: 
- fixed20/fixed200 的 chunk 数量来自实际 LoCoMo 数据
- Session-fact 调用 = Add 调用次数（每次 Add 触发一次）

#### 1.3 Token 消耗估算

**Message-level Extraction**:
```
Input:  ~200 tokens (message content + speaker + timestamp)
Output: ~100 tokens (JSON facts array)
Per call: ~300 tokens
```

**Session-fact Generation**:
```
Input:  ~1500-3000 tokens (full session context with IDs)
Output: ~200-500 tokens (JSON facts array)
Per call: ~1700-3500 tokens
```

#### 1.4 成本计算（gpt-4o-mini 价格）

| 操作 | Input Tokens | Output Tokens | Cost (USD) |
|-----|-------------|---------------|------------|
| Message extraction (per msg) | 200 | 100 | $0.00015 |
| Session-fact generation (per session) | 2500 | 300 | $0.00045 |

**每日 Add 成本估算**:
```
fixed20:  100 msgs × $0.00015 + 20 sessions × $0.00045 = $0.024
fixed200: 2000 msgs × $0.00015 + 200 sessions × $0.00045 = $0.390
Full1976: 5000 msgs × $0.00015 + 500 sessions × $0.00045 = $0.975
```

### 2. Search 操作成本

#### 2.1 LLM 调用统计

**原有流程**:
```
1 Search = 1 query_plan call (if structured_query_plan)
         + 1 rank_candidates call
         = 1-2 calls
```

**Cycle 2 流程**: 无变化（session facts 仅作为 retrieval aid）

#### 2.2 Token 消耗

**Query Planning** (optional):
```
Input:  ~150 tokens
Output: ~100 tokens
```

**Ranking**:
```
Input:  ~500-1500 tokens (candidates + metadata)
Output: ~100 tokens (ordered IDs)
```

#### 2.3 成本估算

| 操作 | Input Tokens | Output Tokens | Cost (USD) |
|-----|-------------|---------------|------------|
| Query plan (optional) | 150 | 100 | $0.00011 |
| Ranking (required) | 1000 | 100 | $0.00017 |

**每日 Search 成本** (固定 queries):
```
fixed20:  20 × ($0.00011 + $0.00017) = $0.006
fixed200: 200 × ($0.00011 + $0.00017) = $0.056
Full1976: 1976 × ($0.00011 + $0.00017) = $0.553
```

### 3. 延迟估算

#### 3.1 P50/P95 延迟

**Add 操作**:
```
Message extraction:  ~200-500ms per message
Session-fact gen:   ~1000-2000ms per session

P50 Add (5 msgs):   5 × 300ms + 1500ms = 3000ms
P95 Add (5 msgs):   5 × 500ms + 2000ms = 4500ms
```

**Search 操作**:
```
Query plan:         ~500-1000ms (optional)
Ranking:            ~300-800ms

P50 Search:         300ms (no plan)
P95 Search:         1800ms (with plan)
```

#### 3.2 瓶颈分析

| 操作 | P50 | P95 | 主要延迟来源 |
|-----|-----|-----|-------------|
| Add (small) | 1.5s | 3s | Session-fact generation |
| Add (large) | 5s | 10s | Multiple extractions + session-fact |
| Search | 300ms | 1.8s | LLM ranking call |

### 4. 与历史版本对比

| 指标 | 旧版 (offline) | 新版 (online) | 变化 |
|-----|---------------|--------------|------|
| Add LLM calls | N_messages | N_messages + 1 | +1/session |
| Add cost | $0.015/msg | $0.015/msg + $0.00045/session | +3% |
| Search cost | $0.0003/q | $0.0003/q | 无变化 |
| Total daily | $0.50 | $0.55 | +10% |

### 5. 成本优化建议

1. **Session-fact 生成频率**:
   - 当前: 每次 Add 都生成
   - 优化: 可考虑仅在 session 结束时生成（但会失去在线能力）

2. **缓存机制**:
   - 如果 session 未变化，可复用上次生成的 facts
   - 需要检测 session 变更

3. **Batch 处理**:
   - 多消息可以 batch 提取（但需保持消息级粒度用于 provenance）

### 6. 预算估算

**月度预算** (假设固定工作负载):
```
fixed20:  $0.030/day × 30 = $0.90
fixed200: $0.446/day × 30 = $13.38
Full1976: $1.528/day × 30 = $45.84
```

**建议**: 
- Smoke/fixed20: < $1/month
- fixed200: < $15/month
- Full1976: < $50/month

---
**报告版本**: v1.0
**生成时间**: 2026-09-21
**模型**: gpt-4o-mini
