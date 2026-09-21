# Add Transaction State Machine

## Cycle 2 Compliance: Online SFv2 Add Protocol

### State Definitions

| State | Description |
|-------|-------------|
| `PENDING` | Request received, processing started |
| `COMPLETED` | All steps successful, request_id marked |
| `FAILED_RETRYABLE` | Session-fact generation failed, raw messages persisted |

### Add Flow

```
1. BEGIN TRANSACTION
2. INSERT INTO ingestions(request_id, ...)
   - If IntegrityError (duplicate request_id):
     → Check state:
       - COMPLETED: return success (idempotent)
       - PENDING/FAILED: allow retry
3. INSERT raw_messages (all chunks)
4. INSERT message-level facts
5. INSERT/update session_porter_fts (full session rebuild)
6. [NEW] Generate session facts via gpt-4o-mini:
   a. get_session_messages_for_facts() → ordered messages
   b. model.generate_session_facts() → parsed facts
   c. replace_session_facts() → atomic REPLACE
7. COMMIT
8. Mark request_id COMPLETED
```

### Failure Handling

| Failure Point | Behavior | Recovery |
|--------------|----------|----------|
| raw_messages INSERT | Rolls back entire transaction | Retry with same request_id |
| message fact extraction | Logged warning, continue | Raw messages still saved |
| session_fact generation | Logged warning, continue | Raw messages + message facts still valid |
| replace_session_facts | ROLLBACK within method | Raw messages preserved |

### Idempotency

- Same `request_id` → immediately returns success (no-op)
- No duplicate raw_messages possible
- Session facts are atomically replaced each Add call

### Provenance Chain

```
session_fact.id
  → session_fact_sources.source_message_id
    → raw_messages.id
      → raw_messages.content (original truth source)
```

Search returns only `raw_messages.content`, never `session_fact` text directly.
