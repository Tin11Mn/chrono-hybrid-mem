import re
import sqlite3
import math
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, List, Optional

from .model import MemoryModel
from .schemas import AddRequest, MemoryResult


class MemoryStore:
    RRF_CONSTANT = 60
    CONTEXT_RRF_WEIGHT = 0.5
    # Recall-only support must not displace core-intent evidence.
    STRUCTURED_SUPPORT_RRF_WEIGHT = 0.01
    # Kept for controlled ablations; full LoCoMo evaluation showed that
    # hard entity binding harms adversarial cross-speaker questions overall.
    ENTITY_RRF_WEIGHT = 0.0
    QUERY_STOP_WORDS = {
        "a", "an", "and", "are", "at", "be", "can", "could", "did", "do", "does", "for",
        "from", "how", "i", "in", "is", "it", "me", "my", "of", "on", "or", "please",
        "tell", "that", "the", "this", "to", "was", "we", "were", "what", "when", "where",
        "which", "who", "why", "with", "would", "you", "your",
    }
    TEMPORAL_QUERY_PATTERN = re.compile(
        r"\b(current|currently|latest|recent|newest|now|today)\b|现在|目前|当前|最新|最近|如今",
        flags=re.IGNORECASE,
    )
    HISTORICAL_QUERY_PATTERN = re.compile(
        r"\b(previous|earlier|before|former|original|initially|used to)\b|之前|以前|曾经|过去|原来|最初",
        flags=re.IGNORECASE,
    )
    def __init__(self, database_path: str, model: Optional[MemoryModel] = None,
                 temporal_bonus: float = 0.0, semantic_retriever: object = None,
                 dense_rrf_weight: float = 1.0,
                 dense_fusion_alpha: Optional[float] = None,
                 dense_context_weight: float = 0.0,
                 dense_time_weight: float = 0.0,
                 local_reranker: object = None, rerank_top_n: int = 10,
                 session_fusion_weight: float = 0.0,
                 session_top_n: int = 0,
                 rerank_fusion_weight: Optional[float] = None,
                 local_instruction_reranker: object = None,
                 local_query_expander: object = None,
                 instruction_rerank_top_n: int = 10,
                 instruction_refine_top_n: int = 0,
                 structured_query_plan: bool = False) -> None:
        self.database_path = database_path
        self.model = model
        self.temporal_bonus = temporal_bonus
        self.semantic_retriever = semantic_retriever
        self.dense_rrf_weight = dense_rrf_weight
        self.dense_fusion_alpha = dense_fusion_alpha
        self.dense_context_weight = dense_context_weight
        self.dense_time_weight = dense_time_weight
        self.local_reranker = local_reranker
        self.rerank_top_n = rerank_top_n
        self.session_fusion_weight = session_fusion_weight
        self.session_top_n = session_top_n
        self.rerank_fusion_weight = rerank_fusion_weight
        self.local_instruction_reranker = local_instruction_reranker
        self.local_query_expander = local_query_expander
        self.instruction_rerank_top_n = instruction_rerank_top_n
        self.instruction_refine_top_n = instruction_refine_top_n
        self.structured_query_plan = structured_query_plan
        Path(database_path).parent.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.database_path, timeout=15)
        connection.row_factory = sqlite3.Row
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def initialize(self) -> None:
        with self._connection() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS ingestions (
                    request_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    completed_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS raw_messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    event_ts INTEGER,
                    sequence INTEGER NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS raw_messages_user_idx ON raw_messages(user_id);
                CREATE TABLE IF NOT EXISTS facts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    source_message_id INTEGER NOT NULL,
                    fact_text TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    UNIQUE(source_message_id, fact_text)
                );
                CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5(
                    message_id UNINDEXED,
                    user_id UNINDEXED,
                    content
                );
                CREATE VIRTUAL TABLE IF NOT EXISTS messages_porter_fts USING fts5(
                    message_id UNINDEXED,
                    user_id UNINDEXED,
                    content,
                    tokenize='porter unicode61'
                );
                CREATE VIRTUAL TABLE IF NOT EXISTS facts_fts USING fts5(
                    fact_id UNINDEXED,
                    user_id UNINDEXED,
                    content
                );
                CREATE VIRTUAL TABLE IF NOT EXISTS facts_porter_fts USING fts5(
                    fact_id UNINDEXED,
                    user_id UNINDEXED,
                    content,
                    tokenize='porter unicode61'
                );
                CREATE VIRTUAL TABLE IF NOT EXISTS context_fts USING fts5(
                    message_id UNINDEXED,
                    user_id UNINDEXED,
                    content
                );
                CREATE VIRTUAL TABLE IF NOT EXISTS context_porter_fts USING fts5(
                    message_id UNINDEXED,
                    user_id UNINDEXED,
                    content,
                    tokenize='porter unicode61'
                );
                CREATE VIRTUAL TABLE IF NOT EXISTS session_porter_fts USING fts5(
                    session_id UNINDEXED,
                    user_id UNINDEXED,
                    content,
                    tokenize='porter unicode61'
                );
                """
            )

    def add(self, request: AddRequest) -> None:
        now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        with self._connection() as connection:
            try:
                connection.execute(
                    "INSERT INTO ingestions(request_id, user_id, session_id, completed_at) VALUES (?, ?, ?, ?)",
                    (request.request_id, request.user_id, request.session_id, now),
                )
            except sqlite3.IntegrityError:
                # The unique constraint makes a retried request safe even if two Adds race.
                return
            inserted_messages = []
            for sequence, message in enumerate(request.messages):
                cursor = connection.execute(
                    """INSERT INTO raw_messages(user_id, session_id, role, content, event_ts, sequence, created_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (request.user_id, request.session_id, message.role, message.content,
                     message.timestamp, sequence, now),
                )
                message_id = cursor.lastrowid
                connection.execute(
                    "INSERT INTO messages_fts(message_id, user_id, content) VALUES (?, ?, ?)",
                    (message_id, request.user_id, message.content),
                )
                connection.execute(
                    "INSERT INTO messages_porter_fts(message_id, user_id, content) VALUES (?, ?, ?)",
                    (message_id, request.user_id, message.content),
                )
                inserted_messages.append((message_id, message.content))
                if self.model:
                    for fact in self.model.extract_facts(message.content):
                        try:
                            fact_cursor = connection.execute(
                                """INSERT INTO facts(user_id, source_message_id, fact_text, created_at)
                                   VALUES (?, ?, ?, ?)""",
                                (request.user_id, message_id, fact, now),
                            )
                        except sqlite3.IntegrityError:
                            continue
                        connection.execute(
                            "INSERT INTO facts_fts(fact_id, user_id, content) VALUES (?, ?, ?)",
                            (fact_cursor.lastrowid, request.user_id, fact),
                        )
                        connection.execute(
                            "INSERT INTO facts_porter_fts(fact_id, user_id, content) VALUES (?, ?, ?)",
                            (fact_cursor.lastrowid, request.user_id, fact),
                        )
            for index, (message_id, _) in enumerate(inserted_messages):
                window = inserted_messages[max(0, index - 1):index + 2]
                context = "\n".join(content for _, content in window)
                connection.execute(
                    "INSERT INTO context_fts(message_id, user_id, content) VALUES (?, ?, ?)",
                    (message_id, request.user_id, context),
                )
                connection.execute(
                    "INSERT INTO context_porter_fts(message_id, user_id, content) VALUES (?, ?, ?)",
                    (message_id, request.user_id, context),
                )
            # Rebuild this session document so repeated chunks with the same
            # session_id remain one lexical unit for hierarchical retrieval.
            session_rows = connection.execute(
                """SELECT content FROM raw_messages
                   WHERE user_id = ? AND session_id = ? ORDER BY id""",
                (request.user_id, request.session_id),
            ).fetchall()
            session_content = "\n".join(str(row["content"]) for row in session_rows)
            connection.execute(
                "DELETE FROM session_porter_fts WHERE user_id = ? AND session_id = ?",
                (request.user_id, request.session_id),
            )
            connection.execute(
                "INSERT INTO session_porter_fts(session_id, user_id, content) VALUES (?, ?, ?)",
                (request.session_id, request.user_id, session_content),
            )

    def search(self, *, user_id: str, query: str, options: Optional[List[str]] = None,
               top_k: int) -> List[MemoryResult]:
        expanded_queries = (
            self.local_query_expander.expand(query, options or [])
            if self.local_query_expander else []
        )
        raw_terms = re.findall(r"[\w]+", query, flags=re.UNICODE)
        terms = [term for term in raw_terms if term.casefold() not in self.QUERY_STOP_WORDS]
        terms = terms or raw_terms
        support_terms = []
        if self.model:
            if self.structured_query_plan:
                plan = self.model.plan_query_structured(query, options or [])
                for key in ("core_terms", "entities", "temporal_cues"):
                    values = plan.get(key, [])
                    if isinstance(values, list):
                        terms.extend(value for value in values if isinstance(value, str))
                for key in ("expansion_terms", "evidence_needs"):
                    values = plan.get(key, [])
                    if isinstance(values, list):
                        support_terms.extend(
                            value for value in values if isinstance(value, str)
                        )
            else:
                terms.extend(self.model.plan_query(query, options or []))
        unique_terms = []
        seen_terms = set()
        for term in terms:
            normalized = term.casefold()
            if normalized and normalized not in seen_terms:
                seen_terms.add(normalized)
                unique_terms.append(term)
        terms = unique_terms
        if not terms:
            return []
        match_query = " OR ".join('"{}"'.format(term.replace('"', '')) for term in terms)
        unique_support_terms = []
        seen_support_terms = set()
        for term in support_terms:
            normalized = term.casefold().strip()
            if normalized and normalized not in seen_terms and normalized not in seen_support_terms:
                seen_support_terms.add(normalized)
                unique_support_terms.append(term)
        support_match_query = (
            " OR ".join(
                '"{}"'.format(term.replace('"', '')) for term in unique_support_terms
            )
            if unique_support_terms else None
        )
        entity_terms = [
            term for index, term in enumerate(raw_terms)
            if index > 0 and len(term) > 1 and term[0].isupper()
        ][:2]
        entity_keys = {term.casefold() for term in entity_terms}
        content_terms = [term for term in terms if term.casefold() not in entity_keys]
        structured_match_query = None
        if entity_terms and content_terms:
            entity_clause = " OR ".join(
                '"{}"'.format(term.replace('"', '')) for term in entity_terms
            )
            content_clause = " OR ".join(
                '"{}"'.format(term.replace('"', '')) for term in content_terms
            )
            structured_match_query = "({}) AND ({})".format(entity_clause, content_clause)
        # Retrieve a broader pool than the response size so fusion and temporal
        # ranking can compare candidates that would otherwise be cut off early.
        candidate_limit = min(max(top_k * 4, 50), 200)
        with self._connection() as connection:
            raw_rows = connection.execute(
                """SELECT raw.id, raw.content, raw.created_at, raw.event_ts
                   FROM messages_fts
                   JOIN raw_messages AS raw ON raw.id = messages_fts.message_id
                   WHERE messages_fts MATCH ? AND messages_fts.user_id = ?
                   ORDER BY bm25(messages_fts), raw.id DESC
                   LIMIT ?""",
                (match_query, user_id, candidate_limit),
            ).fetchall()
            raw_porter_rows = connection.execute(
                """SELECT raw.id, raw.content, raw.created_at, raw.event_ts,
                          bm25(messages_porter_fts) AS bm25_score
                   FROM messages_porter_fts
                   JOIN raw_messages AS raw ON raw.id = messages_porter_fts.message_id
                   WHERE messages_porter_fts MATCH ? AND messages_porter_fts.user_id = ?
                   ORDER BY bm25(messages_porter_fts), raw.id DESC
                   LIMIT ?""",
                (match_query, user_id, candidate_limit),
            ).fetchall()
            fact_rows = connection.execute(
                """SELECT raw.id, raw.content, raw.created_at, raw.event_ts
                   FROM facts_fts
                   JOIN facts AS fact ON fact.id = facts_fts.fact_id
                   JOIN raw_messages AS raw ON raw.id = fact.source_message_id
                   WHERE facts_fts MATCH ? AND facts_fts.user_id = ?
                   ORDER BY bm25(facts_fts), raw.id DESC
                   LIMIT ?""",
                (match_query, user_id, candidate_limit),
            ).fetchall()
            fact_porter_rows = connection.execute(
                """SELECT raw.id, raw.content, raw.created_at, raw.event_ts
                   FROM facts_porter_fts
                   JOIN facts AS fact ON fact.id = facts_porter_fts.fact_id
                   JOIN raw_messages AS raw ON raw.id = fact.source_message_id
                   WHERE facts_porter_fts MATCH ? AND facts_porter_fts.user_id = ?
                   ORDER BY bm25(facts_porter_fts), raw.id DESC
                   LIMIT ?""",
                (match_query, user_id, candidate_limit),
            ).fetchall()
            context_rows = connection.execute(
                """SELECT raw.id, raw.content, raw.created_at, raw.event_ts
                   FROM context_fts
                   JOIN raw_messages AS raw ON raw.id = context_fts.message_id
                   WHERE context_fts MATCH ? AND context_fts.user_id = ?
                   ORDER BY bm25(context_fts), raw.id DESC
                   LIMIT ?""",
                (match_query, user_id, candidate_limit),
            ).fetchall()
            context_porter_rows = connection.execute(
                """SELECT raw.id, raw.content, raw.created_at, raw.event_ts
                   FROM context_porter_fts
                   JOIN raw_messages AS raw ON raw.id = context_porter_fts.message_id
                   WHERE context_porter_fts MATCH ? AND context_porter_fts.user_id = ?
                   ORDER BY bm25(context_porter_fts), raw.id DESC
                   LIMIT ?""",
                (match_query, user_id, candidate_limit),
            ).fetchall()
            support_raw_rows = []
            support_fact_rows = []
            if support_match_query:
                support_raw_rows = connection.execute(
                    """SELECT raw.id, raw.content, raw.created_at, raw.event_ts
                       FROM messages_porter_fts
                       JOIN raw_messages AS raw ON raw.id = messages_porter_fts.message_id
                       WHERE messages_porter_fts MATCH ? AND messages_porter_fts.user_id = ?
                       ORDER BY bm25(messages_porter_fts), raw.id DESC
                       LIMIT ?""",
                    (support_match_query, user_id, candidate_limit),
                ).fetchall()
                support_fact_rows = connection.execute(
                    """SELECT raw.id, raw.content, raw.created_at, raw.event_ts
                       FROM facts_porter_fts
                       JOIN facts AS fact ON fact.id = facts_porter_fts.fact_id
                       JOIN raw_messages AS raw ON raw.id = fact.source_message_id
                       WHERE facts_porter_fts MATCH ? AND facts_porter_fts.user_id = ?
                       ORDER BY bm25(facts_porter_fts), raw.id DESC
                       LIMIT ?""",
                    (support_match_query, user_id, candidate_limit),
                ).fetchall()
            entity_raw_rows = []
            entity_porter_rows = []
            entity_context_rows = []
            if structured_match_query:
                entity_raw_rows = connection.execute(
                    """SELECT raw.id, raw.content, raw.created_at, raw.event_ts
                       FROM messages_fts
                       JOIN raw_messages AS raw ON raw.id = messages_fts.message_id
                       WHERE messages_fts MATCH ? AND messages_fts.user_id = ?
                       ORDER BY bm25(messages_fts), raw.id DESC
                       LIMIT ?""",
                    (structured_match_query, user_id, candidate_limit),
                ).fetchall()
                entity_porter_rows = connection.execute(
                    """SELECT raw.id, raw.content, raw.created_at, raw.event_ts
                       FROM messages_porter_fts
                       JOIN raw_messages AS raw ON raw.id = messages_porter_fts.message_id
                       WHERE messages_porter_fts MATCH ? AND messages_porter_fts.user_id = ?
                       ORDER BY bm25(messages_porter_fts), raw.id DESC
                       LIMIT ?""",
                    (structured_match_query, user_id, candidate_limit),
                ).fetchall()
                entity_context_rows = connection.execute(
                    """SELECT raw.id, raw.content, raw.created_at, raw.event_ts
                       FROM context_porter_fts
                       JOIN raw_messages AS raw ON raw.id = context_porter_fts.message_id
                       WHERE context_porter_fts MATCH ? AND context_porter_fts.user_id = ?
                       ORDER BY bm25(context_porter_fts), raw.id DESC
                       LIMIT ?""",
                    (structured_match_query, user_id, candidate_limit),
                ).fetchall()
            semantic_source_rows = []
            session_porter_rows = []
            if self.semantic_retriever:
                semantic_source_rows = connection.execute(
                    """SELECT id, session_id, role, content, created_at, event_ts
                       FROM raw_messages
                       WHERE user_id = ?
                       ORDER BY id""",
                    (user_id,),
                ).fetchall()
                if self.session_fusion_weight > 0 or self.session_top_n > 0:
                    session_porter_rows = connection.execute(
                        """SELECT session_id, bm25(session_porter_fts) AS bm25_score
                           FROM session_porter_fts
                           WHERE session_porter_fts MATCH ? AND user_id = ?
                           ORDER BY bm25(session_porter_fts)""",
                        (match_query, user_id),
                    ).fetchall()
        dense_rows = []
        dense_scores = {}
        if self.semantic_retriever and semantic_source_rows:
            semantic_candidates = [
                {"id": "mem_{}".format(row["id"]), "content": str(row["content"])}
                for row in semantic_source_rows
            ]
            if self.dense_fusion_alpha is not None:
                dense_scores = self.semantic_retriever.score(
                    query, options or [], semantic_candidates
                )
                for expanded_query in expanded_queries:
                    expanded_scores = self.semantic_retriever.score(
                        expanded_query, [], semantic_candidates
                    )
                    dense_scores = {
                        candidate_id: max(score, expanded_scores[candidate_id])
                        for candidate_id, score in dense_scores.items()
                    }
                if self.dense_context_weight > 0:
                    previous_by_session = {}
                    context_candidates = []
                    for row in semantic_source_rows:
                        session_id = str(row["session_id"])
                        content = str(row["content"])
                        previous = previous_by_session.get(session_id)
                        context_content = (
                            "Previous message: {}\nCurrent message: {}".format(
                                previous, content
                            )
                            if previous else content
                        )
                        context_candidates.append({
                            "id": "mem_{}".format(row["id"]),
                            "content": context_content,
                        })
                        previous_by_session[session_id] = content
                    context_scores = self.semantic_retriever.score(
                        query, options or [], context_candidates
                    )
                    weight = self.dense_context_weight
                    dense_scores = {
                        candidate_id: (
                            (1.0 - weight) * score
                            + weight * context_scores[candidate_id]
                        )
                        for candidate_id, score in dense_scores.items()
                    }
                if self.dense_time_weight > 0:
                    time_candidates = []
                    for row in semantic_source_rows:
                        content = str(row["content"])
                        event_ts = row["event_ts"]
                        time_content = content
                        if event_ts is not None and int(event_ts) >= 86400:
                            event_date = datetime.fromtimestamp(
                                int(event_ts), tz=timezone.utc
                            ).strftime("%d %B %Y")
                            time_content = "Event date: {}\n{}".format(
                                event_date, content
                            )
                        time_candidates.append({
                            "id": "mem_{}".format(row["id"]),
                            "content": time_content,
                        })
                    time_scores = self.semantic_retriever.score(
                        query, options or [], time_candidates
                    )
                    weight = self.dense_time_weight
                    dense_scores = {
                        candidate_id: (
                            (1.0 - weight) * score
                            + weight * time_scores[candidate_id]
                        )
                        for candidate_id, score in dense_scores.items()
                    }
                dense_ids = sorted(dense_scores, key=dense_scores.get, reverse=True)[:candidate_limit]
            else:
                dense_ids = self.semantic_retriever.rank(
                    query, options or [], semantic_candidates, candidate_limit
                )
            semantic_by_id = {
                "mem_{}".format(row["id"]): row for row in semantic_source_rows
            }
            dense_rows = [semantic_by_id[item] for item in dense_ids if item in semantic_by_id]
        if self.dense_fusion_alpha is not None and semantic_source_rows:
            lexical_scores = {
                "mem_{}".format(row["id"]): -float(row["bm25_score"])
                for row in raw_porter_rows
            }
            all_ids = ["mem_{}".format(row["id"]) for row in semantic_source_rows]

            def z_scores(values: List[float]) -> List[float]:
                mean = sum(values) / len(values)
                variance = sum((value - mean) ** 2 for value in values) / len(values)
                deviation = math.sqrt(variance)
                if deviation <= 1e-12:
                    return [0.0 for _ in values]
                return [(value - mean) / deviation for value in values]

            lexical_z = z_scores([lexical_scores.get(item, 0.0) for item in all_ids])
            dense_z = z_scores([dense_scores.get(item, 0.0) for item in all_ids])
            alpha = self.dense_fusion_alpha
            session_scores = {}
            if self.session_fusion_weight > 0 or self.session_top_n > 0:
                session_ids = list(dict.fromkeys(
                    str(row["session_id"]) for row in semantic_source_rows
                ))
                session_lexical = {
                    str(row["session_id"]): -float(row["bm25_score"])
                    for row in session_porter_rows
                }
                session_dense = {session_id: float("-inf") for session_id in session_ids}
                for row in semantic_source_rows:
                    candidate_id = "mem_{}".format(row["id"])
                    session_id = str(row["session_id"])
                    session_dense[session_id] = max(
                        session_dense[session_id], dense_scores[candidate_id]
                    )
                session_lexical_z = z_scores([
                    session_lexical.get(session_id, 0.0) for session_id in session_ids
                ])
                session_dense_z = z_scores([
                    session_dense[session_id] for session_id in session_ids
                ])
                raw_session_scores = [
                    alpha * session_lexical_z[index]
                    + (1.0 - alpha) * session_dense_z[index]
                    for index in range(len(session_ids))
                ]
                normalized_session_scores = z_scores(raw_session_scores)
                session_scores = dict(zip(session_ids, normalized_session_scores))
            fused = []
            for index, row in enumerate(semantic_source_rows):
                score = alpha * lexical_z[index] + (1.0 - alpha) * dense_z[index]
                score += self.session_fusion_weight * session_scores.get(
                    str(row["session_id"]), 0.0
                )
                fused.append({
                    "result": MemoryResult(
                        id=all_ids[index], content=row["content"],
                        score=round(score, 6), created_at=row["created_at"],
                    ),
                    "event_ts": row["event_ts"],
                    "session_id": str(row["session_id"]),
                })
            fused.sort(key=lambda candidate: (
                -candidate["result"].score,
                -(candidate["event_ts"] or 0),
                candidate["result"].id,
            ))
            if self.session_top_n > 0:
                allowed_sessions = set(sorted(
                    session_scores,
                    key=session_scores.get,
                    reverse=True,
                )[:self.session_top_n])
                fused = [
                    candidate for candidate in fused
                    if candidate["session_id"] in allowed_sessions
                ]
            if self.local_reranker:
                rerank_count = min(self.rerank_top_n, len(fused))
                rerank_candidates = [
                    {
                        "id": candidate["result"].id,
                        "content": candidate["result"].content,
                    }
                    for candidate in fused[:rerank_count]
                ]
                if self.rerank_fusion_weight is None:
                    ordered_ids = self.local_reranker.rank(
                        query, options or [], rerank_candidates
                    )
                    positions = {
                        candidate_id: index for index, candidate_id in enumerate(ordered_ids)
                    }
                    reranked = sorted(
                        fused[:rerank_count],
                        key=lambda candidate: positions.get(
                            candidate["result"].id, len(positions)
                        ),
                    )
                else:
                    rerank_scores = self.local_reranker.score(
                        query, options or [], rerank_candidates
                    )
                    base_z = z_scores([
                        candidate["result"].score for candidate in fused[:rerank_count]
                    ])
                    rerank_z = z_scores([
                        rerank_scores[candidate["result"].id]
                        for candidate in fused[:rerank_count]
                    ])
                    weight = self.rerank_fusion_weight
                    combined_scores = {
                        candidate["result"].id: (
                            (1.0 - weight) * base_z[index] + weight * rerank_z[index]
                        )
                        for index, candidate in enumerate(fused[:rerank_count])
                    }
                    reranked = sorted(
                        fused[:rerank_count],
                        key=lambda candidate: -combined_scores[candidate["result"].id],
                    )
                fused = reranked + fused[rerank_count:]
            if self.local_instruction_reranker:
                instruction_count = min(self.instruction_rerank_top_n, len(fused))
                instruction_candidates = [
                    {
                        "id": candidate["result"].id,
                        "content": candidate["result"].content,
                    }
                    for candidate in fused[:instruction_count]
                ]
                ordered_ids = self.local_instruction_reranker.rank(
                    query, options or [], instruction_candidates
                )
                positions = {
                    candidate_id: index for index, candidate_id in enumerate(ordered_ids)
                }
                instruction_ranked = sorted(
                    fused[:instruction_count],
                    key=lambda candidate: positions.get(
                        candidate["result"].id, len(positions)
                    ),
                )
                fused = instruction_ranked + fused[instruction_count:]
                refine_count = min(self.instruction_refine_top_n, instruction_count)
                if refine_count > 1:
                    refine_candidates = [
                        {
                            "id": candidate["result"].id,
                            "content": candidate["result"].content,
                        }
                        for candidate in fused[:refine_count]
                    ]
                    refined_ids = self.local_instruction_reranker.rank(
                        query, options or [], refine_candidates
                    )
                    refined_positions = {
                        candidate_id: index
                        for index, candidate_id in enumerate(refined_ids)
                    }
                    refined = sorted(
                        fused[:refine_count],
                        key=lambda candidate: refined_positions.get(
                            candidate["result"].id, len(refined_positions)
                        ),
                    )
                    fused = refined + fused[refine_count:]
            return [candidate["result"] for candidate in fused[:top_k]]
        candidates = {}
        for rows, channel_weight in (
            (raw_rows, 1.0), (raw_porter_rows, 1.0),
            (fact_rows, 1.0), (fact_porter_rows, 1.0),
            (context_rows, self.CONTEXT_RRF_WEIGHT),
            (context_porter_rows, self.CONTEXT_RRF_WEIGHT),
            (support_raw_rows, self.STRUCTURED_SUPPORT_RRF_WEIGHT),
            (support_fact_rows, self.STRUCTURED_SUPPORT_RRF_WEIGHT),
            (entity_raw_rows, self.ENTITY_RRF_WEIGHT),
            (entity_porter_rows, self.ENTITY_RRF_WEIGHT),
            (entity_context_rows, self.ENTITY_RRF_WEIGHT * self.CONTEXT_RRF_WEIGHT),
            (dense_rows, self.dense_rrf_weight),
        ):
            for rank, row in enumerate(rows):
                candidate_id = "mem_{}".format(row["id"])
                candidate = candidates.setdefault(candidate_id, {
                    "result": MemoryResult(
                        id=candidate_id, content=row["content"], score=0.0,
                        created_at=row["created_at"],
                    ),
                    "event_ts": row["event_ts"],
                })
                candidate["result"].score += channel_weight / (self.RRF_CONSTANT + rank + 1)

        temporal_direction = 0
        if self.TEMPORAL_QUERY_PATTERN.search(query):
            temporal_direction = 1
        elif self.HISTORICAL_QUERY_PATTERN.search(query):
            temporal_direction = -1
        if temporal_direction:
            timestamps = [candidate["event_ts"] for candidate in candidates.values()
                          if candidate["event_ts"] is not None]
            if timestamps and max(timestamps) > min(timestamps):
                oldest, newest = min(timestamps), max(timestamps)
                for candidate in candidates.values():
                    event_ts = candidate["event_ts"]
                    if event_ts is not None:
                        recency = (event_ts - oldest) / (newest - oldest)
                        temporal_score = recency if temporal_direction > 0 else 1.0 - recency
                        # Time should resolve near-ties, not override strong lexical evidence.
                        candidate["result"].score += self.temporal_bonus * temporal_score

        ranked = sorted(
            candidates.values(),
            key=lambda candidate: (
                -candidate["result"].score,
                -(candidate["event_ts"] or 0),
                candidate["result"].id,
            ),
        )
        deduplicated = []
        seen_content = set()
        for candidate in ranked:
            result = candidate["result"]
            normalized = result.content.casefold()
            if normalized not in seen_content:
                seen_content.add(normalized)
                result.score = round(result.score, 6)
                deduplicated.append(result)
        if self.model and deduplicated:
            candidates = [{"id": result.id, "content": result.content} for result in deduplicated]
            ordered_ids = self.model.rank_candidates(query, options or [], candidates)
            positions = {candidate_id: index for index, candidate_id in enumerate(ordered_ids)}
            deduplicated.sort(key=lambda result: positions.get(result.id, len(positions)))
        return deduplicated[:top_k]
