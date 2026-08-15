import re
import sqlite3
import math
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, List, Optional

from .model import MemoryModel
from .local_instruction import mask_candidate_speakers
from .schemas import AddRequest, MemoryResult


def bind_first_person_to_speaker(content: str, speaker: str) -> str:
    """Resolve unambiguous first-person forms in a latent retrieval key."""
    if not speaker:
        return content
    replacements = [
        (r"\bI'm\b", "{} is".format(speaker)),
        (r"\bI've\b", "{} has".format(speaker)),
        (r"\bI'll\b", "{} will".format(speaker)),
        (r"\bI'd\b", "{} would".format(speaker)),
        (r"\bmy\b", "{}'s".format(speaker)),
        (r"\bmine\b", "{}'s".format(speaker)),
        (r"\bme\b", speaker),
        (r"\bI\b", speaker),
    ]
    result = content
    for pattern, replacement in replacements:
        result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)
    return result


def latent_message_text(content: str, speaker: str, event_ts: Optional[int]) -> str:
    """Add source attribution to retrieval keys without changing returned content."""
    parts = []
    if speaker and not content.casefold().startswith((speaker + ":").casefold()):
        parts.append("Speaker: {}".format(speaker))
    if event_ts is not None:
        timestamp = int(event_ts)
        if timestamp >= 100_000_000_000:
            timestamp //= 1000
        if timestamp >= 86400:
            try:
                event_date = datetime.fromtimestamp(
                    timestamp, tz=timezone.utc
                ).strftime("%d %B %Y")
            except (OSError, OverflowError, ValueError):
                event_date = ""
            if event_date:
                parts.append("Event date: {}".format(event_date))
    parts.append(content)
    return "\n".join(parts)


def candidate_ranking_text(
    content: str,
    speaker: str,
    event_ts: Optional[int],
    facts: Optional[List[str]] = None,
    neighbor_context: Optional[List[str]] = None,
) -> str:
    """Expose provenance and extracted facts only to the evidence ranker."""
    parts = ["Original memory:\n{}".format(content)]
    metadata = []
    if speaker:
        metadata.append("Source speaker: {}".format(speaker))
    if event_ts is not None:
        timestamp = int(event_ts)
        if timestamp >= 100_000_000_000:
            timestamp //= 1000
        if timestamp >= 86400:
            try:
                event_date = datetime.fromtimestamp(
                    timestamp, tz=timezone.utc
                ).strftime("%d %B %Y")
            except (OSError, OverflowError, ValueError):
                event_date = ""
            if event_date:
                metadata.append("Event date: {}".format(event_date))
    if metadata:
        parts.insert(0, "\n".join(metadata))
    supported_facts = [item.strip() for item in (facts or []) if item.strip()]
    if supported_facts:
        parts.append(
            "Extracted retrieval annotations:\n- {}".format(
                "\n- ".join(supported_facts)
            )
        )
    neighbors = [item.strip() for item in (neighbor_context or []) if item.strip()]
    if neighbors:
        parts.append("Adjacent source context:\n{}".format("\n".join(neighbors)))
    return "\n".join(parts)


class MemoryStore:
    RRF_CONSTANT = 60
    MODEL_RERANK_LIMIT = 30
    CONTEXT_RRF_WEIGHT = 0.5
    # Support terms widen recall but must not displace core-intent evidence.
    # A public synthetic sweep retained 0.01; weights >=0.05 caused G/H regressions,
    # while 0.02 could still flip a focused near-neighbor conflict case.
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
                 dense_speaker_mask_max: bool = False,
                 dense_speaker_conflict_margin: Optional[float] = None,
                 dense_speaker_conflict_gate_only: bool = False,
                 dense_sentence_weight: float = 0.0,
                 dense_image_carry_weight: float = 0.0,
                 dense_speaker_coref_weight: float = 0.0,
                 dense_speaker_swap_max: bool = False,
                 local_reranker: object = None, rerank_top_n: int = 10,
                 rerank_image_followups: int = 0,
                 session_fusion_weight: float = 0.0,
                 session_top_n: int = 0,
                 rerank_fusion_weight: Optional[float] = None,
                 rerank_near_tie_epsilon: float = 0.0,
                 local_instruction_reranker: object = None,
                 instruction_speaker_conflict_only: bool = False,
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
        self.dense_speaker_mask_max = dense_speaker_mask_max
        self.dense_speaker_conflict_margin = dense_speaker_conflict_margin
        self.dense_speaker_conflict_gate_only = dense_speaker_conflict_gate_only
        self.dense_sentence_weight = dense_sentence_weight
        self.dense_image_carry_weight = dense_image_carry_weight
        self.dense_speaker_coref_weight = dense_speaker_coref_weight
        self.dense_speaker_swap_max = dense_speaker_swap_max
        self.speaker_conflict_trigger_count = 0
        self.last_speaker_conflict = False
        self.local_reranker = local_reranker
        self.rerank_top_n = rerank_top_n
        self.rerank_image_followups = rerank_image_followups
        self.session_fusion_weight = session_fusion_weight
        self.session_top_n = session_top_n
        self.rerank_fusion_weight = rerank_fusion_weight
        self.rerank_near_tie_epsilon = rerank_near_tie_epsilon
        self.local_instruction_reranker = local_instruction_reranker
        self.instruction_speaker_conflict_only = instruction_speaker_conflict_only
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
                indexed_content = latent_message_text(
                    message.content, message.role, message.timestamp
                )
                cursor = connection.execute(
                    """INSERT INTO raw_messages(user_id, session_id, role, content, event_ts, sequence, created_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (request.user_id, request.session_id, message.role, message.content,
                     message.timestamp, sequence, now),
                )
                message_id = cursor.lastrowid
                connection.execute(
                    "INSERT INTO messages_fts(message_id, user_id, content) VALUES (?, ?, ?)",
                    (message_id, request.user_id, indexed_content),
                )
                connection.execute(
                    "INSERT INTO messages_porter_fts(message_id, user_id, content) VALUES (?, ?, ?)",
                    (message_id, request.user_id, indexed_content),
                )
                inserted_messages.append((message_id, indexed_content))
                if self.model:
                    for fact in self.model.extract_facts(
                        message.content,
                        speaker=message.role,
                        timestamp=message.timestamp,
                    ):
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
        self.last_speaker_conflict = False
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
            ranking_metadata = {}
            if self.model:
                metadata_rows = connection.execute(
                    """SELECT raw.id, raw.role, raw.event_ts, fact.fact_text
                       FROM raw_messages AS raw
                       LEFT JOIN facts AS fact ON fact.source_message_id = raw.id
                       WHERE raw.user_id = ?
                       ORDER BY raw.id, fact.id""",
                    (user_id,),
                ).fetchall()
                for row in metadata_rows:
                    item = ranking_metadata.setdefault(
                        int(row["id"]),
                        {
                            "speaker": str(row["role"]),
                            "event_ts": row["event_ts"],
                            "facts": [],
                        },
                    )
                    if row["fact_text"]:
                        item["facts"].append(str(row["fact_text"]))
                neighbor_rows = connection.execute(
                    """SELECT id, session_id, content
                       FROM raw_messages
                       WHERE user_id = ?
                       ORDER BY session_id, id""",
                    (user_id,),
                ).fetchall()
                rows_by_session = {}
                for row in neighbor_rows:
                    rows_by_session.setdefault(str(row["session_id"]), []).append(row)
                for session_rows in rows_by_session.values():
                    for index, row in enumerate(session_rows):
                        neighbors = []
                        if index > 0:
                            neighbors.append(
                                "Previous memory: {}".format(
                                    str(session_rows[index - 1]["content"])
                                )
                            )
                        if index + 1 < len(session_rows):
                            neighbors.append(
                                "Next memory: {}".format(
                                    str(session_rows[index + 1]["content"])
                                )
                            )
                        ranking_metadata[int(row["id"])]["neighbors"] = neighbors
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
                if self.dense_speaker_swap_max:
                    speakers = list(dict.fromkeys(
                        str(row["role"]).strip()
                        for row in semantic_source_rows
                        if str(row["role"]).strip()
                    ))
                    mentioned = [
                        speaker for speaker in speakers
                        if re.search(
                            r"(?<!\w){}(?!\w)".format(re.escape(speaker)),
                            query,
                            flags=re.IGNORECASE,
                        )
                    ]
                    if len(speakers) == 2 and len(mentioned) == 1:
                        source = mentioned[0]
                        target = next(item for item in speakers if item != source)
                        swapped_query = re.sub(
                            r"(?<!\w){}(?!\w)".format(re.escape(source)),
                            target,
                            query,
                            flags=re.IGNORECASE,
                        )
                        swapped_scores = self.semantic_retriever.score(
                            swapped_query, options or [], semantic_candidates
                        )
                        dense_scores = {
                            candidate_id: max(score, swapped_scores[candidate_id])
                            for candidate_id, score in dense_scores.items()
                        }
                for expanded_query in expanded_queries:
                    expanded_scores = self.semantic_retriever.score(
                        expanded_query, [], semantic_candidates
                    )
                    dense_scores = {
                        candidate_id: max(score, expanded_scores[candidate_id])
                        for candidate_id, score in dense_scores.items()
                    }
                if self.dense_sentence_weight > 0:
                    sentence_candidates = []
                    for row in semantic_source_rows:
                        candidate_id = "mem_{}".format(row["id"])
                        content = str(row["content"])
                        speaker = str(row["role"]).strip()
                        body = content
                        prefix = "{}: ".format(speaker)
                        if speaker and content.casefold().startswith(prefix.casefold()):
                            body = content[len(prefix):]
                        sentences = [
                            item.strip()
                            for item in re.split(r"(?<=[.!?;])\s+|\n+", body)
                            if item.strip()
                        ] or [body]
                        for index, sentence in enumerate(sentences):
                            sentence_candidates.append({
                                "id": "{}::sentence:{}".format(candidate_id, index),
                                "content": (
                                    "{}: {}".format(speaker, sentence)
                                    if speaker else sentence
                                ),
                            })
                    sentence_scores = self.semantic_retriever.score(
                        query, options or [], sentence_candidates
                    )
                    max_sentence_scores = {
                        candidate["id"]: float("-inf")
                        for candidate in semantic_candidates
                    }
                    for sentence in sentence_candidates:
                        candidate_id = sentence["id"].split("::sentence:", 1)[0]
                        max_sentence_scores[candidate_id] = max(
                            max_sentence_scores[candidate_id],
                            sentence_scores[sentence["id"]],
                        )
                    weight = self.dense_sentence_weight
                    dense_scores = {
                        candidate_id: (
                            (1.0 - weight) * score
                            + weight * max_sentence_scores[candidate_id]
                        )
                        for candidate_id, score in dense_scores.items()
                    }
                if self.dense_image_carry_weight > 0:
                    image_carry_candidates = []
                    last_image_by_session = {}
                    for row in semantic_source_rows:
                        session_id = str(row["session_id"])
                        content = str(row["content"])
                        image_anchor = last_image_by_session.get(session_id)
                        carried_content = (
                            "Shared image context: {}\nFollowing message: {}".format(
                                image_anchor, content
                            )
                            if image_anchor else content
                        )
                        image_carry_candidates.append({
                            "id": "mem_{}".format(row["id"]),
                            "content": carried_content,
                        })
                        if "Shared image:" in content:
                            last_image_by_session[session_id] = content
                        elif image_anchor:
                            last_image_by_session[session_id] = None
                    image_carry_scores = self.semantic_retriever.score(
                        query, options or [], image_carry_candidates
                    )
                    weight = self.dense_image_carry_weight
                    dense_scores = {
                        candidate_id: (
                            (1.0 - weight) * score
                            + weight * image_carry_scores[candidate_id]
                        )
                        for candidate_id, score in dense_scores.items()
                    }
                if self.dense_speaker_coref_weight > 0:
                    coref_candidates = [
                        {
                            "id": "mem_{}".format(row["id"]),
                            "content": bind_first_person_to_speaker(
                                str(row["content"]), str(row["role"]).strip()
                            ),
                        }
                        for row in semantic_source_rows
                    ]
                    coref_scores = self.semantic_retriever.score(
                        query, options or [], coref_candidates
                    )
                    weight = self.dense_speaker_coref_weight
                    dense_scores = {
                        candidate_id: (
                            (1.0 - weight) * score
                            + weight * coref_scores[candidate_id]
                        )
                        for candidate_id, score in dense_scores.items()
                    }
                if self.dense_speaker_mask_max:
                    masked_query, masked_candidates = mask_candidate_speakers(
                        query, semantic_candidates
                    )
                    masked_scores = self.semantic_retriever.score(
                        masked_query, options or [], masked_candidates
                    )
                    dense_scores = {
                        candidate_id: max(score, masked_scores[candidate_id])
                        for candidate_id, score in dense_scores.items()
                    }
                if self.dense_speaker_conflict_margin is not None:
                    query_speakers = []
                    for row in semantic_source_rows:
                        speaker = str(row["role"]).strip()
                        already_seen = {
                            item.casefold() for item in query_speakers
                        }
                        if (
                            speaker
                            and speaker.casefold() not in already_seen
                            and re.search(
                                r"(?<!\w){}(?!\w)".format(re.escape(speaker)),
                                query,
                                flags=re.IGNORECASE,
                            )
                        ):
                            query_speakers.append(speaker)
                    if len(query_speakers) == 1:
                        target = query_speakers[0].casefold()
                        masked_query, masked_candidates = mask_candidate_speakers(
                            query, semantic_candidates
                        )
                        masked_scores = self.semantic_retriever.score(
                            masked_query, options or [], masked_candidates
                        )
                        target_ids = [
                            "mem_{}".format(row["id"])
                            for row in semantic_source_rows
                            if str(row["role"]).strip().casefold() == target
                        ]
                        other_ids = [
                            "mem_{}".format(row["id"])
                            for row in semantic_source_rows
                            if str(row["role"]).strip().casefold() != target
                        ]
                        if target_ids and other_ids:
                            best_target = max(masked_scores[item] for item in target_ids)
                            best_other = max(masked_scores[item] for item in other_ids)
                            if best_other >= (
                                best_target + self.dense_speaker_conflict_margin
                            ):
                                self.speaker_conflict_trigger_count += 1
                                self.last_speaker_conflict = True
                                if not self.dense_speaker_conflict_gate_only:
                                    dense_scores = dict(masked_scores)
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
                rerank_content = {
                    "mem_{}".format(row["id"]): str(row["content"])
                    for row in semantic_source_rows
                }
                if self.rerank_image_followups > 0:
                    rows_by_session = {}
                    for row in semantic_source_rows:
                        rows_by_session.setdefault(str(row["session_id"]), []).append(row)
                    for session_rows in rows_by_session.values():
                        for index, row in enumerate(session_rows):
                            content = str(row["content"])
                            if "Shared image:" not in content:
                                continue
                            following = session_rows[
                                index + 1:index + 1 + self.rerank_image_followups
                            ]
                            if following:
                                rerank_content["mem_{}".format(row["id"])] = (
                                    content + "\nFollowing conversation:\n" + "\n".join(
                                        str(item["content"]) for item in following
                                    )
                                )
                rerank_candidates = [
                    {
                        "id": candidate["result"].id,
                        "content": rerank_content[candidate["result"].id],
                    }
                    for candidate in fused[:rerank_count]
                ]
                if self.rerank_fusion_weight is None:
                    if self.rerank_near_tie_epsilon > 0:
                        rerank_scores = self.local_reranker.score(
                            query, options or [], rerank_candidates
                        )
                        score_order = sorted(
                            rerank_scores,
                            key=rerank_scores.get,
                            reverse=True,
                        )
                        top_score = rerank_scores[score_order[0]]
                        near_tied = {
                            candidate_id for candidate_id in score_order
                            if rerank_scores[candidate_id] >= (
                                top_score - self.rerank_near_tie_epsilon
                            )
                        }
                        first_stage_choice = next(
                            candidate["id"] for candidate in rerank_candidates
                            if candidate["id"] in near_tied
                        )
                        ordered_ids = [first_stage_choice] + [
                            candidate_id for candidate_id in score_order
                            if candidate_id != first_stage_choice
                        ]
                    else:
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
            if self.local_instruction_reranker and (
                not self.instruction_speaker_conflict_only
                or self.last_speaker_conflict
            ):
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
                    "message_id": int(row["id"]),
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
            message_ids = {
                candidate["result"].id: candidate["message_id"]
                for candidate in ranked
            }
            candidates = []
            for result in deduplicated[:self.MODEL_RERANK_LIMIT]:
                metadata = ranking_metadata.get(message_ids[result.id], {})
                candidates.append({
                    "id": result.id,
                    "content": candidate_ranking_text(
                        result.content,
                        str(metadata.get("speaker", "")),
                        metadata.get("event_ts"),
                        metadata.get("facts", []),
                        metadata.get("neighbors", []),
                    ),
                })
            ordered_ids = self.model.rank_candidates(query, options or [], candidates)
            positions = {candidate_id: index for index, candidate_id in enumerate(ordered_ids)}
            deduplicated.sort(key=lambda result: positions.get(result.id, len(positions)))
        return deduplicated[:top_k]
