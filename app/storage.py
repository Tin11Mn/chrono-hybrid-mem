import re
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, List, Optional

from .model import MemoryModel
from .schemas import AddRequest, MemoryResult


class MemoryStore:
    def __init__(self, database_path: str, model: Optional[MemoryModel] = None) -> None:
        self.database_path = database_path
        self.model = model
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
                CREATE VIRTUAL TABLE IF NOT EXISTS facts_fts USING fts5(
                    fact_id UNINDEXED,
                    user_id UNINDEXED,
                    content
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

    def search(self, *, user_id: str, query: str, options: Optional[List[str]] = None,
               top_k: int) -> List[MemoryResult]:
        terms = re.findall(r"[\w]+", query, flags=re.UNICODE)
        if not terms:
            return []
        match_query = " OR ".join('"{}"'.format(term.replace('"', '')) for term in terms)
        with self._connection() as connection:
            raw_rows = connection.execute(
                """SELECT raw.id, raw.content, raw.created_at, -bm25(messages_fts) AS score
                   FROM messages_fts
                   JOIN raw_messages AS raw ON raw.id = messages_fts.message_id
                   WHERE messages_fts MATCH ? AND messages_fts.user_id = ?
                   ORDER BY bm25(messages_fts), raw.id DESC
                   LIMIT ?""",
                (match_query, user_id, top_k),
            ).fetchall()
            fact_rows = connection.execute(
                """SELECT fact.id, fact.fact_text AS content, fact.created_at, -bm25(facts_fts) AS score
                   FROM facts_fts
                   JOIN facts AS fact ON fact.id = facts_fts.fact_id
                   WHERE facts_fts MATCH ? AND facts_fts.user_id = ?
                   ORDER BY bm25(facts_fts), fact.id DESC
                   LIMIT ?""",
                (match_query, user_id, top_k),
            ).fetchall()
        results = [
            MemoryResult(id="mem_{}".format(row["id"]), content=row["content"],
                         score=round(float(row["score"]), 6), created_at=row["created_at"])
            for row in raw_rows
        ] + [
            MemoryResult(id="fact_{}".format(row["id"]), content=row["content"],
                         score=round(float(row["score"]), 6), created_at=row["created_at"])
            for row in fact_rows
        ]
        results.sort(key=lambda result: result.score, reverse=True)
        deduplicated = []
        seen_content = set()
        for result in results:
            normalized = result.content.casefold()
            if normalized not in seen_content:
                seen_content.add(normalized)
                deduplicated.append(result)
        if self.model and deduplicated:
            candidates = [{"id": result.id, "content": result.content} for result in deduplicated]
            ordered_ids = self.model.rank_candidates(query, options or [], candidates)
            positions = {candidate_id: index for index, candidate_id in enumerate(ordered_ids)}
            deduplicated.sort(key=lambda result: positions.get(result.id, len(positions)))
        return deduplicated[:top_k]
