"""Source-witnessed specialist indexes and conservative, deterministic operators.

Annotations guide retrieval; responses still contain immutable original messages.
No operator executes source instructions or produces unsupported memory facts.
"""

from collections import defaultdict, deque
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from fractions import Fraction
import hashlib
import json
import re
import sqlite3
import unicodedata
from typing import Any, Dict, List, Optional, Sequence
from .source_annotations import CLAUSE_BOUNDARY, first_person_value_witness, source_units


KINDS = {"fact", "state", "rule", "observation", "event", "link", "privacy"}
MODES = {"assert", "update", "correction", "retraction", "future", "hypothetical", "exception"}
CONDITION = re.compile(r"(?<![\w.-])([\w.-]{1,40})\s*(<=|>=|!=|==|=|<|>)\s*(-?\d+(?:\.\d+)?|true|false|[A-Za-z_][\w.-]{0,40})(?![\w-]|\.[\w.])", re.I)


def normalize(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def epoch_seconds(value: Any) -> Optional[int]:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value // 1000 if abs(value) >= 100_000_000_000 else value


def group_id(record: Dict[str, Any]) -> str:
    values = [record[key] for key in ("kind", "subject", "attribute", "scope", "authority")]
    if values[0] == "fact":
        values[0] = "state"  # Plain assertions are prior versions of the same slot.
    return hashlib.sha256(json.dumps(values, ensure_ascii=False).encode("utf-8")).hexdigest()[:24]


def assignments(text: str) -> Dict[str, str]:
    """Read explicit literal assignments, rejecting ambiguous repeated fields."""
    result: Dict[str, str] = {}
    ambiguous = set()
    for match in CONDITION.finditer(text):
        if match[2] not in {"=", "=="}:
            continue
        field, value = normalize(match[1]), normalize(match[3])
        if field in result and result[field] != value:
            ambiguous.add(field)
        result[field] = value
    return {key: value for key, value in result.items() if key not in ambiguous}


def grounded_records(payload: Any, content: str, speaker: str,
                     timestamp: Optional[int]) -> tuple:
    if not isinstance(payload, dict) or not isinstance(payload.get("records"), list):
        return [], False
    items = payload["records"]
    complete = payload.get("complete") is True and len(items) <= 128
    result = []
    for item in items[:128]:
        if not isinstance(item, dict):
            complete = False
            continue
        kind, mode = item.get("kind"), item.get("modality", "assert")
        span, subject, attribute, value = [item.get(key) for key in ("span", "subject", "attribute", "value")]
        scope = item.get("scope", "")
        if (not isinstance(kind, str) or not isinstance(mode, str) or kind not in KINDS or mode not in MODES or not all(isinstance(x, str) for x in
                (span, subject, attribute, value, scope)) or not span or len(span) > 4096
                or not subject or not attribute or not value or len(attribute) > 80
                or len(subject) > 128 or len(value) > 4096 or len(scope) > 128
                or span not in content or value not in span or (scope and scope not in span)):
            complete = False
            continue
        if subject == "@speaker":
            if not first_person_value_witness(span, value):
                complete = False
                continue
            subject = speaker
        elif subject not in span:
            complete = False
            continue
        start = content.index(span)
        prefix = re.split(r"[.!?。\n]", content[:start])[-1][-120:]
        context = prefix + span
        # A full-page witness may contain several independent assertions. Guard
        # the clause containing this value, rather than borrowing modality from
        # an unrelated plan/quote elsewhere in the same immutable source.
        value_at = span.find(value)
        if span.count(value) == 1:
            left, right = 0, len(content)
            absolute_value = start + value_at
            for boundary in re.finditer(CLAUSE_BOUNDARY, content):
                if boundary.end() <= absolute_value:
                    left = max(left, boundary.end())
                elif boundary.start() >= absolute_value:
                    right = min(right, boundary.start())
                    break
            context = content[left:right]
        subject_at = start + max(0, value_at)
        before_subject = content[:subject_at]
        quoted = False
        for opening, closing in [('"', '"'), ('“', '”'), ('‘', '’')]:
            unmatched = (before_subject.count(opening) % 2 == 1 if opening == closing else
                         before_subject.rfind(opening) > before_subject.rfind(closing))
            if unmatched and content.find(closing, subject_at) >= subject_at + len(value):
                quoted = True
        if quoted or re.search(r"\b(might|maybe|hypothetical|suppose|could|said|says|quoted|hearsay)\b|假设|可能|也许|据说|引用", context, re.I):
            mode = "hypothetical"
        elif re.search(r"\b(next year|next month|will\s+(?:be|have|start|move|join|do)|plan to)\b|明年|下个月|计划|将会", context, re.I):
            mode = "future"
        elif kind in {"fact", "state", "link"} and mode != "retraction" and re.search(
                r"\b(not|never|false|didn't|doesn't)\b|不是|并非|不再", context, re.I) and not re.search(
                r"\b(not|never|didn't|doesn't)\b|不是|并非|不再", value, re.I):
            mode = "hypothetical"
        if mode in {"update", "correction", "retraction", "exception"}:
            cues = {"update": r"\b(now|changed|updated|effective)\b|现在|更新|改为|生效",
                    "correction": r"\b(correction|corrected|actually|instead)\b|更正|纠正|改为",
                    "retraction": r"\b(retract(?:s|ed)?|withdraw(?:s|n)?|no longer)\b|撤回|撤销|不再",
                    "exception": r"\b(except|unless|exception|exempt)\b|除非|例外|除外|豁免"}
            if not re.search(cues[mode], context, re.I):
                if kind in {"fact", "state"} and mode in {"update", "correction"}:
                    # Weaken an unsupported transition to a witnessed assertion.
                    # This NEVER authorizes superseding an earlier value.
                    mode = "assert"
                else:
                    complete = False
                    continue
        effective = epoch_seconds(timestamp)
        if item.get("effective_date") is not None:
            date = item["effective_date"]
            try:
                if not isinstance(date, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", date) or date not in context:
                    raise ValueError("unwitnessed date")
                effective = int(datetime.strptime(date, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp())
            except ValueError:
                complete = False
                continue
        conditions = item.get("conditions", [])
        if not isinstance(conditions, list) or len(conditions) > 8:
            complete = False
            continue
        parsed_conditions = []
        valid = True
        for condition in conditions:
            if not isinstance(condition, str) or condition not in span:
                valid = False
                break
            match = CONDITION.fullmatch(condition.strip())
            if not match:
                valid = False
                break
            parsed_conditions.append([normalize(match[1]), match[2], normalize(match[3])])
        if not valid:
            complete = False
            continue
        record = {"kind": kind, "subject": normalize(subject), "attribute": normalize(attribute),
                  "scope": normalize(scope), "authority": normalize(speaker), "value": value,
                  "modality": mode, "effective_ts": effective, "span": span,
                  "start": content.index(span), "end": content.index(span) + len(span),
                  "conditions": parsed_conditions}
        if kind == "rule" and parsed_conditions:
            # Literal comparisons are executable only in a direct conjunctive
            # declaration. Inverted/unless/disjunctive prose remains retrieval-only.
            action_at = span.find(value)
            premise = span[:action_at]
            direct = bool(re.search(r":|\bthen\b", premise, re.I))
            if not direct or re.search(r"\b(unless|not|or)\b|除非|或者|并非", premise, re.I):
                record["conditions"] = []
        if kind == "link":
            cue = {"causes": r"(?:causes|caused|leads to|led to|导致|引起)",
                   "before": r"(?:is before|before|preceded|precedes|早于|先于)",
                   "enables": r"(?:enables|enabled|使得)"}.get(record["attribute"])
            left, right = span.find(item["subject"]), span.find(value)
            bridge = span[left + len(item["subject"]):right] if left >= 0 and right > left else ""
            if mode != "assert" or cue is None or not re.fullmatch(r"\s*" + cue + r"\s*", bridge, re.I):
                complete = False
                continue
        record["group_id"] = group_id(record)
        result.append(record)
    return result, complete


def resolve_states(records: Sequence[Dict[str, Any]], mode: str = "current",
                   cutoff: Optional[int] = None) -> Dict[str, Any]:
    """Resolve a single slot, with event-order versioning and explicit uncertainty."""
    if mode == "current" and cutoff is None:
        cutoff = int(datetime.now(timezone.utc).timestamp())
    ordered = sorted(records, key=lambda r: (r["effective_ts"] is not None, r["effective_ts"] or 0, r["source_id"]))
    if mode in {"history", "order"}:
        return {"status": "history", "required": list(dict.fromkeys(r["source_id"] for r in ordered)), "blocked": []}
    eligible, blocked = [], []
    for record in ordered:
        ts = record["effective_ts"]
        if record["modality"] in {"future", "hypothetical"} or (cutoff is not None and ts is not None and ts > cutoff):
            blocked.append(record["source_id"])
        else:
            eligible.append(record)
    if not eligible:
        # A reported/possible state is still evidence of uncertainty. It must
        # not masquerade as an active value or make the raw source disappear.
        return {"status": "unresolved", "required": list(dict.fromkeys(r["source_id"] for r in ordered)), "blocked": []}
    active = []
    for record in eligible:
        ts, modality = record["effective_ts"], record["modality"]
        if modality in {"update", "correction"} and ts is not None:
            superseded = [r for r in active if r["effective_ts"] is not None and r["effective_ts"] < ts]
            active = [r for r in active if r not in superseded]
            blocked.extend(r["source_id"] for r in superseded)
        if modality == "retraction" and ts is not None:
            removed = [r for r in active if normalize(r["value"]) == normalize(record["value"])
                       and r["effective_ts"] is not None and r["effective_ts"] < ts]
            active = [r for r in active if r not in removed]
            blocked.extend(r["source_id"] for r in removed)
            active.append(record)  # The explicit withdrawal itself is decisive evidence.
        else:
            active.append(record)
    values = {normalize(r["value"]) for r in active if r["modality"] != "retraction"}
    ambiguous = len(values) > 1 or any(r["effective_ts"] is None for r in active)
    required = list(dict.fromkeys(r["source_id"] for r in active))
    # A source may contain multiple assertions. Never hide an entire source still
    # required for another version in this same slot.
    return {"status": "unresolved" if ambiguous else "resolved", "required": required,
            "blocked": [source for source in dict.fromkeys(blocked) if source not in required]}


def evaluate_rules(records: Sequence[Dict[str, Any]], query: str) -> Dict[str, Any]:
    inputs = assignments(query)
    matched, unknown = [], []
    for rule in records:
        if rule["modality"] in {"future", "hypothetical", "retraction"}:
            continue
        if not rule["conditions"]:
            unknown.append(rule["source_id"])
            continue
        status = True
        for field, operator, expected in rule["conditions"]:
            actual = inputs.get(field)
            if actual is None:
                status = None
                continue
            try:
                a, b = Decimal(actual), Decimal(expected)
                if not a.is_finite() or not b.is_finite():
                    status = None
                    continue
            except InvalidOperation:
                if operator not in {"=", "==", "!="}:
                    status = None
                    break
                a, b = actual, expected
            comparison = {"=": lambda: a == b, "==": lambda: a == b,
                          "!=": lambda: a != b, "<=": lambda: a <= b,
                          ">=": lambda: a >= b, "<": lambda: a < b, ">": lambda: a > b}[operator]()
            if not comparison:
                status = False
                break
        if status is True:
            matched.append(rule)
        elif status is None:
            unknown.append(rule["source_id"])
    exceptions = [r for r in matched if r["modality"] == "exception"]
    return {"status": "unknown_inputs" if unknown else "evaluated",
            "required": list(dict.fromkeys(r["source_id"] for r in matched)) + list(dict.fromkeys(unknown)),
            "applicable": list(dict.fromkeys(r["source_id"] for r in exceptions or matched)),
            "unknown": list(dict.fromkeys(unknown))}


def observation_summary(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Describe witnessed literal numeric columns; no answer or future simulation."""
    observations = []
    eligible = []
    for record in records:
        if record["modality"] in {"future", "hypothetical", "retraction"}:
            continue
        eligible.append(record)
        values = {}
        # A full source may contain many distinct rows under one shared header.
        # Each value is an exact witnessed row; parsing the full span would turn
        # repeated input fields into ambiguous assignments and discard the data.
        literal_values = assignments(record["value"])
        if len(literal_values) < 2:
            literal_values = assignments(record["span"])
        for field, value in literal_values.items():
            try:
                number = Decimal(value)
                if number.is_finite():
                    values[field] = number
            except InvalidOperation:
                pass
        if values:
            observations.append((record, values))
    fields = sorted({field for _, values in observations for field in values})
    statistics = {}
    for field in fields:
        data = [values[field] for _, values in observations if field in values]
        timed = [(r["effective_ts"], values[field]) for r, values in observations if field in values]
        trend = "unknown_order"
        if len(timed) > 1 and all(t is not None for t, _ in timed) and len({t for t, _ in timed}) == len(timed):
            sequence = [value for _, value in sorted(timed)]
            changes = [b - a for a, b in zip(sequence, sequence[1:])]
            trend = ("constant" if all(x == 0 for x in changes) else
                     "increasing" if all(x >= 0 for x in changes) else
                     "decreasing" if all(x <= 0 for x in changes) else "mixed")
        statistics[field] = {"count": len(data), "min": str(min(data)), "max": str(max(data)),
                             "mean": str(sum(data) / len(data)), "trend": trend}
    fit = None
    if len(fields) == 2:
        x, y = fields
        pairs = [(values[x], values[y]) for _, values in observations if x in values and y in values]
        by_x = defaultdict(set)
        for a, b in pairs:
            by_x[a].add(b)
        if len(pairs) == len(eligible) and len(by_x) >= 3 and all(len(values) == 1 for values in by_x.values()):
            points = sorted((a, next(iter(b))) for a, b in by_x.items())
            slope = Fraction(points[-1][1] - points[0][1]) / Fraction(points[-1][0] - points[0][0])
            intercept = Fraction(points[0][1]) - slope * Fraction(points[0][0])
            if all(Fraction(b) == slope * Fraction(a) + intercept for a, b in points):
                fit = {"x": x, "y": y, "slope": str(slope), "intercept": str(intercept),
                       "range": [str(points[0][0]), str(points[-1][0])], "extrapolation_validated": False}
    return {"status": "observed_only", "statistics": statistics, "exact_affine_fit": fit,
            "required": list(dict.fromkeys(record["source_id"] for record in eligible))}


def directed_path(records: Sequence[Dict[str, Any]], start: str, end: str,
                  relation: str, max_hops: int = 3, check_reverse: bool = True) -> Dict[str, Any]:
    if relation not in {"causes", "before", "enables"} or not 1 <= max_hops <= 3:
        return {"status": "unsupported", "required": []}
    start, end = normalize(start), normalize(end)
    adjacency = defaultdict(list)
    for record in records:
        if record["attribute"] == relation and record["modality"] == "assert":
            adjacency[record["subject"]].append(record)
    queue = deque([(start, [], {start})])
    while queue:
        node, path, seen = queue.popleft()
        if len(path) >= max_hops:
            continue
        for edge in sorted(adjacency[node], key=lambda r: r["source_id"])[:16]:
            target = normalize(edge["value"])
            if target in seen:
                continue
            next_path = path + [edge]
            if target == end:
                required = list(dict.fromkeys(r["source_id"] for r in next_path))
                if relation == "before" and check_reverse:
                    reverse = directed_path(records, end, start, relation, max_hops, check_reverse=False)
                    if reverse["status"] == "found":
                        return {"status": "conflicting_order", "required": list(dict.fromkeys(required + reverse["required"]))}
                return {"status": "found", "required": required,
                        "hops": len(next_path)}
            queue.append((target, next_path, seen | {target}))
    return {"status": "not_found", "required": []}


def route_memory_requests(catalog: Dict[str, Any], request: Any, query: str) -> Dict[str, Any]:
    """Resolve catalog references by their actual kind, plus explicit literal slots.

    Labels are resolved only against this user's catalog and literal subject,
    attribute and scope in the query. Unknown/ambiguous aliases invent no slot.
    """
    result = dict(request) if isinstance(request, dict) else {}
    groups = catalog["groups"]
    by_id = {row["group_id"]: row for row in groups}
    kinds = {"state": "state_slots", "rule": "rule_groups", "observation": "observation_groups"}
    routed = {key: [] for key in kinds.values()}
    query_text = normalize(query)
    def mentioned(text):
        return (bool(re.search(r"(?<!\w)" + re.escape(text) + r"(?!\w)", query_text))
                if re.fullmatch(r"[a-z0-9_ .-]+", text) else text in query_text)
    for key in routed:
        values = result.get(key, [])
        for value in values if isinstance(values, list) else []:
            if not isinstance(value, str):
                continue
            row = by_id.get(value)
            if row is None:
                matches = [r for r in groups if r["attribute"] == normalize(value)
                           and mentioned(r["subject"]) and mentioned(r["attribute"])
                           and (not r["scope"] or mentioned(r["scope"]))]
                row = matches[0] if len(matches) == 1 else None
            if row is not None:
                routed[kinds[row["kind"]]].append(row["group_id"])
    # Exact known slot terms can recover a planner's omitted group without
    # guessing a paraphrase, authority, person or unmentioned scope.
    for row in groups:
        if (row["kind"] == "state" and mentioned(row["subject"]) and mentioned(row["attribute"])
                and (not row["scope"] or mentioned(row["scope"]))):
            routed["state_slots"].append(row["group_id"])
    for key, values in routed.items():
        result[key] = list(dict.fromkeys(values))
    explicit_asof = re.search(r"(?:as\s+of|截至|截止)\s*(\d{4}-\d{2}-\d{2})", query_text)
    if explicit_asof:
        result.update(temporal_mode="as_of", as_of=explicit_asof[1])
    elif re.search(r"\b(history|changes|changed|previous|earlier|formerly)\b|历史|变化|变更|之前", query_text):
        result["temporal_mode"] = "history"
    elif re.search(r"\b(order|sequence|earliest|chronological|before|after)\b|顺序|排序|最先|先于|早于|晚于|之后", query_text):
        result["temporal_mode"] = "order"
    elif re.search(r"\b(current|currently|now)\b|当前|现在|目前", query_text) or re.search(
            r'\b(latest|most recent)\s+(?:\w+\s+){0,2}(status|state)\b|最新.{0,4}状态', query_text):
        result["temporal_mode"] = "current"
    return result


class StructuredMemoryIndex:
    """Small SQLite index; every row and rollup retains raw-source provenance."""

    @staticmethod
    def initialize(connection: sqlite3.Connection) -> None:
        connection.executescript("""
            CREATE TABLE IF NOT EXISTS structured_sources (
                source_id INTEGER PRIMARY KEY REFERENCES raw_messages(id),
                user_id TEXT NOT NULL, complete INTEGER NOT NULL CHECK(complete IN (0,1)));
            CREATE TABLE IF NOT EXISTS structured_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT, user_id TEXT NOT NULL,
                session_id TEXT NOT NULL, source_id INTEGER NOT NULL REFERENCES raw_messages(id),
                kind TEXT NOT NULL, group_id TEXT NOT NULL, subject TEXT NOT NULL,
                attribute TEXT NOT NULL, scope TEXT NOT NULL, authority TEXT NOT NULL,
                record_json TEXT NOT NULL, fingerprint TEXT NOT NULL UNIQUE);
            CREATE INDEX IF NOT EXISTS structured_records_group_idx
                ON structured_records(user_id,group_id,id);
            CREATE INDEX IF NOT EXISTS structured_records_source_idx
                ON structured_records(user_id,source_id);
            CREATE VIRTUAL TABLE IF NOT EXISTS structured_records_fts USING fts5(
                record_id UNINDEXED,user_id UNINDEXED,content,tokenize='porter unicode61');
            CREATE TABLE IF NOT EXISTS structured_rollups (
                user_id TEXT NOT NULL, level TEXT NOT NULL, group_key TEXT NOT NULL,
                page INTEGER NOT NULL, record_ids TEXT NOT NULL, source_ids TEXT NOT NULL,
                extractive_text TEXT NOT NULL, PRIMARY KEY(user_id,level,group_key,page));
        """)
        connection.execute("CREATE TABLE IF NOT EXISTS structured_index_migrations(version INTEGER PRIMARY KEY)")
        if not connection.execute("SELECT 1 FROM structured_index_migrations WHERE version=1").fetchone():
            for row in connection.execute("SELECT id,user_id,record_json FROM structured_records WHERE kind='fact'").fetchall():
                record = json.loads(row["record_json"])
                canonical = group_id(record)
                if record["group_id"] != canonical:
                    record["group_id"] = canonical
                    encoded = json.dumps(record, ensure_ascii=False, sort_keys=True)
                    fingerprint = hashlib.sha256((row["user_id"] + "\0" + encoded).encode("utf-8")).hexdigest()
                    connection.execute("UPDATE structured_records SET group_id=?,record_json=?,fingerprint=? WHERE id=?",
                                       (canonical, encoded, fingerprint, row["id"]))
            connection.execute("INSERT INTO structured_index_migrations(version) VALUES (1)")

    @staticmethod
    def store(connection: sqlite3.Connection, *, user_id: str, session_id: str,
              source_id: int, content: str, speaker: str, timestamp: Optional[int], payload: Any,
              refresh_rollups: bool = True) -> set:
        records, complete = grounded_records(payload, content, speaker, timestamp)
        connection.execute("INSERT OR REPLACE INTO structured_sources(source_id,user_id,complete) VALUES (?,?,?)",
                           (source_id, user_id, int(complete)))
        for record in records:
            record.update(source_id=source_id, session_id=session_id)
            encoded = json.dumps(record, ensure_ascii=False, sort_keys=True)
            fingerprint = hashlib.sha256((user_id + "\0" + encoded).encode("utf-8")).hexdigest()
            cursor = connection.execute("""INSERT OR IGNORE INTO structured_records(
                user_id,session_id,source_id,kind,group_id,subject,attribute,scope,authority,record_json,fingerprint)
                VALUES (?,?,?,?,?,?,?,?,?,?,?)""", (user_id, session_id, source_id, record["kind"], record["group_id"],
                record["subject"], record["attribute"], record["scope"], record["authority"], encoded, fingerprint))
            if cursor.rowcount:
                connection.execute("INSERT INTO structured_records_fts(record_id,user_id,content) VALUES (?,?,?)",
                                   (cursor.lastrowid, user_id, " ".join([record["subject"], record["attribute"], record["span"]])))
        topics = {record["subject"] for record in records}
        if refresh_rollups:
            StructuredMemoryIndex._rollup(connection, user_id, "session", session_id)
            for topic in topics:
                StructuredMemoryIndex._rollup(connection, user_id, "topic", topic)
        return topics

    @staticmethod
    def _rollup(connection: sqlite3.Connection, user_id: str, level: str, key: str) -> None:
        column = "session_id" if level == "session" else "subject"
        rows = connection.execute("SELECT id,record_json FROM structured_records WHERE user_id=? AND {}=? ORDER BY id".format(column),
                                  (user_id, key)).fetchall()
        connection.execute("DELETE FROM structured_rollups WHERE user_id=? AND level=? AND group_key=?", (user_id, level, key))
        for offset in range(0, len(rows), 16):
            page = rows[offset:offset + 16]
            records = [json.loads(row["record_json"]) for row in page]
            # Exact supported spans, retaining all versions and counterexamples.
            text = "\n".join("[mem_{}:{}-{}] {}".format(r["source_id"], r["start"], r["end"], r["span"]) for r in records)
            connection.execute("INSERT INTO structured_rollups VALUES (?,?,?,?,?,?,?)", (user_id, level, key,
                offset // 16, json.dumps([r["id"] for r in page]), json.dumps(list(dict.fromkeys(r["source_id"] for r in records))), text))

    @staticmethod
    def catalog(connection: sqlite3.Connection, user_id: str, query: str = "") -> Dict[str, Any]:
        counts = connection.execute("""SELECT COUNT(*) AS total,
            SUM(CASE WHEN indexed.complete=1 THEN 1 ELSE 0 END) AS ready
            FROM raw_messages AS raw LEFT JOIN structured_sources AS indexed
            ON indexed.source_id=raw.id AND indexed.user_id=raw.user_id WHERE raw.user_id=?""", (user_id,)).fetchone()
        groups = connection.execute("""SELECT DISTINCT CASE WHEN kind='fact' THEN 'state' ELSE kind END AS kind,
            group_id,subject,attribute,scope,authority
            FROM structured_records WHERE user_id=? AND kind IN ('fact','state','rule','observation')
            ORDER BY kind,group_id LIMIT 128""", (user_id,)).fetchall()
        if query:
            terms = list(dict.fromkeys(re.findall(r"\w+", query)))[:32]
            terms = [term for term in terms if normalize(term) not in {"what", "is", "the", "a", "of", "my", "current", "status", "summarize"}]
            if terms:
                match = " OR ".join('"{}"'.format(term) for term in terms)
                relevant = connection.execute("""SELECT CASE WHEN record.kind='fact' THEN 'state' ELSE record.kind END AS kind,record.group_id,record.subject,
                    record.attribute,record.scope,record.authority FROM structured_records_fts
                    JOIN structured_records AS record ON record.id=structured_records_fts.record_id
                    WHERE structured_records_fts MATCH ? AND record.user_id=?
                    AND record.kind IN ('fact','state','rule','observation')
                    ORDER BY bm25(structured_records_fts),record.id DESC LIMIT 256""", (match, user_id)).fetchall()
                unique = {}
                for row in relevant + groups:
                    unique.setdefault(row["group_id"], row)
                groups = list(unique.values())[:128]
        topics = [row[0] for row in connection.execute(
            "SELECT DISTINCT subject FROM structured_records WHERE user_id=? AND kind<>'privacy' ORDER BY subject LIMIT 128", (user_id,))]
        topics = list(dict.fromkeys([row["subject"] for row in groups] + topics))[:128]
        topic_count = connection.execute("SELECT COUNT(DISTINCT subject) FROM structured_records WHERE user_id=? AND kind<>'privacy'", (user_id,)).fetchone()[0]
        return {"complete": counts["total"] == (counts["ready"] or 0), "sources": counts["total"],
                "groups": [dict(row) for row in groups], "topics": topics, "topic_catalog_truncated": topic_count > len(topics)}

    @staticmethod
    def records(connection: sqlite3.Connection, user_id: str, group: str) -> List[Dict[str, Any]]:
        return [json.loads(row[0]) for row in connection.execute(
            "SELECT record_json FROM structured_records WHERE user_id=? AND group_id=? ORDER BY id", (user_id, group))]

    @staticmethod
    def slot_complete(connection: sqlite3.Connection, user_id: str, subject: str) -> bool:
        """Ignore only simple, explicitly different subjects in partial sources.

        Unknown sources, omitted assertions, pronouns and complex clauses keep
        the slot conservative. This is not a general coreference resolver.
        """
        rows = connection.execute('''SELECT raw.id,raw.content,indexed.complete,record.record_json
            FROM raw_messages AS raw LEFT JOIN structured_sources AS indexed
              ON indexed.source_id=raw.id AND indexed.user_id=raw.user_id
            LEFT JOIN structured_records AS record
              ON record.source_id=raw.id AND record.user_id=raw.user_id
            WHERE raw.user_id=? AND (indexed.complete IS NULL OR indexed.complete<>1)''', (user_id,)).fetchall()
        sources = {}
        for row in rows:
            source = sources.setdefault(row['id'], {'content': row['content'], 'complete': row['complete'], 'records': []})
            if row['record_json'] is not None:
                source['records'].append(json.loads(row['record_json']))
        for source in sources.values():
            records = source['records']
            text = normalize(source['content'])
            mentioned = (bool(re.search(r'(?<!\w)' + re.escape(subject) + r'(?!\w)', text))
                         if re.fullmatch(r'[a-z0-9_ .-]+', subject) else subject in text)
            if source['complete'] is None or len(records) != 1 or mentioned:
                return False
            record = records[0]
            if (record['subject'] == subject or not text.startswith(record['subject'] + ' ')
                    or normalize(record['span']) != text or len(text) > 512
                    or len(source_units(source['content'])) != 1):
                return False
            if re.search(r'\b(i|my|mine|me|we|our|he|she|his|her|they|their|it|its|this|that|'
                         r'if|when|unless|after|before|because|although|however|and|but|or|said|says|reports|reported)\b'
                         r'|我|他|她|它|该|这|那|如果|因为|[;:，,]', text, re.I):
                return False
        return True

    @staticmethod
    def retrieve(connection: sqlite3.Connection, user_id: str, match_query: str,
                 query: str, plan: Dict[str, Any]) -> Dict[str, Any]:
        catalog = StructuredMemoryIndex.catalog(connection, user_id, query)
        by_id = {row["group_id"]: row for row in catalog["groups"]}
        request = route_memory_requests(catalog, plan.get("memory_requests", {}), query)
        required_groups, blocked, diagnostics, hints = [], [], {}, []
        slot_coverage = {}
        coverage_limited = False
        mode = request.get("temporal_mode", "current")
        # Synthesis needs prior stages unless the question explicitly restricts
        # its scope. A planner's default current mode is not such a restriction.
        if plan.get("intent") == "summary" and mode == "current" and not re.search(
                r'\b(current|currently|now|latest|today)\b|当前|目前|现在|最新|今天', query, re.I):
            mode = "history"
        if mode not in {"current", "history", "as_of", "order"}:
            mode = "history"  # Unknown temporal intent cannot justify dropping old states.
        cutoff = None
        if mode == "current":
            cutoff = int(datetime.now(timezone.utc).timestamp())
        if mode == "as_of":
            date = request.get("as_of")
            if isinstance(date, str) and re.fullmatch(r"\d{4}-\d{2}-\d{2}", date) and date in query:
                try:
                    cutoff = int(datetime.strptime(date, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp()) + 86399
                except ValueError:
                    pass
            if cutoff is None:
                mode = "history"
        for key, kind in (("state_slots", "state"), ("rule_groups", "rule"), ("observation_groups", "observation")):
            requested = request.get(key, [])
            if not isinstance(requested, list):
                continue
            for group in list(dict.fromkeys(x for x in requested if isinstance(x, str)))[:8]:
                if group not in by_id or by_id[group]["kind"] != kind:
                    continue
                records = StructuredMemoryIndex.records(connection, user_id, group)
                if kind == "state":
                    detail = resolve_states(records, mode, cutoff)
                    subject = by_id[group]['subject']
                    if subject not in slot_coverage:
                        slot_coverage[subject] = catalog['complete'] or StructuredMemoryIndex.slot_complete(connection, user_id, subject)
                    if slot_coverage[subject]:
                        # A slot decision cannot delete another relation from
                        # the same immutable original message.
                        safe = []
                        for source in detail['blocked']:
                            other = connection.execute('''SELECT 1 FROM structured_records
                                WHERE user_id=? AND source_id=? AND group_id<>? LIMIT 1''',
                                (user_id, source, group)).fetchone()
                            source_records = connection.execute('''SELECT record_json FROM structured_records
                                WHERE user_id=? AND source_id=?''', (user_id, source)).fetchall()
                            uncertain = any(json.loads(row[0])['modality'] in {'future', 'hypothetical'}
                                            for row in source_records)
                            # A plan/quotation may itself be what the question
                            # asks about. It is not an active current value,
                            # but only source selection may judge its relevance.
                            if other is None and not uncertain:
                                safe.append(source)
                        detail['blocked'] = safe
                        blocked.extend(safe)
                    else:
                        detail["status"] = "partial_index"
                        # Incomplete coverage forbids deleting older evidence,
                        # but must not undo the useful ranking of known versions.
                        # Preserve the current candidate first, then all originals.
                        detail["required"] = list(dict.fromkeys(
                            detail["required"] + [r["source_id"] for r in records]))
                elif kind == "rule":
                    detail = evaluate_rules(records, query)
                    if records and all(not r['conditions'] for r in records):
                        detail['status'] = 'source_rules'
                        detail['required'] = list(dict.fromkeys(r['source_id'] for r in records))
                else:
                    detail = observation_summary(records)
                if detail["required"]:
                    required_groups.append(detail["required"])
                diagnostics[group] = {"kind": kind, "status": detail["status"], "source_count": len(detail["required"])}
                hints.append({"group_id": group, "kind": kind, "result": {
                    **detail, "required": detail["required"][:30], "required_count": len(detail["required"])}})
        path = request.get("path")
        if isinstance(path, dict) and all(isinstance(path.get(k), str) for k in ("start", "end", "relation")):
            # Link rows are grounded at ingestion and tenant-filtered here.
            links = [json.loads(row[0]) for row in connection.execute(
                "SELECT record_json FROM structured_records WHERE user_id=? AND kind='link' ORDER BY id LIMIT 5000", (user_id,))]
            if cutoff is not None:
                links = [r for r in links if (r["effective_ts"] is None and mode != "as_of")
                         or (r["effective_ts"] is not None and r["effective_ts"] <= cutoff)]
            detail = directed_path(links, path["start"], path["end"], path["relation"])
            if detail["required"]:
                required_groups.append(detail["required"])
            hints.append({"kind": "path", "result": detail})
            diagnostics["path"] = {"kind": "path", "status": detail["status"], "source_count": len(detail["required"])}
        hits = connection.execute("""SELECT record.source_id FROM structured_records_fts
            JOIN structured_records AS record ON record.id=structured_records_fts.record_id
            WHERE structured_records_fts MATCH ? AND record.user_id=?
            ORDER BY bm25(structured_records_fts),record.id DESC LIMIT 60""", (match_query, user_id)).fetchall()
        candidate_ids = [row[0] for row in hits]
        if plan.get("intent") == "summary":
            # Read every rollup's provenance, not just session endpoints. Pool
            # limits remain explicit when full requested coverage cannot fit.
            topics = request.get("summary_topics", [])
            topics = [normalize(x) for x in topics if isinstance(x, str)] if isinstance(topics, list) else []
            topics = [x for x in topics if x in catalog["topics"]]
            if not topics:
                entities = plan.get("entities", [])
                entities = [normalize(x) for x in entities if isinstance(x, str)] if isinstance(entities, list) else []
                topics = [x for x in catalog["topics"] if x in entities] if entities else catalog["topics"]
                if not entities and catalog["topic_catalog_truncated"]:
                    coverage_limited = True
            pages = []
            if topics:
                pages = connection.execute("SELECT record_ids FROM structured_rollups WHERE user_id=? AND level='topic' AND group_key IN ({}) ORDER BY group_key,page".format(",".join("?" for _ in topics)), [user_id] + topics).fetchall()
            record_ids = list(dict.fromkeys(record for row in pages for record in json.loads(row[0])))
            summary_ids = []
            for offset in range(0, len(record_ids), 400):
                chunk = record_ids[offset:offset + 400]
                summary_ids.extend(row[0] for row in connection.execute(
                    "SELECT DISTINCT source_id FROM structured_records WHERE user_id=? AND kind<>'privacy' AND id IN ({}) ORDER BY source_id".format(",".join("?" for _ in chunk)), [user_id] + chunk))
            summary_ids = list(dict.fromkeys(summary_ids))
            # A planner may name only a few stage-specific topics even when the
            # user explicitly asks for ALL milestones of a literal named project.
            # Recover original named-source coverage, never guessed entity aliases.
            if re.search(r'\b(all|entire|across)\b|全部|所有|全程|历程', query, re.I):
                names = list(plan.get('entities', [])) if isinstance(plan.get('entities'), list) else []
                names += re.findall(r'\b[A-Z][A-Za-z0-9_-]{2,}\b', query)
                ignored_names = {'summarize','summarise','summary','list','give','show','across','all','the','user','please'}
                names = list(dict.fromkeys(normalize(name) for name in names if isinstance(name,str)
                    and len(name)>=3 and normalize(name) not in ignored_names and normalize(name) in normalize(query)))[:8]
                if names:
                    clauses=' OR '.join('lower(raw.content) LIKE ?' for _ in names)
                    named = connection.execute('''SELECT raw.id,raw.content FROM raw_messages AS raw
                        WHERE raw.user_id=? AND ('''+clauses+''')
                        AND NOT EXISTS (SELECT 1 FROM structured_sources AS indexed
                            WHERE indexed.source_id=raw.id AND indexed.user_id=raw.user_id
                            AND EXISTS (SELECT 1 FROM structured_records AS private
                                WHERE private.source_id=raw.id AND private.user_id=raw.user_id AND private.kind='privacy')
                            AND NOT EXISTS (SELECT 1 FROM structured_records AS other
                                WHERE other.source_id=raw.id AND other.user_id=raw.user_id AND other.kind<>'privacy'))
                        ORDER BY raw.id LIMIT 1001''', [user_id]+['%'+name+'%' for name in names]).fetchall()
                    if len(named)>1000:
                        coverage_limited=True
                    for row in named[:1000]:
                        text=normalize(row['content'])
                        if any((bool(re.search(r'(?<!\w)'+re.escape(name)+r'(?!\w)',text))
                                if re.fullmatch(r'[a-z0-9_ .-]+',name) else name in text) for name in names):
                            summary_ids.append(row['id'])
                    summary_ids=list(dict.fromkeys(summary_ids))
            if summary_ids:
                required_groups.append(summary_ids)
            hints.append({'kind':'summary_coverage','result':{
                'index_complete':catalog['complete'], 'indexed_source_count':catalog['sources'],
                'topic_catalog_truncated':catalog['topic_catalog_truncated'],
                'requested_topic_count':len(topics), 'required_source_count':len(summary_ids)}})
            diagnostics['summary']={'kind':'summary','status':'complete_index' if catalog['complete'] else 'partial_index',
                                    'source_count':len(summary_ids)}
        required_groups = [list(group) for group in dict.fromkeys(tuple(group) for group in required_groups)]
        required = list(dict.fromkeys(source for group in required_groups for source in group))
        privacy_only = connection.execute("""SELECT record.source_id,record.attribute
            FROM structured_records AS record JOIN structured_sources AS source ON source.source_id=record.source_id
            AND source.user_id=record.user_id WHERE record.user_id=? AND record.kind='privacy' AND source.complete=1
            AND NOT EXISTS (SELECT 1 FROM structured_records AS other WHERE other.user_id=record.user_id
                            AND other.source_id=record.source_id AND other.kind<>'privacy')""", (user_id,)).fetchall()
        private_fields = defaultdict(list)
        for row in privacy_only:
            private_fields[row["source_id"]].append(row["attribute"])
        minimized = []
        normalized_query = normalize(query)
        for source, attributes in private_fields.items():
            mentioned = any((bool(re.search(r"(?<!\w)" + re.escape(attribute) + r"(?!\w)", normalized_query))
                             if re.fullmatch(r"[a-z0-9_ .-]+", attribute) else attribute in normalized_query)
                            for attribute in attributes)
            if not mentioned:
                minimized.append(source)
        blocked.extend(minimized)
        if minimized:
            diagnostics["privacy"] = {"kind": "privacy", "status": "unrequested_sensitive_sources", "source_count": len(minimized)}
        blocked = [source for source in dict.fromkeys(blocked) if source not in required]
        candidate_ids = list(dict.fromkeys(required + candidate_ids))[:100]
        rows = []
        if candidate_ids:
            rows = connection.execute("SELECT id,content,created_at,event_ts FROM raw_messages WHERE user_id=? AND id IN ({})".format(",".join("?" for _ in candidate_ids)), [user_id] + candidate_ids).fetchall()
            positions = {source: index for index, source in enumerate(candidate_ids)}
            rows.sort(key=lambda row: positions[row["id"]])
        return {"rows": rows, "required_groups": [["mem_{}".format(source) for source in group] for group in required_groups],
                "blocked_ids": ["mem_{}".format(source) for source in blocked], "hints": hints,
                "diagnostics": diagnostics, "index_complete": catalog["complete"], "coverage_limited": coverage_limited}


def enforce_obligations(decision: Dict[str, Any], specialist: Dict[str, Any],
                        pool_ids: Sequence[str]) -> Dict[str, Any]:
    if not decision.get("applied"):
        return decision
    if specialist.get("coverage_limited"):
        return {"applied": False, "reason": "specialist_catalog_incomplete", "selected_ids": []}
    required = {source for group in specialist.get("required_groups", []) for source in group}
    selected = set(decision["selected_ids"])
    if not required.issubset(pool_ids):
        return {"applied": False, "reason": "specialist_pool_incomplete", "selected_ids": []}
    if not required.issubset(selected) or selected.intersection(specialist.get("blocked_ids", [])):
        return {"applied": False, "reason": "specialist_obligation_missing", "selected_ids": []}
    return decision
