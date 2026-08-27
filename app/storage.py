import copy
import hashlib
import re
import sqlite3
import math
import threading
import unicodedata
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

from .model import MemoryModel
from .graph_routing import preferred_graph_predicates
from .local_instruction import mask_candidate_speakers
from .schemas import AddRequest, MemoryResult


# P3-A evidence-support rows deliberately use a normalizer that is independent
# of the parser's legacy sentence splitter.  Offsets are Python Unicode
# codepoint offsets in this exact normalized text:
# ``" ".join(unicodedata.normalize("NFKC", value).casefold().split())``.
SUPPORT_SCHEMA_VERSION = 1
SUPPORT_NORMALIZATION_ID = "nfkc-casefold-ws-v1"
SUPPORT_BINDINGS = frozenset({"named", "trusted_speaker_1p"})
SUPPORT_STATE_CHANGES = frozenset(
    {"assert", "update", "correction", "retraction", "historical"}
)
SUPPORT_TEMPORAL_STATUSES = frozenset(
    {"current", "previous", "historical", "future"}
)

# P3-B1 is deliberately a separate provenance contract from P3-A relation
# support.  An anchor proves only that one normalized entity label has an exact
# occurrence in its own immutable raw source; it never proves an identity or a
# semantic relation.
MENTION_DECLARATION_SCHEMA_VERSION = 1
MENTION_SCHEMA_VERSION = 1
MENTION_NORMALIZATION_ID = "nfkc-casefold-ws-v1"
MENTION_ANCHOR_ID_NAMESPACE = "p3b-anchor-v1"
MENTION_ENTITY_LABEL_MAX_CHARS = 128
_UNSEGMENTED_MENTION_SCRIPT_MARKERS = (
    "CJK UNIFIED IDEOGRAPH",
    "CJK COMPATIBILITY IDEOGRAPH",
    "HIRAGANA",
    "KATAKANA",
    "HANGUL",
    "THAI",
    "LAO",
    "KHMER",
    "MYANMAR",
)
_SUPPORT_STANDARD_SPEC_SUFFIXES = (
    "direct",
    "correction",
    "coordinated",
    "coordinated-first",
    "coordinated-object",
)
_SUPPORT_PREDICATES = (
    "friend_of",
    "parent_of",
    "sibling_of",
    "partner_of",
    "works_at",
    "role_at",
    "lives_in",
    "located_in",
    "likes",
    "prefers",
    "dislikes",
    "member_of",
    "participated_in",
    "owns",
    "created",
    "requires",
    "prohibits",
    "permits",
    "changed_to",
    "replaces",
)
SUPPORT_SPEC_IDS = frozenset(
    "{}:{}".format(predicate, suffix)
    for predicate in _SUPPORT_PREDICATES
    for suffix in _SUPPORT_STANDARD_SPEC_SUFFIXES
).union({"lives_in:moved-from-to", "member_of:retraction"})
_SUPPORT_RULE_PREDICATES = frozenset({"requires", "prohibits", "permits"})
_SUPPORT_DETERMINERS = frozenset(
    {
        "a",
        "an",
        "the",
        "this",
        "that",
        "these",
        "those",
        "my",
        "our",
        "your",
        "his",
        "her",
        "their",
        "another",
    }
)
_SUPPORT_FIRST_PERSON_SUBJECTS = frozenset(
    {"i", "i'm", "i’m", "i've", "i’ve", "i'd", "i’d"}
)
_SUPPORT_PREDICATE_MARKERS = {
    "friend_of": frozenset({"friend"}),
    "parent_of": frozenset({"parent", "mother", "father"}),
    "sibling_of": frozenset({"sibling", "brother", "sister"}),
    "partner_of": frozenset({"partner", "spouse", "husband", "wife"}),
    "works_at": frozenset({"work", "works", "worked", "employed"}),
    "role_at": frozenset(
        {"role", "position", "serve", "serves", "served"}
    ),
    "lives_in": frozenset({"live", "lives", "lived", "moved"}),
    "located_in": frozenset({"located", "based", "situated"}),
    "likes": frozenset(
        {
            "like", "likes", "liked", "love", "loves", "loved",
            "enjoy", "enjoys", "enjoyed", "keen", "fond",
        }
    ),
    "prefers": frozenset({"prefer", "prefers", "preferred"}),
    "dislikes": frozenset(
        {"dislike", "dislikes", "disliked", "hate", "hates", "hated"}
    ),
    "member_of": frozenset(
        {"member", "belong", "belongs", "belonged", "join", "joins", "joined"}
    ),
    "participated_in": frozenset(
        {
            "participate", "participates", "participated", "take", "takes",
            "took", "attend", "attends", "attended", "join", "joins",
            "joined", "went",
        }
    ),
    "owns": frozenset({"own", "owns", "owned", "possess", "possesses", "possessed"}),
    "created": frozenset(
        {
            "create", "creates", "created", "make", "makes", "made",
            "build", "builds", "built", "write", "writes", "wrote",
            "written", "design", "designs", "designed", "compose",
            "composes", "composed", "produce", "produces", "produced",
        }
    ),
    "requires": frozenset({"require", "requires", "required"}),
    "prohibits": frozenset(
        {"prohibit", "prohibits", "prohibited", "forbid", "forbids", "forbade"}
    ),
    "permits": frozenset(
        {"permit", "permits", "permitted", "allow", "allows", "allowed"}
    ),
    "changed_to": frozenset(
        {
            "change", "changes", "changed", "switch", "switches", "switched",
            "transition", "transitions", "transitioned", "became", "becomes",
        }
    ),
    "replaces": frozenset({"replace", "replaces", "replaced"}),
}


def normalize_evidence_mention_source(value: object) -> Optional[str]:
    """Return P3-B1's frozen canonical text representation.

    This must stay independent from the graph parser's entity linker.  Stored
    canonical spans are Python Unicode-codepoint offsets in this string.
    """

    if not isinstance(value, str):
        return None
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def source_local_anchor_entity_id(
    *, user_id: str, source_message_id: int, entity_label: str
) -> str:
    """Build a reproducible P3-B1 source-local label declaration key.

    The key intentionally derives from source-local, independently auditable
    values rather than from a model ID or P3-A's global identity graph.
    """

    material = "\0".join((
        MENTION_ANCHOR_ID_NAMESPACE,
        str(user_id),
        str(source_message_id),
        str(entity_label),
    ))
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def _is_unsegmented_mention_character(value: str) -> bool:
    name = unicodedata.name(value, "")
    return any(marker in name for marker in _UNSEGMENTED_MENTION_SCRIPT_MARKERS)


def _is_mention_word_character(value: str) -> bool:
    category = unicodedata.category(value)
    return (
        value == "_"
        or category[0] in {"L", "N", "M"}
        or category in {"Pc", "Cf"}
    )


def _mention_label_requires_boundaries(entity_label: str) -> bool:
    """Use token boundaries except for entirely unsegmented-script labels."""

    word_characters = [
        character for character in entity_label
        if _is_mention_word_character(character)
    ]
    return bool(word_characters) and not all(
        _is_unsegmented_mention_character(character)
        for character in word_characters
    )


def _normalised_token_boundary_map(
    raw_token: str,
    *,
    raw_offset: int,
) -> Optional[Tuple[str, List[Optional[int]]]]:
    """Map safe canonical token boundaries back to raw codepoint boundaries.

    NFKC/casefold can compose or expand codepoints.  A boundary inside such a
    transformation is deliberately left unmapped so an apparent canonical
    substring cannot be misrepresented as an exact raw-source span.
    """

    canonical_token = unicodedata.normalize("NFKC", raw_token).casefold()
    if not canonical_token or any(character.isspace() for character in canonical_token):
        return None
    candidates: Dict[int, List[int]] = {}
    for raw_index in range(len(raw_token) + 1):
        normalized_prefix = unicodedata.normalize(
            "NFKC", raw_token[:raw_index]
        ).casefold()
        if canonical_token.startswith(normalized_prefix):
            candidates.setdefault(len(normalized_prefix), []).append(
                raw_offset + raw_index
            )
    boundary_map: List[Optional[int]] = [None] * (len(canonical_token) + 1)
    for canonical_index, raw_indexes in candidates.items():
        if len(raw_indexes) == 1:
            boundary_map[canonical_index] = raw_indexes[0]
    if boundary_map[0] != raw_offset or boundary_map[-1] != raw_offset + len(raw_token):
        return None
    return canonical_token, boundary_map


def _canonical_source_with_raw_boundaries(
    source_content: str,
) -> Optional[Tuple[str, List[Optional[int]]]]:
    """Canonicalize a source and map each safe boundary to raw codepoints."""

    canonical_expected = normalize_evidence_mention_source(source_content)
    if canonical_expected is None or not canonical_expected:
        return None
    runs: List[Tuple[bool, int, int]] = []
    run_start = 0
    run_is_space = source_content[0].isspace()
    for index, character in enumerate(source_content[1:], start=1):
        is_space = character.isspace()
        if is_space != run_is_space:
            runs.append((run_is_space, run_start, index))
            run_start = index
            run_is_space = is_space
    runs.append((run_is_space, run_start, len(source_content)))

    canonical_parts: List[str] = []
    boundary_map: List[Optional[int]] = []
    pending_whitespace: Optional[Tuple[int, int]] = None
    for is_space, start, end in runs:
        if is_space:
            if canonical_parts:
                pending_whitespace = (start, end)
            continue
        token = _normalised_token_boundary_map(
            source_content[start:end], raw_offset=start
        )
        if token is None:
            return None
        canonical_token, token_boundaries = token
        if canonical_parts:
            if pending_whitespace is None:
                return None
            whitespace_start, whitespace_end = pending_whitespace
            if boundary_map[-1] != whitespace_start:
                return None
            canonical_parts.append(" ")
            boundary_map.append(whitespace_end)
        else:
            boundary_map.append(token_boundaries[0])
        if boundary_map[-1] != token_boundaries[0]:
            return None
        canonical_parts.append(canonical_token)
        boundary_map.extend(token_boundaries[1:])
        pending_whitespace = None
    canonical_source = "".join(canonical_parts)
    if (
        canonical_source != canonical_expected
        or len(boundary_map) != len(canonical_source) + 1
    ):
        return None
    return canonical_source, boundary_map


def exact_source_entity_mentions(
    source_content: str,
    entity_label: object,
) -> List[Dict[str, Any]]:
    """Find all fail-closed, source-exact P3-B1 mention witnesses.

    Returned spans are half-open Python codepoint offsets.  Canonical spans are
    against the frozen normalized source; raw spans are independently mapped
    back to the immutable original source and are retained for audit/display.
    """

    normalized_label = normalize_evidence_mention_source(entity_label)
    source = _canonical_source_with_raw_boundaries(source_content)
    if (
        not normalized_label
        or len(normalized_label) > MENTION_ENTITY_LABEL_MAX_CHARS
        or source is None
    ):
        return []
    canonical_source, boundary_map = source
    requires_boundaries = _mention_label_requires_boundaries(normalized_label)
    source_hash = hashlib.sha256(canonical_source.encode("utf-8")).hexdigest()
    witnesses: List[Dict[str, Any]] = []
    start = canonical_source.find(normalized_label)
    while start >= 0:
        end = start + len(normalized_label)
        preceding = canonical_source[start - 1] if start else None
        following = canonical_source[end] if end < len(canonical_source) else None
        if not (
            requires_boundaries
            and (
                (preceding is not None and _is_mention_word_character(preceding))
                or (following is not None and _is_mention_word_character(following))
            )
        ):
            raw_start = boundary_map[start]
            raw_end = boundary_map[end]
            if (
                raw_start is not None
                and raw_end is not None
                and raw_start < raw_end
                and normalize_evidence_mention_source(
                    source_content[raw_start:raw_end]
                ) == normalized_label
            ):
                witnesses.append({
                    "mention_schema_version": MENTION_SCHEMA_VERSION,
                    "normalization_id": MENTION_NORMALIZATION_ID,
                    "source_start": 0,
                    "source_end": len(canonical_source),
                    "mention_start": start,
                    "mention_end": end,
                    "raw_source_start": 0,
                    "raw_source_end": len(source_content),
                    "raw_mention_start": raw_start,
                    "raw_mention_end": raw_end,
                    "source_span_sha256": source_hash,
                })
        # Advance one canonical codepoint so overlapping literal CJK matches
        # are retained; a Latin partial remains guarded by boundaries above.
        start = canonical_source.find(normalized_label, start + 1)
    return witnesses


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
    graph_paths: Optional[List[Dict[str, Any]]] = None,
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
    paths = graph_paths or []
    if paths:
        rendered_paths = []
        for path in paths[:4]:
            relations = [
                str(item) for item in path.get("relations", []) if str(item).strip()
            ]
            source_ids = [
                str(item) for item in path.get("source_message_ids", [])
                if str(item).strip()
            ]
            if not relations or not source_ids:
                continue
            rendered_paths.append(
                "{} via {} (source IDs: {})".format(
                    path.get("hop_count", 1),
                    " -> ".join(relations),
                    ", ".join(source_ids),
                )
            )
        if rendered_paths:
            parts.append(
                "Untrusted graph retrieval metadata; use only the supplied source IDs:\n- {}".format(
                    "\n- ".join(rendered_paths)
                )
            )
    return "\n".join(parts)


_BRIDGE_CAPITALIZED_PATTERN = re.compile(
    r"\b[A-Z][a-zA-Z]{1,}(?:\s+[A-Z][a-zA-Z]{1,})*\b"
)


def extract_bridge_terms(
    query: str,
    candidate_texts: Sequence[str],
    known_speakers: Sequence[str],
    max_terms: int = 3,
) -> List[str]:
    """Deterministic bridge terms from first-pass evidence.

    P4-C bridge extraction: capitalized spans from the first-pass evidence,
    excluding the query's own terms, stop words, sentence-initial common words,
    and the known participant names (their speaker prefixes are prefix noise).
    The first hop can be a single candidate, so no cross-candidate co-occurrence
    is required; terms are ordered by frequency then specificity. No model call.
    """
    query_lower = set(re.findall(r"[\w]+", query.casefold()))
    speaker_lower = {
        speaker.casefold().strip()
        for speaker in known_speakers
        if speaker.strip()
    }
    stop_words = MemoryStore.QUERY_STOP_WORDS
    sentence_start_words = {
        "a", "an", "he", "her", "his", "i", "i'm", "im", "it", "its", "me",
        "my", "our", "she", "that", "the", "their", "there", "these", "they",
        "this", "we", "you", "your",
    }
    selected: List[str] = []
    seen: set = set()

    def add_term(term: str) -> bool:
        normalized = term.casefold().strip()
        if (
            not normalized
            or normalized in stop_words
            or normalized in query_lower
            or normalized in speaker_lower
            or normalized in sentence_start_words
            or normalized in seen
        ):
            return False
        seen.add(normalized)
        selected.append(term)
        return True

    span_counts: Dict[str, int] = {}
    span_originals: Dict[str, str] = {}
    for text in candidate_texts:
        for match in _BRIDGE_CAPITALIZED_PATTERN.finditer(text):
            normalized = match.group(0).casefold()
            span_counts[normalized] = span_counts.get(normalized, 0) + 1
            if normalized not in span_originals:
                span_originals[normalized] = match.group(0)
    for normalized, _count in sorted(
        span_counts.items(),
        key=lambda item: (-item[1], -len(item[0])),
    ):
        if len(selected) >= max_terms:
            break
        add_term(span_originals[normalized])
    return selected


class MemoryStore:
    RRF_CONSTANT = 60
    MODEL_RERANK_LIMIT = 30
    GRAPH_SEED_LIMIT = 6
    GRAPH_EDGE_LIMIT_PER_SEED = 20
    ANCHOR_SEED_LIMIT = 6
    ANCHOR_CANDIDATES_PER_SEED = 20
    ANCHOR_MAX_CANDIDATES = 20
    ANCHOR_RRF_WEIGHT = 0.025
    ANCHOR_RERANK_QUOTA = 4
    ADJACENT_SEED_LIMIT = 4
    ADJACENT_CANDIDATE_LIMIT = 4
    CONTEXT_RRF_WEIGHT = 0.5
    # Support terms widen recall but must not displace core-intent evidence.
    # A public synthetic sweep retained 0.01; weights >=0.05 caused G/H regressions,
    # while 0.02 could still flip a focused near-neighbor conflict case.
    STRUCTURED_SUPPORT_RRF_WEIGHT = 0.01
    # P4-A evidence-need channels: each planned evidence need gets its own
    # bounded lexical query; its candidates enter the rerank pool through a
    # reserved quota (default 2) instead of competing at core-intent weight.
    EVIDENCE_NEED_QUOTA_DEFAULT = 2
    EVIDENCE_NEED_RRF_WEIGHT_DEFAULT = 0.01
    # P4-C bridge second-pass: for multi-hop plans, deterministic bridge terms
    # (capitalized spans / speakers) are extracted from the first-pass evidence
    # and re-queried together with each evidence need. The bridge channel gets a
    # reserved pool quota; at most one second pass runs (no recursion).
    BRIDGE_MAX_TERMS_DEFAULT = 3
    BRIDGE_RRF_WEIGHT_DEFAULT = 0.01
    BRIDGE_RERANK_QUOTA_DEFAULT = 2
    # Shared sidecar pool: when >0, P4-A need and P4-C bridge candidates share
    # one total reservation instead of each holding its own fixed quota. This
    # prevents combos from over-compressing the P1 base budget (fixed-200 showed
    # need 2 + bridge 2 pushed P1 to 26 and dropped Hit@1 below baseline).
    SIDECAR_SHARED_QUOTA_DEFAULT = 0
    # P4-D query relaxation: only fires when every evidence-need channel returns
    # zero hits (a pure lexical-miss signal). The relaxed query ORs FTS5 prefix
    # forms (term*) of plan core terms + entities + query terms across the two
    # raw channels at low weight with a reserved quota. Audit (fixed-200 +
    # full 1977) found channel_miss cases are inference-type with zero lexical
    # overlap, so this is expected to add little; kept default-off for a
    # controlled ablation.
    RELAX_RRF_WEIGHT_DEFAULT = 0.01
    RELAX_QUOTA_DEFAULT = 2
    # P5 selective rerank gate: default-off deterministic reordering that only
    # acts when the Search model's Top-1 vs Top-2 are near-tied on the fusion
    # score AND the runner-up carries strictly stronger P1-channel evidence.
    # Full-1977 paired diagnostics showed P4-A converts 56 rank-2/3 cases to
    # Top-1 but displaces 71 Top-1 hits; P5 must gate on evidence and never
    # promote unconditionally. Fusion scores live on MemoryResult.score.
    P5_NEAR_TIE_EPSILON_DEFAULT = 0.0005
    P5_MIN_EVIDENCE_CHANNELS = 2
    # Confidence scale is 0-1, so the dominance margin differs from the fusion
    # score epsilon. Default 0.05: the runner-up must be at least 5 points more
    # confident than the Top-1 for a swap (a conservative, evidence-preserving
    # threshold).
    P5_CONFIDENCE_MARGIN_DEFAULT = 0.05
    # P5 strata: gate only fires for queries in the configured strata (default
    # temporal + correction), where the latest-valid-state rule is most
    # defensible. "all" restores the un-stratified gate.
    P5_STRATA_DEFAULT = "temporal,correction"
    # Correction/state-change language cues (query side).
    P5_CORRECTION_QUERY_PATTERN = re.compile(
        r"\b(change|changes|changed|update|updates|updated|correction|corrected|"
        r"revision|revised|no longer|used to|instead|currently|now|latest|"
        r"recent|newest|previous|earlier|before|after)\b|"
        r"改成|变为|更改|之前|后来|最新|现在|目前",
        flags=re.IGNORECASE,
    )
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
                 structured_query_plan: bool = False,
                 set_aware_rerank: bool = False,
                 evidence_graph: bool = False,
                 graph_max_hops: int = 1,
                 graph_temporal: bool = False,
                 graph_rrf_weight: float = 0.025,
                 graph_max_candidates: int = 20,
                 graph_rerank_quota: int = 4,
                 evidence_anchors: bool = False,
                 anchor_seed_limit: int = ANCHOR_SEED_LIMIT,
                 anchor_max_candidates: int = ANCHOR_MAX_CANDIDATES,
                 anchor_rrf_weight: float = ANCHOR_RRF_WEIGHT,
                 anchor_rerank_quota: int = ANCHOR_RERANK_QUOTA,
                 adjacent_turn_expansion: bool = False,
                 adjacent_seed_limit: int = ADJACENT_SEED_LIMIT,
                 adjacent_candidate_limit: int = ADJACENT_CANDIDATE_LIMIT,
                 evidence_need_retrieval: bool = False,
                 evidence_need_quota: int = EVIDENCE_NEED_QUOTA_DEFAULT,
                 evidence_need_rrf_weight: float = EVIDENCE_NEED_RRF_WEIGHT_DEFAULT,
                 bridge_retrieval: bool = False,
                 bridge_max_terms: int = BRIDGE_MAX_TERMS_DEFAULT,
                 bridge_rrf_weight: float = BRIDGE_RRF_WEIGHT_DEFAULT,
                 bridge_rerank_quota: int = BRIDGE_RERANK_QUOTA_DEFAULT,
                 sidecar_shared_quota: int = SIDECAR_SHARED_QUOTA_DEFAULT,
                 query_relaxation: bool = False,
                 relax_rrf_weight: float = RELAX_RRF_WEIGHT_DEFAULT,
                 relax_quota: int = RELAX_QUOTA_DEFAULT,
                 p5_gate: bool = False,
                 p5_near_tie_epsilon: float = P5_NEAR_TIE_EPSILON_DEFAULT,
                 p5_min_evidence_channels: int = P5_MIN_EVIDENCE_CHANNELS,
                 p5_confidence_margin: float = P5_CONFIDENCE_MARGIN_DEFAULT,
                 p5_strata: str = P5_STRATA_DEFAULT) -> None:
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
        self._last_speaker_conflict = False
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
        if not isinstance(set_aware_rerank, bool):
            raise ValueError("set_aware_rerank must be a boolean")
        if set_aware_rerank and not structured_query_plan:
            raise ValueError("set_aware_rerank requires structured_query_plan")
        self.set_aware_rerank = set_aware_rerank
        self.evidence_graph = evidence_graph
        self.evidence_anchors = evidence_anchors
        if evidence_graph and evidence_anchors:
            raise ValueError(
                "P3-B1 evidence_anchors must run with evidence_graph=false"
            )
        if evidence_graph:
            if not structured_query_plan:
                raise ValueError("evidence_graph requires structured_query_plan")
            if dense_fusion_alpha is not None:
                raise ValueError("evidence_graph cannot be combined with dense_fusion_alpha")
            if graph_max_hops != 1:
                raise ValueError("P3-A evidence_graph requires graph_max_hops=1")
            if graph_temporal:
                raise ValueError("P3-A evidence_graph requires graph_temporal=false")
            if not math.isfinite(graph_rrf_weight) or not 0 <= graph_rrf_weight <= 1:
                raise ValueError("graph_rrf_weight must be finite and between 0 and 1")
            if graph_max_candidates < 0 or graph_max_candidates > 100:
                raise ValueError("graph_max_candidates must be between 0 and 100")
            if graph_rerank_quota < 0 or graph_rerank_quota > self.MODEL_RERANK_LIMIT:
                raise ValueError(
                    "graph_rerank_quota must be between 0 and {}".format(
                        self.MODEL_RERANK_LIMIT
                    )
                )
            self.graph_max_hops = graph_max_hops
            self.graph_temporal = graph_temporal
            self.graph_rrf_weight = graph_rrf_weight
            self.graph_max_candidates = graph_max_candidates
            self.graph_rerank_quota = graph_rerank_quota
        else:
            # Graph-only knobs are intentionally inert when the feature is off.
            # This preserves the P1 constructor and runtime path even if stale
            # graph environment variables are present.
            self.graph_max_hops = 1
            self.graph_temporal = False
            self.graph_rrf_weight = 0.0
            self.graph_max_candidates = 0
            self.graph_rerank_quota = 0
        if evidence_anchors:
            if not structured_query_plan:
                raise ValueError("evidence_anchors requires structured_query_plan")
            if dense_fusion_alpha is not None:
                raise ValueError(
                    "P3-B1 evidence_anchors cannot use dense_fusion_alpha"
                )
            if (
                anchor_seed_limit != self.ANCHOR_SEED_LIMIT
                or anchor_max_candidates != self.ANCHOR_MAX_CANDIDATES
                or anchor_rrf_weight != self.ANCHOR_RRF_WEIGHT
                or anchor_rerank_quota != self.ANCHOR_RERANK_QUOTA
            ):
                raise ValueError(
                    "P3-B1 evidence_anchors is frozen at {}/{} / {} / {}".format(
                        self.ANCHOR_SEED_LIMIT,
                        self.ANCHOR_MAX_CANDIDATES,
                        self.ANCHOR_RRF_WEIGHT,
                        self.ANCHOR_RERANK_QUOTA,
                    )
                )
            self.anchor_seed_limit = anchor_seed_limit
            self.anchor_max_candidates = anchor_max_candidates
            self.anchor_rrf_weight = anchor_rrf_weight
            self.anchor_rerank_quota = anchor_rerank_quota
        else:
            # Anchor-only knobs are intentionally inert while P3-B1 is off.
            self.anchor_seed_limit = 0
            self.anchor_max_candidates = 0
            self.anchor_rrf_weight = 0.0
            self.anchor_rerank_quota = 0
        if not isinstance(adjacent_turn_expansion, bool):
            raise ValueError("adjacent_turn_expansion must be a boolean")
        if (
            isinstance(adjacent_seed_limit, bool)
            or not isinstance(adjacent_seed_limit, int)
            or not 0 <= adjacent_seed_limit <= self.MODEL_RERANK_LIMIT
        ):
            raise ValueError(
                "adjacent_seed_limit must be an integer between 0 and {}".format(
                    self.MODEL_RERANK_LIMIT
                )
            )
        if (
            isinstance(adjacent_candidate_limit, bool)
            or not isinstance(adjacent_candidate_limit, int)
            or not 0 <= adjacent_candidate_limit <= self.MODEL_RERANK_LIMIT
        ):
            raise ValueError(
                "adjacent_candidate_limit must be an integer between 0 and {}".format(
                    self.MODEL_RERANK_LIMIT
                )
            )
        if adjacent_turn_expansion:
            if evidence_graph or evidence_anchors:
                raise ValueError(
                    "P1.1 adjacent_turn_expansion must run with graph features=false"
                )
            if dense_fusion_alpha is not None:
                raise ValueError(
                    "P1.1 adjacent_turn_expansion cannot use dense_fusion_alpha"
                )
            if (
                adjacent_seed_limit != self.ADJACENT_SEED_LIMIT
                or adjacent_candidate_limit != self.ADJACENT_CANDIDATE_LIMIT
            ):
                raise ValueError(
                    "P1.1 adjacent expansion is frozen at {} seeds and {} candidates".format(
                        self.ADJACENT_SEED_LIMIT,
                        self.ADJACENT_CANDIDATE_LIMIT,
                    )
                )
        self.adjacent_turn_expansion = adjacent_turn_expansion
        self.adjacent_seed_limit = adjacent_seed_limit
        self.adjacent_candidate_limit = adjacent_candidate_limit
        if not isinstance(evidence_need_retrieval, bool):
            raise ValueError("evidence_need_retrieval must be a boolean")
        if evidence_need_retrieval and not structured_query_plan:
            raise ValueError("evidence_need_retrieval requires structured_query_plan")
        if evidence_need_retrieval and dense_fusion_alpha is not None:
            raise ValueError(
                "evidence_need_retrieval cannot use dense_fusion_alpha"
            )
        if (
            isinstance(evidence_need_quota, bool)
            or not isinstance(evidence_need_quota, int)
            or not 0 <= evidence_need_quota <= self.MODEL_RERANK_LIMIT
        ):
            raise ValueError(
                "evidence_need_quota must be an integer between 0 and {}".format(
                    self.MODEL_RERANK_LIMIT
                )
            )
        if (
            not isinstance(evidence_need_rrf_weight, (int, float))
            or isinstance(evidence_need_rrf_weight, bool)
            or not 0 <= evidence_need_rrf_weight <= 1
        ):
            raise ValueError(
                "evidence_need_rrf_weight must be a number between 0 and 1"
            )
        self.evidence_need_retrieval = evidence_need_retrieval
        self.evidence_need_quota = evidence_need_quota
        self.evidence_need_rrf_weight = evidence_need_rrf_weight
        if not isinstance(bridge_retrieval, bool):
            raise ValueError("bridge_retrieval must be a boolean")
        if bridge_retrieval and not structured_query_plan:
            raise ValueError("bridge_retrieval requires structured_query_plan")
        if bridge_retrieval and dense_fusion_alpha is not None:
            raise ValueError("bridge_retrieval cannot use dense_fusion_alpha")
        if (
            isinstance(bridge_max_terms, bool)
            or not isinstance(bridge_max_terms, int)
            or not 1 <= bridge_max_terms <= 5
        ):
            raise ValueError("bridge_max_terms must be an integer between 1 and 5")
        if (
            not isinstance(bridge_rrf_weight, (int, float))
            or isinstance(bridge_rrf_weight, bool)
            or not 0 <= bridge_rrf_weight <= 1
        ):
            raise ValueError("bridge_rrf_weight must be a number between 0 and 1")
        if (
            isinstance(bridge_rerank_quota, bool)
            or not isinstance(bridge_rerank_quota, int)
            or not 0 <= bridge_rerank_quota <= self.MODEL_RERANK_LIMIT
        ):
            raise ValueError(
                "bridge_rerank_quota must be an integer between 0 and {}".format(
                    self.MODEL_RERANK_LIMIT
                )
            )
        self.bridge_retrieval = bridge_retrieval
        self.bridge_max_terms = bridge_max_terms
        self.bridge_rrf_weight = bridge_rrf_weight
        self.bridge_rerank_quota = bridge_rerank_quota
        if (
            isinstance(sidecar_shared_quota, bool)
            or not isinstance(sidecar_shared_quota, int)
            or not 0 <= sidecar_shared_quota <= self.MODEL_RERANK_LIMIT
        ):
            raise ValueError(
                "sidecar_shared_quota must be an integer between 0 and {}".format(
                    self.MODEL_RERANK_LIMIT
                )
            )
        if sidecar_shared_quota > 0 and not (
            evidence_need_retrieval or bridge_retrieval
        ):
            raise ValueError(
                "sidecar_shared_quota requires evidence_need_retrieval or bridge_retrieval"
            )
        self.sidecar_shared_quota = sidecar_shared_quota
        if not isinstance(query_relaxation, bool):
            raise ValueError("query_relaxation must be a boolean")
        if query_relaxation and not evidence_need_retrieval:
            raise ValueError(
                "query_relaxation requires evidence_need_retrieval"
            )
        if (
            not isinstance(relax_rrf_weight, (int, float))
            or isinstance(relax_rrf_weight, bool)
            or not 0 <= relax_rrf_weight <= 1
        ):
            raise ValueError("relax_rrf_weight must be a number between 0 and 1")
        if (
            isinstance(relax_quota, bool)
            or not isinstance(relax_quota, int)
            or not 0 <= relax_quota <= self.MODEL_RERANK_LIMIT
        ):
            raise ValueError(
                "relax_quota must be an integer between 0 and {}".format(
                    self.MODEL_RERANK_LIMIT
                )
            )
        self.query_relaxation = query_relaxation
        self.relax_rrf_weight = relax_rrf_weight
        self.relax_quota = relax_quota
        if not isinstance(p5_gate, bool):
            raise ValueError("p5_gate must be a boolean")
        if p5_gate and not structured_query_plan:
            raise ValueError("p5_gate requires structured_query_plan")
        if (
            not isinstance(p5_near_tie_epsilon, (int, float))
            or isinstance(p5_near_tie_epsilon, bool)
            or not 0 <= p5_near_tie_epsilon <= 1
        ):
            raise ValueError("p5_near_tie_epsilon must be a number between 0 and 1")
        if (
            isinstance(p5_min_evidence_channels, bool)
            or not isinstance(p5_min_evidence_channels, int)
            or not 1 <= p5_min_evidence_channels <= 30
        ):
            raise ValueError(
                "p5_min_evidence_channels must be an integer between 1 and 30"
            )
        self.p5_gate = p5_gate
        self.p5_near_tie_epsilon = p5_near_tie_epsilon
        self.p5_min_evidence_channels = p5_min_evidence_channels
        if (
            not isinstance(p5_confidence_margin, (int, float))
            or isinstance(p5_confidence_margin, bool)
            or not 0 <= p5_confidence_margin <= 1
        ):
            raise ValueError("p5_confidence_margin must be a number between 0 and 1")
        self.p5_confidence_margin = p5_confidence_margin
        if not isinstance(p5_strata, str) or not p5_strata.strip():
            raise ValueError("p5_strata must be a non-empty string")
        allowed_strata = {"all", "temporal", "correction"}
        strata_set = {
            part.strip()
            for part in p5_strata.split(",")
            if part.strip()
        }
        if not strata_set or not strata_set.issubset(allowed_strata):
            raise ValueError(
                "p5_strata must be a comma-separated subset of all/temporal/correction"
            )
        self.p5_strata = ",".join(sorted(strata_set))
        self._retrieval_diagnostics_lock = threading.Lock()
        self._last_query_plan: Dict[str, Any] = {}
        self._last_graph_candidate_ids: List[str] = []
        self._last_graph_only_candidate_ids: List[str] = []
        self._last_graph_paths: List[Dict[str, Any]] = []
        self._last_retrieval_trace = self._empty_retrieval_trace()
        Path(database_path).parent.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _empty_retrieval_trace() -> Dict[str, Any]:
        return {
            "plan": {},
            "requested_seeds": [],
            "resolved_seeds": [],
            "unresolved_seeds": [],
            "p1_channels": {},
            "p1_union_ids": [],
            "p1_pre_rerank_ids": [],
            "p1_counterfactual_top30_ids": [],
            "p2_enabled": False,
            "p2_pre_rerank_ids": [],
            "p2_post_rerank_ids": [],
            "p2_evidence_need_tokens": [],
            "p2_newly_covered_tokens": [],
            "adjacent_seed_ids": [],
            "adjacent_candidate_ids": [],
            "adjacent_deduped_ids": [],
            "adjacent_diagnostics": {
                "enabled": False,
                "seed_limit": MemoryStore.ADJACENT_SEED_LIMIT,
                "candidate_limit": MemoryStore.ADJACENT_CANDIDATE_LIMIT,
                "considered_neighbor_ids": [],
                "candidate_limit_reached": False,
            },
            "graph_candidate_ids": [],
            "graph_channel_only_ids": [],
            "graph_paths": [],
            "anchor_candidate_ids": [],
            "anchor_channel_only_ids": [],
            "anchor_diagnostics": {
                "enabled": False,
                "seed_limit": MemoryStore.ANCHOR_SEED_LIMIT,
                "candidate_limit_per_seed": (
                    MemoryStore.ANCHOR_CANDIDATES_PER_SEED
                ),
                "candidate_limit": 0,
                "candidates_fetched_by_seed": [],
                "candidate_visit_seed_labels": [],
                "candidate_count": 0,
                "candidate_cap_reached": False,
            },
            "reserved_graph_ids": [],
            "reserved_anchor_ids": [],
            "reserved_adjacent_ids": [],
            "reserved_need_ids": [],
            "reserved_bridge_ids": [],
            "reserved_relax_ids": [],
            "promoted_graph_ids": [],
            "promoted_anchor_ids": [],
            "promoted_adjacent_ids": [],
            "promoted_need_ids": [],
            "promoted_bridge_ids": [],
            "displaced_p1_ids": [],
            "displaced_p1_for_anchor_ids": [],
            "displaced_p1_for_adjacent_ids": [],
            "displaced_p1_for_need_ids": [],
            "displaced_p1_for_bridge_ids": [],
            "bridge_diagnostics": {
                "enabled": False,
                "max_terms": MemoryStore.BRIDGE_MAX_TERMS_DEFAULT,
                "rrf_weight": MemoryStore.BRIDGE_RRF_WEIGHT_DEFAULT,
                "quota": MemoryStore.BRIDGE_RERANK_QUOTA_DEFAULT,
            },
            "bridge_terms": [],
            "bridge_channels": {},
            "bridge_union_ids": [],
            "evidence_need_diagnostics": {
                "enabled": False,
                "quota": MemoryStore.EVIDENCE_NEED_QUOTA_DEFAULT,
                "rrf_weight": MemoryStore.EVIDENCE_NEED_RRF_WEIGHT_DEFAULT,
                "need_count": 0,
            },
            "evidence_need_channels": {},
            "evidence_need_union_ids": [],
            "relax_diagnostics": {
                "enabled": False,
                "rrf_weight": MemoryStore.RELAX_RRF_WEIGHT_DEFAULT,
                "quota": MemoryStore.RELAX_QUOTA_DEFAULT,
            },
            "relax_channels": {},
            "relax_union_ids": [],
            "promoted_relax_ids": [],
            "displaced_p1_for_relax_ids": [],
            "p5_diagnostics": {
                "enabled": False,
                "near_tie_epsilon": MemoryStore.P5_NEAR_TIE_EPSILON_DEFAULT,
                "min_evidence_channels": MemoryStore.P5_MIN_EVIDENCE_CHANNELS,
                "confidence_margin": MemoryStore.P5_CONFIDENCE_MARGIN_DEFAULT,
                "strata": MemoryStore.P5_STRATA_DEFAULT,
                "strata_matched": False,
                "rank_confidence": {},
                "triggered": False,
                "swapped_ids": [],
                "top1_id": None,
                "top1_score": None,
                "top2_id": None,
                "top2_score": None,
                "top2_evidence_channels": 0,
                "top2_has_fact": False,
                "reason": None,
            },
            "rerank_pool_ids": [],
            "final_ids": [],
            "edge_diagnostics": {
                "seed_limit": MemoryStore.GRAPH_SEED_LIMIT,
                "edge_limit_per_seed": MemoryStore.GRAPH_EDGE_LIMIT_PER_SEED,
                "candidate_limit": 0,
                "edges_fetched_by_seed": [],
                "edges_considered": 0,
                "duplicate_edges_skipped": 0,
                "candidate_cap_reached": False,
                "candidate_count": 0,
                "path_count": 0,
                "edge_visit_seed_ids": [],
            },
        }

    @property
    def last_speaker_conflict(self) -> bool:
        with self._retrieval_diagnostics_lock:
            return self._last_speaker_conflict

    @property
    def last_query_plan(self) -> Dict[str, Any]:
        with self._retrieval_diagnostics_lock:
            return copy.deepcopy(self._last_query_plan)

    @property
    def last_graph_candidate_ids(self) -> List[str]:
        with self._retrieval_diagnostics_lock:
            return list(self._last_graph_candidate_ids)

    @property
    def last_graph_only_candidate_ids(self) -> List[str]:
        with self._retrieval_diagnostics_lock:
            return list(self._last_graph_only_candidate_ids)

    @property
    def last_graph_paths(self) -> List[Dict[str, Any]]:
        with self._retrieval_diagnostics_lock:
            return copy.deepcopy(self._last_graph_paths)

    @property
    def last_retrieval_trace(self) -> Dict[str, Any]:
        """Return an immutable-by-copy diagnostic snapshot for the last Search."""
        with self._retrieval_diagnostics_lock:
            return copy.deepcopy(self._last_retrieval_trace)

    def _publish_retrieval_diagnostics(
        self,
        *,
        query_plan: Dict[str, Any],
        graph_candidate_ids: Sequence[str],
        graph_only_candidate_ids: Sequence[str],
        graph_paths: Sequence[Dict[str, Any]],
        retrieval_trace: Dict[str, Any],
        speaker_conflict: bool,
    ) -> None:
        """Publish one completed Search snapshot without driving retrieval."""
        with self._retrieval_diagnostics_lock:
            self._last_speaker_conflict = speaker_conflict
            self._last_query_plan = copy.deepcopy(query_plan)
            self._last_graph_candidate_ids = list(graph_candidate_ids)
            self._last_graph_only_candidate_ids = list(graph_only_candidate_ids)
            self._last_graph_paths = copy.deepcopy(list(graph_paths))
            self._last_retrieval_trace = copy.deepcopy(retrieval_trace)

    def _normalized_evidence_need_tokens(
        self, query_plan: Dict[str, Any]
    ) -> List[str]:
        """Return bounded, plan-derived lexical coverage tokens for P2."""

        evidence_needs = query_plan.get("evidence_needs", [])
        if not isinstance(evidence_needs, list):
            return []
        tokens: List[str] = []
        seen = set()
        for need in evidence_needs[:4]:
            if not isinstance(need, str):
                continue
            for token in re.findall(r"[\w]+", need, flags=re.UNICODE):
                normalized = token.casefold()
                if (
                    normalized
                    and normalized not in self.QUERY_STOP_WORDS
                    and normalized not in seen
                ):
                    seen.add(normalized)
                    tokens.append(normalized)
        return tokens

    def _apply_set_aware_rerank(
        self,
        results: Sequence[MemoryResult],
        *,
        query_plan: Dict[str, Any],
        retrieval_trace: Dict[str, Any],
    ) -> List[MemoryResult]:
        """Deterministically promote complementary original evidence records."""

        original_results = list(results)
        need_tokens = self._normalized_evidence_need_tokens(query_plan)
        retrieval_trace["p2_enabled"] = self.set_aware_rerank
        retrieval_trace["p2_pre_rerank_ids"] = [
            result.id for result in original_results
        ]
        retrieval_trace["p2_evidence_need_tokens"] = list(need_tokens)
        if (
            not self.set_aware_rerank
            or len(original_results) < 2
            or not need_tokens
        ):
            retrieval_trace["p2_post_rerank_ids"] = list(
                retrieval_trace["p2_pre_rerank_ids"]
            )
            return original_results

        need_token_set = set(need_tokens)
        token_sets = [
            set(re.findall(r"[\w]+", result.content.casefold(), flags=re.UNICODE))
            for result in original_results
        ]
        remaining = list(range(len(original_results)))
        selected_indexes: List[int] = []
        covered_tokens = set()
        selected_diagnostics: List[Dict[str, Any]] = []
        while remaining:
            best_index = max(
                remaining,
                key=lambda index: (
                    bool((token_sets[index] & need_token_set) - covered_tokens),
                    len((token_sets[index] & need_token_set) - covered_tokens),
                    -index,
                ),
            )
            new_tokens = sorted(
                (token_sets[best_index] & need_token_set) - covered_tokens
            )
            if not new_tokens:
                break
            selected_indexes.append(best_index)
            remaining.remove(best_index)
            covered_tokens.update(new_tokens)
            selected_diagnostics.append({
                "candidate_id": original_results[best_index].id,
                "newly_covered_tokens": new_tokens,
            })
        ordered_indexes = selected_indexes + remaining
        ordered_results = [original_results[index] for index in ordered_indexes]
        retrieval_trace["p2_post_rerank_ids"] = [
            result.id for result in ordered_results
        ]
        retrieval_trace["p2_newly_covered_tokens"] = selected_diagnostics
        return ordered_results

    def _apply_selective_rerank_gate(
        self,
        results: Sequence[MemoryResult],
        *,
        query: str,
        query_plan: Dict[str, Any],
        retrieval_trace: Dict[str, Any],
        ranking_metadata: Optional[Dict[Any, Dict[str, Any]]] = None,
        message_ids: Optional[Dict[str, int]] = None,
    ) -> List[MemoryResult]:
        """P5 selective gate: swap Top-1/Top-2 only when they are near-tied on
        the fusion score AND the runner-up has strictly more query-token
        overlap with the question than the current Top-1 (plus an optional fact
        annotation). Default-off; never promotes unconditionally."""

        diagnostics = retrieval_trace["p5_diagnostics"]
        diagnostics.update({
            "enabled": self.p5_gate,
            "near_tie_epsilon": self.p5_near_tie_epsilon,
            "min_evidence_channels": self.p5_min_evidence_channels,
            "confidence_margin": self.p5_confidence_margin,
            "strata": self.p5_strata,
        })
        original = list(results)
        if (
            not self.p5_gate
            or len(original) < 2
            or not retrieval_trace.get("p1_channels")
        ):
            return original

        # Strata gate: only act for queries whose language (or plan intent)
        # matches the configured strata. This is a pre-registration of where the
        # latest-valid-state rule is defensible; outside the strata the gate
        # never fires regardless of near-tie or evidence.
        strata_parts = {
            part.strip()
            for part in self.p5_strata.split(",")
            if part.strip()
        }
        plan_intent = str(query_plan.get("intent", "")).casefold()
        temporal_match = bool(
            self.TEMPORAL_QUERY_PATTERN.search(query)
            or self.HISTORICAL_QUERY_PATTERN.search(query)
            or plan_intent == "temporal"
        )
        correction_match = bool(
            self.P5_CORRECTION_QUERY_PATTERN.search(query)
            or plan_intent == "correction"
        )
        strata_matched = (
            ("all" in strata_parts)
            or ("temporal" in strata_parts and temporal_match)
            or ("correction" in strata_parts and correction_match)
        )
        diagnostics["strata_matched"] = strata_matched
        if not strata_matched:
            diagnostics["reason"] = "strata_excluded"
            return original

        top1, top2 = original[0], original[1]
        top1_score = getattr(top1, "score", None)
        top2_score = getattr(top2, "score", None)
        if top1_score is None or top2_score is None:
            diagnostics["reason"] = "missing_fusion_score"
            return original
        # LLM's own confidence (0-1) for the two leading candidates. This is
        # the model's near-tie signal: if the model ranks A first but gives A
        # and B nearly equal confidence, and B is the one it is more sure of,
        # a swap reflects the model's own uncertainty. When confidence is
        # available it REPLACES the fusion-score near-tie gate (the fusion gap
        # is a retrieval-layer proxy that already failed three times); the
        # fusion gap only gates when confidence is absent.
        rank_confidence = retrieval_trace.get("p5_diagnostics", {}).get(
            "rank_confidence", {}
        )
        top1_confidence = rank_confidence.get(top1.id)
        top2_confidence = rank_confidence.get(top2.id)
        confidence_available = (
            isinstance(top1_confidence, (int, float))
            and isinstance(top2_confidence, (int, float))
        )
        if not confidence_available:
            score_gap = top1_score - top2_score
            if score_gap >= self.p5_near_tie_epsilon:
                diagnostics.update({
                    "top1_id": top1.id,
                    "top1_score": top1_score,
                    "top2_id": top2.id,
                    "top2_score": top2_score,
                    "reason": "not_near_tie",
                })
                return original

        # Query-token overlap: how many non-stopword query tokens appear in each
        # candidate's content. This is a correctness-adjacent signal (a runner-up
        # that literally restates more of the question is more likely the answer),
        # unlike P1 channel counts, which are multi-view projections of the same
        # content and carry no discrimination.
        query_tokens = {
            token.casefold()
            for token in re.findall(r"[\w]+", query, flags=re.UNICODE)
            if token.casefold() not in self.QUERY_STOP_WORDS
        }

        def overlap_count(result: MemoryResult) -> int:
            content_tokens = {
                token.casefold()
                for token in re.findall(
                    r"[\w]+", result.content, flags=re.UNICODE
                )
            }
            return len(query_tokens & content_tokens)

        top1_overlap = overlap_count(top1)
        top2_overlap = overlap_count(top2)

        # P1-channel counts stay as diagnostics only (multi-view, no signal).
        top1_channel_count = 0
        top2_channel_count = 0
        for rows in retrieval_trace.get("p1_channels", {}).values():
            if not isinstance(rows, list):
                continue
            if top1.id in rows:
                top1_channel_count += 1
            if top2.id in rows:
                top2_channel_count += 1
        has_fact = False
        if ranking_metadata and message_ids:
            raw_message_id = message_ids.get(top2.id)
            if raw_message_id is not None and ranking_metadata.get(raw_message_id, {}).get("facts"):
                has_fact = True

        # LLM's own confidence (0-1) for the two leading candidates.
        diagnostics.update({
            "top1_id": top1.id,
            "top1_score": top1_score,
            "top2_id": top2.id,
            "top2_score": top2_score,
            "top1_query_overlap": top1_overlap,
            "top2_query_overlap": top2_overlap,
            "top1_confidence": top1_confidence,
            "top2_confidence": top2_confidence,
            "top1_evidence_channels": top1_channel_count,
            "top2_evidence_channels": top2_channel_count,
            "top2_has_fact": has_fact,
        })
        if confidence_available:
            strictly_stronger = (
                top2_confidence > top1_confidence
                and (top2_confidence - top1_confidence) >= self.p5_confidence_margin
            ) or has_fact
        else:
            # No confidence (truncated long-dialogue question or model without
            # the method): do NOT fall back to retrieval-layer proxies — they
            # were empirically disproven (channels -0.005, overlap -0.035,
            # strata -0.010). Missing confidence means no swap.
            diagnostics["reason"] = "no_confidence_signal"
            return original
        if not strictly_stronger:
            diagnostics["reason"] = "runner_up_not_strictly_stronger"
            return original

        # Both conditions met: swap Top-1/Top-2. The loser keeps rank 2 so no
        # Top-1 evidence is lost from the visible set.
        swapped = [top2, top1] + original[2:]
        diagnostics.update({
            "triggered": True,
            "swapped_ids": [top2.id, top1.id],
            "reason": "near_tie_with_stronger_evidence",
        })
        return swapped

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.database_path, timeout=15)
        connection.row_factory = sqlite3.Row
        try:
            if self.evidence_graph or self.evidence_anchors:
                connection.execute("PRAGMA foreign_keys=ON")
                enabled = connection.execute("PRAGMA foreign_keys").fetchone()
                if not enabled or int(enabled[0]) != 1:
                    raise RuntimeError("SQLite foreign key enforcement is unavailable")
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
            if self.evidence_graph or self.evidence_anchors:
                existing_graph_entity = connection.execute(
                    """SELECT 1 FROM sqlite_master
                       WHERE type = 'table' AND name = 'graph_entities'"""
                ).fetchone()
                if existing_graph_entity:
                    existing_columns = {
                        str(row["name"])
                        for row in connection.execute(
                            "PRAGMA table_info(graph_entities)"
                        ).fetchall()
                    }
                    if "identity_hint" not in existing_columns:
                        # P3 schema v2: additive, nullable scoped-identity hint.
                        connection.execute(
                            "ALTER TABLE graph_entities ADD COLUMN identity_hint TEXT"
                        )
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
            if not (self.evidence_graph or self.evidence_anchors):
                return
            connection.executescript(
                """
                CREATE UNIQUE INDEX IF NOT EXISTS raw_messages_id_user_uq
                    ON raw_messages(id, user_id);
                CREATE TABLE IF NOT EXISTS schema_migrations (
                    version INTEGER PRIMARY KEY,
                    applied_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS graph_entities (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    canonical_name TEXT NOT NULL,
                    display_name TEXT NOT NULL,
                    entity_type TEXT NOT NULL CHECK(entity_type IN (
                        'person', 'organization', 'location', 'event', 'activity',
                        'object', 'product', 'food', 'document', 'rule', 'topic',
                        'group'
                    )),
                    identity_hint TEXT,
                    first_source_message_id INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    UNIQUE(id, user_id),
                    FOREIGN KEY(first_source_message_id, user_id)
                        REFERENCES raw_messages(id, user_id) ON DELETE RESTRICT
                );
                CREATE INDEX IF NOT EXISTS graph_entities_lookup_idx
                    ON graph_entities(user_id, canonical_name, entity_type);
                CREATE INDEX IF NOT EXISTS graph_entities_identity_lookup_idx
                    ON graph_entities(
                        user_id, canonical_name, entity_type, identity_hint
                    );
                CREATE TABLE IF NOT EXISTS graph_aliases (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    entity_id INTEGER NOT NULL,
                    normalized_alias TEXT NOT NULL,
                    display_alias TEXT NOT NULL,
                    source_message_id INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(entity_id, user_id)
                        REFERENCES graph_entities(id, user_id) ON DELETE RESTRICT,
                    FOREIGN KEY(source_message_id, user_id)
                        REFERENCES raw_messages(id, user_id) ON DELETE RESTRICT,
                    UNIQUE(user_id, entity_id, normalized_alias, source_message_id)
                );
                CREATE INDEX IF NOT EXISTS graph_aliases_lookup_idx
                    ON graph_aliases(user_id, normalized_alias);
                CREATE TABLE IF NOT EXISTS graph_edges (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    subject_entity_id INTEGER NOT NULL,
                    predicate TEXT NOT NULL CHECK(predicate IN (
                        'friend_of', 'parent_of', 'sibling_of', 'partner_of',
                        'works_at', 'role_at', 'lives_in', 'located_in', 'likes',
                        'prefers', 'dislikes', 'member_of', 'participated_in',
                        'owns', 'created', 'requires', 'prohibits', 'permits',
                        'changed_to', 'replaces'
                    )),
                    object_entity_id INTEGER,
                    object_value TEXT,
                    source_message_id INTEGER NOT NULL,
                    event_ts INTEGER,
                    state_change TEXT NOT NULL CHECK(state_change IN (
                        'assert', 'update', 'correction', 'retraction', 'historical'
                    )),
                    temporal_status TEXT CHECK(
                        temporal_status IS NULL OR temporal_status IN (
                            'current', 'previous', 'historical', 'future'
                        )
                    ),
                    supersedes_edge_id INTEGER,
                    created_at TEXT NOT NULL,
                    UNIQUE(id, user_id),
                    CHECK((object_entity_id IS NOT NULL) != (object_value IS NOT NULL)),
                    FOREIGN KEY(subject_entity_id, user_id)
                        REFERENCES graph_entities(id, user_id) ON DELETE RESTRICT,
                    FOREIGN KEY(object_entity_id, user_id)
                        REFERENCES graph_entities(id, user_id) ON DELETE RESTRICT,
                    FOREIGN KEY(source_message_id, user_id)
                        REFERENCES raw_messages(id, user_id) ON DELETE RESTRICT,
                    FOREIGN KEY(supersedes_edge_id, user_id)
                        REFERENCES graph_edges(id, user_id) ON DELETE RESTRICT
                );
                CREATE INDEX IF NOT EXISTS graph_edges_subject_idx
                    ON graph_edges(user_id, subject_entity_id, predicate);
                CREATE INDEX IF NOT EXISTS graph_edges_object_idx
                    ON graph_edges(user_id, object_entity_id, predicate);
                CREATE INDEX IF NOT EXISTS graph_edges_source_idx
                    ON graph_edges(user_id, source_message_id);
                CREATE UNIQUE INDEX IF NOT EXISTS graph_edges_entity_uq
                    ON graph_edges(
                        user_id, subject_entity_id, predicate,
                        object_entity_id, source_message_id
                    ) WHERE object_entity_id IS NOT NULL;
                CREATE UNIQUE INDEX IF NOT EXISTS graph_edges_value_uq
                    ON graph_edges(
                        user_id, subject_entity_id, predicate,
                        object_value, source_message_id
                    ) WHERE object_value IS NOT NULL;
                """
            )
            # Schema v3 is additive.  The three-column parent key is required
            # by SQLite before the support table may reference the immutable
            # edge/user/source provenance triple.
            connection.execute(
                """CREATE UNIQUE INDEX IF NOT EXISTS
                   graph_edges_id_user_source_uq
                   ON graph_edges(id, user_id, source_message_id)"""
            )
            support_spec_sql = ", ".join(
                "'{}'".format(spec_id.replace("'", "''"))
                for spec_id in sorted(SUPPORT_SPEC_IDS)
            )
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS graph_edge_support (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    edge_id INTEGER NOT NULL,
                    user_id TEXT NOT NULL,
                    source_message_id INTEGER NOT NULL,
                    support_schema_version INTEGER NOT NULL
                        DEFAULT 1 CHECK(support_schema_version = 1),
                    normalization_id TEXT NOT NULL
                        DEFAULT 'nfkc-casefold-ws-v1'
                        CHECK(normalization_id = 'nfkc-casefold-ws-v1'),
                    spec_id TEXT NOT NULL CHECK(spec_id IN ({})),
                    binding TEXT NOT NULL CHECK(binding IN (
                        'named', 'trusted_speaker_1p'
                    )),
                    source_start INTEGER NOT NULL CHECK(source_start = 0),
                    source_end INTEGER NOT NULL
                        CHECK(source_end > source_start),
                    clause_start INTEGER NOT NULL,
                    clause_end INTEGER NOT NULL,
                    subject_start INTEGER NOT NULL,
                    subject_end INTEGER NOT NULL,
                    predicate_start INTEGER NOT NULL,
                    predicate_end INTEGER NOT NULL,
                    object_start INTEGER NOT NULL,
                    object_end INTEGER NOT NULL,
                    state_change TEXT NOT NULL CHECK(state_change IN (
                        'assert', 'update', 'correction', 'retraction', 'historical'
                    )),
                    temporal_status TEXT CHECK(
                        temporal_status IS NULL OR temporal_status IN (
                            'current', 'previous', 'historical', 'future'
                        )
                    ),
                    source_span_sha256 TEXT NOT NULL CHECK(
                        length(source_span_sha256) = 64
                        AND source_span_sha256 NOT GLOB '*[^0-9a-f]*'
                    ),
                    CHECK(
                        clause_start >= source_start
                        AND clause_start < clause_end
                        AND clause_end <= source_end
                    ),
                    CHECK(
                        subject_start >= clause_start
                        AND subject_start < subject_end
                        AND subject_end <= clause_end
                    ),
                    CHECK(
                        predicate_start >= clause_start
                        AND predicate_start < predicate_end
                        AND predicate_end <= clause_end
                    ),
                    CHECK(
                        object_start >= clause_start
                        AND object_start < object_end
                        AND object_end <= clause_end
                    ),
                    CHECK(
                        subject_end <= predicate_start
                        AND predicate_end <= object_start
                    ),
                    CHECK(
                        (state_change = 'assert' AND temporal_status IS NULL)
                        OR (state_change <> 'assert' AND temporal_status IS NOT NULL)
                    ),
                    UNIQUE(edge_id, user_id, source_message_id),
                    FOREIGN KEY(edge_id, user_id, source_message_id)
                        REFERENCES graph_edges(id, user_id, source_message_id)
                        ON DELETE RESTRICT,
                    FOREIGN KEY(source_message_id, user_id)
                        REFERENCES raw_messages(id, user_id) ON DELETE RESTRICT
                );
                CREATE INDEX IF NOT EXISTS graph_edge_support_source_idx
                    ON graph_edge_support(user_id, source_message_id);
                """.format(support_spec_sql)
            )
            # Schema v4 keeps P3-B1 source-local label declarations out of
            # the P3-A global identity graph.  The declaration/occurrence
            # compound keys make every candidate independently traceable to
            # one immutable raw message without inferring an entity identity.
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS graph_entity_declarations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    source_message_id INTEGER NOT NULL,
                    anchor_entity_id TEXT NOT NULL CHECK(
                        length(anchor_entity_id) = 64
                        AND anchor_entity_id NOT GLOB '*[^0-9a-f]*'
                    ),
                    declaration_schema_version INTEGER NOT NULL
                        DEFAULT 1 CHECK(declaration_schema_version = 1),
                    normalization_id TEXT NOT NULL
                        DEFAULT 'nfkc-casefold-ws-v1'
                        CHECK(normalization_id = 'nfkc-casefold-ws-v1'),
                    entity_label TEXT NOT NULL CHECK(
                        length(trim(entity_label)) > 0
                        AND length(entity_label) <= 128
                    ),
                    created_at TEXT NOT NULL,
                    UNIQUE(id, user_id, source_message_id),
                    UNIQUE(user_id, source_message_id, anchor_entity_id),
                    UNIQUE(user_id, source_message_id, entity_label),
                    FOREIGN KEY(source_message_id, user_id)
                        REFERENCES raw_messages(id, user_id) ON DELETE RESTRICT
                );
                CREATE INDEX IF NOT EXISTS graph_entity_declarations_lookup_idx
                    ON graph_entity_declarations(
                        user_id, entity_label, source_message_id, id
                    );
                CREATE TABLE IF NOT EXISTS graph_entity_mentions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    declaration_id INTEGER NOT NULL,
                    user_id TEXT NOT NULL,
                    source_message_id INTEGER NOT NULL,
                    mention_schema_version INTEGER NOT NULL
                        DEFAULT 1 CHECK(mention_schema_version = 1),
                    normalization_id TEXT NOT NULL
                        DEFAULT 'nfkc-casefold-ws-v1'
                        CHECK(normalization_id = 'nfkc-casefold-ws-v1'),
                    entity_label TEXT NOT NULL CHECK(
                        length(trim(entity_label)) > 0
                        AND length(entity_label) <= 128
                    ),
                    source_start INTEGER NOT NULL CHECK(source_start = 0),
                    source_end INTEGER NOT NULL CHECK(source_end > source_start),
                    mention_start INTEGER NOT NULL,
                    mention_end INTEGER NOT NULL,
                    raw_source_start INTEGER NOT NULL
                        CHECK(raw_source_start = 0),
                    raw_source_end INTEGER NOT NULL
                        CHECK(raw_source_end > raw_source_start),
                    raw_mention_start INTEGER NOT NULL,
                    raw_mention_end INTEGER NOT NULL,
                    source_span_sha256 TEXT NOT NULL CHECK(
                        length(source_span_sha256) = 64
                        AND source_span_sha256 NOT GLOB '*[^0-9a-f]*'
                    ),
                    CHECK(
                        mention_start >= source_start
                        AND mention_start < mention_end
                        AND mention_end <= source_end
                    ),
                    CHECK(
                        raw_mention_start >= raw_source_start
                        AND raw_mention_start < raw_mention_end
                        AND raw_mention_end <= raw_source_end
                    ),
                    UNIQUE(
                        declaration_id, user_id, source_message_id,
                        mention_start, mention_end
                    ),
                    UNIQUE(
                        user_id, source_message_id, entity_label,
                        mention_start, mention_end
                    ),
                    FOREIGN KEY(declaration_id, user_id, source_message_id)
                        REFERENCES graph_entity_declarations(
                            id, user_id, source_message_id
                        ) ON DELETE RESTRICT,
                    FOREIGN KEY(source_message_id, user_id)
                        REFERENCES raw_messages(id, user_id) ON DELETE RESTRICT
                );
                CREATE INDEX IF NOT EXISTS graph_entity_mentions_lookup_idx
                    ON graph_entity_mentions(
                        user_id, entity_label, source_message_id,
                        mention_start, id
                    );
                CREATE INDEX IF NOT EXISTS graph_entity_mentions_source_idx
                    ON graph_entity_mentions(user_id, source_message_id);
                """
            )
            connection.execute(
                "INSERT OR IGNORE INTO schema_migrations(version, applied_at) VALUES (1, ?)",
                (datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),),
            )
            connection.execute(
                "INSERT OR IGNORE INTO schema_migrations(version, applied_at) VALUES (2, ?)",
                (datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),),
            )
            connection.execute(
                "INSERT OR IGNORE INTO schema_migrations(version, applied_at) VALUES (3, ?)",
                (datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),),
            )
            connection.execute(
                "INSERT OR IGNORE INTO schema_migrations(version, applied_at) VALUES (4, ?)",
                (datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),),
            )
            expected_columns = {
                "graph_entities": {
                    "id", "user_id", "canonical_name", "display_name",
                    "entity_type", "identity_hint", "first_source_message_id",
                    "created_at",
                },
                "graph_aliases": {
                    "id", "user_id", "entity_id", "normalized_alias",
                    "display_alias", "source_message_id", "created_at",
                },
                "graph_edges": {
                    "id", "user_id", "subject_entity_id", "predicate",
                    "object_entity_id", "object_value", "source_message_id",
                    "event_ts", "state_change", "temporal_status",
                    "supersedes_edge_id", "created_at",
                },
                "graph_edge_support": {
                    "id", "edge_id", "user_id", "source_message_id",
                    "support_schema_version", "normalization_id", "spec_id",
                    "binding", "source_start", "source_end", "clause_start",
                    "clause_end", "subject_start", "subject_end",
                    "predicate_start", "predicate_end", "object_start",
                    "object_end", "state_change", "temporal_status",
                    "source_span_sha256",
                },
                "graph_entity_declarations": {
                    "id", "user_id", "source_message_id", "anchor_entity_id",
                    "declaration_schema_version", "normalization_id",
                    "entity_label", "created_at",
                },
                "graph_entity_mentions": {
                    "id", "declaration_id", "user_id", "source_message_id",
                    "mention_schema_version", "normalization_id", "entity_label",
                    "source_start", "source_end", "mention_start", "mention_end",
                    "raw_source_start", "raw_source_end", "raw_mention_start",
                    "raw_mention_end", "source_span_sha256",
                },
            }
            for table_name, required in expected_columns.items():
                actual = {
                    str(row["name"])
                    for row in connection.execute(
                        "PRAGMA table_info({})".format(table_name)
                    ).fetchall()
                }
                if not required.issubset(actual):
                    raise RuntimeError(
                        "incompatible {} schema; missing {}".format(
                            table_name, sorted(required - actual)
                        )
                    )

            def unique_indexes(table_name: str) -> set:
                result = set()
                for index_row in connection.execute(
                    "PRAGMA index_list({})".format(table_name)
                ).fetchall():
                    if not int(index_row["unique"]):
                        continue
                    index_name = str(index_row["name"]).replace('"', '""')
                    columns = tuple(
                        str(column["name"])
                        for column in connection.execute(
                            'PRAGMA index_info("{}")'.format(index_name)
                        ).fetchall()
                    )
                    result.add(columns)
                return result

            required_unique_indexes = {
                "raw_messages": {("id", "user_id")},
                "graph_entities": {("id", "user_id")},
                "graph_aliases": {
                    (
                        "user_id", "entity_id", "normalized_alias",
                        "source_message_id",
                    )
                },
                "graph_edges": {
                    ("id", "user_id"),
                    ("id", "user_id", "source_message_id"),
                    (
                        "user_id", "subject_entity_id", "predicate",
                        "object_entity_id", "source_message_id",
                    ),
                    (
                        "user_id", "subject_entity_id", "predicate",
                        "object_value", "source_message_id",
                    ),
                },
                "graph_edge_support": {
                    ("edge_id", "user_id", "source_message_id"),
                },
                "graph_entity_declarations": {
                    ("id", "user_id", "source_message_id"),
                    ("user_id", "source_message_id", "anchor_entity_id"),
                    ("user_id", "source_message_id", "entity_label"),
                },
                "graph_entity_mentions": {
                    (
                        "declaration_id", "user_id", "source_message_id",
                        "mention_start", "mention_end",
                    ),
                    (
                        "user_id", "source_message_id", "entity_label",
                        "mention_start", "mention_end",
                    ),
                },
            }
            for table_name, required in required_unique_indexes.items():
                actual = unique_indexes(table_name)
                if not required.issubset(actual):
                    raise RuntimeError(
                        "incompatible {} schema; missing unique indexes".format(
                            table_name
                        )
                    )

            def foreign_keys(table_name: str) -> set:
                grouped: Dict[int, List[sqlite3.Row]] = {}
                for row in connection.execute(
                    "PRAGMA foreign_key_list({})".format(table_name)
                ).fetchall():
                    grouped.setdefault(int(row["id"]), []).append(row)
                result = set()
                for rows in grouped.values():
                    ordered = sorted(rows, key=lambda row: int(row["seq"]))
                    result.add((
                        str(ordered[0]["table"]),
                        tuple(str(row["from"]) for row in ordered),
                        tuple(str(row["to"]) for row in ordered),
                    ))
                return result

            required_foreign_keys = {
                "graph_entities": {
                    (
                        "raw_messages",
                        ("first_source_message_id", "user_id"),
                        ("id", "user_id"),
                    ),
                },
                "graph_aliases": {
                    (
                        "graph_entities",
                        ("entity_id", "user_id"),
                        ("id", "user_id"),
                    ),
                    (
                        "raw_messages",
                        ("source_message_id", "user_id"),
                        ("id", "user_id"),
                    ),
                },
                "graph_edges": {
                    (
                        "graph_entities",
                        ("subject_entity_id", "user_id"),
                        ("id", "user_id"),
                    ),
                    (
                        "graph_entities",
                        ("object_entity_id", "user_id"),
                        ("id", "user_id"),
                    ),
                    (
                        "raw_messages",
                        ("source_message_id", "user_id"),
                        ("id", "user_id"),
                    ),
                    (
                        "graph_edges",
                        ("supersedes_edge_id", "user_id"),
                        ("id", "user_id"),
                    ),
                },
                "graph_edge_support": {
                    (
                        "graph_edges",
                        ("edge_id", "user_id", "source_message_id"),
                        ("id", "user_id", "source_message_id"),
                    ),
                    (
                        "raw_messages",
                        ("source_message_id", "user_id"),
                        ("id", "user_id"),
                    ),
                },
                "graph_entity_declarations": {
                    (
                        "raw_messages",
                        ("source_message_id", "user_id"),
                        ("id", "user_id"),
                    ),
                },
                "graph_entity_mentions": {
                    (
                        "graph_entity_declarations",
                        ("declaration_id", "user_id", "source_message_id"),
                        ("id", "user_id", "source_message_id"),
                    ),
                    (
                        "raw_messages",
                        ("source_message_id", "user_id"),
                        ("id", "user_id"),
                    ),
                },
            }
            for table_name, required in required_foreign_keys.items():
                if not required.issubset(foreign_keys(table_name)):
                    raise RuntimeError(
                        "incompatible {} schema; missing provenance foreign keys".format(
                            table_name
                        )
                    )

            required_check_fragments = {
                "graph_entities": ("CHECK(ENTITY_TYPE IN",),
                "graph_edges": (
                    "CHECK(PREDICATE IN",
                    "CHECK(STATE_CHANGE IN",
                    "CHECK( TEMPORAL_STATUS IS NULL OR TEMPORAL_STATUS IN",
                    "CHECK((OBJECT_ENTITY_ID IS NOT NULL) != (OBJECT_VALUE IS NOT NULL))",
                ),
                "graph_edge_support": (
                    "CHECK(SUPPORT_SCHEMA_VERSION = 1)",
                    "CHECK(NORMALIZATION_ID = 'NFKC-CASEFOLD-WS-V1')",
                    "CHECK(SPEC_ID IN",
                    "CHECK(BINDING IN",
                    "CHECK(SOURCE_START = 0)",
                    "CHECK(SOURCE_END > SOURCE_START)",
                    "CHECK(STATE_CHANGE IN",
                ),
                "graph_entity_declarations": (
                    "CHECK( LENGTH(ANCHOR_ENTITY_ID) = 64",
                    "CHECK(DECLARATION_SCHEMA_VERSION = 1)",
                    "CHECK(NORMALIZATION_ID = 'NFKC-CASEFOLD-WS-V1')",
                    "CHECK( LENGTH(TRIM(ENTITY_LABEL)) > 0",
                ),
                "graph_entity_mentions": (
                    "CHECK(MENTION_SCHEMA_VERSION = 1)",
                    "CHECK(NORMALIZATION_ID = 'NFKC-CASEFOLD-WS-V1')",
                    "CHECK(SOURCE_START = 0)",
                    "CHECK(SOURCE_END > SOURCE_START)",
                    "CHECK(RAW_SOURCE_START = 0)",
                    "CHECK(RAW_SOURCE_END > RAW_SOURCE_START)",
                    "CHECK( MENTION_START >= SOURCE_START",
                    "CHECK( RAW_MENTION_START >= RAW_SOURCE_START",
                ),
            }
            for table_name, fragments in required_check_fragments.items():
                row = connection.execute(
                    "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = ?",
                    (table_name,),
                ).fetchone()
                schema_sql = re.sub(r"\s+", " ", str(row["sql"])).upper()
                if any(fragment not in schema_sql for fragment in fragments):
                    raise RuntimeError(
                        "incompatible {} schema; missing controlled checks".format(
                            table_name
                        )
                    )
            violations = connection.execute("PRAGMA foreign_key_check").fetchall()
            if violations:
                raise RuntimeError("SQLite foreign key check failed")

    def _resolve_graph_seed_ids(
        self,
        connection: sqlite3.Connection,
        *,
        user_id: str,
        seed_names: Sequence[str],
    ) -> Tuple[List[Tuple[int, str]], List[Dict[str, Any]]]:
        """Resolve only unique, exact, user-scoped graph seeds."""
        from .evidence_graph import normalize_entity_name

        resolved: List[Tuple[int, str]] = []
        unresolved: List[Dict[str, Any]] = []
        seen_ids = set()
        requested = list(seed_names)
        for seed_name in requested[:self.GRAPH_SEED_LIMIT]:
            normalized = normalize_entity_name(seed_name)
            if not normalized:
                unresolved.append({
                    "seed": str(seed_name), "normalized": None,
                    "reason": "invalid",
                })
                continue
            rows = connection.execute(
                """SELECT DISTINCT entity_id FROM (
                       SELECT id AS entity_id
                       FROM graph_entities
                       WHERE user_id = ? AND canonical_name = ?
                       UNION ALL
                       SELECT entity_id
                       FROM graph_aliases
                       WHERE user_id = ? AND normalized_alias = ?
                   )
                   ORDER BY entity_id""",
                (user_id, normalized, user_id, normalized),
            ).fetchall()
            candidate_ids = sorted({int(row["entity_id"]) for row in rows})
            # Same-name ambiguity is deliberately fail-closed.
            if len(candidate_ids) != 1:
                unresolved.append({
                    "seed": str(seed_name),
                    "normalized": normalized,
                    "reason": "not_found" if not candidate_ids else "ambiguous",
                    "candidate_count": len(candidate_ids),
                })
                continue
            entity_id = candidate_ids[0]
            if entity_id in seen_ids:
                unresolved.append({
                    "seed": str(seed_name),
                    "normalized": normalized,
                    "reason": "duplicate",
                    "candidate_count": 1,
                })
                continue
            seen_ids.add(entity_id)
            resolved.append((entity_id, normalized))
        for seed_name in requested[self.GRAPH_SEED_LIMIT:]:
            unresolved.append({
                "seed": str(seed_name),
                "normalized": normalize_entity_name(seed_name),
                "reason": "seed_limit",
            })
        return resolved, unresolved

    def _one_hop_graph_candidates(
        self,
        connection: sqlite3.Connection,
        *,
        user_id: str,
        seed_names: Sequence[str],
        preferred_predicates: Sequence[str] = (),
    ) -> Tuple[List[sqlite3.Row], List[Dict[str, Any]], Dict[str, Any]]:
        from .evidence_graph import PREDICATES

        requested_seeds = [str(item) for item in seed_names]
        predicate_preferences = tuple(dict.fromkeys(
            str(item) for item in preferred_predicates
            if str(item) in PREDICATES
        ))[:10]
        diagnostics: Dict[str, Any] = {
            "requested_seeds": requested_seeds,
            "preferred_predicates": list(predicate_preferences),
            "resolved_seeds": [],
            "unresolved_seeds": [],
            "seed_limit": self.GRAPH_SEED_LIMIT,
            "edge_limit_per_seed": self.GRAPH_EDGE_LIMIT_PER_SEED,
            "candidate_limit": self.graph_max_candidates,
            "edges_fetched_by_seed": [],
            "edges_considered": 0,
            "duplicate_edges_skipped": 0,
            "candidate_cap_reached": False,
            "candidate_count": 0,
            "path_count": 0,
            # This is an audit-only record of every consumed edge.  Together
            # with ``edges_fetched_by_seed`` it lets the evaluator verify the
            # documented round-robin traversal order even when duplicate
            # edges do not yield a graph path.
            "edge_visit_seed_ids": [],
        }
        if not self.evidence_graph:
            return [], [], diagnostics
        seeds, unresolved = self._resolve_graph_seed_ids(
            connection, user_id=user_id, seed_names=seed_names
        )
        diagnostics["resolved_seeds"] = [
            {"entity_id": str(entity_id), "normalized": normalized}
            for entity_id, normalized in seeds
        ]
        diagnostics["unresolved_seeds"] = unresolved
        if not seeds or self.graph_max_candidates <= 0:
            diagnostics["candidate_cap_reached"] = bool(
                seeds and self.graph_max_candidates <= 0
            )
            return [], [], diagnostics
        rows_by_message: Dict[int, sqlite3.Row] = {}
        paths: List[Dict[str, Any]] = []
        seen_edge_ids = set()
        rows_per_seed: List[Tuple[int, str, List[sqlite3.Row]]] = []
        for seed_entity_id, normalized_seed in seeds:
            predicate_order = "edge.id, raw.id"
            predicate_parameters: Tuple[str, ...] = ()
            if predicate_preferences:
                predicate_cases = " ".join(
                    "WHEN ? THEN {}".format(index)
                    for index in range(len(predicate_preferences))
                )
                predicate_order = (
                    "CASE edge.predicate {} ELSE {} END, edge.id, raw.id".format(
                        predicate_cases, len(predicate_preferences)
                    )
                )
                predicate_parameters = predicate_preferences
            edge_rows = connection.execute(
                """SELECT raw.id, raw.content, raw.created_at, raw.event_ts,
                           edge.id AS edge_id, edge.predicate,
                           edge.subject_entity_id, edge.object_entity_id,
                           subject.display_name AS subject_name,
                           object.display_name AS object_name,
                           support.id AS support_id,
                           support.support_schema_version,
                           support.normalization_id AS support_normalization_id,
                           support.spec_id AS support_spec_id,
                           support.binding AS support_binding,
                           support.source_start AS support_source_start,
                           support.source_end AS support_source_end,
                           support.clause_start AS support_clause_start,
                           support.clause_end AS support_clause_end,
                           support.subject_start AS support_subject_start,
                           support.subject_end AS support_subject_end,
                           support.predicate_start AS support_predicate_start,
                           support.predicate_end AS support_predicate_end,
                           support.object_start AS support_object_start,
                           support.object_end AS support_object_end,
                           support.state_change AS support_state_change,
                           support.temporal_status AS support_temporal_status,
                           support.source_span_sha256 AS support_source_span_sha256
                   FROM graph_edges AS edge
                   JOIN graph_edge_support AS support
                     ON support.edge_id = edge.id
                    AND support.user_id = edge.user_id
                    AND support.source_message_id = edge.source_message_id
                   JOIN graph_entities AS subject
                     ON subject.id = edge.subject_entity_id
                    AND subject.user_id = edge.user_id
                   LEFT JOIN graph_entities AS object
                     ON object.id = edge.object_entity_id
                    AND object.user_id = edge.user_id
                   JOIN raw_messages AS raw
                     ON raw.id = edge.source_message_id
                    AND raw.user_id = edge.user_id
                   WHERE edge.user_id = ?
                     AND raw.user_id = ?
                     AND (edge.subject_entity_id = ? OR edge.object_entity_id = ?)
                   ORDER BY {}
                   LIMIT ?""".format(predicate_order),
                (
                    user_id, user_id, seed_entity_id, seed_entity_id,
                    *predicate_parameters,
                    self.GRAPH_EDGE_LIMIT_PER_SEED,
                ),
            ).fetchall()
            rows_per_seed.append((seed_entity_id, normalized_seed, list(edge_rows)))
            diagnostics["edges_fetched_by_seed"].append({
                "entity_id": str(seed_entity_id),
                "normalized": normalized_seed,
                "count": len(edge_rows),
                "limit_reached": len(edge_rows) >= self.GRAPH_EDGE_LIMIT_PER_SEED,
            })

        positions = [0 for _ in rows_per_seed]
        while len(rows_by_message) < self.graph_max_candidates:
            made_progress = False
            for index, (seed_entity_id, _, edge_rows) in enumerate(rows_per_seed):
                if positions[index] >= len(edge_rows):
                    continue
                row = edge_rows[positions[index]]
                positions[index] += 1
                made_progress = True
                diagnostics["edges_considered"] += 1
                diagnostics["edge_visit_seed_ids"].append(str(seed_entity_id))
                edge_id = int(row["edge_id"])
                if edge_id in seen_edge_ids:
                    diagnostics["duplicate_edges_skipped"] += 1
                    continue
                seen_edge_ids.add(edge_id)
                message_id = int(row["id"])
                if message_id not in rows_by_message:
                    rows_by_message[message_id] = row
                paths.append({
                    "path_id": "edge_{}".format(edge_id),
                    "hop_count": 1,
                    "seed_entity_id": str(seed_entity_id),
                    "bridge_entity_id": None,
                    "relations": [str(row["predicate"])],
                    "source_message_ids": ["mem_{}".format(message_id)],
                    "subject": str(row["subject_name"]),
                    "object": str(row["object_name"] or ""),
                    # These values are selected from the support sidecar above,
                    # never reconstructed from an in-memory parser payload.
                    "support_id": int(row["support_id"]),
                    "support_schema_version": int(
                        row["support_schema_version"]
                    ),
                    "support_normalization_id": str(
                        row["support_normalization_id"]
                    ),
                    "support_spec_id": str(row["support_spec_id"]),
                    "support_binding": str(row["support_binding"]),
                    "support_source_span": [
                        int(row["support_source_start"]),
                        int(row["support_source_end"]),
                    ],
                    "support_clause_span": [
                        int(row["support_clause_start"]),
                        int(row["support_clause_end"]),
                    ],
                    "support_subject_span": [
                        int(row["support_subject_start"]),
                        int(row["support_subject_end"]),
                    ],
                    "support_predicate_span": [
                        int(row["support_predicate_start"]),
                        int(row["support_predicate_end"]),
                    ],
                    "support_object_span": [
                        int(row["support_object_start"]),
                        int(row["support_object_end"]),
                    ],
                    "support_state_change": str(row["support_state_change"]),
                    "support_temporal_status": row["support_temporal_status"],
                    "support_source_span_sha256": str(
                        row["support_source_span_sha256"]
                    ),
                })
                if len(rows_by_message) >= self.graph_max_candidates:
                    break
            if not made_progress:
                break
        diagnostics["candidate_cap_reached"] = bool(
            len(rows_by_message) >= self.graph_max_candidates
            and any(
                positions[index] < len(edge_rows)
                for index, (_, _, edge_rows) in enumerate(rows_per_seed)
            )
        )
        diagnostics["candidate_count"] = len(rows_by_message)
        diagnostics["path_count"] = len(paths)
        return list(rows_by_message.values()), paths, diagnostics

    @staticmethod
    def _stored_anchor_row_is_witnessed(
        row: sqlite3.Row,
        *,
        user_id: str,
        entity_label: str,
    ) -> bool:
        """Recheck a selected P3-B1 sidecar row against its raw source.

        Formal paired evaluation has a separate evaluator-owned audit.  This
        local read check is deliberately redundant: application retrieval will
        not turn a malformed/legacy sidecar row into a candidate merely because
        its SQLite shape happens to be valid.
        """

        try:
            source_message_id = int(row["id"])
            if (
                str(row["mention_entity_label"]) != entity_label
                or str(row["declaration_entity_label"]) != entity_label
                or int(row["mention_schema_version"])
                != MENTION_SCHEMA_VERSION
                or int(row["declaration_schema_version"])
                != MENTION_DECLARATION_SCHEMA_VERSION
                or str(row["mention_normalization_id"])
                != MENTION_NORMALIZATION_ID
                or str(row["declaration_normalization_id"])
                != MENTION_NORMALIZATION_ID
                or str(row["anchor_entity_id"])
                != source_local_anchor_entity_id(
                    user_id=user_id,
                    source_message_id=source_message_id,
                    entity_label=entity_label,
                )
            ):
                return False
            expected_rows = exact_source_entity_mentions(
                str(row["content"]), entity_label
            )
            return any(
                all(
                    witness[field] == row["mention_{}".format(field)]
                    for field in (
                        "source_start", "source_end", "mention_start",
                        "mention_end", "raw_source_start", "raw_source_end",
                        "raw_mention_start", "raw_mention_end",
                        "source_span_sha256",
                    )
                )
                for witness in expected_rows
            )
        except (KeyError, TypeError, ValueError):
            return False

    def _provenance_anchor_candidates(
        self,
        connection: sqlite3.Connection,
        *,
        user_id: str,
        seed_names: Sequence[str],
    ) -> Tuple[List[sqlite3.Row], Dict[str, Any]]:
        """Resolve source-local label anchors into bounded raw candidates.

        This is not a graph traversal: it follows only
        ``label -> exact occurrence -> raw message`` and does not inspect P3-A
        identities, aliases, relations, paths or relation-support witnesses.
        """

        requested_seeds = [str(item) for item in seed_names]
        diagnostics: Dict[str, Any] = {
            "enabled": self.evidence_anchors,
            "requested_seeds": requested_seeds,
            "resolved_seeds": [],
            "unresolved_seeds": [],
            "seed_limit": self.ANCHOR_SEED_LIMIT,
            "candidate_limit_per_seed": self.ANCHOR_CANDIDATES_PER_SEED,
            "candidate_limit": self.anchor_max_candidates,
            "candidates_fetched_by_seed": [],
            "candidates_considered": 0,
            "duplicate_candidates_skipped": 0,
            "invalid_witness_candidates_skipped": 0,
            "candidate_cap_reached": False,
            "candidate_count": 0,
            "candidate_visit_seed_labels": [],
        }
        if not self.evidence_anchors:
            return [], diagnostics

        rows_per_seed: List[Tuple[str, List[sqlite3.Row]]] = []
        seen_labels = set()
        for raw_seed in requested_seeds[:self.ANCHOR_SEED_LIMIT]:
            normalized_label = normalize_evidence_mention_source(raw_seed)
            if (
                not normalized_label
                or len(normalized_label) > MENTION_ENTITY_LABEL_MAX_CHARS
            ):
                diagnostics["unresolved_seeds"].append({
                    "seed": raw_seed,
                    "normalized": normalized_label,
                    "reason": "invalid",
                })
                continue
            if normalized_label in seen_labels:
                diagnostics["unresolved_seeds"].append({
                    "seed": raw_seed,
                    "normalized": normalized_label,
                    "reason": "duplicate",
                })
                continue
            seen_labels.add(normalized_label)
            candidate_rows = connection.execute(
                """SELECT * FROM (
                       SELECT raw.id, raw.content, raw.created_at, raw.event_ts,
                              declaration.anchor_entity_id,
                              declaration.declaration_schema_version,
                              declaration.normalization_id
                                  AS declaration_normalization_id,
                              declaration.entity_label
                                  AS declaration_entity_label,
                              mention.id AS mention_id,
                              mention.mention_schema_version,
                              mention.normalization_id
                                  AS mention_normalization_id,
                              mention.entity_label AS mention_entity_label,
                              mention.source_start AS mention_source_start,
                              mention.source_end AS mention_source_end,
                              mention.mention_start AS mention_mention_start,
                              mention.mention_end AS mention_mention_end,
                              mention.raw_source_start
                                  AS mention_raw_source_start,
                              mention.raw_source_end AS mention_raw_source_end,
                              mention.raw_mention_start
                                  AS mention_raw_mention_start,
                              mention.raw_mention_end
                                  AS mention_raw_mention_end,
                              mention.source_span_sha256
                                  AS mention_source_span_sha256,
                              ROW_NUMBER() OVER (
                                  PARTITION BY raw.id
                                  ORDER BY mention.mention_start, mention.id
                              ) AS source_rank
                       FROM graph_entity_mentions AS mention
                       JOIN graph_entity_declarations AS declaration
                         ON declaration.id = mention.declaration_id
                        AND declaration.user_id = mention.user_id
                        AND declaration.source_message_id = mention.source_message_id
                       JOIN raw_messages AS raw
                         ON raw.id = mention.source_message_id
                        AND raw.user_id = mention.user_id
                       WHERE mention.user_id = ?
                         AND declaration.user_id = ?
                         AND raw.user_id = ?
                         AND mention.entity_label = ?
                         AND declaration.entity_label = mention.entity_label
                   ) AS source_local_anchors
                   WHERE source_rank = 1
                   ORDER BY mention_mention_start, id, mention_id
                   LIMIT ?""",
                (
                    user_id,
                    user_id,
                    user_id,
                    normalized_label,
                    self.ANCHOR_CANDIDATES_PER_SEED,
                ),
            ).fetchall()
            valid_rows = []
            for row in candidate_rows:
                if self._stored_anchor_row_is_witnessed(
                    row, user_id=user_id, entity_label=normalized_label
                ):
                    valid_rows.append(row)
                else:
                    diagnostics["invalid_witness_candidates_skipped"] += 1
            diagnostics["candidates_fetched_by_seed"].append({
                "entity_label": normalized_label,
                "count": len(valid_rows),
                "unwitnessed_count": len(candidate_rows) - len(valid_rows),
                "limit_reached": len(candidate_rows)
                >= self.ANCHOR_CANDIDATES_PER_SEED,
            })
            if not valid_rows:
                diagnostics["unresolved_seeds"].append({
                    "seed": raw_seed,
                    "normalized": normalized_label,
                    "reason": "not_found",
                })
                continue
            diagnostics["resolved_seeds"].append({
                "entity_label": normalized_label,
                "candidate_count": len(valid_rows),
            })
            rows_per_seed.append((normalized_label, valid_rows))
        for raw_seed in requested_seeds[self.ANCHOR_SEED_LIMIT:]:
            diagnostics["unresolved_seeds"].append({
                "seed": raw_seed,
                "normalized": normalize_evidence_mention_source(raw_seed),
                "reason": "seed_limit",
            })
        if not rows_per_seed or self.anchor_max_candidates <= 0:
            diagnostics["candidate_cap_reached"] = bool(
                rows_per_seed and self.anchor_max_candidates <= 0
            )
            return [], diagnostics

        rows_by_message: Dict[int, sqlite3.Row] = {}
        positions = [0 for _ in rows_per_seed]
        while len(rows_by_message) < self.anchor_max_candidates:
            made_progress = False
            for index, (entity_label, rows) in enumerate(rows_per_seed):
                if positions[index] >= len(rows):
                    continue
                row = rows[positions[index]]
                positions[index] += 1
                made_progress = True
                diagnostics["candidates_considered"] += 1
                diagnostics["candidate_visit_seed_labels"].append(entity_label)
                message_id = int(row["id"])
                if message_id in rows_by_message:
                    diagnostics["duplicate_candidates_skipped"] += 1
                    continue
                rows_by_message[message_id] = row
                if len(rows_by_message) >= self.anchor_max_candidates:
                    break
            if not made_progress:
                break
        diagnostics["candidate_cap_reached"] = bool(
            len(rows_by_message) >= self.anchor_max_candidates
            and any(
                positions[index] < len(rows)
                for index, (_, rows) in enumerate(rows_per_seed)
            )
        )
        diagnostics["candidate_count"] = len(rows_by_message)
        return list(rows_by_message.values()), diagnostics

    def _adjacent_turn_candidates(
        self,
        connection: sqlite3.Connection,
        *,
        user_id: str,
        seed_ids: Sequence[str],
        excluded_ids: Sequence[str],
        excluded_content_keys: Sequence[str],
    ) -> Tuple[List[sqlite3.Row], Dict[str, Any]]:
        """Return bounded immediate same-session raw-message neighbors.

        This is deliberately a P1 evidence-completeness channel rather than a
        graph traversal.  A neighbor has no synthetic payload or inferred
        relation: the existing raw row is the only candidate that reaches the
        ranker.
        """

        diagnostics: Dict[str, Any] = {
            "enabled": self.adjacent_turn_expansion,
            "seed_limit": self.adjacent_seed_limit,
            "candidate_limit": self.adjacent_candidate_limit,
            "seed_ids": [],
            "candidate_ids": [],
            "deduped_ids": [],
            "considered_neighbor_ids": [],
            "candidate_limit_reached": False,
        }
        if not self.adjacent_turn_expansion or self.adjacent_candidate_limit == 0:
            return [], diagnostics

        excluded_id_set = {str(item) for item in excluded_ids}
        seen_ids = set(excluded_id_set)
        seen_content_keys = {str(item) for item in excluded_content_keys}
        deduped_ids: set[str] = set()
        rows: List[sqlite3.Row] = []
        for raw_seed_id in seed_ids[:self.adjacent_seed_limit]:
            seed_id = str(raw_seed_id)
            if not seed_id.startswith("mem_") or not seed_id[4:].isdigit():
                continue
            seed_row = connection.execute(
                """SELECT id, session_id FROM raw_messages
                   WHERE id = ? AND user_id = ?""",
                (int(seed_id[4:]), user_id),
            ).fetchone()
            if seed_row is None:
                continue
            diagnostics["seed_ids"].append(seed_id)
            for operator, ordering in (("<", "DESC"), (">", "ASC")):
                neighbor = connection.execute(
                    """SELECT id, content, created_at, event_ts
                       FROM raw_messages
                       WHERE user_id = ? AND session_id = ? AND id {} ?
                       ORDER BY id {} LIMIT 1""".format(operator, ordering),
                    (user_id, str(seed_row["session_id"]), int(seed_row["id"])),
                ).fetchone()
                if neighbor is None:
                    continue
                candidate_id = "mem_{}".format(int(neighbor["id"]))
                diagnostics["considered_neighbor_ids"].append(candidate_id)
                content_key = str(neighbor["content"]).casefold()
                if candidate_id in seen_ids or content_key in seen_content_keys:
                    if candidate_id not in deduped_ids:
                        diagnostics["deduped_ids"].append(candidate_id)
                        deduped_ids.add(candidate_id)
                    continue
                rows.append(neighbor)
                diagnostics["candidate_ids"].append(candidate_id)
                seen_ids.add(candidate_id)
                seen_content_keys.add(content_key)
                if len(rows) >= self.adjacent_candidate_limit:
                    diagnostics["candidate_limit_reached"] = True
                    return rows, diagnostics
        return rows, diagnostics

    @staticmethod
    def _normalize_evidence_support_source(value: str) -> str:
        """Normalize an evidence source using the persisted v1 contract.

        This intentionally differs from ``evidence_graph._normalized_source``:
        all Unicode whitespace is collapsed, and every returned offset is a
        Python Unicode-codepoint offset into this exact value.
        """
        return " ".join(unicodedata.normalize("NFKC", value).casefold().split())

    @staticmethod
    def _legacy_parser_support_source(value: str) -> str:
        """Reproduce only the legacy parser offset domain for remapping.

        This is not a parser invocation and is deliberately kept local so the
        sidecar can convert a carried ``SupportWitness`` without treating its
        old offsets as canonical offsets.
        """
        normalized = unicodedata.normalize("NFKC", value).casefold()
        normalized = re.sub(r"[\t\r\f\v ]+", " ", normalized)
        normalized = re.sub(r"\s*\n\s*", "\n", normalized)
        return normalized.strip()

    @staticmethod
    def _legacy_support_boundary_map(
        legacy_source: str,
        canonical_source: str,
    ) -> Optional[List[Optional[int]]]:
        """Map legacy parser boundaries to canonical v1 codepoint boundaries.

        Whitespace runs in the old source may be a newline or multiple Unicode
        whitespace codepoints, while v1 represents the same run as one space.
        A boundary inside such a collapsed run has no unique canonical offset,
        so it is represented by ``None`` and callers must reject that witness.
        """
        boundaries: List[Optional[int]] = [None] * (len(legacy_source) + 1)
        legacy_index = 0
        canonical_index = 0
        boundaries[0] = 0
        while legacy_index < len(legacy_source):
            if canonical_index >= len(canonical_source):
                return None
            legacy_character = legacy_source[legacy_index]
            canonical_character = canonical_source[canonical_index]
            if legacy_character.isspace():
                if not canonical_character.isspace():
                    return None
                legacy_start = legacy_index
                canonical_start = canonical_index
                while (
                    legacy_index < len(legacy_source)
                    and legacy_source[legacy_index].isspace()
                ):
                    legacy_index += 1
                while (
                    canonical_index < len(canonical_source)
                    and canonical_source[canonical_index].isspace()
                ):
                    canonical_index += 1
                boundaries[legacy_start] = canonical_start
                # Interior boundaries are intentionally left unmapped: they
                # cannot be reconstructed after a many-to-one collapse.
                boundaries[legacy_index] = canonical_index
                continue
            if canonical_character.isspace() or legacy_character != canonical_character:
                return None
            boundaries[legacy_index] = canonical_index
            legacy_index += 1
            canonical_index += 1
            boundaries[legacy_index] = canonical_index
        if canonical_index != len(canonical_source):
            return None
        return boundaries

    @staticmethod
    def _coerce_support_span(
        value: object,
        *,
        limit: int,
    ) -> Optional[Tuple[int, int]]:
        if not isinstance(value, (tuple, list)) or len(value) != 2:
            return None
        start, end = value
        if (
            isinstance(start, bool)
            or isinstance(end, bool)
            or not isinstance(start, int)
            or not isinstance(end, int)
            or start < 0
            or start >= end
            or end > limit
        ):
            return None
        return start, end

    @staticmethod
    def _support_surface_matches_entity(
        surface: str,
        canonical_name: str,
        *,
        predicate: str,
        is_subject: bool,
    ) -> bool:
        """Verify an endpoint span without re-running parser extraction."""
        from .evidence_graph import normalize_entity_name

        expected = normalize_entity_name(canonical_name)
        actual = normalize_entity_name(surface)
        if not expected or not actual:
            return False
        allowed = {expected}
        expected_words = expected.split()
        if expected_words and expected_words[0] not in _SUPPORT_DETERMINERS:
            if is_subject:
                allowed.update({
                    "a different {}".format(expected),
                    "an different {}".format(expected),
                })
                if predicate in _SUPPORT_RULE_PREDICATES:
                    allowed.add("the {}".format(expected))
            else:
                allowed.update(
                    "{} {}".format(determiner, expected)
                    for determiner in _SUPPORT_DETERMINERS
                )
        return actual in allowed

    def _prepare_graph_edge_support(
        self,
        *,
        relation: object,
        support_witness: object,
        subject_entity: object,
        object_entity: object,
        user_id: str,
        source_message_id: int,
        source_content: str,
        source_role: str,
    ) -> Optional[Dict[str, Any]]:
        """Validate and canonically remap one carried parser witness.

        The parser remains the authority that found the witness.  Storage only
        persists a relation when it can independently tie every carried field
        back to the immutable raw message and the sanitized relation payload.
        """
        if support_witness is None or not isinstance(source_content, str):
            return None
        expected_source_id = "mem_{}".format(source_message_id)
        if (
            str(getattr(relation, "user_id", "")) != user_id
            or str(getattr(relation, "source_message_id", "")) != expected_source_id
        ):
            return None
        predicate = getattr(relation, "predicate", None)
        if not isinstance(predicate, str) or predicate not in _SUPPORT_PREDICATE_MARKERS:
            return None
        spec_id = getattr(support_witness, "spec_id", None)
        binding = getattr(support_witness, "binding", None)
        witness_state_change = getattr(support_witness, "state_change", None)
        witness_temporal_status = getattr(support_witness, "temporal_status", None)
        relation_state_change = getattr(relation, "state_change", None)
        relation_temporal_status = getattr(relation, "temporal_status", None)
        if (
            not isinstance(spec_id, str)
            or spec_id not in SUPPORT_SPEC_IDS
            or spec_id.partition(":")[0] != predicate
            or binding not in SUPPORT_BINDINGS
            or witness_state_change not in SUPPORT_STATE_CHANGES
            or relation_state_change not in SUPPORT_STATE_CHANGES
            or witness_state_change != relation_state_change
            or witness_temporal_status != relation_temporal_status
            or (
                witness_temporal_status is not None
                and witness_temporal_status not in SUPPORT_TEMPORAL_STATUSES
            )
            or (
                relation_temporal_status is not None
                and relation_temporal_status not in SUPPORT_TEMPORAL_STATUSES
            )
            or (relation_state_change == "assert") != (relation_temporal_status is None)
        ):
            return None

        clause = getattr(support_witness, "clause", None)
        if not isinstance(clause, str) or not clause:
            return None
        legacy_source = self._legacy_parser_support_source(source_content)
        canonical_source = self._normalize_evidence_support_source(source_content)
        if not legacy_source or not canonical_source:
            return None
        boundaries = self._legacy_support_boundary_map(
            legacy_source,
            canonical_source,
        )
        if boundaries is None:
            return None
        source_span = self._coerce_support_span(
            getattr(support_witness, "source_span", None),
            limit=len(legacy_source),
        )
        # ``SupportWitness.source_span`` starts in the parser's legacy source
        # domain, but its end is calculated from the already collapsed clause
        # length.  Therefore its numeric end is not safe to map after a run of
        # U+2028/CRLF/etc.  The start is an explicit source boundary; map only
        # that boundary, require the constructor's length invariant, then
        # prove the full canonical clause at the mapped position below.
        if (
            source_span is None
            or source_span[1] != source_span[0] + len(clause)
            or source_span[0] >= len(boundaries)
        ):
            return None
        clause_start = boundaries[source_span[0]]
        canonical_clause = self._normalize_evidence_support_source(clause)
        if (
            clause_start is None
            or canonical_clause != clause
            or clause_start + len(canonical_clause) > len(canonical_source)
            or canonical_source[
                clause_start:clause_start + len(canonical_clause)
            ] != canonical_clause
        ):
            return None
        clause_span = (clause_start, clause_start + len(canonical_clause))
        local_spans = {
            "subject": self._coerce_support_span(
                getattr(support_witness, "subject_span", None),
                limit=len(clause),
            ),
            "predicate": self._coerce_support_span(
                getattr(support_witness, "predicate_span", None),
                limit=len(clause),
            ),
            "object": self._coerce_support_span(
                getattr(support_witness, "object_span", None),
                limit=len(clause),
            ),
        }
        if any(span is None for span in local_spans.values()):
            return None
        absolute_spans: Dict[str, Tuple[int, int]] = {}
        for name, local_span in local_spans.items():
            assert local_span is not None
            absolute_spans[name] = (
                clause_span[0] + local_span[0],
                clause_span[0] + local_span[1],
            )

        clause_start, clause_end = clause_span
        subject_start, subject_end = absolute_spans["subject"]
        predicate_start, predicate_end = absolute_spans["predicate"]
        object_start, object_end = absolute_spans["object"]
        if not (
            clause_start <= subject_start < subject_end <= predicate_start
            < predicate_end <= object_start < object_end <= clause_end
        ):
            return None
        if canonical_source[clause_start:clause_end] != canonical_clause:
            return None
        for name, local_span in local_spans.items():
            absolute_span = absolute_spans[name]
            if (
                canonical_source[absolute_span[0]:absolute_span[1]]
                != clause[local_span[0]:local_span[1]]
            ):
                return None

        subject_name = getattr(subject_entity, "canonical_name", None)
        object_name = getattr(object_entity, "canonical_name", None)
        if not isinstance(subject_name, str) or not isinstance(object_name, str):
            return None
        subject_surface = canonical_source[subject_start:subject_end]
        object_surface = canonical_source[object_start:object_end]
        predicate_surface = canonical_source[predicate_start:predicate_end]
        predicate_tokens = frozenset(
            re.findall(r"[\w]+", predicate_surface, flags=re.UNICODE)
        )
        if not predicate_tokens.intersection(_SUPPORT_PREDICATE_MARKERS[predicate]):
            return None
        if binding == "named":
            if not self._support_surface_matches_entity(
                subject_surface,
                subject_name,
                predicate=predicate,
                is_subject=True,
            ):
                return None
        else:
            from .evidence_graph import normalize_entity_name

            if (
                subject_surface not in _SUPPORT_FIRST_PERSON_SUBJECTS
                or normalize_entity_name(source_role) != normalize_entity_name(subject_name)
            ):
                return None
        if not self._support_surface_matches_entity(
            object_surface,
            object_name,
            predicate=predicate,
            is_subject=False,
        ):
            return None

        source_start = 0
        source_end = len(canonical_source)
        source_hash = hashlib.sha256(
            canonical_source[source_start:source_end].encode("utf-8")
        ).hexdigest()
        return {
            "support_schema_version": SUPPORT_SCHEMA_VERSION,
            "normalization_id": SUPPORT_NORMALIZATION_ID,
            "spec_id": spec_id,
            "binding": binding,
            "source_start": source_start,
            "source_end": source_end,
            "clause_start": clause_start,
            "clause_end": clause_end,
            "subject_start": subject_start,
            "subject_end": subject_end,
            "predicate_start": predicate_start,
            "predicate_end": predicate_end,
            "object_start": object_start,
            "object_end": object_end,
            "state_change": witness_state_change,
            "temporal_status": witness_temporal_status,
            "source_span_sha256": source_hash,
        }

    def _store_source_local_entity_mentions(
        self,
        connection: sqlite3.Connection,
        *,
        user_id: str,
        source_message_id: int,
        created_at: str,
        source_content: str,
        payload_entities: Sequence[object],
    ) -> None:
        """Persist P3-B1 label witnesses before P3-A identity linking.

        The parser supplies candidate labels only.  A label becomes an anchor
        only when this storage-owned, exact matcher finds one or more safe raw
        occurrences in the same message.  Per-source de-duplication keeps
        parser duplicates/type disagreements from consuming retrieval slots.
        """

        labels: List[str] = []
        seen_labels = set()
        for entity in payload_entities:
            normalized_label = normalize_evidence_mention_source(
                getattr(entity, "canonical_name", None)
            )
            if (
                not normalized_label
                or len(normalized_label) > MENTION_ENTITY_LABEL_MAX_CHARS
                or normalized_label in seen_labels
            ):
                continue
            seen_labels.add(normalized_label)
            labels.append(normalized_label)
        for entity_label in labels:
            witnesses = exact_source_entity_mentions(source_content, entity_label)
            if not witnesses:
                continue
            anchor_entity_id = source_local_anchor_entity_id(
                user_id=user_id,
                source_message_id=source_message_id,
                entity_label=entity_label,
            )
            connection.execute(
                """INSERT OR IGNORE INTO graph_entity_declarations(
                       user_id, source_message_id, anchor_entity_id,
                       declaration_schema_version, normalization_id,
                       entity_label, created_at
                   ) VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    user_id,
                    source_message_id,
                    anchor_entity_id,
                    MENTION_DECLARATION_SCHEMA_VERSION,
                    MENTION_NORMALIZATION_ID,
                    entity_label,
                    created_at,
                ),
            )
            declaration_row = connection.execute(
                """SELECT id, entity_label, declaration_schema_version,
                          normalization_id
                   FROM graph_entity_declarations
                   WHERE user_id = ? AND source_message_id = ?
                     AND anchor_entity_id = ?""",
                (user_id, source_message_id, anchor_entity_id),
            ).fetchone()
            if (
                declaration_row is None
                or str(declaration_row["entity_label"]) != entity_label
                or int(declaration_row["declaration_schema_version"])
                != MENTION_DECLARATION_SCHEMA_VERSION
                or str(declaration_row["normalization_id"])
                != MENTION_NORMALIZATION_ID
            ):
                # A colliding/legacy declaration can never safely receive a
                # child mention.  The unique constraints make this fail closed.
                continue
            declaration_id = int(declaration_row["id"])
            for witness in witnesses:
                connection.execute(
                    """INSERT OR IGNORE INTO graph_entity_mentions(
                           declaration_id, user_id, source_message_id,
                           mention_schema_version, normalization_id,
                           entity_label, source_start, source_end,
                           mention_start, mention_end, raw_source_start,
                           raw_source_end, raw_mention_start,
                           raw_mention_end, source_span_sha256
                       ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        declaration_id,
                        user_id,
                        source_message_id,
                        witness["mention_schema_version"],
                        witness["normalization_id"],
                        entity_label,
                        witness["source_start"],
                        witness["source_end"],
                        witness["mention_start"],
                        witness["mention_end"],
                        witness["raw_source_start"],
                        witness["raw_source_end"],
                        witness["raw_mention_start"],
                        witness["raw_mention_end"],
                        witness["source_span_sha256"],
                    ),
                )

    def _store_graph_payload(
        self,
        connection: sqlite3.Connection,
        *,
        user_id: str,
        session_id: str,
        source_message_id: int,
        event_ts: Optional[int],
        created_at: str,
        payload: object,
    ) -> None:
        """Persist a sanitized extraction while preserving source provenance."""
        source_row = connection.execute(
            """SELECT content, role FROM raw_messages
               WHERE id = ? AND user_id = ?""",
            (source_message_id, user_id),
        ).fetchone()
        if source_row is None:
            # The raw row is inserted in this transaction before graph
            # materialization.  If provenance cannot be re-read, fail closed
            # rather than writing an unattested edge.
            return
        payload_entities = list(getattr(payload, "entities", []))
        self._store_source_local_entity_mentions(
            connection,
            user_id=user_id,
            source_message_id=source_message_id,
            created_at=created_at,
            source_content=str(source_row["content"]),
            payload_entities=payload_entities,
        )
        if not self.evidence_graph:
            # P3-B1 materializes only its source-exact label sidecar.  It
            # deliberately does not write an identity, semantic edge or P3-A
            # relation witness while graph traversal is disabled.
            return
        entity_id_map: Dict[str, int] = {}
        payload_entities_by_id = {
            str(entity.entity_id): entity for entity in payload_entities
        }
        payload_identity_counts = Counter(
            (
                str(entity.canonical_name), str(entity.entity_type),
                getattr(entity, "identity_hint", None),
            )
            for entity in payload_entities
        )
        for entity in payload_entities:
            canonical_name = str(entity.canonical_name)
            entity_type = str(entity.entity_type)
            identity_hint = getattr(entity, "identity_hint", None)
            identity_is_ambiguous_in_payload = (
                payload_identity_counts[
                    (canonical_name, entity_type, identity_hint)
                ] > 1
            )
            if identity_is_ambiguous_in_payload:
                matches = []
            elif (
                isinstance(identity_hint, str)
                and identity_hint.startswith("api-speaker:")
            ):
                # Exact speaker metadata is the only identity key trusted
                # across sessions.  A verified occupation/appositive is useful
                # for local disambiguation, but is not globally unique.
                matches = connection.execute(
                    """SELECT id, identity_hint FROM graph_entities
                       WHERE user_id = ? AND canonical_name = ? AND entity_type = ?
                          AND identity_hint = ?
                       ORDER BY id""",
                    (user_id, canonical_name, entity_type, identity_hint),
                ).fetchall()
            elif identity_hint:
                matches = connection.execute(
                    """SELECT entity.id, entity.identity_hint
                       FROM graph_entities AS entity
                       JOIN raw_messages AS first_source
                         ON first_source.id = entity.first_source_message_id
                        AND first_source.user_id = entity.user_id
                       WHERE entity.user_id = ?
                         AND entity.canonical_name = ?
                         AND entity.entity_type = ?
                         AND entity.identity_hint = ?
                         AND first_source.session_id = ?
                       ORDER BY entity.id""",
                    (
                        user_id, canonical_name, entity_type, identity_hint,
                        session_id,
                    ),
                ).fetchall()
            else:
                hinted_match = connection.execute(
                    """SELECT 1 FROM graph_entities
                       WHERE user_id = ? AND canonical_name = ? AND entity_type = ?
                         AND identity_hint IS NOT NULL
                       LIMIT 1""",
                    (user_id, canonical_name, entity_type),
                ).fetchone()
                if hinted_match:
                    # A generic mention must not silently attach to, or create
                    # a competing identity beside, a specifically
                    # disambiguated same-name entity.
                    continue
                matches = connection.execute(
                    """SELECT entity.id, entity.identity_hint
                       FROM graph_entities AS entity
                       JOIN raw_messages AS first_source
                         ON first_source.id = entity.first_source_message_id
                        AND first_source.user_id = entity.user_id
                       WHERE entity.user_id = ?
                         AND entity.canonical_name = ?
                         AND entity.entity_type = ?
                         AND entity.identity_hint IS NULL
                         AND first_source.session_id = ?
                       ORDER BY entity.id""",
                    (user_id, canonical_name, entity_type, session_id),
                ).fetchall()
            if len(matches) > 1:
                # The identity is genuinely ambiguous; never pick one by order.
                continue
            if matches:
                database_entity_id = int(matches[0]["id"])
            else:
                cursor = connection.execute(
                    """INSERT INTO graph_entities(
                           user_id, canonical_name, display_name, entity_type,
                           identity_hint, first_source_message_id, created_at
                       ) VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (
                        user_id,
                        canonical_name,
                        str(entity.display_name),
                        entity_type,
                        identity_hint,
                        source_message_id,
                        created_at,
                    ),
                )
                database_entity_id = int(cursor.lastrowid)
            entity_id_map[str(entity.entity_id)] = database_entity_id

        for relation in getattr(payload, "relations", []):
            subject_id = entity_id_map.get(str(relation.subject_entity_id))
            object_id = entity_id_map.get(str(relation.object_entity_id))
            subject_entity = payload_entities_by_id.get(
                str(relation.subject_entity_id)
            )
            object_entity = payload_entities_by_id.get(
                str(relation.object_entity_id)
            )
            if (
                subject_id is None
                or object_id is None
                or subject_entity is None
                or object_entity is None
            ):
                continue
            # The parser must carry the exact SupportWitness that justified the
            # relation.  Do not recompute a boolean support decision here: a
            # relation without that immutable witness is intentionally absent
            # from the graph until a clean rematerialization.
            support = self._prepare_graph_edge_support(
                relation=relation,
                support_witness=getattr(relation, "support_witness", None),
                subject_entity=subject_entity,
                object_entity=object_entity,
                user_id=user_id,
                source_message_id=source_message_id,
                source_content=str(source_row["content"]),
                source_role=str(source_row["role"]),
            )
            if support is None:
                continue
            edge_cursor = connection.execute(
                """INSERT OR IGNORE INTO graph_edges(
                       user_id, subject_entity_id, predicate, object_entity_id,
                       object_value, source_message_id, event_ts, state_change,
                       temporal_status, supersedes_edge_id, created_at
                   ) VALUES (?, ?, ?, ?, NULL, ?, ?, ?, ?, NULL, ?)""",
                (
                    user_id,
                    subject_id,
                    str(relation.predicate),
                    object_id,
                    source_message_id,
                    event_ts,
                    str(relation.state_change),
                    relation.temporal_status,
                    created_at,
                ),
            )
            if edge_cursor.rowcount != 1:
                # An existing exact edge may already have its original support
                # (an idempotent retry), may be a legacy unsupported edge, or
                # may carry a different witness.  In every case never attach
                # or overwrite a witness after the edge was created.
                continue
            edge_id = int(edge_cursor.lastrowid)
            connection.execute(
                """INSERT INTO graph_edge_support(
                       edge_id, user_id, source_message_id,
                       support_schema_version, normalization_id, spec_id,
                       binding, source_start, source_end, clause_start,
                       clause_end, subject_start, subject_end, predicate_start,
                       predicate_end, object_start, object_end, state_change,
                       temporal_status, source_span_sha256
                   ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    edge_id,
                    user_id,
                    source_message_id,
                    support["support_schema_version"],
                    support["normalization_id"],
                    support["spec_id"],
                    support["binding"],
                    support["source_start"],
                    support["source_end"],
                    support["clause_start"],
                    support["clause_end"],
                    support["subject_start"],
                    support["subject_end"],
                    support["predicate_start"],
                    support["predicate_end"],
                    support["object_start"],
                    support["object_end"],
                    support["state_change"],
                    support["temporal_status"],
                    support["source_span_sha256"],
                ),
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
                    graph_payload = None
                    if (self.evidence_graph or self.evidence_anchors) and callable(
                        getattr(self.model, "extract_memory", None)
                    ):
                        extraction = self.model.extract_memory(
                            message.content,
                            speaker=message.role,
                            timestamp=message.timestamp,
                        )
                        facts = extraction.get("facts", []) if isinstance(
                            extraction, dict
                        ) else []
                        from .evidence_graph import parse_graph_payload
                        try:
                            graph_payload = parse_graph_payload(
                                extraction if isinstance(extraction, dict) else {},
                                user_id=request.user_id,
                                source_message_id="mem_{}".format(message_id),
                                event_ts=message.timestamp,
                                created_at=now,
                                source_text=message.content,
                                speaker=message.role,
                            )
                        except ValueError:
                            # A malformed provenance identifier or timestamp must
                            # drop the graph annotation, not the original memory.
                            graph_payload = None
                    else:
                        facts = self.model.extract_facts(
                            message.content,
                            speaker=message.role,
                            timestamp=message.timestamp,
                        )
                    for fact in facts:
                        if not isinstance(fact, str) or not fact.strip():
                            continue
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
                    if graph_payload is not None:
                        self._store_graph_payload(
                            connection,
                            user_id=request.user_id,
                            session_id=request.session_id,
                            source_message_id=int(message_id),
                            event_ts=message.timestamp,
                            created_at=now,
                            payload=graph_payload,
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
        speaker_conflict = False
        retrieval_trace = self._empty_retrieval_trace()
        message_ids: Dict[str, int] = {}
        ranking_metadata: Dict[Any, Dict[str, Any]] = {}
        retrieval_trace["edge_diagnostics"][
            "candidate_limit"
        ] = self.graph_max_candidates
        retrieval_trace["adjacent_diagnostics"].update({
            "enabled": self.adjacent_turn_expansion,
            "seed_limit": self.adjacent_seed_limit,
            "candidate_limit": self.adjacent_candidate_limit,
        })
        retrieval_trace["anchor_diagnostics"].update({
            "enabled": self.evidence_anchors,
            "seed_limit": self.anchor_seed_limit,
            "candidate_limit_per_seed": self.ANCHOR_CANDIDATES_PER_SEED,
            "candidate_limit": self.anchor_max_candidates,
        })
        graph_candidate_ids: List[str] = []
        graph_only_candidate_ids: List[str] = []
        graph_paths: List[Dict[str, Any]] = []
        anchor_candidate_ids: List[str] = []
        expanded_queries = (
            self.local_query_expander.expand(query, options or [])
            if self.local_query_expander else []
        )
        raw_terms = re.findall(r"[\w]+", query, flags=re.UNICODE)
        terms = [term for term in raw_terms if term.casefold() not in self.QUERY_STOP_WORDS]
        terms = terms or raw_terms
        support_terms = []
        query_plan: Dict[str, Any] = {}
        if self.model:
            if self.structured_query_plan:
                plan = self.model.plan_query_structured(query, options or [])
                if isinstance(plan, dict):
                    query_plan = plan
                    retrieval_trace["plan"] = copy.deepcopy(plan)
                for key in ("core_terms", "entities", "temporal_cues"):
                    values = plan.get(key, [])
                    if isinstance(values, list):
                        terms.extend(value for value in values if isinstance(value, str))
                for key in ("expansion_terms", "evidence_needs"):
                    values = plan.get(key, [])
                    if not isinstance(values, list):
                        continue
                    if key == "evidence_needs" and self.evidence_need_retrieval:
                        # P4-A gives each evidence need its own bounded retrieval
                        # channel; they no longer share the low-weight support query.
                        continue
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
            self._publish_retrieval_diagnostics(
                query_plan=query_plan,
                graph_candidate_ids=graph_candidate_ids,
                graph_only_candidate_ids=graph_only_candidate_ids,
                graph_paths=graph_paths,
                retrieval_trace=retrieval_trace,
                speaker_conflict=speaker_conflict,
            )
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
        need_match_queries: List[str] = []
        if self.evidence_need_retrieval:
            planned_needs = query_plan.get("evidence_needs", [])
            if isinstance(planned_needs, list):
                for need in planned_needs:
                    if not isinstance(need, str):
                        continue
                    need_terms = [
                        term for term in re.findall(r"[\w]+", need, flags=re.UNICODE)
                        if term.casefold() not in self.QUERY_STOP_WORDS
                    ]
                    if not need_terms:
                        continue
                    need_match_queries.append(
                        " OR ".join(
                            '"{}"'.format(term.replace('"', '')) for term in need_terms
                        )
                    )
            retrieval_trace["evidence_need_diagnostics"] = {
                "enabled": bool(need_match_queries),
                "quota": self.evidence_need_quota,
                "rrf_weight": self.evidence_need_rrf_weight,
                "need_count": len(need_match_queries),
            }
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
        graph_rows: List[sqlite3.Row] = []
        anchor_rows: List[sqlite3.Row] = []
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
            need_channel_rows: Dict[str, List[sqlite3.Row]] = {}
            if need_match_queries:
                need_query_specs = (
                    ("raw", "messages_fts", ""),
                    ("raw_porter", "messages_porter_fts", ""),
                    ("fact", "facts_fts",
                     "JOIN facts AS fact ON fact.id = {fts}.fact_id "
                     "JOIN raw_messages AS raw ON raw.id = fact.source_message_id"),
                    ("fact_porter", "facts_porter_fts",
                     "JOIN facts AS fact ON fact.id = {fts}.fact_id "
                     "JOIN raw_messages AS raw ON raw.id = fact.source_message_id"),
                    ("context", "context_fts", ""),
                    ("context_porter", "context_porter_fts", ""),
                )
                for need_index, need_query in enumerate(need_match_queries):
                    for channel_name, fts_table, join_clause in need_query_specs:
                        raw_alias = "raw" if join_clause else "raw"
                        select_clause = (
                            "SELECT raw.id, raw.content, raw.created_at, raw.event_ts"
                        )
                        from_clause = "FROM {fts}".format(fts=fts_table)
                        effective_join = join_clause.format(fts=fts_table)
                        if effective_join:
                            from_clause += " " + effective_join
                        else:
                            from_clause += " JOIN raw_messages AS raw ON raw.id = {fts}.message_id".format(
                                fts=fts_table
                            )
                        sql = (
                            "{select} {from_clause} "
                            "WHERE {fts} MATCH ? AND {fts}.user_id = ? "
                            "ORDER BY bm25({fts}), raw.id DESC LIMIT ?"
                        ).format(
                            select=select_clause,
                            from_clause=from_clause,
                            fts=fts_table,
                        )
                        rows = connection.execute(
                            sql, (need_query, user_id, candidate_limit)
                        ).fetchall()
                        need_channel_rows[
                            "need_{}_{}".format(need_index, channel_name)
                        ] = rows
            need_union_ids: List[str] = []
            seen_need_ids = set()
            for rows in need_channel_rows.values():
                for row in rows:
                    candidate_id = "mem_{}".format(row["id"])
                    if candidate_id not in seen_need_ids:
                        seen_need_ids.add(candidate_id)
                        need_union_ids.append(candidate_id)
            retrieval_trace["evidence_need_channels"] = {
                name: ["mem_{}".format(row["id"]) for row in rows]
                for name, rows in need_channel_rows.items()
            }
            retrieval_trace["evidence_need_union_ids"] = list(need_union_ids)
            relax_channel_rows: Dict[str, List[sqlite3.Row]] = {}
            relax_union_ids: List[str] = []
            relax_triggered = False
            if (
                self.query_relaxation
                and need_match_queries
                and not need_union_ids
            ):
                # Pure lexical-miss signal: every evidence-need channel came back
                # empty. One bounded relaxation pass: OR FTS5 prefix forms
                # (word*) of plan core terms + entities + query words across the
                # two raw channels. No extra Search call; candidate_limit keeps
                # the pass bounded.
                relax_words: List[str] = []
                seen_relax_words = set()
                for key in ("core_terms", "entities", "expansion_terms"):
                    values = query_plan.get(key, [])
                    if not isinstance(values, list):
                        continue
                    for value in values:
                        if not isinstance(value, str):
                            continue
                        for word in re.findall(r"[\w]+", value, flags=re.UNICODE):
                            normalized = word.casefold()
                            if (
                                len(word) >= 3
                                and normalized not in self.QUERY_STOP_WORDS
                                and normalized not in seen_relax_words
                            ):
                                seen_relax_words.add(normalized)
                                relax_words.append(word)
                for word in raw_terms:
                    normalized = word.casefold()
                    if (
                        len(word) >= 3
                        and normalized not in self.QUERY_STOP_WORDS
                        and normalized not in seen_relax_words
                    ):
                        seen_relax_words.add(normalized)
                        relax_words.append(word)
                if relax_words:
                    relax_triggered = True
                    relax_clause = " OR ".join(
                        '"{}*"'.format(word.replace('"', '')) for word in relax_words
                    )
                    relax_query_specs = (
                        ("raw", "messages_fts", ""),
                        ("raw_porter", "messages_porter_fts", ""),
                    )
                    for channel_name, fts_table, _join in relax_query_specs:
                        sql = (
                            "SELECT raw.id, raw.content, raw.created_at, raw.event_ts "
                            "FROM {fts} JOIN raw_messages AS raw "
                            "ON raw.id = {fts}.message_id "
                            "WHERE {fts} MATCH ? AND {fts}.user_id = ? "
                            "ORDER BY bm25({fts}), raw.id DESC LIMIT ?"
                        ).format(fts=fts_table)
                        rows = connection.execute(
                            sql, (relax_clause, user_id, candidate_limit)
                        ).fetchall()
                        relax_channel_rows[channel_name] = rows
                    seen_relax_ids = set()
                    for rows in relax_channel_rows.values():
                        for row in rows:
                            candidate_id = "mem_{}".format(row["id"])
                            if candidate_id not in seen_relax_ids:
                                seen_relax_ids.add(candidate_id)
                                relax_union_ids.append(candidate_id)
            retrieval_trace["relax_diagnostics"] = {
                "enabled": relax_triggered,
                "rrf_weight": self.relax_rrf_weight,
                "quota": self.relax_quota,
                "relax_clause": (
                    " OR ".join('"{}*"'.format(word.replace('"', '')) for word in relax_words)
                    if relax_triggered else None
                ),
            }
            retrieval_trace["relax_channels"] = {
                name: ["mem_{}".format(row["id"]) for row in rows]
                for name, rows in relax_channel_rows.items()
            }
            retrieval_trace["relax_union_ids"] = list(relax_union_ids)
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
            if self.evidence_graph and query_plan:
                planned_entities = query_plan.get("entities", [])
                if isinstance(planned_entities, list):
                    graph_predicates = preferred_graph_predicates(
                        query, query_plan
                    )
                    graph_rows, graph_paths, graph_diagnostics = (
                        self._one_hop_graph_candidates(
                        connection,
                        user_id=user_id,
                        seed_names=[
                            value for value in planned_entities
                            if isinstance(value, str)
                        ],
                        preferred_predicates=graph_predicates,
                        )
                    )
                    graph_candidate_ids = [
                        "mem_{}".format(row["id"]) for row in graph_rows
                    ]
                    retrieval_trace.update({
                        "requested_seeds": graph_diagnostics["requested_seeds"],
                        "resolved_seeds": graph_diagnostics["resolved_seeds"],
                        "unresolved_seeds": graph_diagnostics["unresolved_seeds"],
                        "graph_candidate_ids": list(graph_candidate_ids),
                        "graph_paths": copy.deepcopy(graph_paths),
                        "edge_diagnostics": {
                            key: copy.deepcopy(value)
                            for key, value in graph_diagnostics.items()
                            if key not in {
                                "requested_seeds", "resolved_seeds",
                                "unresolved_seeds",
                            }
                        },
                    })
            elif self.evidence_anchors and query_plan:
                planned_entities = query_plan.get("entities", [])
                if isinstance(planned_entities, list):
                    anchor_rows, anchor_diagnostics = (
                        self._provenance_anchor_candidates(
                            connection,
                            user_id=user_id,
                            seed_names=[
                                value for value in planned_entities
                                if isinstance(value, str)
                            ],
                        )
                    )
                    anchor_candidate_ids = [
                        "mem_{}".format(row["id"]) for row in anchor_rows
                    ]
                    retrieval_trace["anchor_candidate_ids"] = list(
                        anchor_candidate_ids
                    )
                    retrieval_trace["anchor_diagnostics"] = copy.deepcopy(
                        anchor_diagnostics
                    )
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
                                with self._retrieval_diagnostics_lock:
                                    self.speaker_conflict_trigger_count += 1
                                speaker_conflict = True
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
        p1_channel_rows = (
            ("raw", raw_rows, 1.0),
            ("raw_porter", raw_porter_rows, 1.0),
            ("fact", fact_rows, 1.0),
            ("fact_porter", fact_porter_rows, 1.0),
            ("context", context_rows, self.CONTEXT_RRF_WEIGHT),
            ("context_porter", context_porter_rows, self.CONTEXT_RRF_WEIGHT),
            (
                "support_raw",
                support_raw_rows,
                self.STRUCTURED_SUPPORT_RRF_WEIGHT,
            ),
            (
                "support_fact",
                support_fact_rows,
                self.STRUCTURED_SUPPORT_RRF_WEIGHT,
            ),
            ("entity_raw", entity_raw_rows, self.ENTITY_RRF_WEIGHT),
            ("entity_porter", entity_porter_rows, self.ENTITY_RRF_WEIGHT),
            (
                "entity_context",
                entity_context_rows,
                self.ENTITY_RRF_WEIGHT * self.CONTEXT_RRF_WEIGHT,
            ),
            ("dense", dense_rows, self.dense_rrf_weight),
        )
        retrieval_trace["p1_channels"] = {
            name: ["mem_{}".format(row["id"]) for row in rows]
            for name, rows, _ in p1_channel_rows
        }
        p1_channels = tuple(
            (rows, weight) for _, rows, weight in p1_channel_rows
        )
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
                or speaker_conflict
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
            dense_ranked_results = [candidate["result"] for candidate in fused]
            dense_ids = [result.id for result in dense_ranked_results]
            retrieval_trace["p1_union_ids"] = list(dense_ids)
            retrieval_trace["p1_pre_rerank_ids"] = list(dense_ids)
            retrieval_trace["p1_counterfactual_top30_ids"] = list(
                dense_ids[:self.MODEL_RERANK_LIMIT]
            )
            retrieval_trace["rerank_pool_ids"] = list(
                dense_ids[:self.MODEL_RERANK_LIMIT]
            )
            dense_ranked_results = self._apply_set_aware_rerank(
                dense_ranked_results,
                query_plan=query_plan,
                retrieval_trace=retrieval_trace,
            )
            dense_results = dense_ranked_results[:top_k]
            retrieval_trace["final_ids"] = [
                result.id for result in dense_results
            ]
            self._publish_retrieval_diagnostics(
                query_plan=query_plan,
                graph_candidate_ids=graph_candidate_ids,
                graph_only_candidate_ids=graph_only_candidate_ids,
                graph_paths=graph_paths,
                retrieval_trace=retrieval_trace,
                speaker_conflict=speaker_conflict,
            )
            return dense_results
        p1_union_ids: List[str] = []
        p1_candidate_ids = set()
        for rows, _ in p1_channels:
            for row in rows:
                candidate_id = "mem_{}".format(row["id"])
                if candidate_id not in p1_candidate_ids:
                    p1_candidate_ids.add(candidate_id)
                    p1_union_ids.append(candidate_id)
        retrieval_trace["p1_union_ids"] = list(p1_union_ids)
        graph_only_candidate_ids = [
            candidate_id for candidate_id in graph_candidate_ids
            if candidate_id not in p1_candidate_ids
        ]
        anchor_only_candidate_ids = [
            candidate_id for candidate_id in anchor_candidate_ids
            if candidate_id not in p1_candidate_ids
        ]
        graph_candidate_id_set = set(graph_candidate_ids)
        anchor_candidate_id_set = set(anchor_candidate_ids)
        provenance_candidate_id_set = (
            graph_candidate_id_set | anchor_candidate_id_set
        )

        def fuse_channels(channels: Sequence[Tuple[Sequence[sqlite3.Row], float]]):
            channel_candidates: Dict[str, Dict[str, Any]] = {}
            for rows, channel_weight in channels:
                for rank, row in enumerate(rows):
                    candidate_id = "mem_{}".format(row["id"])
                    candidate = channel_candidates.setdefault(candidate_id, {
                        "result": MemoryResult(
                            id=candidate_id, content=row["content"], score=0.0,
                            created_at=row["created_at"],
                        ),
                        "event_ts": row["event_ts"],
                        "message_id": int(row["id"]),
                    })
                    candidate["result"].score += (
                        channel_weight / (self.RRF_CONSTANT + rank + 1)
                    )

            temporal_direction = 0
            if self.TEMPORAL_QUERY_PATTERN.search(query):
                temporal_direction = 1
            elif self.HISTORICAL_QUERY_PATTERN.search(query):
                temporal_direction = -1
            if temporal_direction:
                timestamps = [
                    candidate["event_ts"] for candidate in channel_candidates.values()
                    if candidate["event_ts"] is not None
                ]
                if timestamps and max(timestamps) > min(timestamps):
                    oldest, newest = min(timestamps), max(timestamps)
                    for candidate in channel_candidates.values():
                        event_ts = candidate["event_ts"]
                        if event_ts is not None:
                            recency = (event_ts - oldest) / (newest - oldest)
                            temporal_score = (
                                recency if temporal_direction > 0 else 1.0 - recency
                            )
                            candidate["result"].score += (
                                self.temporal_bonus * temporal_score
                            )
            return sorted(
                channel_candidates.values(),
                key=lambda candidate: (
                    -candidate["result"].score,
                    -(candidate["event_ts"] or 0),
                    candidate["result"].id,
                ),
            )

        def deduplicate_ranked(
            ranked_candidates: Sequence[Dict[str, Any]], *, prefer_provenance: bool
        ) -> List[MemoryResult]:
            deduplicated_results: List[MemoryResult] = []
            content_positions: Dict[str, int] = {}
            for candidate in ranked_candidates:
                result = candidate["result"]
                normalized = result.content.casefold()
                existing_index = content_positions.get(normalized)
                if existing_index is None:
                    result.score = round(result.score, 6)
                    content_positions[normalized] = len(deduplicated_results)
                    deduplicated_results.append(result)
                    continue
                existing = deduplicated_results[existing_index]
                if (
                    prefer_provenance
                    and result.id in provenance_candidate_id_set
                    and existing.id not in provenance_candidate_id_set
                ):
                    result.score = round(max(result.score, existing.score), 6)
                    deduplicated_results[existing_index] = result
            return deduplicated_results

        p1_ranked = fuse_channels(p1_channels)
        p1_counterfactual = deduplicate_ranked(
            p1_ranked, prefer_provenance=False
        )
        retrieval_trace["p1_pre_rerank_ids"] = [
            result.id for result in p1_counterfactual
        ]
        p1_counterfactual_ids = [
            result.id for result in p1_counterfactual[:self.MODEL_RERANK_LIMIT]
        ]
        retrieval_trace[
            "p1_counterfactual_top30_ids"
        ] = p1_counterfactual_ids

        adjacent_rows: List[sqlite3.Row] = []
        if self.adjacent_turn_expansion:
            with self._connection() as adjacent_connection:
                adjacent_rows, adjacent_diagnostics = (
                    self._adjacent_turn_candidates(
                        adjacent_connection,
                        user_id=user_id,
                        seed_ids=[result.id for result in p1_counterfactual],
                        excluded_ids=p1_candidate_ids,
                        excluded_content_keys=[
                            result.content.casefold()
                            for result in p1_counterfactual
                        ],
                    )
                )
            retrieval_trace["adjacent_seed_ids"] = list(
                adjacent_diagnostics["seed_ids"]
            )
            retrieval_trace["adjacent_candidate_ids"] = list(
                adjacent_diagnostics["candidate_ids"]
            )
            retrieval_trace["adjacent_deduped_ids"] = list(
                adjacent_diagnostics["deduped_ids"]
            )
            retrieval_trace["adjacent_diagnostics"] = adjacent_diagnostics

        adjacent_channel = (
            ((adjacent_rows, 0.0),) if self.adjacent_turn_expansion else ()
        )
        anchor_channel = (
            ((anchor_rows, self.anchor_rrf_weight),)
            if self.evidence_anchors else ()
        )
        need_channels = (
            tuple(
                (rows, self.evidence_need_rrf_weight)
                for rows in need_channel_rows.values()
            )
            if need_match_queries else ()
        )
        relax_channels = (
            tuple(
                (rows, self.relax_rrf_weight)
                for rows in relax_channel_rows.values()
            )
            if relax_channel_rows else ()
        )
        ranked = fuse_channels(
            p1_channels
            + need_channels
            + relax_channels
            + ((graph_rows, self.graph_rrf_weight),)
            + anchor_channel
            + adjacent_channel
        )
        deduplicated = deduplicate_ranked(ranked, prefer_provenance=True)
        retrieval_trace["graph_channel_only_ids"] = list(
            graph_only_candidate_ids
        )
        retrieval_trace["anchor_channel_only_ids"] = list(
            anchor_only_candidate_ids
        )
        bridge_channel_rows: Dict[str, List[sqlite3.Row]] = {}
        bridge_union_ids: List[str] = []
        bridge_triggered = False
        if self.bridge_retrieval:
            planned_intent = query_plan.get("intent")
            planned_needs = query_plan.get("evidence_needs", [])
            if isinstance(planned_needs, list):
                planned_needs = [
                    value for value in planned_needs if isinstance(value, str)
                ]
            bridge_triggered = (
                planned_intent == "multi_hop"
                or len(planned_needs) >= 2
            ) and bool(deduplicated)
            retrieval_trace["bridge_diagnostics"] = {
                "enabled": bridge_triggered,
                "max_terms": self.bridge_max_terms,
                "rrf_weight": self.bridge_rrf_weight,
                "quota": self.bridge_rerank_quota,
            }
            if bridge_triggered:
                first_pass_texts = [
                    candidate.content for candidate in deduplicated[:8]
                ]
                with self._connection() as bridge_connection:
                    speaker_rows = bridge_connection.execute(
                        "SELECT DISTINCT role FROM raw_messages WHERE user_id = ?",
                        (user_id,),
                    ).fetchall()
                    known_speakers = [
                        str(row["role"]).strip()
                        for row in speaker_rows
                        if str(row["role"]).strip()
                    ]
                    bridge_terms = extract_bridge_terms(
                        query,
                        first_pass_texts,
                        known_speakers,
                        max_terms=self.bridge_max_terms,
                    )
                    retrieval_trace["bridge_terms"] = list(bridge_terms)
                    if bridge_terms:
                        bridge_query_specs = (
                            ("raw", "messages_fts", ""),
                            ("raw_porter", "messages_porter_fts", ""),
                            ("context", "context_fts", ""),
                            ("context_porter", "context_porter_fts", ""),
                        )
                        bridge_clause = " OR ".join(
                            '"{}"'.format(term.replace('"', ''))
                            for term in bridge_terms
                        )
                        for need_index, need in enumerate(planned_needs[:2]):
                            need_terms = [
                                term
                                for term in re.findall(
                                    r"[\w]+", need, flags=re.UNICODE
                                )
                                if term.casefold() not in self.QUERY_STOP_WORDS
                            ]
                            if not need_terms:
                                continue
                            need_clause = " OR ".join(
                                '"{}"'.format(term.replace('"', ''))
                                for term in need_terms
                            )
                            second_query = (
                                "({}) AND ({})".format(
                                    bridge_clause, need_clause
                                )
                            )
                            for channel_name, fts_table, join_clause in (
                                bridge_query_specs
                            ):
                                from_clause = "FROM {fts}".format(fts=fts_table)
                                if join_clause:
                                    from_clause += " " + join_clause.format(
                                        fts=fts_table
                                    )
                                else:
                                    from_clause += (
                                        " JOIN raw_messages AS raw ON raw.id = "
                                        "{fts}.message_id"
                                    ).format(fts=fts_table)
                                sql = (
                                    "{select} {from_clause} "
                                    "WHERE {fts} MATCH ? AND {fts}.user_id = ? "
                                    "ORDER BY bm25({fts}), raw.id DESC LIMIT ?"
                                ).format(
                                    select=(
                                        "SELECT raw.id, raw.content, "
                                        "raw.created_at, raw.event_ts"
                                    ),
                                    from_clause=from_clause,
                                    fts=fts_table,
                                )
                                rows = bridge_connection.execute(
                                    sql,
                                    (second_query, user_id, candidate_limit),
                                ).fetchall()
                                bridge_channel_rows[
                                    "bridge_{}_{}".format(
                                        need_index, channel_name
                                    )
                                ] = rows
                seen_bridge_ids = set()
                for rows in bridge_channel_rows.values():
                    for row in rows:
                        candidate_id = "mem_{}".format(row["id"])
                        if candidate_id not in seen_bridge_ids:
                            seen_bridge_ids.add(candidate_id)
                            bridge_union_ids.append(candidate_id)
                retrieval_trace["bridge_channels"] = {
                    name: ["mem_{}".format(row["id"]) for row in rows]
                    for name, rows in bridge_channel_rows.items()
                }
                retrieval_trace["bridge_union_ids"] = list(bridge_union_ids)
        if self.model and deduplicated:
            message_ids = {
                candidate["result"].id: candidate["message_id"]
                for candidate in ranked
            }
            result_by_id = {result.id: result for result in deduplicated}
            if bridge_union_ids:
                for rows in bridge_channel_rows.values():
                    for row in rows:
                        candidate_id = "mem_{}".format(row["id"])
                        if candidate_id in result_by_id:
                            continue
                        message_ids[candidate_id] = int(row["id"])
                        result_by_id[candidate_id] = MemoryResult(
                            id=candidate_id,
                            content=str(row["content"]),
                            score=0.0,
                            created_at=row["created_at"],
                        )
            if self.adjacent_turn_expansion:
                special_quota = self.adjacent_candidate_limit
            elif self.evidence_anchors:
                special_quota = self.anchor_rerank_quota
            else:
                special_quota = self.graph_rerank_quota
            need_quota = self.evidence_need_quota if need_match_queries else 0
            bridge_quota = self.bridge_rerank_quota if bridge_union_ids else 0
            relax_quota = self.relax_quota if relax_union_ids else 0
            if self.sidecar_shared_quota > 0:
                # Shared sidecar pool: need, bridge and relax candidates compete
                # for ONE total reservation, so the P1 base budget is compressed
                # by at most sidecar_shared_quota slots regardless of how many
                # sidecar components are active.
                base_budget = (
                    self.MODEL_RERANK_LIMIT
                    - special_quota
                    - self.sidecar_shared_quota
                )
            else:
                base_budget = (
                    self.MODEL_RERANK_LIMIT
                    - special_quota
                    - need_quota
                    - bridge_quota
                    - relax_quota
                )
            rerank_pool = list(deduplicated[:base_budget])
            pool_ids = {result.id for result in rerank_pool}
            reserved_graph_ids: List[str] = []
            reserved_anchor_ids: List[str] = []
            reserved_adjacent_ids: List[str] = []
            reserved_need_ids: List[str] = []
            reserved_bridge_ids: List[str] = []
            reserved_relax_ids: List[str] = []
            if self.adjacent_turn_expansion:
                for candidate_id in retrieval_trace["adjacent_candidate_ids"]:
                    if len(reserved_adjacent_ids) >= self.adjacent_candidate_limit:
                        break
                    if candidate_id in result_by_id and candidate_id not in pool_ids:
                        reserved_adjacent_ids.append(candidate_id)
                        pool_ids.add(candidate_id)
            elif self.evidence_anchors:
                for candidate_id in anchor_candidate_ids:
                    if len(reserved_anchor_ids) >= self.anchor_rerank_quota:
                        break
                    if candidate_id in result_by_id and candidate_id not in pool_ids:
                        reserved_anchor_ids.append(candidate_id)
                        pool_ids.add(candidate_id)
            elif self.graph_rerank_quota > 0:
                for candidate_id in graph_candidate_ids:
                    if len(reserved_graph_ids) >= self.graph_rerank_quota:
                        break
                    if candidate_id in result_by_id and candidate_id not in pool_ids:
                        reserved_graph_ids.append(candidate_id)
                        pool_ids.add(candidate_id)
            sidecar_reserved = 0
            if need_quota > 0:
                need_limit = (
                    self.sidecar_shared_quota
                    if self.sidecar_shared_quota > 0
                    else need_quota
                )
                for candidate_id in need_union_ids:
                    if sidecar_reserved >= need_limit:
                        break
                    if candidate_id in result_by_id and candidate_id not in pool_ids:
                        reserved_need_ids.append(candidate_id)
                        pool_ids.add(candidate_id)
                        sidecar_reserved += 1
            if bridge_quota > 0:
                bridge_limit = (
                    self.sidecar_shared_quota
                    if self.sidecar_shared_quota > 0
                    else bridge_quota
                )
                for candidate_id in bridge_union_ids:
                    if sidecar_reserved >= bridge_limit:
                        break
                    if candidate_id in result_by_id and candidate_id not in pool_ids:
                        reserved_bridge_ids.append(candidate_id)
                        pool_ids.add(candidate_id)
                        sidecar_reserved += 1
            if relax_quota > 0:
                relax_limit = (
                    self.sidecar_shared_quota
                    if self.sidecar_shared_quota > 0
                    else relax_quota
                )
                for candidate_id in relax_union_ids:
                    if sidecar_reserved >= relax_limit:
                        break
                    if candidate_id in result_by_id and candidate_id not in pool_ids:
                        reserved_relax_ids.append(candidate_id)
                        pool_ids.add(candidate_id)
                        sidecar_reserved += 1
            rerank_pool.extend(
                result_by_id[item]
                for item in (
                    reserved_graph_ids
                    + reserved_anchor_ids
                    + reserved_adjacent_ids
                    + reserved_need_ids
                    + reserved_bridge_ids
                    + reserved_relax_ids
                )
            )
            for result in deduplicated:
                if len(rerank_pool) >= self.MODEL_RERANK_LIMIT:
                    break
                if result.id not in pool_ids:
                    rerank_pool.append(result)
                    pool_ids.add(result.id)
            if len(rerank_pool) > self.MODEL_RERANK_LIMIT:
                raise RuntimeError("model rerank pool exceeded its hard limit")
            rerank_pool_ids = [result.id for result in rerank_pool]
            retrieval_trace["reserved_graph_ids"] = list(reserved_graph_ids)
            retrieval_trace["reserved_anchor_ids"] = list(reserved_anchor_ids)
            retrieval_trace["reserved_adjacent_ids"] = list(reserved_adjacent_ids)
            retrieval_trace["reserved_need_ids"] = list(reserved_need_ids)
            retrieval_trace["reserved_bridge_ids"] = list(reserved_bridge_ids)
            retrieval_trace["reserved_relax_ids"] = list(reserved_relax_ids)
            retrieval_trace["rerank_pool_ids"] = rerank_pool_ids
            retrieval_trace["promoted_graph_ids"] = [
                candidate_id for candidate_id in rerank_pool_ids
                if (
                    candidate_id in graph_candidate_id_set
                    and candidate_id not in p1_counterfactual_ids
                )
            ]
            retrieval_trace["promoted_anchor_ids"] = [
                candidate_id for candidate_id in rerank_pool_ids
                if (
                    candidate_id in anchor_candidate_id_set
                    and candidate_id not in p1_counterfactual_ids
                )
            ]
            adjacent_candidate_id_set = set(
                retrieval_trace["adjacent_candidate_ids"]
            )
            retrieval_trace["promoted_adjacent_ids"] = [
                candidate_id for candidate_id in rerank_pool_ids
                if (
                    candidate_id in adjacent_candidate_id_set
                    and candidate_id not in p1_counterfactual_ids
                )
            ]
            need_candidate_id_set = set(need_union_ids)
            retrieval_trace["promoted_need_ids"] = [
                candidate_id for candidate_id in rerank_pool_ids
                if (
                    candidate_id in need_candidate_id_set
                    and candidate_id not in p1_counterfactual_ids
                )
            ]
            bridge_candidate_id_set = set(bridge_union_ids)
            retrieval_trace["promoted_bridge_ids"] = [
                candidate_id for candidate_id in rerank_pool_ids
                if (
                    candidate_id in bridge_candidate_id_set
                    and candidate_id not in p1_counterfactual_ids
                )
            ]
            relax_candidate_id_set = set(relax_union_ids)
            retrieval_trace["promoted_relax_ids"] = [
                candidate_id for candidate_id in rerank_pool_ids
                if (
                    candidate_id in relax_candidate_id_set
                    and candidate_id not in p1_counterfactual_ids
                )
            ]
            rerank_pool_id_set = set(rerank_pool_ids)
            displaced_p1_ids = [
                candidate_id for candidate_id in p1_counterfactual_ids
                if candidate_id not in rerank_pool_id_set
            ]
            if self.adjacent_turn_expansion:
                retrieval_trace["displaced_p1_for_adjacent_ids"] = (
                    displaced_p1_ids
                )
            elif self.evidence_anchors:
                retrieval_trace["displaced_p1_for_anchor_ids"] = (
                    displaced_p1_ids
                )
            elif need_quota > 0:
                retrieval_trace["displaced_p1_for_need_ids"] = displaced_p1_ids
            elif bridge_quota > 0:
                retrieval_trace["displaced_p1_for_bridge_ids"] = displaced_p1_ids
            elif relax_quota > 0:
                retrieval_trace["displaced_p1_for_relax_ids"] = displaced_p1_ids
            else:
                retrieval_trace["displaced_p1_ids"] = displaced_p1_ids
            paths_by_source: Dict[str, List[Dict[str, Any]]] = {}
            for path in graph_paths:
                for source_id in path.get("source_message_ids", []):
                    paths_by_source.setdefault(str(source_id), []).append(path)
            candidates = []
            for result in rerank_pool:
                metadata = ranking_metadata.get(message_ids[result.id], {})
                candidates.append({
                    "id": result.id,
                    "content": candidate_ranking_text(
                        result.content,
                        str(metadata.get("speaker", "")),
                        metadata.get("event_ts"),
                        metadata.get("facts", []),
                        metadata.get("neighbors", []),
                        paths_by_source.get(result.id, []),
                    ),
                })
            ordered_ids = self.model.rank_candidates(query, options or [], candidates)
            rank_confidence: Dict[str, float] = {}
            if self.p5_gate and hasattr(self.model, "rank_candidates_with_confidence"):
                ordered_ids, rank_confidence = (
                    self.model.rank_candidates_with_confidence(
                        query, options or [], candidates
                    )
                )
            retrieval_trace["p5_diagnostics"]["rank_confidence"] = dict(
                rank_confidence
            )
            positions = {candidate_id: index for index, candidate_id in enumerate(ordered_ids)}
            deduplicated.sort(key=lambda result: positions.get(result.id, len(positions)))
        elif deduplicated:
            retrieval_trace["rerank_pool_ids"] = [
                result.id for result in deduplicated[:self.MODEL_RERANK_LIMIT]
            ]
        deduplicated = self._apply_set_aware_rerank(
            deduplicated,
            query_plan=query_plan,
            retrieval_trace=retrieval_trace,
        )
        deduplicated = self._apply_selective_rerank_gate(
            deduplicated,
            query=query,
            query_plan=query_plan,
            retrieval_trace=retrieval_trace,
            ranking_metadata=ranking_metadata,
            message_ids=message_ids if self.model else None,
        )
        final_results = deduplicated[:top_k]
        retrieval_trace["final_ids"] = [
            result.id for result in final_results
        ]
        self._publish_retrieval_diagnostics(
            query_plan=query_plan,
            graph_candidate_ids=graph_candidate_ids,
            graph_only_candidate_ids=graph_only_candidate_ids,
            graph_paths=graph_paths,
            retrieval_trace=retrieval_trace,
            speaker_conflict=speaker_conflict,
        )
        return final_results
