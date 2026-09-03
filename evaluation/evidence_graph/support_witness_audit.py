"""Evaluator-owned verification for persisted P3-A graph witnesses.

This module is deliberately independent from the production graph parser.  In
particular, it neither imports production parsing helpers nor asks a
production support boolean to certify an edge.  The production parser may be
changed independently; an evaluation artifact is accepted only when this
frozen, evaluator-owned contract can reproduce the stored witness from the
immutable source row.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import sqlite3
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Sequence
import unicodedata


# These identifiers are persisted with each support row and in the prepared
# sidecar.  Changing either is a materialization-compatibility change, not a
# best-effort evaluator upgrade.
SUPPORT_SCHEMA_VERSION = 1
NORMALIZATION_ID = "nfkc-casefold-ws-v1"
FROZEN_SPEC_REGISTRY_VERSION = 1

CONTROLLED_PREDICATES = (
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
_PREDICATE_SET = frozenset(CONTROLLED_PREDICATES)
_STATE_CHANGES = frozenset(
    {"assert", "update", "correction", "retraction", "historical"}
)
_TEMPORAL_STATUSES = frozenset({"current", "previous", "historical", "future"})
_BINDINGS = frozenset({"named", "trusted_speaker_1p"})


@dataclass(frozen=True)
class FrozenWitnessSpec:
    """Closed evaluator rule for a persisted support witness."""

    spec_id: str
    predicate: str
    kind: str
    state_status_pairs: tuple[tuple[str, str | None], ...]

    def canonical(self) -> dict[str, object]:
        return {
            "spec_id": self.spec_id,
            "predicate": self.predicate,
            "kind": self.kind,
            "state_status_pairs": [list(pair) for pair in self.state_status_pairs],
        }


def _build_frozen_registry() -> Mapping[str, FrozenWitnessSpec]:
    """Build only from literals in this evaluator module.

    The registry intentionally contains no production imports or production
    function references.  The generated mapping is immediately wrapped as a
    read-only mapping below.
    """

    specs: dict[str, FrozenWitnessSpec] = {}
    for predicate in CONTROLLED_PREDICATES:
        specs[f"{predicate}:direct"] = FrozenWitnessSpec(
            spec_id=f"{predicate}:direct",
            predicate=predicate,
            kind="direct",
            state_status_pairs=(
                ("assert", None),
                ("historical", "previous"),
                ("update", "current"),
            ),
        )
        specs[f"{predicate}:correction"] = FrozenWitnessSpec(
            spec_id=f"{predicate}:correction",
            predicate=predicate,
            kind="correction",
            state_status_pairs=(("correction", "current"),),
        )
        for suffix in ("coordinated", "coordinated-first", "coordinated-object"):
            specs[f"{predicate}:{suffix}"] = FrozenWitnessSpec(
                spec_id=f"{predicate}:{suffix}",
                predicate=predicate,
                kind=suffix,
                state_status_pairs=(("assert", None),),
            )
    specs["lives_in:moved-from-to"] = FrozenWitnessSpec(
        spec_id="lives_in:moved-from-to",
        predicate="lives_in",
        kind="moved-from-to",
        state_status_pairs=(("update", "current"),),
    )
    specs["member_of:retraction"] = FrozenWitnessSpec(
        spec_id="member_of:retraction",
        predicate="member_of",
        kind="retraction",
        state_status_pairs=(("retraction", "previous"),),
    )
    return MappingProxyType(dict(sorted(specs.items())))


FROZEN_SPEC_REGISTRY = _build_frozen_registry()


def _canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


_COMPUTED_FROZEN_SPEC_REGISTRY_FINGERPRINT = hashlib.sha256(
    _canonical_json(
        {
            "registry_version": FROZEN_SPEC_REGISTRY_VERSION,
            "normalization_id": NORMALIZATION_ID,
            "specs": [spec.canonical() for spec in FROZEN_SPEC_REGISTRY.values()],
        }
    ).encode("utf-8")
).hexdigest()
# This literal makes a grammar/spec change deliberate: changing the registry
# without intentionally advancing the evaluator contract fails at import time
# instead of silently certifying an old prepared database under new rules.
FROZEN_SPEC_REGISTRY_FINGERPRINT = (
    "57678cad8d6b9464e38e9686367354827adbc39e583fca3205594a3758f1632b"
)
if _COMPUTED_FROZEN_SPEC_REGISTRY_FINGERPRINT != FROZEN_SPEC_REGISTRY_FINGERPRINT:
    raise RuntimeError("frozen evaluator witness registry fingerprint mismatch")


def normalize_witness_source(value: object) -> str | None:
    """Return the evaluator's documented canonical source representation.

    The operation is exactly NFKC, casefold, then collapse *all* whitespace to
    one ASCII space.  It is intentionally not the production normalizer, and
    is kept here rather than delegated to an application helper so persisted
    numeric offsets have a reproducible evaluator-owned meaning.
    """

    if not isinstance(value, str):
        return None
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def normalized_source_span_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _normalise_identifier(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = normalize_witness_source(value)
    return normalized if normalized else None


def _span_value(row: Mapping[str, Any], prefix: str) -> tuple[int, int] | None:
    start_key = f"{prefix}_start"
    end_key = f"{prefix}_end"
    start = row.get(start_key)
    end = row.get(end_key)
    # bool is an int subclass but is never a legitimate source offset.
    if isinstance(start, bool) or isinstance(end, bool):
        return None
    if not isinstance(start, int) or not isinstance(end, int):
        return None
    return start, end


def _span_list(span: tuple[int, int]) -> list[int]:
    return [span[0], span[1]]


def _valid_span(span: tuple[int, int], *, lower: int, upper: int) -> bool:
    return lower <= span[0] < span[1] <= upper


def _endpoint_surface_matches(surface: str, endpoint: str, *, subject: bool) -> bool:
    """Validate an endpoint without using production entity normalization.

    Grammar permits a small visible determiner set around an endpoint.  This
    is a witness-surface rule, not a fuzzy linker: after that exact grammar
    cleanup, text must equal the stored entity display/value under the frozen
    normalizer.
    """

    observed = normalize_witness_source(surface)
    expected = normalize_witness_source(endpoint)
    if not observed or not expected:
        return False
    observed = observed.strip(" \"'“”‘’")
    if observed == expected:
        return True
    if observed.startswith(("a ", "an ", "the ", "this ", "that ", "these ", "those ", "my ", "our ", "your ", "his ", "her ", "their ", "another ")):
        observed = observed.split(" ", 1)[1]
        if observed == expected:
            return True
    if subject and re.fullmatch(r"(?:a|an) different " + re.escape(expected), observed):
        return True
    return False


# These are evaluator-owned surface frames.  They intentionally mirror only
# the closed predicate vocabulary; no loose word-distance or co-occurrence
# fallback exists.  A source must prove the entire relation through its stored
# S/P/O spans and one of these frames.
_FRAME_PATTERNS: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {
        "friend_of": (r"(?:am|is|are|was|were)\s+(?:a\s+)?friend\s+of",),
        "parent_of": (r"(?:am|is|are|was|were)\s+(?:a\s+)?(?:parent|mother|father)\s+of",),
        "sibling_of": (r"(?:am|is|are|was|were)\s+(?:a\s+)?(?:sibling|brother|sister)\s+of",),
        "partner_of": (r"(?:am|is|are|was|were)\s+(?:a\s+)?(?:partner|spouse|husband|wife)\s+of",),
        "works_at": (
            r"(?:work|works|worked)\s+(?:at|for)",
            r"(?:am|is|are|was|were)\s+employed\s+by",
            r"(?:work|works|worked)\s+as\s+(?:(?:the|an?|their)\s+)?(?:[a-z]+\s+){0,3}[a-z]+\s+at",
        ),
        "role_at": (
            r"(?:have|has|had|hold|holds|held)\s+(?:an?\s+|the\s+)?(?:formal\s+)?(?:role|position)\s+at",
            r"(?:serve|serves|served)\s+(?:in\s+)?(?:(?:the|an?|their)\s+)?(?:[a-z]+\s+){0,3}[a-z]+\s+at",
        ),
        "lives_in": (r"(?:live|lives|lived)\s+in",),
        "located_in": (r"(?:am|is|are|was|were)\s+(?:located|based|situated)\s+in",),
        "likes": (
            r"(?:like|likes|liked|love|loves|loved|enjoy|enjoys|enjoyed)",
            r"(?:am|is|are|was|were)\s+(?:keen\s+on|fond\s+of)",
            r"(?:keen\s+on|fond\s+of)",
            r"(?:have|has|had)\s+(?:always\s+)?(?:had\s+)?(?:a\s+)?love\s+(?:of|for)",
        ),
        "prefers": (r"(?:prefer|prefers|preferred)",),
        "dislikes": (r"(?:dislike|dislikes|disliked|hate|hates|hated)",),
        "member_of": (
            r"(?:am|is|are|was|were|became)\s+(?:a\s+)?member\s+of",
            r"(?:belong|belongs|belonged)\s+to",
            r"(?:join|joins|joined)",
        ),
        "participated_in": (
            r"(?:participate|participates|participated)\s+in",
            r"(?:take|takes|took)\s+part\s+in",
            r"(?:attend|attends|attended)",
            r"(?:join|joins|joined)",
            r"went\s+to",
        ),
        "owns": (r"(?:own|owns|owned|possess|possesses|possessed)",),
        "created": (
            r"(?:create|creates|created|make|makes|made|build|builds|built|write|writes|wrote|written|design|designs|designed|compose|composes|composed|produce|produces|produced)",
        ),
        "requires": (r"(?:explicitly\s+)?(?:require|requires|required)",),
        "prohibits": (r"(?:explicitly\s+)?(?:prohibit|prohibits|prohibited|forbid|forbids|forbade)",),
        "permits": (r"(?:explicitly\s+)?(?:permit|permits|permitted|allow|allows|allowed)",),
        "changed_to": (
            r"(?:change|changes|changed|switch|switches|switched|transition|transitions|transitioned)\s+to",
            r"(?:became|becomes)",
        ),
        "replaces": (r"(?:replace|replaces|replaced)",),
    }
)

_FIRST_PERSON_FRAMES: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {
        "friend_of": (r"(?:am|was)\s+(?:a\s+)?friend\s+of",),
        "parent_of": (r"(?:am|was)\s+(?:a\s+)?(?:parent|mother|father)\s+of",),
        "sibling_of": (r"(?:am|was)\s+(?:a\s+)?(?:sibling|brother|sister)\s+of",),
        "partner_of": (r"(?:am|was)\s+(?:a\s+)?(?:partner|spouse|husband|wife)\s+of",),
        "works_at": (r"(?:work|worked)\s+(?:at|for)", r"(?:am|was)\s+employed\s+by", r"(?:work|worked)\s+as\s+.+\s+at"),
        "role_at": (r"(?:have|had|hold|held)\s+.+\s+at", r"(?:serve|served)\s+(?:in\s+)?.+\s+at"),
        "lives_in": (r"(?:live|lived)\s+in", r"moved\s+from"),
        "located_in": (r"(?:am|was)\s+(?:located|based|situated)\s+in",),
        "likes": (r"(?:like|liked|love|loved|enjoy|enjoyed)", r"(?:am|was)\s+(?:keen\s+on|fond\s+of)", r"(?:have|had)\s+.+love\s+(?:of|for)"),
        "prefers": (r"(?:prefer|preferred)",),
        "dislikes": (r"(?:dislike|disliked|hate|hated)",),
        "member_of": (r"(?:am|was|became)\s+(?:a\s+)?member\s+of", r"(?:belong|belonged)\s+to", r"(?:join|joined)"),
        "participated_in": (r"(?:participate|participated)\s+in", r"(?:take|took)\s+part\s+in", r"(?:attend|attended)", r"(?:join|joined)", r"went\s+to"),
        "owns": (r"(?:own|owned|possess|possessed)",),
        "created": (r"(?:create|created|make|made|build|built|write|wrote|design|designed|compose|composed|produce|produced)",),
        "requires": (r"(?:explicitly\s+)?(?:require|required)",),
        "prohibits": (r"(?:explicitly\s+)?(?:prohibit|prohibited|forbid|forbade)",),
        "permits": (r"(?:explicitly\s+)?(?:permit|permitted|allow|allowed)",),
        "changed_to": (r"(?:change|changed|switch|switched|transition|transitioned)\s+to", r"became"),
        "replaces": (r"(?:replace|replaced)",),
    }
)

_UNTRUSTED_CLAIM_TOKENS = re.compile(
    r"\b(?:ignore|invent|pretend|fabricate|hypothetical|fictional|"
    r"not\s+a\s+fact|not\s+true|false\s+claim|untrusted\s+instruction|"
    r"example\s+only|merely\s+an\s+example|maybe|perhaps|if|would|could|"
    r"should|report(?:ed|s|ing)?|said|says|claimed|claims|according\s+to)\b"
)

# Source-level scope is intentionally a separate verifier from the local
# predicate frame.  A valid-looking substring is not a witness when the full
# source quotes it, labels it as an example, governs it with a forward marker,
# or later withdraws the same subject's assertion.
_SOURCE_META_MARKERS = re.compile(
    r"\b(?:ignore|disregard|pretend|fabricate|invent|hypothetical|fictional|"
    r"example|scenario|report(?:ed|s|ing)?|said|says|claimed|claims|"
    r"according\s+to|untrusted|not\s+a\s+fact|not\s+true|false\s+claim)\b"
)
_FORWARD_GOVERNANCE_MARKER = re.compile(
    r"\b(?:ignore|disregard|treat|consider|read|take)\b[^.!?。！？]{0,80}"
    r"\b(?:next|following)\s+(?:sentence|statement|claim)\b|"
    r"\b(?:next|following)\s+(?:sentence|statement|claim)\b[^.!?。！？]{0,80}"
    r"\b(?:not\s+true|false|untrusted|ignore|fictional|hypothetical)\b|"
    r"\b(?:hypothetical|fictional|example)\b[^.!?。！？]{0,80}"
    r"\b(?:next|following)\b"
)
_GENERIC_RETRACTION_MARKER = re.compile(
    r"\b(?:the\s+)?(?:earlier|previous|above)\s+(?:statement|claim|assertion)"
    r"\s+(?:is|was)\s+(?:retracted|revoked|false|untrue)\b"
)
_WITHDRAWAL_TAIL = re.compile(
    r"\b(?:do|does|did)\s+not\b|\b(?:don't|doesn't|didn't)\b|"
    r"\b(?:am|is|are|was|were)\s+not\b|\bno\s+longer\b|\bnever\b|"
    r"\b(?:retract|retracted|revoke|revoked|withdraw|withdrew|withdrawn)\b"
)
# A source-level evaluator cannot safely resolve a later pronoun, shortened
# name, or ellipsis back to the stored subject.  Once a later sentence has an
# explicit withdrawal/negation construction, retain no witness from the
# earlier sentence.  This is intentionally broader than the local parser's
# subject-aware rule and is part of the frozen evaluation contract.
_LATER_WITHDRAWAL_OR_NEGATION_MARKER = re.compile(
    r"\b(?:do|does|did)\s+not\b|"
    r"\b(?:don't|doesn't|didn't|don’t|doesn’t|didn’t)\b|"
    r"\b(?:am|is|are|was|were)\s+not\b|"
    r"\b(?:isn't|aren't|wasn't|weren't|isn’t|aren’t|wasn’t|weren’t)\b|"
    r"\b(?:cannot|can't|won't|can't|cannot|can’t|won’t)\b|"
    r"\b(?:no\s+longer|never)\b|"
    r"\b(?:retract|retracted|revoke|revoked|withdraw|withdrew|withdrawn)\b"
)


def _source_container_spans(source: str) -> tuple[tuple[int, int], ...]:
    """Return fail-closed quote/bracket containers in canonical source text.

    This is not copied from the parser.  It is a small evaluator-owned stack
    that treats unmatched delimiters conservatively while leaving apostrophes
    in contractions/possessives lexical.
    """

    pairs = {
        "“": "”",
        "‘": "’",
        "«": "»",
        "(": ")",
        "[": "]",
        "{": "}",
        "「": "」",
        "『": "』",
        "‹": "›",
        "《": "》",
        "【": "】",
        "⟦": "⟧",
    }
    # The explicit pairs cover common ASCII and language-specific forms.  The
    # Unicode punctuation categories below deliberately cover the rest (for
    # example 〈〉, 〔〕, and less common editorial quotation marks) so a new
    # paired delimiter cannot create an evaluator blind spot.
    opening_categories = frozenset({"Pi", "Ps"})
    closing_categories = frozenset({"Pf", "Pe"})
    stack: list[tuple[str | None, str, int]] = []
    spans: list[tuple[int, int]] = []
    for index, character in enumerate(source):
        previous = source[index - 1] if index else ""
        following = source[index + 1] if index + 1 < len(source) else ""
        # Quote marks in ordinary contractions / possessives do not open a
        # source container.  The remaining straight quote handling is a
        # conservative toggle, sufficient for a persisted witness audit.
        if character in {"'", "’"} and previous.isalnum() and following.isalnum():
            continue
        if character in {"'", '"'}:
            if stack and stack[-1][0] == character:
                _, _, start = stack.pop()
                spans.append((start, index + 1))
            elif following:
                stack.append((character, "straight", index))
            else:
                spans.append((0, index + 1))
            continue
        if character in pairs:
            stack.append(
                (
                    pairs[character],
                    unicodedata.category(pairs[character]),
                    index,
                )
            )
            continue
        category = unicodedata.category(character)
        if category in opening_categories:
            expected_category = "Pf" if category == "Pi" else "Pe"
            stack.append((None, expected_category, index))
            continue
        if category in closing_categories:
            if stack and (
                stack[-1][0] == character
                or (
                    stack[-1][0] is None
                    and stack[-1][1] == category
                )
            ):
                _, _, start = stack.pop()
                spans.append((start, index + 1))
            else:
                # A mismatched closing delimiter makes preceding source
                # structurally ambiguous, so mask it rather than guessing.
                spans.append((0, index + 1))
    spans.extend((start, len(source)) for _, _, start in stack)
    return tuple(spans)


def _source_sentence_spans(source: str) -> tuple[tuple[int, int], ...]:
    """Split canonical source into top-level sentence-sized containers."""

    spans: list[tuple[int, int]] = []
    start = 0
    for match in re.finditer(r"[.!?。！？]+", source):
        end = match.end()
        if start < end:
            spans.append((start, end))
        start = end
        while start < len(source) and source[start].isspace():
            start += 1
    if start < len(source):
        spans.append((start, len(source)))
    return tuple(spans)


def _span_overlaps(left: tuple[int, int], right: tuple[int, int]) -> bool:
    return left[0] < right[1] and right[0] < left[1]


def _same_subject_withdrawal(
    sentence: str,
    *,
    subject_name: str,
    binding: str,
) -> bool:
    """Detect a later explicit withdrawal by this witness's subject only."""

    normalized_subject = normalize_witness_source(subject_name)
    if not normalized_subject:
        return False
    subject_forms = [re.escape(normalized_subject).replace(r"\ ", r"\s+")]
    if binding == "trusted_speaker_1p":
        subject_forms.append(r"i(?:'m|'ve|'d|’m|’ve|’d)?")
    for subject_pattern in subject_forms:
        match = re.search(
            r"(?<![\w])(?:{})\b(?P<tail>.{{0,160}})".format(subject_pattern),
            sentence,
        )
        if match and _WITHDRAWAL_TAIL.search(match.group("tail")):
            return True
    return False


def _source_scope_violations(
    *,
    source: str,
    clause_span: tuple[int, int],
    subject_name: str,
    binding: str,
    raw_role: object,
) -> list[str]:
    """Prove that a local clause remains an atomic assertion in its source."""

    issues: list[str] = []
    for container in _source_container_spans(source):
        if _span_overlaps(clause_span, container):
            issues.append("claimed clause lies in a quoted/bracketed source container")
            break
    sentence_spans = _source_sentence_spans(source)
    sentence_index = next(
        (
            index
            for index, span in enumerate(sentence_spans)
            if span[0] <= clause_span[0] and clause_span[1] <= span[1]
        ),
        None,
    )
    if sentence_index is None:
        return [*issues, "claimed clause is not contained by a source sentence"]
    sentence_start, sentence_end = sentence_spans[sentence_index]
    prefix = source[sentence_start:clause_span[0]]
    suffix = source[clause_span[1]:sentence_end]
    normalized_speaker = normalize_witness_source(raw_role)
    allowed_prefix = not prefix.strip()
    if not allowed_prefix and normalized_speaker:
        allowed_prefix = bool(
            re.fullmatch(r"{}\s*:\s*".format(re.escape(normalized_speaker)), prefix)
        )
    if not allowed_prefix or suffix.strip():
        issues.append("claimed clause is not the full atomic source sentence")
    sentence = source[sentence_start:sentence_end]
    prior_sentences = tuple(
        source[start:end] for start, end in sentence_spans[:sentence_index]
    )
    if _SOURCE_META_MARKERS.search(sentence) or any(
        _SOURCE_META_MARKERS.search(previous) for previous in prior_sentences
    ):
        issues.append("claimed clause has source-level meta/reporting governance")
    if any(
        _FORWARD_GOVERNANCE_MARKER.search(previous)
        for previous in prior_sentences
    ):
        issues.append("claimed clause is governed by a prior forward marker")
    for later_start, later_end in sentence_spans[sentence_index + 1:]:
        later_sentence = source[later_start:later_end]
        if _GENERIC_RETRACTION_MARKER.search(later_sentence):
            issues.append("later source sentence retracts the earlier assertion")
            break
        if _LATER_WITHDRAWAL_OR_NEGATION_MARKER.search(later_sentence):
            if _same_subject_withdrawal(
                later_sentence,
                subject_name=subject_name,
                binding=binding,
            ):
                issues.append(
                    "later source sentence withdraws the same subject assertion"
                )
            else:
                issues.append(
                    "later source sentence contains an explicit withdrawal/negation marker"
                )
            break
    return issues


def _frame_matches(predicate: str, predicate_surface: str, binding: str, subject_surface: str) -> bool:
    patterns = _FRAME_PATTERNS.get(predicate, ())
    if not any(re.fullmatch(pattern, predicate_surface) for pattern in patterns):
        return False
    if binding != "trusted_speaker_1p":
        # Production intentionally does not infer a named first-person-style
        # copular predicate as a relation.  Keep this independent equivalent.
        first = predicate_surface.split(" ", 1)[0]
        return first not in {"am", "keen", "fond", "written"}
    if subject_surface not in {"i", "i'm", "i’m", "i've", "i’ve", "i'd", "i’d"}:
        return False
    if subject_surface == "i":
        allowed = _FIRST_PERSON_FRAMES.get(predicate, ())
    elif subject_surface in {"i'm", "i’m"}:
        allowed = (r"(?:keen\s+on|fond\s+of)",) if predicate == "likes" else ()
    else:
        # Perfect/contraction forms are deliberately narrower than plain I.
        allowed = tuple(
            pattern
            for pattern in _FIRST_PERSON_FRAMES.get(predicate, ())
            if any(token in pattern for token in ("ed", "had", "was", "were", "written", "took", "went"))
        )
    return any(re.fullmatch(pattern, predicate_surface) for pattern in allowed)


def _safe_direct_prefix(value: str) -> tuple[str | None, bool]:
    """Return wrapper and explicit ``still`` status for direct frames."""

    # The ordinary case is just the grammar's required separator.  Keep it
    # explicit instead of letting an optional appositive regex accidentally
    # consume bare whitespace as a malformed role phrase.
    if not value.strip():
        return "", False
    match = re.fullmatch(
        r"\s*(?:(?:,\s*(?:who\s+(?:is|was)\s+)?[a-z][a-z ]{0,80},)|(?:\s+(?:(?:is|was)\s+)?[a-z][a-z ]{0,80}))?\s*"
        r"(?:(used\s+to|formerly|previously|now)\s+)?"
        r"(?:(still)\s+)?",
        value,
    )
    if match is None:
        return None, False
    return match.group(1), bool(match.group(2))


def _semantic_state_for_direct(
    *, wrapper: str | None, suffix: str
) -> tuple[str, str | None] | None:
    suffix = suffix.strip()
    suffix = re.sub(r"(?:[.!。！])+?$", "", suffix).strip()
    has_instead = suffix == "instead"
    if suffix and not has_instead and not re.fullmatch(r"on\s+(?:weekdays|weekends)", suffix):
        return None
    if wrapper in {"used to", "formerly", "previously"}:
        return "historical", "previous"
    if wrapper == "now" and has_instead:
        return "update", "current"
    if has_instead:
        return None
    return "assert", None


def _semantic_witness_violations(row: Mapping[str, Any]) -> list[str]:
    """Re-derive a sidecar proof using only evaluator-owned rules."""

    issues: list[str] = []
    predicate = _normalise_identifier(row.get("edge_predicate"))
    spec_id = row.get("spec_id")
    binding = row.get("binding")
    state_change = row.get("state_change")
    temporal_status = row.get("temporal_status")
    if predicate not in _PREDICATE_SET:
        return ["edge predicate is outside the frozen controlled registry"]
    if not isinstance(spec_id, str) or spec_id not in FROZEN_SPEC_REGISTRY:
        return ["support spec_id is absent from the frozen registry"]
    spec = FROZEN_SPEC_REGISTRY[spec_id]
    if spec.predicate != predicate:
        issues.append("support spec predicate does not match edge predicate")
    if binding not in _BINDINGS:
        issues.append("support binding is not evaluator-recognized")
    if state_change not in _STATE_CHANGES:
        issues.append("support state_change is not controlled")
    if temporal_status is not None and temporal_status not in _TEMPORAL_STATUSES:
        issues.append("support temporal_status is not controlled")
    if (state_change, temporal_status) not in spec.state_status_pairs:
        issues.append("support state/status is incompatible with its frozen spec")
    if row.get("edge_state_change") != state_change:
        issues.append("support state_change differs from graph edge")
    if row.get("edge_temporal_status") != temporal_status:
        issues.append("support temporal_status differs from graph edge")
    if issues:
        return issues

    source = row.get("normalized_source")
    subject_span = _span_value(row, "subject")
    predicate_span = _span_value(row, "predicate")
    object_span = _span_value(row, "object")
    clause_span = _span_value(row, "clause")
    if not isinstance(source, str) or not all((subject_span, predicate_span, object_span, clause_span)):
        return ["support semantics cannot load normalized spans"]
    subject_surface = source[subject_span[0]:subject_span[1]]
    predicate_surface = source[predicate_span[0]:predicate_span[1]]
    object_surface = source[object_span[0]:object_span[1]]
    clause = source[clause_span[0]:clause_span[1]]
    expected_subject = row.get("subject_name")
    expected_object = row.get("object_name")
    if not isinstance(expected_subject, str) or not isinstance(expected_object, str):
        return ["edge endpoint display/value is missing"]
    if binding == "named":
        if not _endpoint_surface_matches(subject_surface, expected_subject, subject=True):
            issues.append("subject span does not prove the stored subject")
    elif subject_surface not in {"i", "i'm", "i’m", "i've", "i’ve", "i'd", "i’d"}:
        issues.append("trusted first-person support has an invalid subject surface")
    if not _endpoint_surface_matches(object_surface, expected_object, subject=False):
        issues.append("object span does not prove the stored object")
    normalized_subject = normalize_witness_source(expected_subject)
    normalized_speaker = normalize_witness_source(row.get("raw_role"))
    if binding == "named":
        if normalize_witness_source(subject_surface) in {"i", "i'm", "i’m", "i've", "i’ve", "i'd", "i’d"}:
            issues.append("named support cannot use a first-person subject")
    elif normalized_subject != normalized_speaker:
        issues.append("trusted first-person binding does not match source speaker")
    if not _frame_matches(predicate, predicate_surface, str(binding), subject_surface):
        issues.append("predicate span does not match the frozen frame")

    # Every ordinary support is a full clause proof.  The explicit wrappers
    # below are the only exceptions; no lexical-distance fallback is allowed.
    if _UNTRUSTED_CLAIM_TOKENS.search(clause):
        issues.append("clause contains an untrusted/modal/reporting claim marker")
    if '"' in clause or "“" in clause or "”" in clause:
        issues.append("clause is quoted rather than an atomic direct assertion")

    prefix = source[clause_span[0]:subject_span[0]]
    between_subject_predicate = source[subject_span[1]:predicate_span[0]]
    between_predicate_object = source[predicate_span[1]:object_span[0]]
    suffix = source[object_span[1]:clause_span[1]]
    if spec.kind == "direct":
        trusted_time_prelude = bool(
            binding == "trusted_speaker_1p"
            and re.fullmatch(
                r"\s*(?:today|yesterday|last\s+(?:monday|tuesday|wednesday|"
                r"thursday|friday|saturday|sunday|week|weekend|night)|"
                r"on\s+(?:monday|tuesday|wednesday|thursday|friday|saturday|"
                r"sunday)|last\s+week)\s+",
                prefix,
            )
        )
        if prefix.strip() and not trusted_time_prelude:
            issues.append("direct spec has text before its subject")
        wrapper, _ = _safe_direct_prefix(between_subject_predicate)
        if wrapper is None:
            issues.append("direct spec has unsupported subject/predicate gap")
        if not re.fullmatch(r"\s+", between_predicate_object):
            issues.append("direct spec lacks a direct predicate/object complement")
        if wrapper is not None:
            expected_state = _semantic_state_for_direct(wrapper=wrapper, suffix=suffix)
            if expected_state is None:
                issues.append("direct spec has an unsupported suffix/state wrapper")
            elif expected_state != (state_change, temporal_status):
                issues.append("direct source semantics disagree with stored state")
    elif spec.kind == "correction":
        if not re.fullmatch(r"\s*correction:\s*", prefix):
            issues.append("correction spec lacks its closed correction prefix")
        if not re.fullmatch(r"\s+actually\s+", between_subject_predicate):
            issues.append("correction spec lacks its closed actual predicate wrapper")
        if not re.fullmatch(r"\s+", between_predicate_object):
            issues.append("correction spec lacks a direct complement")
        if not re.fullmatch(r"\s*,\s*not\s+[^,;.!?。！？]+\s*[.!。！]?\s*", suffix):
            issues.append("correction spec lacks its closed old-object suffix")
    elif spec.kind == "moved-from-to":
        if prefix.strip():
            issues.append("moved-from-to spec has text before its subject")
        if predicate_surface != "moved from":
            issues.append("moved-from-to spec has the wrong predicate surface")
        if not re.fullmatch(r"\s+[^,;.!?。！？]+\s+to\s+", between_predicate_object):
            issues.append("moved-from-to spec lacks an explicit old-to-new bridge")
        if not re.fullmatch(r"\s*[.!。！]?\s*", suffix):
            issues.append("moved-from-to spec has an unsupported suffix")
    elif spec.kind == "retraction":
        if prefix.strip():
            issues.append("retraction spec has text before its subject")
        if not re.fullmatch(r"(?:am|is|are|was|were)\s+no\s+longer\s+(?:a\s+)?member\s+of", predicate_surface):
            issues.append("retraction spec has the wrong predicate surface")
        if not re.fullmatch(r"\s+", between_predicate_object):
            issues.append("retraction spec lacks a direct complement")
        if not re.fullmatch(r"\s*;\s*the\s+earlier\s+statement\s+is\s+(?:retracted|revoked)\s*[.!。！]?\s*", suffix):
            issues.append("retraction spec lacks its closed withdrawal suffix")
    else:
        # Coordinated witnesses are only a closed assert/null extension.  The
        # stored spans must still prove one direct frame, and an explicit
        # coordination is required so a tampered spec cannot relabel direct
        # evidence as a multi-clause proof.
        if " and " not in clause:
            issues.append("coordinated spec has no coordination boundary")
        if any(token in clause for token in (";", " but ", " however ", " not ", " never ")):
            issues.append("coordinated spec contains a non-atomic contradiction boundary")
        if not re.fullmatch(r"\s+", between_predicate_object):
            issues.append("coordinated spec lacks a direct predicate/object complement")
    return issues


_SUPPORT_REQUIRED_COLUMNS = frozenset(
    {
        "id",
        "edge_id",
        "user_id",
        "source_message_id",
        "support_schema_version",
        "normalization_id",
        "spec_id",
        "binding",
        "source_start",
        "source_end",
        "clause_start",
        "clause_end",
        "subject_start",
        "subject_end",
        "predicate_start",
        "predicate_end",
        "object_start",
        "object_end",
        "state_change",
        "temporal_status",
        "source_span_sha256",
    }
)

_EDGE_REQUIRED_COLUMNS = frozenset(
    {
        "id",
        "user_id",
        "subject_entity_id",
        "predicate",
        "object_entity_id",
        "object_value",
        "source_message_id",
        "state_change",
        "temporal_status",
    }
)


def _table_exists(connection: sqlite3.Connection, table_name: str) -> bool:
    return connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?", (table_name,)
    ).fetchone() is not None


def _table_columns(connection: sqlite3.Connection, table_name: str) -> set[str]:
    return {
        str(row["name"])
        for row in connection.execute(f"PRAGMA table_info({table_name})").fetchall()
    }


def _unique_indexes(connection: sqlite3.Connection, table_name: str) -> set[tuple[str, ...]]:
    indexes: set[tuple[str, ...]] = set()
    for index in connection.execute(f"PRAGMA index_list({table_name})").fetchall():
        if not int(index["unique"]):
            continue
        index_name = str(index["name"]).replace('"', '""')
        indexes.add(
            tuple(
                str(column["name"])
                for column in connection.execute(
                    f'PRAGMA index_info("{index_name}")'
                ).fetchall()
            )
        )
    return indexes


def _foreign_key_shapes(
    connection: sqlite3.Connection, table_name: str
) -> set[tuple[str, tuple[str, ...], tuple[str, ...]]]:
    grouped: dict[int, list[sqlite3.Row]] = {}
    for row in connection.execute(f"PRAGMA foreign_key_list({table_name})").fetchall():
        grouped.setdefault(int(row["id"]), []).append(row)
    result = set()
    for rows in grouped.values():
        ordered = sorted(rows, key=lambda item: int(item["seq"]))
        result.add(
            (
                str(ordered[0]["table"]),
                tuple(str(item["from"]) for item in ordered),
                tuple(str(item["to"]) for item in ordered),
            )
        )
    return result


def _normalized_table_sql(connection: sqlite3.Connection, table_name: str) -> str:
    row = connection.execute(
        "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = ?", (table_name,)
    ).fetchone()
    return re.sub(r"\s+", " ", str(row["sql"] if row else "")).upper()


def _support_schema_violations(connection: sqlite3.Connection) -> list[str]:
    violations: list[str] = []
    required_tables = ("raw_messages", "graph_entities", "graph_edges", "graph_edge_support")
    for table_name in required_tables:
        if not _table_exists(connection, table_name):
            violations.append(f"missing required witness table {table_name}")
    if violations:
        return violations
    edge_columns = _table_columns(connection, "graph_edges")
    missing_edge_columns = sorted(_EDGE_REQUIRED_COLUMNS - edge_columns)
    if missing_edge_columns:
        violations.append(f"graph_edges missing columns: {missing_edge_columns}")
        return violations
    columns = _table_columns(connection, "graph_edge_support")
    missing = sorted(_SUPPORT_REQUIRED_COLUMNS - columns)
    if missing:
        violations.append(f"graph_edge_support missing columns: {missing}")
        return violations
    required_unique = ("edge_id", "user_id", "source_message_id")
    if required_unique not in _unique_indexes(connection, "graph_edge_support"):
        violations.append("graph_edge_support lacks one-to-one composite unique index")
    expected_foreign_keys = {
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
    }
    missing_fks = expected_foreign_keys - _foreign_key_shapes(connection, "graph_edge_support")
    if missing_fks:
        violations.append("graph_edge_support lacks composite provenance foreign keys")
    graph_edge_unique = ("id", "user_id", "source_message_id")
    if graph_edge_unique not in _unique_indexes(connection, "graph_edges"):
        violations.append("graph_edges lacks parent unique index for support provenance")
    edge_foreign_keys = {
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
    }
    if not edge_foreign_keys.issubset(_foreign_key_shapes(connection, "graph_edges")):
        violations.append("graph_edges lacks entity/raw provenance foreign keys")
    required_checks = {
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
            "CHECK( SUBJECT_END <= PREDICATE_START AND PREDICATE_END <= OBJECT_START )",
        ),
    }
    for table_name, fragments in required_checks.items():
        schema_sql = _normalized_table_sql(connection, table_name)
        # SQLite's pretty-printer preserves semantically relevant whitespace
        # inside a few multi-line checks.  Compare a fully compact form too so
        # the audit validates the constraint, not formatting.
        compact_sql = schema_sql.replace(" ", "")
        for fragment in fragments:
            if fragment not in schema_sql and fragment.replace(" ", "") not in compact_sql:
                violations.append(
                    f"{table_name} lacks frozen witness CHECK constraint: {fragment}"
                )
    if connection.execute("PRAGMA foreign_key_check").fetchone() is not None:
        violations.append("SQLite foreign_key_check reports a provenance violation")
    return violations


def _iter_edge_support_rows(connection: sqlite3.Connection, user_id: str | None) -> list[sqlite3.Row]:
    query = """
        SELECT edge.id AS edge_id,
               edge.user_id AS edge_user_id,
               edge.predicate AS edge_predicate,
               edge.object_value AS edge_object_value,
               edge.source_message_id AS edge_source_message_id,
               edge.state_change AS edge_state_change,
               edge.temporal_status AS edge_temporal_status,
               edge.subject_entity_id AS edge_subject_entity_id,
               edge.object_entity_id AS edge_object_entity_id,
               support.id AS support_id,
               support.edge_id AS support_edge_id,
               support.user_id AS support_user_id,
               support.source_message_id AS support_source_message_id,
               support.support_schema_version AS support_schema_version,
               support.normalization_id AS normalization_id,
               support.spec_id AS spec_id,
               support.binding AS binding,
               support.source_start AS source_start,
               support.source_end AS source_end,
               support.clause_start AS clause_start,
               support.clause_end AS clause_end,
               support.subject_start AS subject_start,
               support.subject_end AS subject_end,
               support.predicate_start AS predicate_start,
               support.predicate_end AS predicate_end,
               support.object_start AS object_start,
               support.object_end AS object_end,
               support.state_change AS state_change,
               support.temporal_status AS temporal_status,
               support.source_span_sha256 AS source_span_sha256,
               raw.user_id AS raw_user_id,
               raw.content AS raw_content,
               raw.role AS raw_role,
               subject.user_id AS subject_user_id,
               subject.display_name AS subject_name,
               object.user_id AS object_user_id,
               object.display_name AS object_entity_name
          FROM graph_edges AS edge
          LEFT JOIN graph_edge_support AS support
            ON support.edge_id = edge.id
           AND support.user_id = edge.user_id
           AND support.source_message_id = edge.source_message_id
          LEFT JOIN raw_messages AS raw
            ON raw.id = edge.source_message_id
          LEFT JOIN graph_entities AS subject
            ON subject.id = edge.subject_entity_id
          LEFT JOIN graph_entities AS object
            ON object.id = edge.object_entity_id
         WHERE (? IS NULL OR edge.user_id = ?)
         ORDER BY edge.id
    """
    return connection.execute(query, (user_id, user_id)).fetchall()


def _edge_support_record(row: sqlite3.Row) -> tuple[bool, list[str], dict[str, Any] | None]:
    """Validate one graph edge and return its DB-authoritative trace shape."""

    record = dict(row)
    edge_id = int(record["edge_id"])
    prefix = f"edge_{edge_id}: "
    violations: list[str] = []
    if record.get("support_id") is None:
        violations.append(prefix + "missing one-to-one support witness")
        return False, violations, None
    if record.get("support_edge_id") != record.get("edge_id"):
        violations.append(prefix + "support edge_id does not bind the graph edge")
    if record.get("support_user_id") != record.get("edge_user_id"):
        violations.append(prefix + "support user_id does not bind the graph edge")
    if record.get("support_source_message_id") != record.get("edge_source_message_id"):
        violations.append(prefix + "support source_message_id does not bind the graph edge")
    if record.get("raw_user_id") != record.get("edge_user_id"):
        violations.append(prefix + "raw source user does not bind the graph edge")
    if record.get("subject_user_id") != record.get("edge_user_id") or not record.get("subject_name"):
        violations.append(prefix + "subject entity provenance is invalid")
    if record.get("edge_object_entity_id") is not None:
        if record.get("object_user_id") != record.get("edge_user_id") or not record.get("object_entity_name"):
            violations.append(prefix + "object entity provenance is invalid")
        object_name: object = record.get("object_entity_name")
    else:
        object_name = record.get("edge_object_value")
    if not isinstance(object_name, str) or not object_name.strip():
        violations.append(prefix + "graph edge has no valid object endpoint")
    record["object_name"] = object_name
    normalized_source = normalize_witness_source(record.get("raw_content"))
    if not normalized_source:
        violations.append(prefix + "raw source cannot be normalized")
        return False, violations, None
    record["normalized_source"] = normalized_source
    if record.get("support_schema_version") != SUPPORT_SCHEMA_VERSION:
        violations.append(prefix + "support schema version is not frozen version 1")
    if record.get("normalization_id") != NORMALIZATION_ID:
        violations.append(prefix + "support normalization ID is not evaluator-owned v1")
    source_span = _span_value(record, "source")
    clause_span = _span_value(record, "clause")
    subject_span = _span_value(record, "subject")
    predicate_span = _span_value(record, "predicate")
    object_span = _span_value(record, "object")
    spans = {
        "source": source_span,
        "clause": clause_span,
        "subject": subject_span,
        "predicate": predicate_span,
        "object": object_span,
    }
    for label, span in spans.items():
        if span is None:
            violations.append(prefix + f"{label} span is not integer half-open data")
    if all(span is not None for span in spans.values()):
        assert source_span is not None
        assert clause_span is not None
        assert subject_span is not None
        assert predicate_span is not None
        assert object_span is not None
        source_length = len(normalized_source)
        if source_span != (0, source_length):
            violations.append(prefix + "source span is not the exact normalized message")
        if not _valid_span(clause_span, lower=source_span[0], upper=source_span[1]):
            violations.append(prefix + "clause span is not a valid absolute half-open span")
        for label, span in (("subject", subject_span), ("predicate", predicate_span), ("object", object_span)):
            if not _valid_span(span, lower=clause_span[0], upper=clause_span[1]):
                violations.append(prefix + f"{label} span is outside the claimed clause")
        if not (
            subject_span[0] < subject_span[1] <= predicate_span[0] < predicate_span[1] <= object_span[0] < object_span[1]
        ):
            violations.append(prefix + "S/P/O spans are not source-ordered")
        expected_hash = normalized_source_span_sha256(
            normalized_source[source_span[0]:source_span[1]]
        )
        actual_hash = record.get("source_span_sha256")
        if not isinstance(actual_hash, str) or actual_hash.casefold() != expected_hash:
            violations.append(prefix + "normalized source span SHA-256 mismatch")
        subject_name = record.get("subject_name")
        if isinstance(subject_name, str):
            violations.extend(
                prefix + item
                for item in _source_scope_violations(
                    source=normalized_source,
                    clause_span=clause_span,
                    subject_name=subject_name,
                    binding=str(record.get("binding", "")),
                    raw_role=record.get("raw_role"),
                )
            )
        else:
            violations.append(prefix + "source-level scope cannot load subject")
    semantic_issues = _semantic_witness_violations(record)
    violations.extend(prefix + item for item in semantic_issues)
    valid = not violations
    trace_witness: dict[str, Any] | None = None
    if all(span is not None for span in spans.values()):
        assert source_span is not None
        assert clause_span is not None
        assert subject_span is not None
        assert predicate_span is not None
        assert object_span is not None
        trace_witness = {
            "support_id": int(record["support_id"]),
            "support_schema_version": record.get("support_schema_version"),
            "support_normalization_id": record.get("normalization_id"),
            "support_spec_id": record.get("spec_id"),
            "support_binding": record.get("binding"),
            "support_source_span": _span_list(source_span),
            "support_clause_span": _span_list(clause_span),
            "support_subject_span": _span_list(subject_span),
            "support_predicate_span": _span_list(predicate_span),
            "support_object_span": _span_list(object_span),
            "support_state_change": record.get("state_change"),
            "support_temporal_status": record.get("temporal_status"),
            "support_source_span_sha256": record.get("source_span_sha256"),
        }
    return valid, violations, trace_witness


def audit_persisted_graph_support(
    database_path: str | Path, *, user_id: str | None = None
) -> dict[str, Any]:
    """Audit every persisted edge/support pair in a prepared database.

    This deliberately audits all graph rows in scope, rather than only paths
    that happened to be retrieved in one query.  The report is plain data so
    callers can persist counts/invariants in a prepared sidecar without
    trusting an in-memory production object.
    """

    path = Path(database_path)
    if not path.exists():
        raise FileNotFoundError(path)
    connection = sqlite3.connect(str(path))
    connection.row_factory = sqlite3.Row
    try:
        schema_violations = _support_schema_violations(connection)
        if schema_violations:
            return {
                "edge_count": 0,
                "support_count": 0,
                "supported_edge_count": 0,
                "unsupported_edge_count": 0,
                "all_edges_witnessed": False,
                "unsupported_edge_ids": [],
                "edge_validity": {},
                "violations": schema_violations,
            }
        rows = _iter_edge_support_rows(connection, user_id)
        support_count = int(
            connection.execute(
                "SELECT COUNT(*) FROM graph_edge_support WHERE (? IS NULL OR user_id = ?)",
                (user_id, user_id),
            ).fetchone()[0]
        )
        orphan_count = int(
            connection.execute(
                """SELECT COUNT(*)
                       FROM graph_edge_support AS support
                       LEFT JOIN graph_edges AS edge
                         ON edge.id = support.edge_id
                        AND edge.user_id = support.user_id
                        AND edge.source_message_id = support.source_message_id
                      WHERE (? IS NULL OR support.user_id = ?)
                        AND edge.id IS NULL""",
                (user_id, user_id),
            ).fetchone()[0]
        )
        violations: list[str] = []
        if orphan_count:
            violations.append(f"graph_edge_support has {orphan_count} orphan provenance rows")
        if support_count != len(rows):
            violations.append(
                "graph_edge_support row count does not equal graph edge count "
                f"({support_count} != {len(rows)})"
            )
        edge_validity: dict[str, dict[str, Any]] = {}
        unsupported_ids: list[str] = []
        supported_count = 0
        for row in rows:
            valid, row_violations, trace_witness = _edge_support_record(row)
            edge_key = f"edge_{int(row['edge_id'])}"
            edge_validity[edge_key] = {
                "valid": valid,
                "support_id": int(row["support_id"]) if row["support_id"] is not None else None,
                "source_message_id": f"mem_{int(row['edge_source_message_id'])}",
                "predicate": str(row["edge_predicate"]),
                "subject": row["subject_name"],
                "object": row["object_entity_name"] or row["edge_object_value"] or "",
                "trace_witness": trace_witness,
            }
            if valid:
                supported_count += 1
            else:
                unsupported_ids.append(edge_key)
                violations.extend(row_violations)
        edge_count = len(rows)
        return {
            "edge_count": edge_count,
            "support_count": support_count,
            "supported_edge_count": supported_count,
            "unsupported_edge_count": len(unsupported_ids),
            "all_edges_witnessed": (
                not violations
                and edge_count == support_count == supported_count
                and not unsupported_ids
            ),
            "unsupported_edge_ids": unsupported_ids,
            "edge_validity": edge_validity,
            "violations": violations,
        }
    finally:
        connection.close()


def _trace_value_matches(expected: object, observed: object) -> bool:
    if isinstance(expected, list):
        return isinstance(observed, list) and observed == expected
    return observed == expected


def audit_graph_trace_paths(
    database_path: str | Path,
    *,
    user_id: str,
    paths: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Reload and compare each trace witness to authoritative DB support.

    Trace values are diagnostics only.  A tampered trace cannot certify a
    traversal because this function first recomputes each edge's validity from
    the persisted support row, then requires every copied trace field to match
    that freshly loaded record.
    """

    persisted = audit_persisted_graph_support(database_path, user_id=user_id)
    edge_validity = persisted["edge_validity"]
    violations: list[str] = []
    unsupported_edge_ids: list[str] = []
    seen_path_ids: set[str] = set()
    verified_paths = 0
    required_trace_fields = (
        "support_id",
        "support_schema_version",
        "support_normalization_id",
        "support_spec_id",
        "support_binding",
        "support_source_span",
        "support_clause_span",
        "support_subject_span",
        "support_predicate_span",
        "support_object_span",
        "support_state_change",
        "support_temporal_status",
        "support_source_span_sha256",
    )
    for path in paths:
        path_id = path.get("path_id")
        if not isinstance(path_id, str) or not re.fullmatch(r"edge_[0-9]+", path_id):
            violations.append(f"unverifiable graph path ID {path_id!r}")
            continue
        if path_id in seen_path_ids:
            violations.append(f"duplicate graph path ID {path_id!r}")
        seen_path_ids.add(path_id)
        authoritative = edge_validity.get(path_id)
        if authoritative is None:
            violations.append(f"missing persisted edge/support for {path_id}")
            unsupported_edge_ids.append(path_id)
            continue
        if not authoritative["valid"]:
            violations.append(f"unsupported traversed edge {path_id}")
            unsupported_edge_ids.append(path_id)
        expected_witness = authoritative.get("trace_witness")
        if not isinstance(expected_witness, Mapping):
            violations.append(f"missing authoritative witness fields for {path_id}")
            unsupported_edge_ids.append(path_id)
            continue
        for field in required_trace_fields:
            if field not in path:
                violations.append(f"trace witness missing {field} for {path_id}")
            elif not _trace_value_matches(expected_witness[field], path[field]):
                violations.append(f"trace witness mismatch for {path_id}: {field}")
        source_ids = path.get("source_message_ids")
        if source_ids != [authoritative["source_message_id"]]:
            violations.append(f"source mismatch for {path_id}")
        if path.get("relations") != [authoritative["predicate"]]:
            violations.append(f"relation mismatch for {path_id}")
        if path.get("subject") != authoritative["subject"]:
            violations.append(f"subject mismatch for {path_id}")
        if path.get("object") != authoritative["object"]:
            violations.append(f"object mismatch for {path_id}")
        try:
            hop_count = int(path.get("hop_count"))
        except (TypeError, ValueError):
            hop_count = 0
        if hop_count != 1:
            violations.append(f"non-one-hop graph path {path_id}")
        if authoritative["valid"]:
            verified_paths += 1
    return {
        "traversed_edges": len(paths),
        "supported_traversed_edges": verified_paths,
        "unsupported_traversed_edges": len(set(unsupported_edge_ids)),
        "unsupported_edge_ids": sorted(set(unsupported_edge_ids)),
        "persisted_edge_count": persisted["edge_count"],
        "persisted_support_count": persisted["support_count"],
        "persisted_all_edges_witnessed": persisted["all_edges_witnessed"],
        "persisted_violations": list(persisted["violations"]),
        "violations": violations,
    }
