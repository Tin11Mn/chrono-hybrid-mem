"""Strict, model-independent contracts for the P3 evidence graph.

This module deliberately does not know how to call a model, persist a graph, or
answer a query.  Its only job is to turn an untrusted graph-extraction payload
into a small set of provenance-carrying entities and relations.  Parsing is
fail-closed: malformed or ambiguous data is omitted rather than guessed.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import re
import unicodedata
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple


MAX_ENTITIES_PER_PAYLOAD = 20
MAX_RELATIONS_PER_PAYLOAD = 20
MAX_ENTITY_NAME_CHARS = 128
MAX_ENTITY_NAME_WORDS = 12
MAX_IDENTIFIER_CHARS = 256
MAX_CREATED_AT_CHARS = 64
MAX_CONTROLLED_TOKEN_CHARS = 32


ENTITY_TYPES = frozenset(
    {
        "person",
        "organization",
        "location",
        "event",
        "activity",
        "object",
        "product",
        "food",
        "document",
        "rule",
        "topic",
        "group",
    }
)

PREDICATES = frozenset(
    {
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
    }
)

STATE_CHANGES = frozenset(
    {"assert", "update", "correction", "retraction", "historical"}
)

TEMPORAL_STATUSES = frozenset({"current", "previous", "historical", "future"})


@dataclass(frozen=True)
class Entity:
    """An evidence-graph entity with a scoped, non-semantic identity."""

    entity_id: str
    user_id: str
    canonical_name: str
    display_name: str
    entity_type: str
    first_source_message_id: str
    created_at: str
    identity_hint: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# A descriptive alias for callers that prefer the longer name.
GraphEntity = Entity


@dataclass(frozen=True)
class GraphRelation:
    """A controlled relation whose evidence is one original source message."""

    relation_id: str
    user_id: str
    subject_entity_id: str
    predicate: str
    object_entity_id: str
    source_message_id: str
    state_change: str
    temporal_status: Optional[str]
    event_ts: Optional[int]
    supersedes_edge_id: Optional[str]
    created_at: str
    explicit: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EntityResolution:
    """Result of conservative entity linking.

    ``entity`` is populated only for the ``resolved`` state.  In particular,
    callers cannot accidentally select the first of multiple same-name
    candidates.
    """

    status: str
    entity: Optional[Entity]
    candidate_ids: Tuple[str, ...] = ()

    @property
    def resolved(self) -> bool:
        return self.status == "resolved" and self.entity is not None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "entity": self.entity.to_dict() if self.entity else None,
            "candidate_ids": list(self.candidate_ids),
        }


@dataclass(frozen=True)
class GraphPayload:
    """Sanitized graph extraction for exactly one source message."""

    entities: Tuple[Entity, ...] = ()
    relations: Tuple[GraphRelation, ...] = ()
    dropped_entities: int = 0
    dropped_relations: int = 0
    ambiguous_relations: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entities": [entity.to_dict() for entity in self.entities],
            "relations": [relation.to_dict() for relation in self.relations],
            "dropped_entities": self.dropped_entities,
            "dropped_relations": self.dropped_relations,
            "ambiguous_relations": self.ambiguous_relations,
        }


_CONTROLLED_SEPARATOR_PATTERN = re.compile(r"[\s-]+", re.UNICODE)


def _strip_surrounding_punctuation(value: str) -> str:
    """Remove punctuation only at the outside of a display name."""

    previous = None
    while value and value != previous:
        previous = value
        value = value.strip()
        while value and unicodedata.category(value[0]).startswith("P"):
            value = value[1:].lstrip()
        while value and unicodedata.category(value[-1]).startswith("P"):
            value = value[:-1].rstrip()
    return value


def _display_entity_name(value: Any) -> Optional[str]:
    if not isinstance(value, str) or len(value) > MAX_ENTITY_NAME_CHARS:
        return None
    value = unicodedata.normalize("NFKC", value)
    value = " ".join(value.split())
    value = _strip_surrounding_punctuation(value)
    value = " ".join(value.split())
    if not value or len(value) > MAX_ENTITY_NAME_CHARS:
        return None
    if len(value.split()) > MAX_ENTITY_NAME_WORDS:
        return None
    if any(
        unicodedata.category(character).startswith("C")
        for character in value
    ):
        return None
    return value


def normalize_entity_name(value: Any) -> Optional[str]:
    """Normalize a name for conservative exact-name linking.

    NFKC, whitespace trimming/collapse, surrounding-punctuation removal, and
    case folding are intentional.  No fuzzy matching, honorific removal, or
    partial-name matching is performed.
    """

    display_name = _display_entity_name(value)
    if display_name is None:
        return None
    return display_name.casefold()


def _normalize_controlled_token(
    value: Any, vocabulary: frozenset[str]
) -> Optional[str]:
    if not isinstance(value, str) or not value:
        return None
    if len(value) > MAX_CONTROLLED_TOKEN_CHARS:
        return None
    normalized = unicodedata.normalize("NFKC", value).strip().casefold()
    normalized = _CONTROLLED_SEPARATOR_PATTERN.sub("_", normalized)
    if len(normalized) > MAX_CONTROLLED_TOKEN_CHARS:
        return None
    return normalized if normalized in vocabulary else None


def normalize_entity_type(value: Any) -> Optional[str]:
    return _normalize_controlled_token(value, ENTITY_TYPES)


def normalize_predicate(value: Any) -> Optional[str]:
    return _normalize_controlled_token(value, PREDICATES)


_PREFERENCE_PREDICATES = frozenset({"likes", "prefers", "dislikes"})
_RULE_PREDICATES = frozenset({"requires", "prohibits", "permits"})
_FOOD_OR_DRINK_WORDS = frozenset(
    {
        "beverage",
        "beer",
        "biscuit",
        "bread",
        "broth",
        "cake",
        "candy",
        "cheese",
        "chocolate",
        "cocoa",
        "cocktail",
        "coffee",
        "cookie",
        "cookies",
        "curry",
        "dessert",
        "dinner",
        "dish",
        "drink",
        "dumpling",
        "dumplings",
        "food",
        "fruit",
        "jam",
        "jelly",
        "juice",
        "lunch",
        "meal",
        "melon",
        "milk",
        "noodle",
        "noodles",
        "pasta",
        "pie",
        "pizza",
        "rice",
        "salad",
        "sandwich",
        "sauce",
        "smoothie",
        "snack",
        "soda",
        "soup",
        "stew",
        "tart",
        "tea",
        "water",
        "wine",
        "yogurt",
    }
)

_GROUP_HEAD_WORDS = frozenset(
    {
        "association",
        "circle",
        "club",
        "collective",
        "committee",
        "community",
        "council",
        "forum",
        "guild",
        "league",
        "network",
        "society",
        "team",
        "union",
    }
)
_RULE_HEAD_WORDS = frozenset(
    {
        "code",
        "guideline",
        "guidelines",
        "manual",
        "policy",
        "protocol",
        "regulation",
        "regulations",
        "rule",
        "rules",
        "standard",
        "standards",
    }
)
_HUMAN_ROLE_HEAD_WORDS = frozenset(
    {
        "accountant",
        "administrator",
        "analyst",
        "architect",
        "artist",
        "attorney",
        "author",
        "baker",
        "chemist",
        "chef",
        "consultant",
        "curator",
        "designer",
        "developer",
        "director",
        "doctor",
        "editor",
        "educator",
        "engineer",
        "journalist",
        "lawyer",
        "librarian",
        "manager",
        "mechanic",
        "musician",
        "nurse",
        "pharmacist",
        "photographer",
        "physician",
        "pilot",
        "professor",
        "programmer",
        "researcher",
        "scientist",
        "teacher",
        "technician",
        "writer",
    }
)


def _semantic_head_word(value: Any) -> Optional[str]:
    canonical_name = normalize_entity_name(value)
    if canonical_name is None:
        return None
    words = re.findall(r"[^\W_]+", canonical_name, flags=re.UNICODE)
    return words[-1] if words else None


def _strip_grammatical_leading_article(value: Any) -> Optional[str]:
    """Strip only a visibly grammatical, lowercase English article.

    This helper is intentionally not part of global entity normalization.  A
    lowercase article in an extraction such as ``the Cycling Code`` is often
    sentence grammar, while a capitalized article in a proper name such as
    ``The Beatles`` is part of the name and must be retained.
    """

    display_name = _display_entity_name(value)
    if display_name is None:
        return None
    for article in ("the ", "an ", "a "):
        if display_name.startswith(article):
            return normalize_entity_name(display_name[len(article):])
    return normalize_entity_name(display_name)


def normalize_relation_object_type(
    predicate: Any, object_name: Any, object_type: Any
) -> Optional[str]:
    """Conservatively repair a narrow food-vs-product model ambiguity.

    ``product`` remains the default for manufactured goods.  It is changed to
    ``food`` only when a preference relation targets a name containing a
    generic food or drink head word.  No fuzzy, brand, or world-knowledge
    classification is attempted.
    """

    normalized_type = normalize_entity_type(object_type)
    normalized_predicate = normalize_predicate(predicate)
    canonical_name = normalize_entity_name(object_name)
    if (
        normalized_type != "product"
        or normalized_predicate not in _PREFERENCE_PREDICATES
        or canonical_name is None
    ):
        return normalized_type
    words = re.findall(r"[^\W_]+", canonical_name, flags=re.UNICODE)
    # Use only the English semantic head (the final token), so names such as
    # "coffee maker" and "water bottle" remain manufactured products.
    return (
        "food"
        if words and words[-1] in _FOOD_OR_DRINK_WORDS
        else normalized_type
    )


def normalize_state_change(value: Any) -> Optional[str]:
    return _normalize_controlled_token(value, STATE_CHANGES)


def normalize_temporal_status(value: Any) -> Optional[str]:
    if value is None:
        return None
    return _normalize_controlled_token(value, TEMPORAL_STATUSES)


_PEOPLE = frozenset({"person"})
_ORGANIZATIONS = frozenset({"organization", "group"})
_PLACES = frozenset({"location"})
_EVENTS = frozenset({"event", "activity"})
_OWNABLE = frozenset(
    {"organization", "group", "object", "product", "document"}
)
_CREATABLE = frozenset(
    {"event", "activity", "object", "product", "document", "rule", "topic"}
)
_RULE_TARGETS = frozenset(
    {"event", "activity", "object", "product", "document", "rule", "topic"}
)
_PREFERENCE_SUBJECTS = frozenset({"person", "organization", "group"})


# ``None`` means all controlled entity types are accepted at that endpoint.
PREDICATE_ENDPOINT_TYPES: Mapping[
    str, Tuple[Optional[frozenset[str]], Optional[frozenset[str]]]
] = {
    "friend_of": (_PEOPLE, _PEOPLE),
    "parent_of": (_PEOPLE, _PEOPLE),
    "sibling_of": (_PEOPLE, _PEOPLE),
    "partner_of": (_PEOPLE, _PEOPLE),
    "works_at": (_PEOPLE, _ORGANIZATIONS),
    "role_at": (_PEOPLE, _ORGANIZATIONS),
    "lives_in": (_PEOPLE, _PLACES),
    "located_in": (
        frozenset(
            {"organization", "group", "event", "activity", "object", "product"}
        ),
        _PLACES,
    ),
    "likes": (_PREFERENCE_SUBJECTS, None),
    "prefers": (_PREFERENCE_SUBJECTS, None),
    "dislikes": (_PREFERENCE_SUBJECTS, None),
    "member_of": (_PEOPLE, _ORGANIZATIONS),
    "participated_in": (
        frozenset({"person", "organization", "group"}),
        _EVENTS,
    ),
    "owns": (frozenset({"person", "organization", "group"}), _OWNABLE),
    "created": (frozenset({"person", "organization", "group"}), _CREATABLE),
    "requires": (frozenset({"rule", "document", "organization"}), _RULE_TARGETS),
    "prohibits": (frozenset({"rule", "document", "organization"}), _RULE_TARGETS),
    "permits": (frozenset({"rule", "document", "organization"}), _RULE_TARGETS),
    # ``changed_to`` and ``replaces`` additionally require the same endpoint
    # type in ``endpoint_types_are_compatible`` below.
    "changed_to": (None, None),
    "replaces": (None, None),
}


_ROLE_MODIFIER_WORDS = (
    "assistant", "associate", "chief", "graphic", "lead", "licensed",
    "principal", "product", "professional", "research", "senior",
    "software", "staff",
)


def _role_np_frame_pattern() -> str:
    role_heads = "|".join(
        re.escape(role) for role in sorted(_HUMAN_ROLE_HEAD_WORDS)
    )
    modifiers = "|".join(
        re.escape(modifier) for modifier in _ROLE_MODIFIER_WORDS
    )
    return (
        r"(?:(?:the|an?|their)\s+)?"
        r"(?:(?:{})\s+){{0,2}}(?:{})"
    ).format(modifiers, role_heads)


@dataclass(frozen=True)
class SupportWitness:
    """A complete, local surface proof for one candidate relation.

    Offsets are measured in the normalized clause/source used by the verifier.
    They are deliberately diagnostic only: graph persistence continues to use
    the immutable source-message id as provenance.
    """

    spec_id: str
    clause: str
    subject_span: Tuple[int, int]
    predicate_span: Tuple[int, int]
    object_span: Tuple[int, int]
    binding: str
    state_change: str
    temporal_status: Optional[str]
    source_span: Tuple[int, int]


# Every accepted predicate is a closed, governed S-P-O frame.  ``{object}``
# is replaced with the candidate endpoint itself; there is no wildcard gap
# between the predicate and its direct complement.
_PREDICATE_FRAME_CORES: Mapping[str, Tuple[str, ...]] = {
    "friend_of": (
        r"(?P<predicate>(?:am|is|are|was|were)\s+(?:a\s+)?friend\s+of)\s+{object}",
    ),
    "parent_of": (
        r"(?P<predicate>(?:am|is|are|was|were)\s+(?:a\s+)?(?:parent|mother|father)\s+of)\s+{object}",
    ),
    "sibling_of": (
        r"(?P<predicate>(?:am|is|are|was|were)\s+(?:a\s+)?(?:sibling|brother|sister)\s+of)\s+{object}",
    ),
    "partner_of": (
        r"(?P<predicate>(?:am|is|are|was|were)\s+(?:a\s+)?(?:partner|spouse|husband|wife)\s+of)\s+{object}",
    ),
    "works_at": (
        r"(?P<predicate>(?:work|works|worked)\s+(?:at|for))\s+{object}",
        r"(?P<predicate>(?:am|is|are|was|were)\s+employed\s+by)\s+{object}",
        r"(?P<predicate>(?:work|works|worked)\s+as\s+{role}\s+at)\s+{object}",
    ),
    "role_at": (
        r"(?P<predicate>(?:have|has|had|hold|holds|held)\s+(?:an?\s+|the\s+)?(?:formal\s+)?(?:role|position)\s+at)\s+{object}",
        r"(?P<predicate>(?:serve|serves|served)\s+(?:in\s+)?{role}\s+at)\s+{object}",
    ),
    "lives_in": (
        r"(?P<predicate>(?:live|lives|lived)\s+in)\s+{object}",
    ),
    "located_in": (
        r"(?P<predicate>(?:am|is|are|was|were)\s+(?:located|based|situated)\s+in)\s+{object}",
    ),
    "likes": (
        r"(?P<predicate>(?:like|likes|liked|love|loves|loved|enjoy|enjoys|enjoyed))\s+{object}",
        r"(?P<predicate>(?:am|is|are|was|were)\s+(?:keen\s+on|fond\s+of))\s+{object}",
        r"(?P<predicate>(?:keen\s+on|fond\s+of))\s+{object}",
        r"(?P<predicate>(?:have|has|had)\s+(?:always\s+)?(?:had\s+)?(?:a\s+)?love\s+(?:of|for))\s+{object}",
    ),
    "prefers": (
        r"(?P<predicate>(?:prefer|prefers|preferred))\s+{object}",
    ),
    "dislikes": (
        r"(?P<predicate>(?:dislike|dislikes|disliked|hate|hates|hated))\s+{object}",
    ),
    "member_of": (
        r"(?P<predicate>(?:am|is|are|was|were|became)\s+(?:a\s+)?member\s+of)\s+{object}",
        r"(?P<predicate>(?:belong|belongs|belonged)\s+to)\s+{object}",
        r"(?P<predicate>(?:join|joins|joined))\s+{object}",
    ),
    "participated_in": (
        r"(?P<predicate>(?:participate|participates|participated)\s+in)\s+{object}",
        r"(?P<predicate>(?:take|takes|took)\s+part\s+in)\s+{object}",
        r"(?P<predicate>(?:attend|attends|attended))\s+{object}",
        r"(?P<predicate>(?:join|joins|joined))\s+{object}",
        r"(?P<predicate>went\s+to)\s+{object}",
    ),
    "owns": (
        r"(?P<predicate>(?:own|owns|owned|possess|possesses|possessed))\s+{object}",
    ),
    "created": (
        r"(?P<predicate>(?:create|creates|created|make|makes|made|build|builds|built|write|writes|wrote|written|design|designs|designed|compose|composes|composed|produce|produces|produced))\s+{object}",
    ),
    "requires": (
        r"(?P<predicate>(?:explicitly\s+)?(?:require|requires|required))\s+{object}",
    ),
    "prohibits": (
        r"(?P<predicate>(?:explicitly\s+)?(?:prohibit|prohibits|prohibited|forbid|forbids|forbade))\s+{object}",
    ),
    "permits": (
        r"(?P<predicate>(?:explicitly\s+)?(?:permit|permits|permitted|allow|allows|allowed))\s+{object}",
    ),
    "changed_to": (
        r"(?P<predicate>(?:change|changes|changed|switch|switches|switched|transition|transitions|transitioned)\s+to)\s+{object}",
        r"(?P<predicate>(?:became|becomes))\s+{object}",
    ),
    "replaces": (
        r"(?P<predicate>(?:replace|replaces|replaced))\s+{object}",
    ),
}

_DETERMINERS = frozenset(
    {
        "a", "an", "the", "this", "that", "these", "those", "my",
        "our", "your", "his", "her", "their", "another",
    }
)
_TIME_PRELUDE_PATTERN = (
    r"(?:today|yesterday|last\s+(?:monday|tuesday|wednesday|thursday|"
    r"friday|saturday|sunday|week|weekend|night)|on\s+(?:monday|tuesday|"
    r"wednesday|thursday|friday|saturday|sunday))\s+"
)
_FIRST_PERSON_SUBJECTS = frozenset(
    {"i", "i'm", "i’m", "i've", "i’ve", "i'd", "i’d"}
)
_PAST_PARTICIPLE_CUES = frozenset(
    {
        "worked", "employed", "lived", "located", "based", "situated",
        "liked", "loved", "enjoyed", "preferred", "disliked", "hated",
        "participated", "attended", "joined", "owned", "possessed",
        "created", "made", "built", "written", "designed", "composed",
        "produced", "required", "prohibited", "forbade", "permitted",
        "allowed", "changed", "switched", "transitioned", "replaced",
    }
)

_UNSUPPORTED_CLAIM_PATTERN = re.compile(
    r"\b(ignore|invent|pretend|fabricate|hypothetical|fictional|not a fact|"
    r"not true|false claim|untrusted instruction|example only|merely an example)\b",
    flags=re.IGNORECASE,
)

_ATOMIC_RELATION_SEPARATOR = re.compile(
    r"\b(?:but|whereas|however|although|while|yet)\b",
    flags=re.IGNORECASE,
)


def _entity_spans(text: str, normalized_name: str) -> Tuple[Tuple[int, int], ...]:
    """Find exact entity mentions without Latin partial-name matches."""

    if any("\u3400" <= character <= "\u9fff" for character in normalized_name):
        result = []
        start = 0
        while True:
            index = text.find(normalized_name, start)
            if index < 0:
                break
            result.append((index, index + len(normalized_name)))
            start = index + len(normalized_name)
        return tuple(result)
    pattern = re.compile(
        r"(?<![\w]){}(?![\w])".format(re.escape(normalized_name)),
        flags=re.IGNORECASE,
    )
    return tuple((match.start(), match.end()) for match in pattern.finditer(text))


def _escaped_exact_name(value: str) -> str:
    return r"\s+".join(re.escape(part) for part in value.split())


def _object_capture_pattern(normalized_object: str) -> str:
    words = normalized_object.split()
    if words and words[0] in {"no", "not", "neither", "nor", "never"}:
        # Negative quantifiers are not members of the positive determiner
        # grammar, even if an untrusted extractor copied one into the endpoint.
        return r"(?P<object>(?!))"
    determiner = ""
    if not words or words[0] not in _DETERMINERS:
        determiner = (
            r"(?:(?:a|an|the|this|that|these|those|my|our|your|his|her|"
            r"their|another)\s+)?"
        )
    exact = _escaped_exact_name(normalized_object)
    return (
        r"(?:[\"'“‘]\s*)?"
        r"(?P<object>{}{})"
        r"(?:\s*[\"'”’])?"
    ).format(determiner, exact)


def _subject_specs(
    *, normalized_subject: str, normalized_speaker: Optional[str], predicate: str
) -> Tuple[Tuple[str, str], ...]:
    exact = _escaped_exact_name(normalized_subject)
    if (
        predicate in _RULE_PREDICATES
        and normalized_subject.split()[0] not in _DETERMINERS
    ):
        exact = r"(?:the\s+)?{}".format(exact)
    # ``a different Jordan`` is a complete, explicit subject NP used by the
    # same-name isolation contract.  The modifier is part of the subject span;
    # it is not a free gap before a later mention.
    specs = [("named", r"(?:an?\s+different\s+)?{}".format(exact))]
    if normalized_speaker == normalized_subject:
        specs.append(
            (
                "trusted_speaker_1p",
                r"(?:i'm|i’m|i've|i’ve|i'd|i’d|i)(?![\w])",
            )
        )
    return tuple(specs)


def _normalized_source(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).casefold()
    value = re.sub(r"[\t\r\f\v ]+", " ", value)
    value = re.sub(r"\s*\n\s*", "\n", value)
    return value.strip()


def _normalized_sentences(source_text: str) -> Tuple[Tuple[str, int], ...]:
    normalized = _normalized_source(source_text)
    parts = re.split(r"(?<=[.!?。！？])\s+|\n+", normalized)
    result = []
    cursor = 0
    for part in parts:
        clause = " ".join(part.split())
        if not clause:
            continue
        start = normalized.find(part, cursor)
        if start < 0:
            start = cursor
        result.append((clause, start))
        cursor = start + len(part)
    return tuple(result)


def _strip_speaker_prefix(
    clause: str, normalized_speaker: Optional[str]
) -> Tuple[str, int]:
    if not normalized_speaker:
        return clause, 0
    match = re.match(
        r"{}\s*:\s*".format(_escaped_exact_name(normalized_speaker)),
        clause,
    )
    if match is None:
        return clause, 0
    return clause[match.end():].lstrip(), match.end()


def _appositive_pattern(predicate: str) -> str:
    if predicate not in {"works_at", "role_at"}:
        return ""
    role_np = _role_np_frame_pattern()
    return (
        r"(?:"
        r"\s*,\s*(?:who\s+(?:is|was)\s+)?{}\s*,"
        r"|\s+(?:(?:is|was)\s+)?{}(?:\s+who)?"
        r")?"
    ).format(role_np, role_np)


def _first_person_frame_is_grammatical(
    *, subject_text: str, predicate_text: str
) -> bool:
    subject_text = subject_text.casefold()
    first_cue = predicate_text.casefold().split()[0]
    if subject_text in {"i'm", "i’m"}:
        return first_cue in {"keen", "fond"}
    if subject_text in {"i've", "i’ve", "i'd", "i’d"}:
        return first_cue in _PAST_PARTICIPLE_CUES
    return subject_text == "i"


def _witness_from_match(
    *,
    match: re.Match[str],
    clause: str,
    source_start: int,
    predicate: str,
    spec_suffix: str,
    binding: str,
    state_change: str = "assert",
    temporal_status: Optional[str] = None,
) -> Optional[SupportWitness]:
    if binding == "trusted_speaker_1p" and not _first_person_frame_is_grammatical(
        subject_text=match.group("subject"),
        predicate_text=match.group("predicate"),
    ):
        return None
    first_cue = match.group("predicate").casefold().split()[0]
    if binding == "named" and first_cue in {"am", "keen", "fond", "written"}:
        return None
    if (
        binding == "trusted_speaker_1p"
        and match.group("subject").casefold() == "i"
        and first_cue in {"is", "are"}
    ):
        return None
    return SupportWitness(
        spec_id="{}:{}".format(predicate, spec_suffix),
        clause=clause,
        subject_span=match.span("subject"),
        predicate_span=match.span("predicate"),
        object_span=match.span("object"),
        binding=binding,
        state_change=state_change,
        temporal_status=temporal_status,
        source_span=(source_start, source_start + len(clause)),
    )


def _match_complete_frame(
    *,
    clause: str,
    source_start: int,
    normalized_subject: str,
    normalized_object: str,
    normalized_speaker: Optional[str],
    predicate: str,
) -> Tuple[SupportWitness, ...]:
    """Full-match a clause against the predicate's positive grammar."""

    if (
        not clause
        or "?" in clause
        or "？" in clause
        or _UNSUPPORTED_CLAIM_PATTERN.search(clause)
    ):
        return ()
    object_pattern = _object_capture_pattern(normalized_object)
    cores = _PREDICATE_FRAME_CORES.get(predicate, ())
    if not cores:
        return ()
    suffix = r"(?:\s+on\s+(?:weekdays|weekends))?\s*[.!。！]?"
    witnesses = []
    for binding, subject_pattern in _subject_specs(
        normalized_subject=normalized_subject,
        normalized_speaker=normalized_speaker,
        predicate=predicate,
    ):
        time_prelude = _TIME_PRELUDE_PATTERN if binding == "trusted_speaker_1p" else ""
        subject = r"(?P<subject>{})".format(subject_pattern)
        appositive = _appositive_pattern(predicate)
        for core_template in cores:
            core = core_template.format(
                object=object_pattern, role=_role_np_frame_pattern()
            )
            regular = re.compile(
                r"^(?:{})?{}{}\s+"
                r"(?:(?P<wrapper>used\s+to|formerly|previously|now)\s+)?"
                r"(?:(?P<positive_modifier>still)\s+)?"
                r"{}(?P<instead>\s+instead)?{}$".format(
                    time_prelude, subject, appositive, core, suffix
                )
            )
            match = regular.fullmatch(clause)
            if match:
                wrapper = match.group("wrapper")
                instead = bool(match.group("instead"))
                if wrapper in {"used to", "formerly", "previously"}:
                    state_change, temporal_status = "historical", "previous"
                elif wrapper == "now" and instead:
                    state_change, temporal_status = "update", "current"
                elif instead:
                    # ``instead`` changes state only when its controlled
                    # companion ``now`` is present in this same frame.
                    continue
                else:
                    state_change, temporal_status = "assert", None
                witness = _witness_from_match(
                    match=match,
                    clause=clause,
                    source_start=source_start,
                    predicate=predicate,
                    spec_suffix="direct",
                    binding=binding,
                    state_change=state_change,
                    temporal_status=temporal_status,
                )
                if witness:
                    witnesses.append(witness)

            correction = re.compile(
                r"^correction:\s*{}{}\s+actually\s+{}"
                r"\s*,\s*not\s+(?:[\"'“‘]\s*)?"
                r"(?P<old_object>(?:[^,;.!?。！？\s]+\s*){{1,12}})"
                r"(?:\s*[\"'”’])?\s*[.!。！]?$".format(
                    subject, appositive, core
                )
            )
            match = correction.fullmatch(clause)
            if match:
                witness = _witness_from_match(
                    match=match,
                    clause=clause,
                    source_start=source_start,
                    predicate=predicate,
                    spec_suffix="correction",
                    binding=binding,
                    state_change="correction",
                    temporal_status="current",
                )
                if witness:
                    witnesses.append(witness)

        if predicate == "lives_in":
            moved = re.compile(
                r"^(?:{})?{}\s+(?P<predicate>moved\s+from)\s+"
                r"(?P<old_object>(?:[^,;.!?。！？\s]+\s+){{1,12}}?)"
                r"to\s+{}\s*[.!。！]?$".format(
                    time_prelude, subject, object_pattern
                )
            )
            match = moved.fullmatch(clause)
            if match:
                witness = _witness_from_match(
                    match=match,
                    clause=clause,
                    source_start=source_start,
                    predicate=predicate,
                    spec_suffix="moved-from-to",
                    binding=binding,
                    state_change="update",
                    temporal_status="current",
                )
                if witness:
                    witnesses.append(witness)

        if predicate == "member_of":
            retraction = re.compile(
                r"^{}\s+(?P<predicate>(?:am|is|are|was|were)\s+no\s+longer\s+"
                r"(?:a\s+)?member\s+of)\s+{}\s*;\s*the\s+earlier\s+"
                r"statement\s+is\s+(?:retracted|revoked)\s*[.!。！]?$".format(
                    subject, object_pattern
                )
            )
            match = retraction.fullmatch(clause)
            if match:
                witness = _witness_from_match(
                    match=match,
                    clause=clause,
                    source_start=source_start,
                    predicate=predicate,
                    spec_suffix="retraction",
                    binding=binding,
                    state_change="retraction",
                    temporal_status="previous",
                )
                if witness:
                    witnesses.append(witness)
    return tuple(witnesses)


_COORDINATED_PREDICATE_HEAD = re.compile(
    r"^(?:(?:used\s+to|formerly|previously|now)\s+)?(?:"
    r"(?:am|is|are|was|were)\s+(?:a\s+)?(?:friend|parent|mother|father|"
    r"sibling|brother|sister|partner|spouse|husband|wife|member)\s+(?:of)|"
    r"(?:work|works|worked)\s+(?:at|for)|"
    r"(?:have|has|had|hold|holds|held)\s+(?:an?\s+|the\s+)?(?:formal\s+)?(?:role|position)\s+at|"
    r"(?:live|lives|lived)\s+in|"
    r"(?:like|likes|liked|love|loves|loved|enjoy|enjoys|enjoyed|prefer|prefers|preferred|"
    r"dislike|dislikes|disliked|hate|hates|hated|attend|attends|attended|join|joins|joined|"
    r"own|owns|owned|possess|possesses|possessed|create|creates|created|make|makes|made|"
    r"build|builds|built|write|writes|wrote|design|designs|designed|compose|composes|composed|"
    r"produce|produces|produced|replace|replaces|replaced)|"
    r"(?:participate|participates|participated)\s+in|(?:take|takes|took)\s+part\s+in|"
    r"went\s+to|(?:explicitly\s+)?(?:require|requires|required|prohibit|prohibits|prohibited|"
    r"forbid|forbids|forbade|permit|permits|permitted|allow|allows|allowed)|"
    r"(?:change|changes|changed|switch|switches|switched|transition|transitions|transitioned)\s+to"
    r")\s+(?!no\b)"
    r"(?:(?:a|an|the|this|that|these|those|my|our|your|his|her|their|another)\s+)?"
    r"[^\W_][\w'’+\-]*(?:\s+[^\W_][\w'’+\-]*){0,11}\s*[.!。！]?$"
)


def _coordinating_subject_binding(
    *,
    first_conjunct: str,
    normalized_subject: str,
    normalized_speaker: Optional[str],
    predicate: str,
) -> Optional[Tuple[str, Tuple[int, int], str]]:
    for binding, subject_pattern in _subject_specs(
        normalized_subject=normalized_subject,
        normalized_speaker=normalized_speaker,
        predicate=predicate,
    ):
        time_prelude = _TIME_PRELUDE_PATTERN if binding == "trusted_speaker_1p" else ""
        pattern = re.compile(
            r"^(?:{})?(?P<subject>{})\s+(?P<body>[^,;.!?。！？]{{1,256}})$".format(
                time_prelude, subject_pattern
            )
        )
        match = pattern.fullmatch(first_conjunct.strip())
        if not match or not _COORDINATED_PREDICATE_HEAD.fullmatch(match.group("body")):
            continue
        subject_text = match.group("subject")
        if binding == "trusted_speaker_1p" and subject_text not in _FIRST_PERSON_SUBJECTS:
            continue
        return binding, match.span("subject"), subject_text
    return None


def _relation_support_witnesses(
    *,
    source_text: Any,
    subject: Any,
    predicate: Any,
    object_name: Any,
    speaker: Any = None,
) -> Tuple[SupportWitness, ...]:
    """Return complete positive-grammar witnesses for a candidate edge."""

    if not isinstance(source_text, str):
        return ()
    normalized_predicate = normalize_predicate(predicate)
    normalized_subject = normalize_entity_name(subject)
    normalized_object = normalize_entity_name(object_name)
    if not normalized_predicate or not normalized_subject or not normalized_object:
        return ()
    normalized_speaker = normalize_entity_name(speaker)
    witnesses = []
    seen = set()
    for sentence, sentence_start in _normalized_sentences(source_text):
        body, prefix_chars = _strip_speaker_prefix(sentence, normalized_speaker)
        body_start = sentence_start + prefix_chars

        # Retraction is the sole semicolon-spanning frame.  All ordinary
        # assertions below are evaluated one atomic clause at a time.
        for witness in _match_complete_frame(
            clause=body,
            source_start=body_start,
            normalized_subject=normalized_subject,
            normalized_object=normalized_object,
            normalized_speaker=normalized_speaker,
            predicate=normalized_predicate,
        ):
            key = (witness.spec_id, witness.subject_span, witness.predicate_span, witness.object_span)
            if key not in seen:
                seen.add(key)
                witnesses.append(witness)

        for semicolon_clause in re.split(r"\s*;\s*", body):
            for atomic_clause in _ATOMIC_RELATION_SEPARATOR.split(semicolon_clause):
                atomic_clause = atomic_clause.strip()
                if not atomic_clause:
                    continue
                atomic_start = body_start + max(0, body.find(atomic_clause))
                conjunctions = re.split(r"\s+and\s+", atomic_clause)
                variants = [atomic_clause]
                if len(conjunctions) > 1:
                    variants.extend(part.strip() for part in conjunctions)
                for variant in variants:
                    variant_start = atomic_start + max(0, atomic_clause.find(variant))
                    for witness in _match_complete_frame(
                        clause=variant,
                        source_start=variant_start,
                        normalized_subject=normalized_subject,
                        normalized_object=normalized_object,
                        normalized_speaker=normalized_speaker,
                        predicate=normalized_predicate,
                    ):
                        key = (
                            witness.spec_id, witness.source_span,
                            witness.predicate_span, witness.object_span,
                        )
                        if key not in seen:
                            seen.add(key)
                            witnesses.append(witness)

                if len(conjunctions) <= 1:
                    continue
                binding_info = _coordinating_subject_binding(
                    first_conjunct=conjunctions[0],
                    normalized_subject=normalized_subject,
                    normalized_speaker=normalized_speaker,
                    predicate=normalized_predicate,
                )
                if binding_info is None:
                    continue
                binding, subject_span, subject_text = binding_info
                for conjunct in conjunctions[1:]:
                    conjunct = conjunct.strip()
                    synthetic = "{} {}".format(subject_text, conjunct)
                    for match_witness in _match_complete_frame(
                        clause=synthetic,
                        source_start=atomic_start,
                        normalized_subject=normalized_subject,
                        normalized_object=normalized_object,
                        normalized_speaker=normalized_speaker,
                        predicate=normalized_predicate,
                    ):
                        if match_witness.binding != binding:
                            continue
                        offset = max(0, atomic_clause.find(conjunct))
                        prefix_length = len(subject_text) + 1
                        witness = SupportWitness(
                            spec_id="{}:coordinated".format(normalized_predicate),
                            clause=atomic_clause,
                            subject_span=subject_span,
                            predicate_span=(
                                offset + match_witness.predicate_span[0] - prefix_length,
                                offset + match_witness.predicate_span[1] - prefix_length,
                            ),
                            object_span=(
                                offset + match_witness.object_span[0] - prefix_length,
                                offset + match_witness.object_span[1] - prefix_length,
                            ),
                            binding=binding,
                            state_change=match_witness.state_change,
                            temporal_status=match_witness.temporal_status,
                            source_span=(atomic_start, atomic_start + len(atomic_clause)),
                        )
                        key = (
                            witness.spec_id, witness.source_span,
                            witness.predicate_span, witness.object_span,
                        )
                        if key not in seen:
                            seen.add(key)
                            witnesses.append(witness)
    return tuple(witnesses)


def _supported_relation_clauses(
    *,
    source_text: Any,
    subject: Any,
    predicate: Any,
    object_name: Any,
    speaker: Any = None,
) -> Tuple[str, ...]:
    """Return clauses backed by the unified positive grammar."""

    return tuple(
        dict.fromkeys(
            witness.clause
            for witness in _relation_support_witnesses(
                source_text=source_text,
                subject=subject,
                predicate=predicate,
                object_name=object_name,
                speaker=speaker,
            )
        )
    )


def relation_is_textually_supported(
    *,
    source_text: Any,
    subject: Any,
    predicate: Any,
    object_name: Any,
    speaker: Any = None,
) -> bool:
    """Require one non-adversarial source clause to state the whole relation.

    This is deliberately lexical and conservative. It is a provenance guard,
    not a relation extractor: uncertain paraphrases are dropped rather than
    promoted into graph edges.
    """

    return bool(
        _relation_support_witnesses(
            source_text=source_text,
            subject=subject,
            predicate=predicate,
            object_name=object_name,
            speaker=speaker,
        )
    )


def _normalize_human_role_hint(value: Any) -> Optional[str]:
    role = normalize_entity_name(value)
    if role is None:
        return None
    role = re.sub(r"^(?:the|an?|their)\s+", "", role)
    words = re.findall(r"[^\W_]+", role, flags=re.UNICODE)
    if not words or len(words) > 4 or words[-1] not in _HUMAN_ROLE_HEAD_WORDS:
        return None
    return " ".join(words)


_API_SPEAKER_IDENTITY_HINT_PREFIX = "api-speaker:"


def _api_speaker_identity_hint(
    *, speaker: Any, canonical_name: Any
) -> Optional[str]:
    """Return a stable identity hint only for an exact API speaker match.

    ``speaker`` is trusted request metadata rather than model output.  The
    explicit prefix keeps this namespace separate from model-extracted role
    descriptors, while normal entity-name bounds prevent an unbounded identity
    key from reaching storage.
    """

    normalized_speaker = normalize_entity_name(speaker)
    normalized_entity = normalize_entity_name(canonical_name)
    if (
        normalized_speaker is None
        or normalized_entity is None
        or normalized_speaker != normalized_entity
    ):
        return None
    candidate = "{}{}".format(
        _API_SPEAKER_IDENTITY_HINT_PREFIX, normalized_speaker
    )
    if len(candidate) > MAX_ENTITY_NAME_CHARS:
        return None
    normalized_candidate = normalize_entity_name(candidate)
    if (
        normalized_candidate is None
        or not normalized_candidate.startswith(_API_SPEAKER_IDENTITY_HINT_PREFIX)
    ):
        return None
    return normalized_candidate


def _source_body_mentions_api_speaker(
    *, source_text: Any, speaker: Any
) -> bool:
    """Detect an explicit same-name mention outside attribution metadata.

    When ``Alice:`` is followed by ``another Alice``, the payload may contain
    two genuinely different same-name people.  In that case no local entity
    may receive or be folded by the otherwise-stable API speaker hint.
    """

    if not isinstance(source_text, str):
        return False
    normalized_speaker = normalize_entity_name(speaker)
    if normalized_speaker is None:
        return False
    body = unicodedata.normalize("NFKC", source_text).casefold()
    body = re.sub(
        r"^{}\s*:\s*".format(re.escape(normalized_speaker)),
        "",
        body,
        count=1,
        flags=re.IGNORECASE,
    )
    return bool(_entity_spans(body, normalized_speaker))


_SAME_NAME_AMBIGUITY_PATTERN = re.compile(
    r"\b(?:namesake|same\s+name|someone\s+with\s+my\s+name|"
    r"another\s+person\s+(?:called|named)|"
    r"different\s+person\s+(?:called|named))\b",
    flags=re.IGNORECASE,
)


def _source_body_has_same_name_ambiguity(
    *, source_text: Any, speaker: Any
) -> bool:
    """Reject API identity folding when the body signals a namesake."""

    if not isinstance(source_text, str):
        return False
    normalized_speaker = normalize_entity_name(speaker)
    if normalized_speaker is None:
        return False
    body = _normalized_source(source_text)
    body, _ = _strip_speaker_prefix(body, normalized_speaker)
    return bool(
        _entity_spans(body, normalized_speaker)
        or _SAME_NAME_AMBIGUITY_PATTERN.search(body)
    )


_GENERATED_ENTITY_ID_FIELDS = frozenset(
    {"id", "entity_id", "local_id", "local_entity_id"}
)


def _duplicate_speaker_records_are_identical(
    records: Sequence[Mapping[str, Any]],
) -> bool:
    """Compare raw declarations while ignoring only generated local ids."""

    if not records:
        return False
    fingerprints = []
    for record in records:
        fingerprint = tuple(
            sorted(
                (str(key), repr(value))
                for key, value in record.items()
                if str(key) not in _GENERATED_ENTITY_ID_FIELDS
            )
        )
        fingerprints.append(fingerprint)
    return len(set(fingerprints)) == 1


def _explicit_human_role_hint(
    *,
    source_text: Any,
    subject: Any,
    predicate: Any,
    object_name: Any,
    speaker: Any = None,
) -> Optional[str]:
    """Extract a narrow occupation/appositive cue from the supporting clause."""

    normalized_subject = normalize_entity_name(subject)
    if normalized_subject is None:
        return None
    subject_pattern = re.escape(normalized_subject)
    patterns = (
        # ``Jordan 7, the designer, works at ...``
        rf"{subject_pattern}\s*,\s*(?!who\b)([^,]{{1,64}}?)\s*,",
        # ``Jordan 7, who is a designer, works at ...``
        rf"{subject_pattern}\s*,\s*who\s+(?:is|was)\s+([^,]{{1,64}}?)\s*,",
        # ``Jordan 7 works as a designer at ...``
        rf"{subject_pattern}\s+(?:works?|worked|serves?|served)\s+as\s+(.{{1,64}}?)\s+at\b",
        # ``Jordan 7 is a designer who works at ...``
        rf"{subject_pattern}\s+(?:is|was)\s+(.{{1,64}}?)\s+(?:who\s+)?(?:works?|worked|serves?|served)\b",
        # ``Jordan 7 the designer works at ...``
        rf"{subject_pattern}\s+(?:the|an?|their)\s+(.{{1,64}}?)\s+(?:works?|worked|serves?|served)\b",
    )
    hints = set()
    for clause in _supported_relation_clauses(
        source_text=source_text,
        subject=subject,
        predicate=predicate,
        object_name=object_name,
        speaker=speaker,
    ):
        for pattern in patterns:
            match = re.search(pattern, clause)
            if match:
                hint = _normalize_human_role_hint(match.group(1))
                if hint:
                    hints.add(hint)
    return next(iter(hints)) if len(hints) == 1 else None


def _explicit_temporal_state(
    *,
    source_text: Any,
    subject: Any,
    predicate: Any,
    object_name: Any,
    speaker: Any = None,
) -> Optional[Tuple[str, str]]:
    """Return the unique state carried by a complete support witness."""

    witnesses = _relation_support_witnesses(
        source_text=source_text,
        subject=subject,
        predicate=predicate,
        object_name=object_name,
        speaker=speaker,
    )
    explicit_states = {
        (witness.state_change, witness.temporal_status)
        for witness in witnesses
        if witness.state_change != "assert"
        and witness.temporal_status is not None
    }
    return next(iter(explicit_states)) if len(explicit_states) == 1 else None


def endpoint_types_are_compatible(
    predicate: str, subject_type: str, object_type: str
) -> bool:
    """Return whether a controlled predicate accepts both endpoint types."""

    normalized_predicate = normalize_predicate(predicate)
    normalized_subject = normalize_entity_type(subject_type)
    normalized_object = normalize_entity_type(object_type)
    if not normalized_predicate or not normalized_subject or not normalized_object:
        return False
    constraints = PREDICATE_ENDPOINT_TYPES.get(normalized_predicate)
    if constraints is None:
        return False
    subject_types, object_types = constraints
    if subject_types is not None and normalized_subject not in subject_types:
        return False
    if object_types is not None and normalized_object not in object_types:
        return False
    if normalized_predicate in {"changed_to", "replaces"}:
        return normalized_subject == normalized_object
    return True


def resolve_entity_reference(
    entities: Iterable[Entity],
    name: Any,
    entity_type: Any,
    *,
    user_id: Optional[str] = None,
) -> EntityResolution:
    """Resolve only a unique normalized-name and exact-type match.

    Two same-name entities of the same type are intentionally ambiguous.  A
    same-name entity of another type never acts as a fallback.
    """

    canonical_name = normalize_entity_name(name)
    normalized_type = normalize_entity_type(entity_type)
    if canonical_name is None or normalized_type is None:
        return EntityResolution("invalid", None)
    if user_id is not None:
        try:
            user_id = _bounded_identifier(user_id, "user_id")
        except ValueError:
            return EntityResolution("invalid", None)
    scoped_entities = tuple(
        entity for entity in entities if user_id is None or entity.user_id == user_id
    )
    name_matches = tuple(
        entity for entity in scoped_entities if entity.canonical_name == canonical_name
    )
    type_matches = tuple(
        entity for entity in name_matches if entity.entity_type == normalized_type
    )
    candidate_ids = tuple(entity.entity_id for entity in type_matches)
    if len(type_matches) == 1:
        return EntityResolution("resolved", type_matches[0], candidate_ids)
    if len(type_matches) > 1:
        return EntityResolution("ambiguous", None, candidate_ids)
    if name_matches:
        return EntityResolution(
            "type_mismatch",
            None,
            tuple(entity.entity_id for entity in name_matches),
        )
    return EntityResolution("missing", None)


def _bounded_identifier(value: Any, field_name: str) -> str:
    if not isinstance(value, str):
        raise ValueError("{} must be a string".format(field_name))
    # User and source identifiers are opaque security/provenance boundaries.
    # Do not Unicode-normalize them: distinct caller IDs must stay distinct.
    value = value.strip()
    if not value or len(value) > MAX_IDENTIFIER_CHARS:
        raise ValueError("{} must contain 1 to {} characters".format(
            field_name, MAX_IDENTIFIER_CHARS
        ))
    if any(unicodedata.category(character).startswith("C") for character in value):
        raise ValueError("{} contains a control character".format(field_name))
    return value


def _bounded_created_at(value: Optional[str]) -> str:
    if value is None:
        return datetime.now(timezone.utc).isoformat()
    if not isinstance(value, str):
        raise ValueError("created_at must be a string")
    value = value.strip()
    if not value or len(value) > MAX_CREATED_AT_CHARS:
        raise ValueError("created_at must contain 1 to {} characters".format(
            MAX_CREATED_AT_CHARS
        ))
    return value


def _stable_id(prefix: str, *parts: Any) -> str:
    encoded = "\0".join(str(part) for part in parts).encode("utf-8")
    return "{}_{}".format(prefix, hashlib.sha256(encoded).hexdigest()[:24])


def _raw_sequence(payload: Mapping[str, Any], field: str) -> Sequence[Any]:
    value = payload.get(field, ())
    if isinstance(value, (list, tuple)):
        return value
    return ()


def parse_graph_payload(
    payload: Any,
    *,
    user_id: str,
    source_message_id: str,
    event_ts: Optional[int] = None,
    created_at: Optional[str] = None,
    source_text: Optional[str] = None,
    speaker: Optional[str] = None,
) -> GraphPayload:
    """Sanitize one model payload into bounded, provenance-backed graph data.

    Relations are accepted only when they are explicitly asserted, use every
    controlled field correctly, have predicate-compatible endpoint types, and
    resolve both endpoints uniquely to entities declared in this same payload.
    Unknown or ambiguous data is dropped; no implicit entity is manufactured.
    """

    user_id = _bounded_identifier(user_id, "user_id")
    source_message_id = _bounded_identifier(source_message_id, "source_message_id")
    created_at = _bounded_created_at(created_at)
    if event_ts is not None and (not isinstance(event_ts, int) or isinstance(event_ts, bool)):
        raise ValueError("event_ts must be an integer or None")
    if not isinstance(payload, Mapping):
        return GraphPayload()

    raw_entities = _raw_sequence(payload, "entities")
    raw_relations = _raw_sequence(payload, "relations")

    raw_relation_witnesses: dict[int, Tuple[SupportWitness, ...]] = {}
    if source_text is not None:
        for index, raw_relation in enumerate(
            raw_relations[:MAX_RELATIONS_PER_PAYLOAD]
        ):
            if (
                not isinstance(raw_relation, Mapping)
                or raw_relation.get("explicit") is not True
            ):
                continue
            raw_relation_witnesses[index] = _relation_support_witnesses(
                source_text=source_text,
                subject=raw_relation.get("subject"),
                predicate=raw_relation.get("relation"),
                object_name=raw_relation.get("object"),
                speaker=speaker,
            )

    # Context-sensitive endpoint repairs are proposed from explicit,
    # source-supported relations, then applied only when every supported use
    # of the same raw endpoint agrees and exactly one raw entity can receive
    # the repair.  This prevents one relation from silently retyping a shared
    # or same-name entity used differently elsewhere in the message.
    raw_entities_by_key: dict[
        tuple[str, str], list[Mapping[str, Any]]
    ] = {}
    for raw_entity in raw_entities[:MAX_ENTITIES_PER_PAYLOAD]:
        if not isinstance(raw_entity, Mapping):
            continue
        raw_canonical_name = normalize_entity_name(raw_entity.get("name"))
        raw_entity_type = normalize_entity_type(raw_entity.get("type"))
        if raw_canonical_name and raw_entity_type:
            raw_entities_by_key.setdefault(
                (raw_canonical_name, raw_entity_type), []
            ).append(raw_entity)

    endpoint_outcomes: dict[
        tuple[str, str], set[tuple[str, str, Optional[str]]]
    ] = {}
    if source_text is not None:
        for relation_index, raw_relation in enumerate(
            raw_relations[:MAX_RELATIONS_PER_PAYLOAD]
        ):
            if (
                not isinstance(raw_relation, Mapping)
                or raw_relation.get("explicit") is not True
            ):
                continue
            predicate = normalize_predicate(raw_relation.get("relation"))
            subject_name = normalize_entity_name(raw_relation.get("subject"))
            object_name = normalize_entity_name(raw_relation.get("object"))
            subject_type = normalize_entity_type(raw_relation.get("subject_type"))
            object_type = normalize_entity_type(raw_relation.get("object_type"))
            if (
                not predicate
                or not subject_name
                or not object_name
                or not subject_type
                or not object_type
                or not raw_relation_witnesses.get(relation_index)
            ):
                continue

            desired_subject_name = subject_name
            desired_subject_type = subject_type
            desired_subject_hint: Optional[str] = None
            desired_object_name = object_name
            desired_object_type = object_type

            if (
                predicate == "member_of"
                and object_type == "organization"
                and _semantic_head_word(object_name) in _GROUP_HEAD_WORDS
            ):
                desired_object_type = "group"

            if predicate in _RULE_PREDICATES:
                desired_subject_name = (
                    _strip_grammatical_leading_article(
                        raw_relation.get("subject")
                    )
                    or subject_name
                )
                desired_object_name = (
                    _strip_grammatical_leading_article(
                        raw_relation.get("object")
                    )
                    or object_name
                )
                # A capitalized sentence-initial article is still grammatical
                # when the endpoint's semantic head explicitly names a rule.
                # This remains narrower than global article stripping and does
                # not affect proper names such as ``The Beatles``.
                subject_without_article = re.sub(
                    r"^(?:the|an|a)\s+", "", subject_name
                )
                if (
                    subject_type in {"document", "rule"}
                    and subject_without_article != subject_name
                    and _semantic_head_word(subject_without_article)
                    in _RULE_HEAD_WORDS
                ):
                    desired_subject_name = subject_without_article
                if (
                    subject_type == "document"
                    and _semantic_head_word(desired_subject_name)
                    in _RULE_HEAD_WORDS
                ):
                    desired_subject_type = "rule"
                if object_type == "product":
                    desired_object_type = "object"

            if predicate in {"works_at", "role_at"}:
                desired_subject_hint = _explicit_human_role_hint(
                    source_text=source_text,
                    subject=raw_relation.get("subject"),
                    predicate=predicate,
                    object_name=raw_relation.get("object"),
                    speaker=speaker,
                )
                if desired_subject_hint:
                    desired_subject_type = "person"

            if not endpoint_types_are_compatible(
                predicate, desired_subject_type, desired_object_type
            ):
                # An endpoint repair must make a complete, controlled relation
                # valid; never let an invalid/reversed edge mutate or veto a
                # repair proposed by a valid edge.
                continue

            endpoint_outcomes.setdefault(
                (subject_name, subject_type), set()
            ).add(
                (
                    desired_subject_name,
                    desired_subject_type,
                    desired_subject_hint,
                )
            )
            endpoint_outcomes.setdefault(
                (object_name, object_type), set()
            ).add((desired_object_name, desired_object_type, None))

    contextual_entity_corrections: dict[
        tuple[str, str], tuple[str, str, Optional[str]]
    ] = {}
    for raw_key, outcomes in endpoint_outcomes.items():
        raw_matches = raw_entities_by_key.get(raw_key, [])
        if len(outcomes) != 1 or len(raw_matches) != 1:
            continue
        outcome = next(iter(outcomes))
        desired_name, desired_type, desired_hint = outcome
        if outcome == (raw_key[0], raw_key[1], None):
            continue
        desired_key = (desired_name, desired_type)
        if desired_key != raw_key and raw_entities_by_key.get(desired_key):
            continue
        contextual_entity_corrections[raw_key] = outcome

    product_predicates_by_object: dict[str, set[str]] = {}
    for relation_index, raw_relation in enumerate(
        raw_relations[:MAX_RELATIONS_PER_PAYLOAD]
    ):
        if not isinstance(raw_relation, Mapping) or raw_relation.get("explicit") is not True:
            continue
        canonical_object = normalize_entity_name(raw_relation.get("object"))
        predicate = normalize_predicate(raw_relation.get("relation"))
        if (
            canonical_object
            and predicate
            and normalize_entity_type(raw_relation.get("object_type")) == "product"
        ):
            product_predicates_by_object.setdefault(canonical_object, set()).add(
                predicate
            )
    mixed_product_objects = {
        name
        for name, predicates in product_predicates_by_object.items()
        if predicates & _PREFERENCE_PREDICATES
        and predicates - _PREFERENCE_PREDICATES
    }
    entity_type_corrections: dict[tuple[str, str], set[str]] = {}
    for relation_index, raw_relation in enumerate(
        raw_relations[:MAX_RELATIONS_PER_PAYLOAD]
    ):
        if not isinstance(raw_relation, Mapping) or raw_relation.get("explicit") is not True:
            continue
        if source_text is None or not raw_relation_witnesses.get(relation_index):
            continue
        canonical_object = normalize_entity_name(raw_relation.get("object"))
        original_type = normalize_entity_type(raw_relation.get("object_type"))
        predicate = normalize_predicate(raw_relation.get("relation"))
        subject_type = normalize_entity_type(raw_relation.get("subject_type"))
        corrected_type = normalize_relation_object_type(
            predicate,
            raw_relation.get("object"),
            raw_relation.get("object_type"),
        )
        if (
            canonical_object
            and original_type
            and corrected_type
            and corrected_type != original_type
            and canonical_object not in mixed_product_objects
            and endpoint_types_are_compatible(
                predicate, subject_type, corrected_type
            )
        ):
            entity_type_corrections.setdefault(
                (canonical_object, original_type), set()
            ).add(corrected_type)

    normalized_speaker = normalize_entity_name(speaker)
    speaker_raw_key = (
        (normalized_speaker, "person") if normalized_speaker else None
    )
    speaker_records = (
        raw_entities_by_key.get(speaker_raw_key, [])
        if speaker_raw_key is not None
        else []
    )
    source_body_has_speaker_ambiguity = _source_body_has_same_name_ambiguity(
        source_text=source_text, speaker=speaker
    )
    speaker_group_policy = "preserve"
    speaker_bound_relation_indexes: set[int] = set()
    if source_text is not None and len(speaker_records) == 1:
        if not source_body_has_speaker_ambiguity:
            speaker_group_policy = "unique"
    elif source_text is not None and len(speaker_records) >= 3:
        all_hints_empty = all(
            record.get("identity_hint") in (None, "")
            for record in speaker_records
        )
        duplicate_records_are_safe = (
            all_hints_empty
            and not source_body_has_speaker_ambiguity
            and _duplicate_speaker_records_are_identical(speaker_records)
        )
        if duplicate_records_are_safe:
            for relation_index, raw_relation in enumerate(
                raw_relations[:MAX_RELATIONS_PER_PAYLOAD]
            ):
                if not isinstance(raw_relation, Mapping):
                    continue
                relation_subject_key = (
                    normalize_entity_name(raw_relation.get("subject")),
                    normalize_entity_type(raw_relation.get("subject_type")),
                )
                predicate = normalize_predicate(raw_relation.get("relation"))
                object_name = normalize_entity_name(raw_relation.get("object"))
                object_type = normalize_entity_type(
                    raw_relation.get("object_type")
                )
                has_object_declaration = bool(
                    object_name
                    and object_type
                    and raw_entities_by_key.get((object_name, object_type))
                )
                if (
                    raw_relation.get("explicit") is True
                    and relation_subject_key == speaker_raw_key
                    and predicate
                    and object_type
                    and endpoint_types_are_compatible(
                        predicate, "person", object_type
                    )
                    and has_object_declaration
                    and any(
                        witness.binding == "trusted_speaker_1p"
                        for witness in raw_relation_witnesses.get(
                            relation_index, ()
                        )
                    )
                ):
                    speaker_bound_relation_indexes.add(relation_index)
            speaker_group_policy = (
                "fold" if speaker_bound_relation_indexes else "drop"
            )

    entities = []
    trusted_api_speaker_entities: dict[tuple[str, str, str], Entity] = {}
    dropped_entities = max(0, len(raw_entities) - MAX_ENTITIES_PER_PAYLOAD)
    for index, raw_entity in enumerate(raw_entities[:MAX_ENTITIES_PER_PAYLOAD]):
        if not isinstance(raw_entity, Mapping):
            dropped_entities += 1
            continue
        display_name = _display_entity_name(raw_entity.get("name"))
        canonical_name = normalize_entity_name(raw_entity.get("name"))
        entity_type = normalize_entity_type(raw_entity.get("type"))
        raw_entity_key = (canonical_name, entity_type)
        if (
            source_text is not None
            and raw_entity_key == speaker_raw_key
            and speaker_group_policy == "drop"
        ):
            dropped_entities += 1
            continue
        contextual_correction = contextual_entity_corrections.get(
            (canonical_name, entity_type)
        )
        contextual_identity_hint = None
        if contextual_correction is not None:
            canonical_name, entity_type, contextual_identity_hint = (
                contextual_correction
            )
        corrected_types = entity_type_corrections.get(
            (canonical_name, entity_type), set()
        )
        if len(corrected_types) == 1:
            entity_type = next(iter(corrected_types))
        raw_identity_hint = raw_entity.get("identity_hint")
        normalized_raw_identity_hint = None
        if raw_identity_hint not in (None, ""):
            normalized_raw_identity_hint = normalize_entity_name(raw_identity_hint)
        speaker_identity_hint = None
        if (
            source_text is not None
            and raw_entity_key == speaker_raw_key
            and entity_type == "person"
            and speaker_group_policy in {"unique", "fold"}
        ):
            speaker_identity_hint = _api_speaker_identity_hint(
                speaker=speaker, canonical_name=canonical_name
            )
        if source_text is not None:
            # Model-provided hints are untrusted.  With source evidence
            # available, only a verified same-clause role/appositive or an
            # exact API speaker identity may become a persistent link key.
            identity_hint = speaker_identity_hint or contextual_identity_hint
        else:
            # Preserve the source-less parser contract used by offline callers;
            # production ingestion always supplies source_text.
            identity_hint = normalized_raw_identity_hint
        if (
            not display_name
            or not canonical_name
            or not entity_type
            or (
                source_text is None
                and raw_identity_hint not in (None, "")
                and not identity_hint
            )
        ):
            dropped_entities += 1
            continue
        entity = Entity(
            entity_id=_stable_id(
                "ent", user_id, source_message_id, index, canonical_name,
                entity_type, identity_hint or ""
            ),
            user_id=user_id,
            canonical_name=canonical_name,
            display_name=display_name,
            entity_type=entity_type,
            first_source_message_id=source_message_id,
            created_at=created_at,
            identity_hint=identity_hint,
        )
        if (
            source_text is not None
            and speaker_identity_hint is not None
            and speaker_group_policy == "fold"
        ):
            # Only trusted request metadata may collapse repeated local
            # declarations.  Model/null/occupation hints intentionally keep
            # same-name entities distinct so ordinary ambiguity still fails
            # closed.  Relations resolve by canonical name and exact type, so
            # retaining the first trusted declaration gives every duplicate
            # local speaker reference one unique endpoint.
            trusted_key = (
                canonical_name, entity_type, speaker_identity_hint
            )
            if trusted_key in trusted_api_speaker_entities:
                dropped_entities += 1
                continue
            trusted_api_speaker_entities[trusted_key] = entity
        entities.append(entity)

    relations = []
    dropped_relations = max(0, len(raw_relations) - MAX_RELATIONS_PER_PAYLOAD)
    ambiguous_relations = 0
    seen_relations = set()
    for index, raw_relation in enumerate(raw_relations[:MAX_RELATIONS_PER_PAYLOAD]):
        if not isinstance(raw_relation, Mapping) or raw_relation.get("explicit") is not True:
            dropped_relations += 1
            continue

        predicate = normalize_predicate(raw_relation.get("relation"))
        raw_subject_name = normalize_entity_name(raw_relation.get("subject"))
        raw_object_name = normalize_entity_name(raw_relation.get("object"))
        raw_subject_type = normalize_entity_type(raw_relation.get("subject_type"))
        raw_object_type = normalize_entity_type(raw_relation.get("object_type"))
        relation_witnesses = raw_relation_witnesses.get(index, ())
        textually_supported = source_text is None or bool(relation_witnesses)
        if (
            source_text is not None
            and speaker_group_policy == "fold"
            and (raw_subject_name, raw_subject_type) == speaker_raw_key
            and index not in speaker_bound_relation_indexes
        ):
            # A folded duplicate endpoint is valid only for the exact
            # first-person relation whose witness justified the fold.
            textually_supported = False
        subject_correction = contextual_entity_corrections.get(
            (raw_subject_name, raw_subject_type)
        )
        object_correction = contextual_entity_corrections.get(
            (raw_object_name, raw_object_type)
        )
        subject_name = (
            subject_correction[0] if subject_correction else raw_subject_name
        )
        subject_type = (
            subject_correction[1] if subject_correction else raw_subject_type
        )
        object_name = object_correction[0] if object_correction else raw_object_name
        if object_correction:
            object_type = object_correction[1]
        else:
            object_type = (
                normalize_relation_object_type(
                    predicate,
                    raw_relation.get("object"),
                    raw_relation.get("object_type"),
                )
                if source_text is not None
                and textually_supported
                and raw_object_name not in mixed_product_objects
                else raw_object_type
            )

        raw_temporal_status = raw_relation.get("temporal_status")
        witness_states = {
            (witness.state_change, witness.temporal_status)
            for witness in relation_witnesses
        }
        explicit_temporal_state = (
            next(iter(witness_states))
            if source_text is not None
            and textually_supported
            and len(witness_states) == 1
            and next(iter(witness_states))[0] != "assert"
            else None
        )
        if explicit_temporal_state is not None:
            state_change, temporal_status = explicit_temporal_state
            invalid_temporal_status = False
        elif source_text is not None:
            # Model-provided state labels are untrusted annotations. When the
            # source has no controlled, explicit state-change cue, fail closed
            # to an ordinary assertion instead of preserving a speculative
            # ``current``/``previous`` label. The source-less parser contract
            # below remains backward compatible for trusted internal callers.
            state_change = "assert"
            temporal_status = None
            invalid_temporal_status = False
        else:
            state_change = normalize_state_change(
                raw_relation.get("state_change", "assert")
            )
            temporal_status = normalize_temporal_status(raw_temporal_status)
            invalid_temporal_status = (
                raw_temporal_status is not None and temporal_status is None
            )
        if (
            not predicate
            or not subject_name
            or not object_name
            or not subject_type
            or not object_type
            or not state_change
            or invalid_temporal_status
            or not endpoint_types_are_compatible(
                predicate, subject_type, object_type
            )
            or not textually_supported
        ):
            dropped_relations += 1
            continue

        subject = resolve_entity_reference(
            entities, subject_name, subject_type
        )
        object_ = resolve_entity_reference(
            entities, object_name, object_type
        )
        if not subject.resolved or not object_.resolved:
            if subject.status == "ambiguous" or object_.status == "ambiguous":
                ambiguous_relations += 1
            dropped_relations += 1
            continue
        assert subject.entity is not None and object_.entity is not None

        relation_key = (
            subject.entity.entity_id,
            predicate,
            object_.entity.entity_id,
            state_change,
            temporal_status,
        )
        if relation_key in seen_relations:
            dropped_relations += 1
            continue
        seen_relations.add(relation_key)
        relations.append(
            GraphRelation(
                relation_id=_stable_id(
                    "edge",
                    user_id,
                    source_message_id,
                    index,
                    *relation_key,
                ),
                user_id=user_id,
                subject_entity_id=subject.entity.entity_id,
                predicate=predicate,
                object_entity_id=object_.entity.entity_id,
                source_message_id=source_message_id,
                state_change=state_change,
                temporal_status=temporal_status,
                event_ts=event_ts,
                # Model output is never trusted to choose an existing edge.
                # Storage may set this later only after explicit update checks.
                supersedes_edge_id=None,
                created_at=created_at,
                explicit=True,
            )
        )

    return GraphPayload(
        entities=tuple(entities),
        relations=tuple(relations),
        dropped_entities=dropped_entities,
        dropped_relations=dropped_relations,
        ambiguous_relations=ambiguous_relations,
    )
