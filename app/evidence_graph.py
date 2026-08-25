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
    # Internal provenance carrier.  It is deliberately omitted from the
    # serialized public graph contract; storage persists and independently
    # validates it before an edge can be traversed.
    support_witness: Optional[SupportWitness] = None

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result.pop("support_witness", None)
        return result


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
_FIRST_PERSON_I_PREDICATE_FRAMES: Mapping[str, Tuple[str, ...]] = {
    "friend_of": (r"(?:am|was)\s+(?:a\s+)?friend\s+of",),
    "parent_of": (r"(?:am|was)\s+(?:a\s+)?(?:parent|mother|father)\s+of",),
    "sibling_of": (r"(?:am|was)\s+(?:a\s+)?(?:sibling|brother|sister)\s+of",),
    "partner_of": (r"(?:am|was)\s+(?:a\s+)?(?:partner|spouse|husband|wife)\s+of",),
    "works_at": (
        r"(?:work|worked)\s+(?:at|for)",
        r"(?:am|was)\s+employed\s+by",
        r"(?:work|worked)\s+as\s+{}\s+at".format(
            _role_np_frame_pattern()
        ),
    ),
    "role_at": (
        r"(?:have|had|hold|held)\s+(?:an?\s+|the\s+)?(?:formal\s+)?(?:role|position)\s+at",
        r"(?:serve|served)\s+(?:in\s+)?{}\s+at".format(
            _role_np_frame_pattern()
        ),
    ),
    "lives_in": (r"(?:live|lived)\s+in", r"moved\s+from"),
    "located_in": (r"(?:am|was)\s+(?:located|based|situated)\s+in",),
    "likes": (
        r"(?:like|liked|love|loved|enjoy|enjoyed)",
        r"(?:am|was)\s+(?:keen\s+on|fond\s+of)",
        r"(?:have|had)\s+(?:always\s+)?(?:had\s+)?(?:a\s+)?love\s+(?:of|for)",
    ),
    "prefers": (r"(?:prefer|preferred)",),
    "dislikes": (r"(?:dislike|disliked|hate|hated)",),
    "member_of": (
        r"(?:am|was|became)\s+(?:a\s+)?member\s+of",
        r"(?:belong|belonged)\s+to",
        r"(?:join|joined)",
        r"(?:am|was)\s+no\s+longer\s+(?:a\s+)?member\s+of",
    ),
    "participated_in": (
        r"(?:participate|participated)\s+in",
        r"(?:take|took)\s+part\s+in",
        r"(?:attend|attended)",
        r"(?:join|joined)",
        r"went\s+to",
    ),
    "owns": (r"(?:own|owned|possess|possessed)",),
    "created": (
        r"(?:create|created|make|made|build|built|write|wrote|design|designed|"
        r"compose|composed|produce|produced)",
    ),
    "requires": (r"(?:explicitly\s+)?(?:require|required)",),
    "prohibits": (
        r"(?:explicitly\s+)?(?:prohibit|prohibited|forbid|forbade)",
    ),
    "permits": (
        r"(?:explicitly\s+)?(?:permit|permitted|allow|allowed)",
    ),
    "changed_to": (
        r"(?:change|changed|switch|switched|transition|transitioned)\s+to",
        r"became",
    ),
    "replaces": (r"(?:replace|replaced)",),
}

_FIRST_PERSON_PERFECT_PREDICATE_FRAMES: Mapping[str, Tuple[str, ...]] = {
    "friend_of": (),
    "parent_of": (),
    "sibling_of": (),
    "partner_of": (),
    "works_at": (r"worked\s+(?:at|for)", r"employed\s+by"),
    "role_at": (
        r"(?:had|held)\s+(?:an?\s+|the\s+)?(?:formal\s+)?(?:role|position)\s+at",
        r"served\s+(?:in\s+)?{}\s+at".format(_role_np_frame_pattern()),
    ),
    "lives_in": (r"lived\s+in", r"moved\s+from"),
    "located_in": (),
    "likes": (
        r"(?:liked|loved|enjoyed)",
        r"had\s+(?:always\s+)?(?:had\s+)?(?:a\s+)?love\s+(?:of|for)",
    ),
    "prefers": (r"preferred",),
    "dislikes": (r"(?:disliked|hated)",),
    "member_of": (r"(?:belonged\s+to|joined)",),
    "participated_in": (
        r"participated\s+in", r"taken\s+part\s+in", r"attended", r"joined",
    ),
    "owns": (r"(?:owned|possessed)",),
    "created": (
        r"(?:created|made|built|written|designed|composed|produced)",
    ),
    "requires": (r"(?:explicitly\s+)?required",),
    "prohibits": (r"(?:explicitly\s+)?prohibited",),
    "permits": (r"(?:explicitly\s+)?(?:permitted|allowed)",),
    "changed_to": (r"(?:changed|switched|transitioned)\s+to",),
    "replaces": (r"replaced",),
}

_UNSUPPORTED_CLAIM_PATTERN = re.compile(
    r"\b(ignore|invent|pretend|fabricate|hypothetical|fictional|not a fact|"
    r"not true|false claim|untrusted instruction|example only|merely an example)\b",
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


def _surface_sentences(source_text: str) -> Tuple[str, ...]:
    surface = unicodedata.normalize("NFKC", source_text)
    surface = re.sub(r"[\t\r\f\v ]+", " ", surface)
    surface = re.sub(r"\s*\n\s*", "\n", surface).strip()
    return tuple(
        " ".join(part.split())
        for part in re.split(r"(?<=[.!?。！？])\s+|\n+", surface)
        if part.strip()
    )


def _lexical_word_character(character: str) -> bool:
    return bool(character) and (character.isalnum() or character == "_")


_REPORTING_COLON_PATTERN = re.compile(
    r"\b(?:said|says|wrote|writes|reported|reports|claimed|claims|"
    r"asked|asks|replied|replies|noted|notes|stated|states)\b"
    r"[^:;.!?。！？]{0,80}:|\baccording\s+to\b[^:;.!?。！？]{1,80}:"
)


def _untrusted_source_spans(normalized_source: str) -> Tuple[Tuple[int, int], ...]:
    """Return lexical quote, bracket, and attributed-reporting scopes.

    The stack spans sentence boundaries.  Straight/smart apostrophes inside a
    word (and postfix possessives without an opening quote) remain lexical,
    while paired single quotes are treated exactly like double quotes.  Any
    unmatched opening delimiter governs the rest of the message fail-closed.
    """

    paired_openings = {
        "“": "”",
        "„": "”",
        "‘": "’",
        "‚": "’",
        "«": "»",
        "(": ")",
        "[": "]",
        "{": "}",
    }
    paired_closings = frozenset(paired_openings.values())
    symmetric = frozenset({'"', "'"})
    opening_to_closing_category = {"Pi": "Pf", "Ps": "Pe"}
    closing_categories = frozenset(opening_to_closing_category.values())
    # Each stack item carries an exact closer when the pair is known.  Unicode
    # delimiters outside the small ASCII/smart-punctuation pair table instead
    # carry only the matching closing category.  LIFO category matching keeps
    # arbitrary Pi/Pf and Ps/Pe nesting scoped across sentence boundaries.
    stack: list[tuple[Optional[str], Optional[str], int]] = []
    spans = []

    for index, character in enumerate(normalized_source):
        previous = normalized_source[index - 1] if index else ""
        following = (
            normalized_source[index + 1]
            if index + 1 < len(normalized_source)
            else ""
        )
        previous_is_word = _lexical_word_character(previous)
        following_is_word = _lexical_word_character(following)
        category = unicodedata.category(character)

        if character in {"'", "’"} and previous_is_word:
            matching_single_quote = bool(
                stack and stack[-1][0] == character
            )
            if following_is_word:
                # Word-internal contractions and possessives: I'm, don't,
                # Alice's.  A matching outer quote remains on the stack.
                continue
            if matching_single_quote and previous.casefold() == "s":
                # A terminal/postfix plural or s-ending-name possessive inside
                # a single-quoted scope is lexically indistinguishable from a
                # closer.  Keeping the opener active is fail-closed: a real
                # closer may conservatively govern extra prose, whereas a
                # possessive can never terminate the untrusted container.
                continue
            remainder = normalized_source[index + 1 :]
            next_nonspace = remainder.lstrip()[:1]
            if (
                following.isspace()
                and _lexical_word_character(next_nonspace)
            ):
                # Postfix possessives such as ``students' studio`` remain
                # lexical even inside a single-quoted container.  Treating the
                # possessive as the outer closer could expose every later
                # sentence when the real quote is missing.  If this was
                # instead a close followed by ordinary prose, retaining the
                # opener to message end is the conservative outcome.
                continue

        if character in symmetric:
            if stack and stack[-1][0] == character:
                _, _, start = stack.pop()
                spans.append((start, index + 1))
            elif following or following_is_word:
                stack.append((character, None, index))
            else:
                # A closing delimiter without a trustworthy opener makes the
                # preceding source ambiguous; mask the prefix fail-closed.
                spans.append((0, index + 1))
            continue

        if character in paired_openings:
            expected_character = paired_openings[character]
            stack.append(
                (
                    expected_character,
                    unicodedata.category(expected_character),
                    index,
                )
            )
            continue
        if category in opening_to_closing_category:
            stack.append((None, opening_to_closing_category[category], index))
            continue
        if character in paired_closings or category in closing_categories:
            exact_match = bool(stack and stack[-1][0] == character)
            category_match = bool(
                stack
                and stack[-1][0] is None
                and stack[-1][1] == category
            )
            if exact_match or category_match:
                _, _, start = stack.pop()
                spans.append((start, index + 1))
            else:
                # Do not search down through a mismatched nesting level: both
                # the unmatched closer's prefix and any still-open scopes are
                # untrusted.  This prevents a malformed close from exposing a
                # complete middle sentence as an assertion.
                spans.append((0, index + 1))

    spans.extend((start, len(normalized_source)) for _, _, start in stack)
    # ``Bob wrote: ...`` without a lexical delimiter is still a closed
    # attributed scope.  It lasts to message end because no trusted boundary
    # tells us where the reported material stops.
    for match in _REPORTING_COLON_PATTERN.finditer(normalized_source):
        remainder = normalized_source[match.end():].lstrip()
        if remainder and (
            remainder[0] in symmetric
            or remainder[0] in paired_openings
            or unicodedata.category(remainder[0])
            in opening_to_closing_category
        ):
            continue
        spans.append((match.end() - 1, len(normalized_source)))
    if not spans:
        return ()
    merged = []
    for start, end in sorted(spans):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return tuple(merged)


_PRIOR_GOVERNANCE_MARKER_PATTERN = re.compile(
    r"\b(?:report(?:s|ed|ing)?|say|says|said|saying|write|writes|wrote|"
    r"written|writing|quote|quotes|quoted|quoting|quotation|claim|claims|"
    r"claimed|claiming|allege|alleges|alleged|alleging|tell|tells|told|"
    r"telling|ask|asks|asked|asking|hear|hears|heard|hearing|read|reads|"
    r"reading|suppose|supposes|supposed|supposing|pretend|pretends|"
    r"pretended|pretending|imagine|imagines|imagined|imagining|treat|"
    r"treats|treated|treating|ignore|ignores|ignored|ignoring|disregard|"
    r"disregards|disregarded|disregarding|hypothetical|fiction|fictional|"
    r"lie|lies|lied|lying|false|untrue|example|examples|scenario|scenarios|"
    r"fabricate|fabricates|fabricated|fabricating|invent|invents|invented|"
    r"inventing|attribut(?:e|es|ed|ing)|according|correction|"
    r"made[\s-]+up|not\s+(?:a\s+)?fact|not\s+factual|not\s+true)\b"
)


def _spans_overlap(
    left: Tuple[int, int], right: Tuple[int, int]
) -> bool:
    return left[0] < right[1] and right[0] < left[1]


def _sentence_index_for_witness(
    *,
    witness: SupportWitness,
    sentences: Sequence[Tuple[str, int]],
) -> Optional[int]:
    for index, (sentence, start) in enumerate(sentences):
        if start <= witness.source_span[0] < start + len(sentence):
            return index
    return None


def _top_level_sentence_body(
    *,
    sentence: str,
    sentence_start: int,
    untrusted_spans: Sequence[Tuple[int, int]],
    normalized_speaker: Optional[str],
) -> str:
    masked = _sentence_with_masked_spans(
        sentence=sentence,
        sentence_start=sentence_start,
        spans=untrusted_spans,
    )
    body = " ".join(masked.split())
    body, _ = _strip_speaker_prefix(body, normalized_speaker)
    return body


def _sentence_with_masked_spans(
    *,
    sentence: str,
    sentence_start: int,
    spans: Sequence[Tuple[int, int]],
) -> str:
    characters = list(sentence)
    sentence_end = sentence_start + len(sentence)
    for span_start, span_end in spans:
        overlap_start = max(sentence_start, span_start)
        overlap_end = min(sentence_end, span_end)
        if overlap_start >= overlap_end:
            continue
        local_start = overlap_start - sentence_start
        local_end = overlap_end - sentence_start
        characters[local_start:local_end] = " " * (local_end - local_start)
    return "".join(characters)


_TOKEN_NEGATOR_PATTERN = re.compile(
    r"(?<![\w])(?:not|no|never)(?![\w])|n['’]t\b"
)
_RETRACTION_OR_FALSE_PATTERN = re.compile(
    r"\bretract(?:ed|s|ing)?\b|"
    r"\b(?:take|takes|took)\b[^.!?。！？]{0,32}\bback\b|"
    r"\b(?:withdraw(?:n|s|ing)?|recant(?:ed|s|ing)?|revoke(?:d|s|ing)?|"
    r"disavow(?:ed|s|ing)?|scratch(?:ed|es|ing)?|correction|correcting|"
    r"false|untrue|wrong|lie|fiction(?:al)?|made[\s-]+up|"
    r"not\s+(?:a\s+)?fact|not\s+true)\b"
)
_SCOPE_WORD_PATTERN = re.compile(r"[^\W_]+(?:['’][^\W_]+)?", re.UNICODE)
_FORWARD_TRIGGER_TOKEN_PATTERN = re.compile(
    r"(?:report(?:s|ed|ing)?|say|says|said|saying|write|writes|wrote|"
    r"written|writing|quote|quotes|quoted|quoting|quotation|claim|claims|"
    r"claimed|claiming|allege|alleges|alleged|alleging|tell|tells|told|"
    r"telling|ask|asks|asked|asking|hear|hears|heard|hearing|read|reads|"
    r"reading|suppose|supposes|supposed|supposing|pretend|pretends|"
    r"pretended|pretending|imagine|imagines|imagined|imagining|treat|"
    r"treats|treated|treating|ignore|ignores|ignored|ignoring|disregard|"
    r"disregards|disregarded|disregarding|hypothetical|example|examples|"
    r"scenario|scenarios|fabricate|fabricates|fabricated|fabricating|"
    r"invent|invents|invented|inventing|attribut(?:e|es|ed|ing)|according)"
)
_FORWARD_TARGET_NOUNS = frozenset(
    {
        "claim",
        "example",
        "instruction",
        "passage",
        "sentence",
        "statement",
        "text",
    }
)
_FORWARD_SCAFFOLD_TAIL_TOKENS = frozenset(
    {
        "a",
        "allegedly",
        "an",
        "are",
        "as",
        "attributed",
        "be",
        "being",
        "by",
        "false",
        "fact",
        "factual",
        "fiction",
        "fictional",
        "from",
        "hypothetical",
        "is",
        "lie",
        "made",
        "merely",
        "not",
        "only",
        "quote",
        "quoted",
        "reported",
        "said",
        "the",
        "true",
        "untrusted",
        "up",
        "was",
        "were",
        "will",
        "written",
    }
).union(_FORWARD_TARGET_NOUNS)
_FIRST_PERSON_ANAPHORA_PATTERN = re.compile(
    r"\b(?:i|me|my|mine|myself|we|us|our|ours|ourselves)\b"
)
_GENERAL_ANAPHORA_PATTERN = re.compile(
    r"\b(?:i|me|my|mine|myself|we|us|our|ours|ourselves|he|him|his|"
    r"himself|she|her|hers|herself|they|them|their|theirs|themselves|"
    r"it|its|itself|this|that|these|those|former|latter)\b"
)
_EXPLICIT_NEGATIVE_REMAINDER_PATTERN = re.compile(
    r"^(?:[^\W_][\w'’\-]*[\s,]+){0,3}(?:"
    r"(?:do|does|did|am|is|are|was|were|have|has|had|can|could|will|"
    r"would|should|must)\s+(?:[^\W_][\w'’\-]*\s+){0,2}not\b|"
    r"(?:don['’]t|doesn['’]t|didn['’]t|isn['’]t|aren['’]t|wasn['’]t|"
    r"weren['’]t|haven['’]t|hasn['’]t|hadn['’]t|can['’]t|couldn['’]t|"
    r"won['’]t|wouldn['’]t|shouldn['’]t|mustn['’]t)\b|never\b|"
    r"(?:have|has|had)\s+no\b)"
)
_INITIALS_SURFACE_SUBJECT_PATTERN = re.compile(
    r"^(?P<name>[A-Z](?![\w])(?:(?:\s*[.,-]\s*|\s+)[A-Z](?![\w]))+)"
)
_NAME_HONORIFICS = frozenset(
    {
        "capt",
        "captain",
        "dame",
        "dr",
        "lady",
        "lord",
        "miss",
        "mr",
        "mrs",
        "ms",
        "prof",
        "professor",
        "rev",
        "reverend",
        "sir",
    }
)
_NAME_SUFFIXES = frozenset({"ii", "iii", "iv", "jr", "sr"})


def _substantive_name_tokens(normalized_name: str) -> Tuple[str, ...]:
    tokens = re.findall(r"[^\W_]+", normalized_name, flags=re.UNICODE)
    while tokens and tokens[0] in _NAME_HONORIFICS:
        tokens.pop(0)
    while tokens and tokens[-1] in _NAME_SUFFIXES:
        tokens.pop()
    return tuple(tokens)


def _names_have_identity_overlap(left: str, right: str) -> bool:
    """Conservatively detect full, partial, or alias-like name overlap."""

    left_tokens = _substantive_name_tokens(left)
    right_tokens = _substantive_name_tokens(right)
    if not left_tokens or not right_tokens:
        return True
    if set(left_tokens).intersection(right_tokens):
        return True
    # Ordinary multi-character prefix abbreviations cannot prove a new identity.
    if any(
        (
            len(right_token) >= 2
            and left_token.startswith(right_token)
        )
        or (
            len(left_token) >= 2
            and right_token.startswith(left_token)
        )
        for left_token in left_tokens
        for right_token in right_tokens
    ):
        return True

    left_compact = "".join(left_tokens)
    right_compact = "".join(right_tokens)
    if left_compact == right_compact:
        return True

    left_acronym = "".join(token[0] for token in left_tokens)
    right_acronym = "".join(token[0] for token in right_tokens)

    def acronym_abbreviation(tokens: Tuple[str, ...]) -> Optional[str]:
        if len(tokens) == 1 and len(tokens[0]) >= 2:
            return tokens[0]
        if len(tokens) >= 2 and all(len(token) == 1 for token in tokens):
            return "".join(tokens)
        return None

    def acronym_is_ordered_subsequence(
        abbreviation: str, full_initials: str
    ) -> bool:
        if len(abbreviation) < 2 or len(abbreviation) > len(full_initials):
            return False
        abbreviation_index = 0
        for initial in full_initials:
            if abbreviation[abbreviation_index] == initial:
                abbreviation_index += 1
                if abbreviation_index == len(abbreviation):
                    return True
        return False

    left_abbreviation = acronym_abbreviation(left_tokens)
    right_abbreviation = acronym_abbreviation(right_tokens)
    if (
        left_abbreviation is not None
        and len(right_tokens) >= 2
        and acronym_is_ordered_subsequence(
            left_abbreviation, right_acronym
        )
    ) or (
        right_abbreviation is not None
        and len(left_tokens) >= 2
        and acronym_is_ordered_subsequence(
            right_abbreviation, left_acronym
        )
    ):
        return True

    has_unsegmented_non_latin = (
        (" " not in left and any(ord(character) > 127 for character in left))
        or (" " not in right and any(ord(character) > 127 for character in right))
    )
    if has_unsegmented_non_latin and (
        left_compact in right_compact or right_compact in left_compact
    ):
        return True
    return False


def _forward_marker_scaffold_spans(
    sentence: str,
) -> Tuple[Tuple[int, int], ...]:
    """Locate minimal forward-governance token scaffolds in one sentence."""

    tokens = tuple(
        (match.group(0), match.start(), match.end())
        for match in _SCOPE_WORD_PATTERN.finditer(sentence)
    )
    if not tokens:
        return ()

    def is_trigger(index: int) -> bool:
        return bool(
            _FORWARD_TRIGGER_TOKEN_PATTERN.fullmatch(tokens[index][0])
        )

    def prior_trigger(target_index: int) -> Optional[int]:
        for index in range(target_index - 1, max(-1, target_index - 8), -1):
            token = tokens[index][0]
            if token in {"not", "no", "never"} or token.endswith("n't"):
                break
            if is_trigger(index):
                return index
        return None

    spans = []
    for index, (token, _, _) in enumerate(tokens):
        target_end = None
        if (
            token in {"next", "following"}
            and index + 1 < len(tokens)
            and tokens[index + 1][0] in _FORWARD_TARGET_NOUNS
        ):
            target_end = index + 1
        elif (
            token == "what"
            and index + 1 < len(tokens)
            and tokens[index + 1][0] == "follows"
        ):
            target_end = index + 1
        elif token in {"this", "that", "following"}:
            trigger_index = prior_trigger(index)
            if trigger_index is not None:
                target_end = index
        if target_end is None:
            continue

        trigger_index = prior_trigger(index)
        scaffold_start = trigger_index if trigger_index is not None else index
        scaffold_end = target_end
        while (
            scaffold_end + 1 < len(tokens)
            and tokens[scaffold_end + 1][0]
            in _FORWARD_SCAFFOLD_TAIL_TOKENS
        ):
            scaffold_end += 1
        spans.append(
            (tokens[scaffold_start][1], tokens[scaffold_end][2])
        )

    has_backward_signal = bool(
        _TOKEN_NEGATOR_PATTERN.search(sentence)
        or _RETRACTION_OR_FALSE_PATTERN.search(sentence)
    )
    if not spans and not has_backward_signal:
        spans.extend(
            (start, end)
            for token, start, end in tokens
            if _FORWARD_TRIGGER_TOKEN_PATTERN.fullmatch(token)
        )

    merged = []
    for start, end in sorted(spans):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return tuple(merged)


def _sentence_scope_indexes(
    *,
    sentences: Sequence[Tuple[str, int]],
    untrusted_spans: Sequence[Tuple[int, int]],
    normalized_speaker: Optional[str],
) -> Tuple[
    frozenset[int],
    frozenset[int],
    Tuple[Tuple[Tuple[int, int], ...], ...],
]:
    """Return broad markers, governed targets, and forward clause spans.

    A broad marker blocks witnesses in every later sentence, matching the
    all-prior positive-grammar contract.  A forward marker clause cannot act
    as a backward withdrawal and governs its complete immediate target.  Any
    independent material outside that clause remains eligible to withdraw an
    earlier assertion.  Chained forward markers each govern their own target.
    """

    marker_indexes = set()
    governed_target_indexes = set()
    marker_spans_by_sentence = [[] for _ in sentences]
    for index, (sentence, start) in enumerate(sentences):
        masked_sentence = _sentence_with_masked_spans(
            sentence=sentence,
            sentence_start=start,
            spans=untrusted_spans,
        )
        body = " ".join(masked_sentence.split())
        if _PRIOR_GOVERNANCE_MARKER_PATTERN.search(body):
            marker_indexes.add(index)
        for scaffold_start, scaffold_end in _forward_marker_scaffold_spans(
            masked_sentence
        ):
            marker_spans_by_sentence[index].append(
                (start + scaffold_start, start + scaffold_end)
            )
            if index + 1 < len(sentences):
                governed_target_indexes.add(index + 1)
    return (
        frozenset(marker_indexes),
        frozenset(governed_target_indexes),
        tuple(tuple(spans) for spans in marker_spans_by_sentence),
    )


def _surface_body_without_speaker(
    sentence: str, normalized_speaker: Optional[str]
) -> str:
    if not normalized_speaker:
        return sentence
    match = re.match(
        r"{}\s*:\s*".format(_escaped_exact_name(normalized_speaker)),
        sentence,
        flags=re.IGNORECASE,
    )
    if match is None:
        return sentence
    return sentence[match.end():].lstrip()


def _strip_leading_clause_joiners(value: str) -> str:
    return re.sub(
        r"^(?:(?:[;,]\s*)|(?:(?:and|but)\s+))+",
        "",
        value.lstrip(),
        flags=re.IGNORECASE,
    )


def _later_sentence_proves_different_subject(
    *,
    normalized_body: str,
    surface_body: str,
    witness: SupportWitness,
    normalized_subject: str,
) -> bool:
    if _entity_spans(normalized_body, normalized_subject):
        return False
    anaphora_pattern = (
        _FIRST_PERSON_ANAPHORA_PATTERN
        if witness.binding == "trusted_speaker_1p"
        else _GENERAL_ANAPHORA_PATTERN
    )
    if anaphora_pattern.search(normalized_body):
        return False
    named = _INITIALS_SURFACE_SUBJECT_PATTERN.match(surface_body)
    if named is None:
        named = re.match(
            r"^(?P<name>(?!The\b|A\b|An\b|This\b|That\b|These\b|Those\b)"
            r"[A-Z][\w'’\-]*(?:\s+[A-Z][\w'’\-]*){0,3})\b",
            surface_body,
        )
    if named is None:
        return False
    first_name_token = named.group("name").split(maxsplit=1)[0].casefold()
    if first_name_token.endswith("ly"):
        return False
    explicit_subject = normalize_entity_name(named.group("name"))
    if not explicit_subject or explicit_subject == normalized_subject:
        return False
    if _names_have_identity_overlap(explicit_subject, normalized_subject):
        return False
    if not normalized_body.startswith(explicit_subject):
        return False
    remainder = normalized_body[len(explicit_subject):].lstrip()
    return bool(_EXPLICIT_NEGATIVE_REMAINDER_PATTERN.match(remainder))


def _later_source_withdraws_assertion(
    *,
    witness: SupportWitness,
    sentences: Sequence[Tuple[str, int]],
    surface_sentences: Sequence[str],
    untrusted_spans: Sequence[Tuple[int, int]],
    governed_target_indexes: frozenset[int],
    forward_marker_spans_by_sentence: Sequence[
        Sequence[Tuple[int, int]]
    ],
    normalized_subject: str,
    normalized_speaker: Optional[str],
) -> bool:
    if witness.state_change != "assert":
        return False
    current_index = _sentence_index_for_witness(
        witness=witness, sentences=sentences
    )
    if current_index is None or current_index + 1 >= len(sentences):
        return False
    surfaces_align = len(surface_sentences) == len(sentences)
    for index in range(current_index + 1, len(sentences)):
        if index in governed_target_indexes:
            continue
        sentence, start = sentences[index]
        masked_spans = (
            *untrusted_spans,
            *forward_marker_spans_by_sentence[index],
        )
        body = _top_level_sentence_body(
            sentence=sentence,
            sentence_start=start,
            untrusted_spans=masked_spans,
            normalized_speaker=normalized_speaker,
        )
        body = _strip_leading_clause_joiners(body)
        if not body or not (
            _TOKEN_NEGATOR_PATTERN.search(body)
            or _RETRACTION_OR_FALSE_PATTERN.search(body)
        ):
            continue
        surface_body = ""
        if (
            surfaces_align
            and len(surface_sentences[index]) == len(sentence)
        ):
            masked_surface = _sentence_with_masked_spans(
                sentence=surface_sentences[index],
                sentence_start=start,
                spans=masked_spans,
            )
            surface_body = _surface_body_without_speaker(
                " ".join(masked_surface.split()), normalized_speaker
            )
            surface_body = _strip_leading_clause_joiners(surface_body)
        if surface_body and _later_sentence_proves_different_subject(
            normalized_body=body,
            surface_body=surface_body,
            witness=witness,
            normalized_subject=normalized_subject,
        ):
            continue
        return True
    return False


def _witness_is_in_assertion_scope(
    *,
    witness: SupportWitness,
    sentences: Sequence[Tuple[str, int]],
    surface_sentences: Sequence[str],
    untrusted_spans: Sequence[Tuple[int, int]],
    marker_sentence_indexes: frozenset[int],
    governed_target_indexes: frozenset[int],
    forward_marker_spans_by_sentence: Sequence[
        Sequence[Tuple[int, int]]
    ],
    normalized_subject: str,
    normalized_speaker: Optional[str],
) -> bool:
    if any(
        _spans_overlap(witness.source_span, untrusted_span)
        for untrusted_span in untrusted_spans
    ):
        return False
    sentence_index = _sentence_index_for_witness(
        witness=witness, sentences=sentences
    )
    if (
        sentence_index is None
        or sentence_index in governed_target_indexes
        or bool(forward_marker_spans_by_sentence[sentence_index])
        or any(
            marker_index < sentence_index
            for marker_index in marker_sentence_indexes
        )
    ):
        return False
    return not _later_source_withdraws_assertion(
        witness=witness,
        sentences=sentences,
        surface_sentences=surface_sentences,
        untrusted_spans=untrusted_spans,
        governed_target_indexes=governed_target_indexes,
        forward_marker_spans_by_sentence=(
            forward_marker_spans_by_sentence
        ),
        normalized_subject=normalized_subject,
        normalized_speaker=normalized_speaker,
    )


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
    *, subject_text: str, predicate: str, predicate_text: str
) -> bool:
    subject_text = subject_text.casefold()
    predicate_text = " ".join(predicate_text.casefold().split())
    if subject_text == "i":
        frames = _FIRST_PERSON_I_PREDICATE_FRAMES.get(predicate, ())
    elif subject_text in {"i'm", "i’m"}:
        frames = (
            (r"(?:keen\s+on|fond\s+of)",)
            if predicate == "likes"
            else ()
        )
    elif subject_text in {"i've", "i’ve", "i'd", "i’d"}:
        frames = _FIRST_PERSON_PERFECT_PREDICATE_FRAMES.get(predicate, ())
    else:
        return False
    return any(re.fullmatch(frame, predicate_text) for frame in frames)


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
        predicate=predicate,
        predicate_text=match.group("predicate"),
    ):
        return None
    first_cue = match.group("predicate").casefold().split()[0]
    if binding == "named" and first_cue in {"am", "keen", "fond", "written"}:
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
    r"^(?:"
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

_COORDINATED_GENERIC_OBJECT = (
    r"(?:(?:a|an|the|this|that|these|those|my|our|your|his|her|their|"
    r"another)\s+)?(?!no\b)[^\W_][\w'’+\-]*"
    r"(?:\s+[^\W_][\w'’+\-]*){0,11}\s*[.!。！]?"
)
_COORDINATED_LIST_OBJECT = (
    r"(?:(?:a|an|the|this|that|these|those|my|our|your|his|her|their|"
    r"another)\s+)?(?!no\b)[^\W_][\w'’+\-]*"
    r"(?:\s+[^\W_][\w'’+\-]*)?"
)
_COORDINATED_NON_NOMINAL_TOKEN = re.compile(
    r"\b(?:and|but|however|yet|although|whereas|while|because|if|unless|"
    r"except|after|before|since|not|no|never|maybe|perhaps|possibly)\b|"
    r"n['’]t\b"
)


def _coordinated_predicate_head_is_grammatical(
    *, conjunct_body: str, binding: str, subject_text: str
) -> bool:
    if not _COORDINATED_PREDICATE_HEAD.fullmatch(conjunct_body):
        return False
    if _COORDINATED_NON_NOMINAL_TOKEN.search(conjunct_body):
        return False
    if binding != "trusted_speaker_1p":
        return True
    # Inherited first-person coordination is a closed ``I P O and P O``
    # grammar.  Contracted auxiliaries do not safely govern an arbitrary next
    # predicate, so they remain unsupported here.
    if subject_text != "i":
        return False
    return any(
        re.fullmatch(
            r"(?:{})\s+{}".format(frame, _COORDINATED_GENERIC_OBJECT),
            conjunct_body,
        )
        for frames in _FIRST_PERSON_I_PREDICATE_FRAMES.values()
        for frame in frames
    )


def _coordinated_first_predicate(
    *,
    first_conjunct: str,
    subject_span: Tuple[int, int],
    binding: str,
    subject_text: str,
) -> Optional[Tuple[str, str, Tuple[int, int]]]:
    remainder = first_conjunct[subject_span[1]:]
    leading = len(remainder) - len(remainder.lstrip())
    body = remainder.strip()
    body_offset = subject_span[1] + leading
    matches = []
    for predicate, core_templates in _PREDICATE_FRAME_CORES.items():
        for core_template in core_templates:
            core = core_template.format(
                object=_COORDINATED_GENERIC_OBJECT,
                role=_role_np_frame_pattern(),
            )
            match = re.fullmatch(core, body)
            if match is None:
                continue
            predicate_text = match.group("predicate")
            if binding == "trusted_speaker_1p" and not _first_person_frame_is_grammatical(
                subject_text=subject_text,
                predicate=predicate,
                predicate_text=predicate_text,
            ):
                continue
            first_cue = predicate_text.casefold().split()[0]
            if binding == "named" and first_cue in {"am", "keen", "fond", "written"}:
                continue
            matches.append(
                (
                    predicate,
                    predicate_text,
                    (
                        body_offset + match.start("predicate"),
                        body_offset + match.end("predicate"),
                    ),
                )
            )
    unique = tuple(dict.fromkeys(matches))
    if len(unique) != 1:
        return None
    return unique[0]


def _coordinated_object_continuation(conjunct: str) -> Optional[str]:
    # An inherited-object list item is accepted only in the explicit
    # ``P O1 and O2, and P2 O3`` shape.  The terminal comma is the closed
    # boundary that prevents an arbitrary subject/verb tail from becoming O2.
    if not conjunct.endswith(","):
        return None
    object_text = conjunct[:-1].strip()
    if (
        not object_text
        or _COORDINATED_NON_NOMINAL_TOKEN.search(object_text)
        or not re.fullmatch(_COORDINATED_LIST_OBJECT, object_text)
    ):
        return None
    return object_text


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
        if not match:
            continue
        subject_text = match.group("subject")
        if binding == "trusted_speaker_1p" and subject_text not in _FIRST_PERSON_SUBJECTS:
            continue
        if not _coordinated_predicate_head_is_grammatical(
            conjunct_body=match.group("body"),
            binding=binding,
            subject_text=subject_text,
        ):
            continue
        return binding, match.span("subject"), subject_text
    return None


def _coordinated_conjuncts(body: str) -> Tuple[Tuple[str, int], ...]:
    separators = tuple(re.finditer(r"\s+and\s+", body))
    if not separators:
        return ()
    result = []
    raw_start = 0
    for separator in (*separators, None):
        raw_end = separator.start() if separator is not None else len(body)
        raw = body[raw_start:raw_end]
        leading = len(raw) - len(raw.lstrip())
        conjunct = raw.strip()
        if not conjunct:
            return ()
        result.append((conjunct, raw_start + leading))
        if separator is not None:
            raw_start = separator.end()
    return tuple(result)


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
    normalized_source = _normalized_source(source_text)
    sentences = _normalized_sentences(source_text)
    surface_sentences = _surface_sentences(source_text)
    untrusted_spans = _untrusted_source_spans(normalized_source)
    scope_indexes = _sentence_scope_indexes(
        sentences=sentences,
        untrusted_spans=untrusted_spans,
        normalized_speaker=normalized_speaker,
    )
    (
        marker_sentence_indexes,
        governed_target_indexes,
        forward_marker_spans_by_sentence,
    ) = scope_indexes
    witnesses = []
    seen = set()

    def add_witness(witness: SupportWitness) -> None:
        if not _witness_is_in_assertion_scope(
            witness=witness,
            sentences=sentences,
            surface_sentences=surface_sentences,
            untrusted_spans=untrusted_spans,
            marker_sentence_indexes=marker_sentence_indexes,
            governed_target_indexes=governed_target_indexes,
            forward_marker_spans_by_sentence=(
                forward_marker_spans_by_sentence
            ),
            normalized_subject=normalized_subject,
            normalized_speaker=normalized_speaker,
        ):
            return
        key = (
            witness.spec_id,
            witness.source_span,
            witness.subject_span,
            witness.predicate_span,
            witness.object_span,
            witness.binding,
            witness.state_change,
            witness.temporal_status,
        )
        if key in seen:
            return
        seen.add(key)
        witnesses.append(witness)

    for sentence, sentence_start in sentences:
        body, prefix_chars = _strip_speaker_prefix(sentence, normalized_speaker)
        body_start = sentence_start + prefix_chars

        # Every ordinary assertion must full-match this complete top-level
        # sentence.  The only semicolon-spanning acceptance is the explicit
        # retraction frame inside ``_match_complete_frame``.
        for witness in _match_complete_frame(
            clause=body,
            source_start=body_start,
            normalized_subject=normalized_subject,
            normalized_object=normalized_object,
            normalized_speaker=normalized_speaker,
            predicate=normalized_predicate,
        ):
            add_witness(witness)

        conjuncts = _coordinated_conjuncts(body)
        if not conjuncts:
            continue
        binding_info = _coordinating_subject_binding(
            first_conjunct=conjuncts[0][0],
            normalized_subject=normalized_subject,
            normalized_speaker=normalized_speaker,
            predicate=normalized_predicate,
        )
        if binding_info is None:
            continue
        binding, subject_span, subject_text = binding_info
        tail_descriptors = []
        seen_predicate_tail = False
        invalid_tail = False
        for conjunct, offset in conjuncts[1:]:
            if _coordinated_predicate_head_is_grammatical(
                conjunct_body=conjunct,
                binding=binding,
                subject_text=subject_text,
            ):
                seen_predicate_tail = True
                tail_descriptors.append(("predicate", conjunct, offset))
                continue
            object_text = _coordinated_object_continuation(conjunct)
            if object_text is None or seen_predicate_tail:
                invalid_tail = True
                break
            tail_descriptors.append(("object", object_text, offset))
        if invalid_tail:
            continue

        first_conjunct, first_offset = conjuncts[0]
        first_predicate = _coordinated_first_predicate(
            first_conjunct=first_conjunct,
            subject_span=subject_span,
            binding=binding,
            subject_text=subject_text,
        )
        if any(kind == "object" for kind, _, _ in tail_descriptors):
            if first_predicate is None or not seen_predicate_tail:
                continue
        for match_witness in _match_complete_frame(
            clause=first_conjunct,
            source_start=body_start + first_offset,
            normalized_subject=normalized_subject,
            normalized_object=normalized_object,
            normalized_speaker=normalized_speaker,
            predicate=normalized_predicate,
        ):
            if match_witness.binding != binding or match_witness.state_change != "assert":
                continue
            add_witness(
                SupportWitness(
                    spec_id="{}:coordinated-first".format(normalized_predicate),
                    clause=body,
                    subject_span=match_witness.subject_span,
                    predicate_span=match_witness.predicate_span,
                    object_span=match_witness.object_span,
                    binding=binding,
                    state_change="assert",
                    temporal_status=None,
                    source_span=(body_start, body_start + len(body)),
                )
            )

        prefix_length = len(subject_text) + 1
        if first_predicate is not None:
            first_predicate_name, predicate_text, predicate_span = first_predicate
            object_prefix_length = (
                len(subject_text) + 1 + len(predicate_text) + 1
            )
            for kind, object_text, offset in tail_descriptors:
                if kind != "object" or normalized_predicate != first_predicate_name:
                    continue
                synthetic = "{} {} {}".format(
                    subject_text, predicate_text, object_text
                )
                for match_witness in _match_complete_frame(
                    clause=synthetic,
                    source_start=body_start,
                    normalized_subject=normalized_subject,
                    normalized_object=normalized_object,
                    normalized_speaker=normalized_speaker,
                    predicate=normalized_predicate,
                ):
                    if (
                        match_witness.binding != binding
                        or match_witness.state_change != "assert"
                    ):
                        continue
                    add_witness(
                        SupportWitness(
                            spec_id="{}:coordinated-object".format(
                                normalized_predicate
                            ),
                            clause=body,
                            subject_span=subject_span,
                            predicate_span=predicate_span,
                            object_span=(
                                offset
                                + match_witness.object_span[0]
                                - object_prefix_length,
                                offset
                                + match_witness.object_span[1]
                                - object_prefix_length,
                            ),
                            binding=binding,
                            state_change="assert",
                            temporal_status=None,
                            source_span=(body_start, body_start + len(body)),
                        )
                    )

        for kind, conjunct, offset in tail_descriptors:
            if kind != "predicate":
                continue
            synthetic = "{} {}".format(subject_text, conjunct)
            for match_witness in _match_complete_frame(
                clause=synthetic,
                source_start=body_start,
                normalized_subject=normalized_subject,
                normalized_object=normalized_object,
                normalized_speaker=normalized_speaker,
                predicate=normalized_predicate,
            ):
                if (
                    match_witness.binding != binding
                    or match_witness.state_change != "assert"
                ):
                    continue
                add_witness(
                    SupportWitness(
                        spec_id="{}:coordinated".format(normalized_predicate),
                        clause=body,
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
                        state_change="assert",
                        temporal_status=None,
                        source_span=(body_start, body_start + len(body)),
                    )
                )
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
        selected_witness: Optional[SupportWitness] = None
        if source_text is not None:
            # Carry one exact parser witness only when its state is identical
            # to the parser-normalized relation state.  A source can contain
            # conflicting state frames that the historical parser preserves as
            # an ordinary relation; do not change that parser contract here.
            # In that rare case the absent carrier makes storage fail closed
            # and refuse an edge rather than recomputing support grammar.
            matching_witnesses = [
                witness
                for witness in relation_witnesses
                if (
                    witness.state_change == state_change
                    and witness.temporal_status == temporal_status
                )
            ]
            if matching_witnesses:
                selected_witness = min(
                    matching_witnesses,
                    key=lambda witness: (
                        witness.source_span,
                        witness.subject_span,
                        witness.predicate_span,
                        witness.object_span,
                        witness.spec_id,
                        witness.binding,
                    ),
                )
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
                support_witness=selected_witness,
            )
        )

    return GraphPayload(
        entities=tuple(entities),
        relations=tuple(relations),
        dropped_entities=dropped_entities,
        dropped_relations=dropped_relations,
        ambiguous_relations=ambiguous_relations,
    )
