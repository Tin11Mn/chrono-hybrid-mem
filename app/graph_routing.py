"""Deterministic query-plan routing for the bounded evidence-graph channel.

The graph remains a recall source, never an answer engine.  These helpers only
prefer controlled predicates that are explicitly suggested by the query and
its already-paid-for P1 plan.  Callers must keep non-preferred edges as bounded
fallbacks so a routing miss cannot turn into a hard recall filter.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any, Mapping, Sequence, Tuple

from .evidence_graph import PREDICATES


_SPACE = re.compile(r"\s+")


def _normalized_plan_text(query: str, plan: Mapping[str, Any]) -> str:
    values = [query]
    intent = plan.get("intent")
    if isinstance(intent, str):
        values.append(intent.replace("_", " "))
    for field in ("evidence_needs", "temporal_cues", "core_terms"):
        raw_values = plan.get(field, ())
        if not isinstance(raw_values, Sequence) or isinstance(raw_values, str):
            continue
        values.extend(value for value in raw_values if isinstance(value, str))
    return _SPACE.sub(
        " ", unicodedata.normalize("NFKC", " ".join(values)).casefold()
    ).strip()


def _has(text: str, pattern: str) -> bool:
    return re.search(pattern, text, flags=re.IGNORECASE) is not None


def preferred_graph_predicates(
    query: str, plan: Mapping[str, Any]
) -> Tuple[str, ...]:
    """Return a small ordered predicate preference from a frozen P1 plan.

    The mapping is intentionally generic and conservative.  It neither names
    benchmark answers nor removes edges.  The result is suitable as a stable
    sort preference ahead of a deterministic all-edge fallback.
    """

    text = _normalized_plan_text(query, plan)
    preferred: list[str] = []

    def add(*predicates: str) -> None:
        for predicate in predicates:
            if predicate in PREDICATES and predicate not in preferred:
                preferred.append(predicate)

    # Specific relation language precedes generic question words.
    if _has(
        text,
        r"\b(?:must|required?|requires?|rule|policy|protocol|allowed?|permits?|"
        r"prohibit(?:ed|s)?|forbid(?:den|s)?|not allowed)\b|必须|允许|禁止|规则|政策",
    ):
        add("requires", "permits", "prohibits")
    if _has(
        text,
        r"\b(?:work(?:s|ed|ing)?|workplace|job|career|profession|occupation|employ(?:er|ed|ment)?|"
        r"role|position|company|organization)\b|工作|职业|雇主|职位",
    ):
        add("works_at", "role_at")
    if _has(
        text,
        r"\b(?:member(?:ship)?|club|group|society|team|community|joined?|belong(?:s|ed)?)\b|"
        r"成员|加入|社团|团体",
    ):
        add("member_of", "participated_in")
    if _has(
        text,
        r"\b(?:likes?|liked|favorite|favourite|prefers?|preference|dislikes?|"
        r"enjoys?|fond of|into)\b|喜欢|偏好|讨厌",
    ):
        add("likes", "prefers", "dislikes")
    if _has(text, r"\b(?:friend|friends|friendship)\b|朋友"):
        add("friend_of")
    if _has(
        text,
        r"\b(?:partner|spouse|wife|husband|married|dating|relationship status|single)\b|"
        r"伴侣|配偶|婚姻|单身",
    ):
        add("partner_of")
    if _has(text, r"\b(?:parent|mother|father|mom|mum|dad|child|children|kids?)\b|父母|母亲|父亲|孩子"):
        add("parent_of")
    if _has(text, r"\b(?:sibling|brother|sister)\b|兄弟|姐妹"):
        add("sibling_of")
    if _has(
        text,
        r"\b(?:owns?|owned|possess(?:es|ed|ion)?|belongs to|has)\b|拥有|属于",
    ):
        add("owns")
    if _has(
        text,
        r"\b(?:created?|creates?|made|makes|built|paint(?:ed|s)?|wrote|written|"
        r"designed?|produced?)\b|创作|制作|画|写",
    ):
        add("created")

    activity_signal = _has(
        text,
        r"\b(?:attend(?:ed|ing|s)?|participat(?:e|ed|ing|es|ion)|went to|go(?:ing)? to|"
        r"visit(?:ed|ing|s)?|event|activity|activities|class|course|workshop|conference|"
        r"race|camp(?:ed|ing)?|trip|meeting|meet up|speech|exhibit)\b|参加|活动|课程|会议|旅行",
    )
    if activity_signal:
        add("participated_in", "member_of", "located_in")

    residence_signal = _has(
        text,
        r"\b(?:lives?|lived|living|resides?|resided|home|hometown|based|moved?|"
        r"relocated?|address|residence)\b|居住|住在|搬到|家乡",
    )
    if residence_signal:
        add("lives_in", "located_in")
    elif _has(text, r"\bwhere\b|哪里|何处"):
        # Generic where-questions can concern either an entity location or an
        # event venue; keep both as preferences and never filter by them.
        add("located_in", "participated_in", "lives_in")

    if _has(
        text,
        r"\b(?:changed?|updated?|replaced?|instead|no longer|formerly|previously|"
        r"used to|current|latest|newest|old|earlier)\b|改变|更新|替代|不再|以前|当前|最新",
    ):
        add("changed_to", "replaces")

    if _has(text, r"\bwhen\b|\bhow long ago\b|何时|什么时候|多久前"):
        # Time questions commonly target the source turn for an event.  This
        # is deliberately last so explicit relation cues above win.
        add("participated_in", "created", "member_of")

    return tuple(preferred[:10])


def graph_predicate_priority(
    predicate: str, preferred: Sequence[str]
) -> tuple[int, int | str]:
    """Stable sort key: preferred order first, controlled fallback second."""

    try:
        return (0, preferred.index(predicate))
    except ValueError:
        return (1, predicate)
