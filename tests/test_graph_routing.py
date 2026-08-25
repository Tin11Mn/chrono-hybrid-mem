from app.graph_routing import graph_predicate_priority, preferred_graph_predicates


def test_route_uses_all_non_entity_plan_signals_without_answering():
    plan = {
        "intent": "temporal",
        "core_terms": ["Caroline", "support group"],
        "entities": ["Caroline"],
        "evidence_needs": ["find when Caroline attended the support group"],
        "temporal_cues": ["yesterday"],
    }

    routed = preferred_graph_predicates(
        "When did Caroline go to the support group?", plan
    )

    assert routed[:2] == ("member_of", "participated_in")
    assert "Caroline" not in routed


def test_route_prioritizes_explicit_relation_over_generic_where():
    routed = preferred_graph_predicates(
        "Where does Mina work?",
        {"intent": "fact", "evidence_needs": ["find Mina's employer"]},
    )

    assert routed[:2] == ("works_at", "role_at")
    assert routed.index("located_in") > routed.index("works_at")


def test_route_covers_preferences_family_rules_and_changes():
    preference = preferred_graph_predicates(
        "What food does Bob prefer?", {"evidence_needs": []}
    )
    family = preferred_graph_predicates(
        "Who is Ana's sister?", {"evidence_needs": []}
    )
    rules = preferred_graph_predicates(
        "What does the policy prohibit?", {"evidence_needs": []}
    )
    changed = preferred_graph_predicates(
        "What replaced the old badge?", {"temporal_cues": ["old"]}
    )

    assert preference[:3] == ("likes", "prefers", "dislikes")
    assert family[0] == "sibling_of"
    assert rules[:3] == ("requires", "permits", "prohibits")
    assert changed[:2] == ("changed_to", "replaces")


def test_route_keeps_unknown_query_as_no_preference():
    assert preferred_graph_predicates(
        "Explain the memory.", {"intent": "other", "entities": ["memory"]}
    ) == ()


def test_predicate_priority_is_deterministic_and_non_filtering():
    preferred = ("participated_in", "member_of")

    assert graph_predicate_priority("member_of", preferred) == (0, 1)
    assert graph_predicate_priority("works_at", preferred) == (1, "works_at")
