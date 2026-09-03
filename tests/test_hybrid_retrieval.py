from app.schemas import AddRequest
from app.storage import (
    MemoryStore,
    bind_first_person_to_speaker,
    candidate_ranking_text,
    latent_message_text,
)


def test_latent_message_text_adds_missing_speaker_and_date_only_to_key():
    assert latent_message_text(
        "I ran a race.", "Melanie", 1683417600
    ) == "Speaker: Melanie\nEvent date: 07 May 2023\nI ran a race."
    assert latent_message_text("Melanie: I ran.", "Melanie", None) == (
        "Melanie: I ran."
    )


def test_candidate_ranking_text_exposes_provenance_without_replacing_memory():
    rendered = candidate_ranking_text(
        "I no longer prefer tea.",
        "Mina",
        1683417600000,
        ["Mina RETRACTED her preference for tea."],
        ["Previous memory: Mina used to prefer tea."],
    )

    assert "Source speaker: Mina" in rendered
    assert "Event date: 07 May 2023" in rendered
    assert "Original memory:\nI no longer prefer tea." in rendered
    assert "Mina RETRACTED her preference for tea." in rendered
    assert "Previous memory: Mina used to prefer tea." in rendered


class FakeSemanticRetriever:
    def rank(self, query, options, candidates, limit):
        preferred = [item["id"] for item in candidates if "jasmine" in item["content"]]
        remaining = [item["id"] for item in candidates if item["id"] not in preferred]
        return (preferred + remaining)[:limit]


class FakeSemanticScorer:
    def score(self, query, options, candidates):
        return {
            item["id"]: 0.9 if "jasmine" in item["content"] else 0.1
            for item in candidates
        }


class FakeSpeakerSensitiveScorer:
    def score(self, query, options, candidates):
        masked = "[PERSON]" in query
        return {
            item["id"]: (
                (0.95 if "adoption" in item["content"] else 0.2)
                if masked else
                (0.1 if "adoption" in item["content"] else 0.9)
            )
            for item in candidates
        }


class FakeSentenceSensitiveScorer:
    def score(self, query, options, candidates):
        scores = {}
        for item in candidates:
            sentence = "::sentence:" in item["id"]
            relevant = "adoption" in item["content"]
            if sentence:
                scores[item["id"]] = 0.95 if relevant else 0.2
            else:
                scores[item["id"]] = 0.1 if relevant else 0.9
        return scores


class FakeImageCarryScorer:
    def score(self, query, options, candidates):
        return {
            item["id"]: (
                0.95 if "Shared image context:" in item["content"]
                and "painted it" in item["content"] else 0.1
            )
            for item in candidates
        }


class FakeLateInteractionReranker:
    def rank(self, query, options, candidates):
        return list(reversed([item["id"] for item in candidates]))

    def score(self, query, options, candidates):
        return {item["id"]: float(index) for index, item in enumerate(candidates)}


class RecordingReranker(FakeLateInteractionReranker):
    def __init__(self):
        self.candidates = []

    def rank(self, query, options, candidates):
        self.candidates = candidates
        return [item["id"] for item in candidates]


class NearTieReranker:
    def score(self, query, options, candidates):
        return {
            item["id"]: (
                0.99 if "grinder" in item["content"] else
                0.985 if "jasmine" in item["content"] else 0.5
            )
            for item in candidates
        }


class FakeHierarchicalScorer:
    def score(self, query, options, candidates):
        scores = {}
        for item in candidates:
            content = item["content"]
            if "jasmine" in content:
                score = 0.9
            elif "supplies" in content:
                score = 0.95
            else:
                score = 0.1
            scores[item["id"]] = score
        return scores


class FakeInstructionReranker:
    def rank(self, query, options, candidates):
        return list(reversed([item["id"] for item in candidates]))


class TrackingInstructionReranker(FakeInstructionReranker):
    def __init__(self):
        self.pool_sizes = []

    def rank(self, query, options, candidates):
        self.pool_sizes.append(len(candidates))
        return super().rank(query, options, candidates)


class FirstInstructionReranker(FakeInstructionReranker):
    def __init__(self):
        self.calls = 0

    def rank(self, query, options, candidates):
        self.calls += 1
        return [item["id"] for item in candidates]


def test_temporal_query_prefers_newer_event_over_later_ingestion(tmp_path):
    store = MemoryStore(str(tmp_path / "memory.db"), temporal_bonus=0.001)
    store.initialize()
    store.add(AddRequest(
        request_id="new-event", user_id="user-a", session_id="session-a",
        messages=[{
            "role": "user", "timestamp": 200,
            "content": "Ravi's preferred drink is tea.",
        }],
    ))
    store.add(AddRequest(
        request_id="old-event", user_id="user-a", session_id="session-a",
        messages=[{
            "role": "user", "timestamp": 100,
            "content": "Ravi's preferred drink is coffee.",
        }],
    ))

    results = store.search(
        user_id="user-a", query="What is Ravi's current preferred drink?", top_k=1
    )

    assert results[0].content == "Ravi's preferred drink is tea."


def test_historical_query_prefers_older_event_over_later_ingestion(tmp_path):
    store = MemoryStore(str(tmp_path / "memory.db"), temporal_bonus=0.001)
    store.initialize()
    store.add(AddRequest(
        request_id="old-event", user_id="user-a", session_id="session-a",
        messages=[{
            "role": "user", "timestamp": 100,
            "content": "Ravi's preferred drink is hot coffee.",
        }],
    ))
    store.add(AddRequest(
        request_id="new-event", user_id="user-a", session_id="session-a",
        messages=[{
            "role": "user", "timestamp": 200,
            "content": "Ravi's preferred drink is black tea.",
        }],
    ))

    results = store.search(
        user_id="user-a", query="What did Ravi prefer before?", top_k=1
    )

    assert results[0].content == "Ravi's preferred drink is hot coffee."


def test_query_stop_words_do_not_hide_content_terms(tmp_path):
    store = MemoryStore(str(tmp_path / "memory.db"))
    store.initialize()
    store.add(AddRequest(
        request_id="content", user_id="user-a", session_id="session-a",
        messages=[{"role": "user", "content": "Milo prefers jasmine tea."}],
    ))

    results = store.search(user_id="user-a", query="What does Milo prefer?", top_k=1)

    assert results[0].content == "Milo prefers jasmine tea."


def test_first_person_coreference_binds_only_unambiguous_forms():
    result = bind_first_person_to_speaker(
        "Caroline: I'm proud of my work; it matters to me, and we agree.",
        "Caroline",
    )

    assert "Caroline is proud of Caroline's work" in result
    assert "matters to Caroline" in result
    assert "we agree" in result


def test_neighbor_context_retrieves_pronominal_evidence(tmp_path):
    store = MemoryStore(str(tmp_path / "memory.db"))
    store.initialize()
    store.add(AddRequest(
        request_id="context", user_id="user-a", session_id="session-a",
        messages=[
            {"role": "user", "content": "Mina is choosing a drink for breakfast."},
            {"role": "user", "content": "She settles on jasmine tea."},
            {"role": "user", "content": "Bela plans to hike tomorrow."},
        ],
    ))

    results = store.search(
        user_id="user-a", query="Which tea did Mina choose?", top_k=3
    )

    assert any(result.content == "She settles on jasmine tea." for result in results)


def test_porter_channel_matches_inflected_content_word(tmp_path):
    store = MemoryStore(str(tmp_path / "memory.db"))
    store.initialize()
    store.add(AddRequest(
        request_id="porter", user_id="user-a", session_id="session-a",
        messages=[{"role": "user", "content": "Milo relocated to Qingdao."}],
    ))

    results = store.search(user_id="user-a", query="Where will Milo relocate?", top_k=1)

    assert results[0].content == "Milo relocated to Qingdao."


def test_entity_channel_prefers_the_named_person(tmp_path):
    store = MemoryStore(str(tmp_path / "memory.db"))
    store.initialize()
    store.add(AddRequest(
        request_id="entities", user_id="user-a", session_id="session-a",
        messages=[
            {"role": "user", "content": "Caroline researched adoption agencies."},
            {"role": "user", "content": "Melanie researched travel agencies."},
        ],
    ))

    results = store.search(user_id="user-a", query="What did Caroline research?", top_k=1)

    assert results[0].content == "Caroline researched adoption agencies."


def test_dense_channel_can_retrieve_a_semantic_match_without_lexical_overlap(tmp_path):
    store = MemoryStore(
        str(tmp_path / "memory.db"),
        semantic_retriever=FakeSemanticRetriever(),
        dense_rrf_weight=4.0,
    )
    store.initialize()
    store.add(AddRequest(
        request_id="dense", user_id="user-a", session_id="session-a",
        messages=[
            {"role": "user", "content": "Mina's morning beverage is jasmine tea."},
            {"role": "user", "content": "Mina bought a coffee grinder."},
        ],
    ))

    results = store.search(user_id="user-a", query="What does Mina drink at breakfast?", top_k=1)

    assert results[0].content == "Mina's morning beverage is jasmine tea."


def test_score_fusion_can_use_dense_only_endpoint(tmp_path):
    store = MemoryStore(
        str(tmp_path / "memory.db"),
        semantic_retriever=FakeSemanticScorer(),
        dense_fusion_alpha=0.0,
    )
    store.initialize()
    store.add(AddRequest(
        request_id="score-fusion", user_id="user-a", session_id="session-a",
        messages=[
            {"role": "user", "content": "Mina's morning beverage is jasmine tea."},
            {"role": "user", "content": "Mina bought a coffee grinder."},
        ],
    ))

    results = store.search(user_id="user-a", query="What does Mina drink at breakfast?", top_k=1)

    assert results[0].content == "Mina's morning beverage is jasmine tea."


def test_dense_speaker_mask_max_can_recover_wrong_person_counterevidence(tmp_path):
    store = MemoryStore(
        str(tmp_path / "memory.db"),
        semantic_retriever=FakeSpeakerSensitiveScorer(),
        dense_fusion_alpha=0.0,
        dense_speaker_mask_max=True,
    )
    store.initialize()
    store.add(AddRequest(
        request_id="speaker-mask", user_id="user-a", session_id="session-a",
        messages=[
            {"role": "Caroline", "content": "Caroline: I chose the adoption agency."},
            {"role": "Melanie", "content": "Melanie: I booked a travel agency."},
        ],
    ))

    results = store.search(
        user_id="user-a",
        query="Why did Melanie choose the adoption agency?",
        top_k=1,
    )

    assert results[0].content == "Caroline: I chose the adoption agency."


def test_dense_speaker_conflict_only_triggers_for_stronger_other_speaker(tmp_path):
    store = MemoryStore(
        str(tmp_path / "memory.db"),
        semantic_retriever=FakeSpeakerSensitiveScorer(),
        dense_fusion_alpha=0.0,
        dense_speaker_conflict_margin=0.05,
    )
    store.initialize()
    store.add(AddRequest(
        request_id="speaker-conflict", user_id="user-a", session_id="session-a",
        messages=[
            {"role": "Caroline", "content": "Caroline: I chose the adoption agency."},
            {"role": "Melanie", "content": "Melanie: I booked a travel agency."},
        ],
    ))

    results = store.search(
        user_id="user-a",
        query="Why did Melanie choose the adoption agency?",
        top_k=1,
    )

    assert results[0].content == "Caroline: I chose the adoption agency."
    assert store.speaker_conflict_trigger_count == 1


def test_sentence_dense_key_returns_original_long_turn(tmp_path):
    relevant = (
        "Caroline: We discussed many unrelated updates. "
        "I chose the adoption agency because it supports LGBTQ families."
    )
    store = MemoryStore(
        str(tmp_path / "memory.db"),
        semantic_retriever=FakeSentenceSensitiveScorer(),
        dense_fusion_alpha=0.0,
        dense_sentence_weight=1.0,
    )
    store.initialize()
    store.add(AddRequest(
        request_id="sentence-key", user_id="user-a", session_id="session-a",
        messages=[
            {"role": "Caroline", "content": relevant},
            {"role": "Melanie", "content": "Melanie: I went running yesterday."},
        ],
    ))

    results = store.search(
        user_id="user-a", query="Why choose the adoption agency?", top_k=1
    )

    assert results[0].content == relevant
    assert "::sentence:" not in results[0].id


def test_image_carry_dense_key_returns_following_original_turn(tmp_path):
    followup = "Ari: Yes, I painted it last year."
    store = MemoryStore(
        str(tmp_path / "memory.db"),
        semantic_retriever=FakeImageCarryScorer(),
        dense_fusion_alpha=0.0,
        dense_image_carry_weight=1.0,
    )
    store.initialize()
    store.add(AddRequest(
        request_id="image-carry", user_id="user-a", session_id="session-a",
        messages=[
            {"role": "Ari", "content": "Ari: Look. Shared image: a sunrise painting"},
            {"role": "Ari", "content": followup},
            {"role": "Bela", "content": "Bela: I went running."},
        ],
    ))

    results = store.search(user_id="user-a", query="Who painted the image?", top_k=1)

    assert results[0].content == followup
    assert "Shared image context:" not in results[0].content


def test_instruction_reranker_can_be_gated_by_speaker_conflict(tmp_path):
    instruction = FirstInstructionReranker()
    store = MemoryStore(
        str(tmp_path / "memory.db"),
        semantic_retriever=FakeSpeakerSensitiveScorer(),
        dense_fusion_alpha=0.0,
        dense_speaker_conflict_margin=0.05,
        dense_speaker_conflict_gate_only=True,
        local_instruction_reranker=instruction,
        instruction_speaker_conflict_only=True,
    )
    store.initialize()
    store.add(AddRequest(
        request_id="gated-instruction", user_id="user-a", session_id="session-a",
        messages=[
            {"role": "Caroline", "content": "Caroline: I chose the adoption agency."},
            {"role": "Melanie", "content": "Melanie: I booked a travel agency."},
        ],
    ))

    store.search(
        user_id="user-a",
        query="Why did Melanie choose the adoption agency?",
        top_k=1,
    )

    assert instruction.calls == 1
    assert store.last_speaker_conflict is True


def test_local_reranker_only_reorders_the_bounded_fused_pool(tmp_path):
    store = MemoryStore(
        str(tmp_path / "memory.db"),
        semantic_retriever=FakeSemanticScorer(),
        dense_fusion_alpha=0.0,
        local_reranker=FakeLateInteractionReranker(),
        rerank_top_n=2,
    )
    store.initialize()
    store.add(AddRequest(
        request_id="rerank", user_id="user-a", session_id="session-a",
        messages=[
            {"role": "user", "content": "Mina's morning beverage is jasmine tea."},
            {"role": "user", "content": "Mina bought a coffee grinder."},
        ],
    ))

    results = store.search(user_id="user-a", query="What does Mina drink?", top_k=2)

    assert results[0].content == "Mina bought a coffee grinder."


def test_near_tie_reranker_keeps_first_stage_choice_within_epsilon(tmp_path):
    store = MemoryStore(
        str(tmp_path / "memory.db"),
        semantic_retriever=FakeSemanticScorer(),
        dense_fusion_alpha=0.0,
        local_reranker=NearTieReranker(),
        rerank_top_n=2,
        rerank_near_tie_epsilon=0.01,
    )
    store.initialize()
    store.add(AddRequest(
        request_id="near-tie", user_id="user-a", session_id="session-a",
        messages=[
            {"role": "user", "content": "Mina's morning beverage is jasmine tea."},
            {"role": "user", "content": "Mina bought a coffee grinder."},
        ],
    ))

    results = store.search(user_id="user-a", query="What does Mina drink?", top_k=2)

    assert results[0].content == "Mina's morning beverage is jasmine tea."


def test_image_anchor_can_use_following_turns_only_for_reranker_scoring(tmp_path):
    reranker = RecordingReranker()
    store = MemoryStore(
        str(tmp_path / "memory.db"),
        semantic_retriever=FakeSemanticScorer(),
        dense_fusion_alpha=0.0,
        local_reranker=reranker,
        rerank_top_n=3,
        rerank_image_followups=2,
    )
    store.initialize()
    store.add(AddRequest(
        request_id="image-context", user_id="user-a", session_id="session-a",
        messages=[
            {"role": "Ari", "content": "Take a look. Shared image: a sunrise painting"},
            {"role": "Bela", "content": "Did you paint it?"},
            {"role": "Ari", "content": "Yes, I painted it last year."},
        ],
    ))

    results = store.search(user_id="user-a", query="When was it painted?", top_k=3)

    by_id = {item["id"]: item["content"] for item in reranker.candidates}
    assert "Following conversation:" in by_id[results[0].id]
    assert "painted it last year" in by_id[results[0].id]
    assert all(
        "Following conversation:" not in content
        for candidate_id, content in by_id.items()
        if candidate_id != results[0].id
    )
    assert results[0].content == "Take a look. Shared image: a sunrise painting"


def test_session_fusion_promotes_evidence_from_the_relevant_session(tmp_path):
    store = MemoryStore(
        str(tmp_path / "memory.db"),
        semantic_retriever=FakeHierarchicalScorer(),
        dense_fusion_alpha=0.5,
        session_fusion_weight=4.0,
    )
    store.initialize()
    store.add(AddRequest(
        request_id="relevant-session", user_id="user-a", session_id="session-a",
        messages=[
            {"role": "user", "content": "Mina mentioned breakfast plans."},
            {"role": "user", "content": "She chose a jasmine beverage."},
        ],
    ))
    store.add(AddRequest(
        request_id="distractor-session", user_id="user-a", session_id="session-b",
        messages=[
            {"role": "user", "content": "Mina bought breakfast supplies."},
            {"role": "user", "content": "The delivery arrived on Tuesday."},
        ],
    ))

    results = store.search(
        user_id="user-a", query="What beverage did Mina choose for breakfast?", top_k=1
    )

    assert results[0].content == "She chose a jasmine beverage."


def test_hard_session_filter_returns_only_top_session_turns(tmp_path):
    store = MemoryStore(
        str(tmp_path / "memory.db"),
        semantic_retriever=FakeSemanticScorer(),
        dense_fusion_alpha=0.0,
        session_top_n=1,
    )
    store.initialize()
    store.add(AddRequest(
        request_id="session-a", user_id="user-a", session_id="session-a",
        messages=[
            {"role": "user", "content": "Mina drinks jasmine tea."},
            {"role": "user", "content": "She prepares it every morning."},
        ],
    ))
    store.add(AddRequest(
        request_id="session-b", user_id="user-a", session_id="session-b",
        messages=[
            {"role": "user", "content": "Mina bought a coffee grinder."},
            {"role": "user", "content": "The package arrived yesterday."},
        ],
    ))

    results = store.search(user_id="user-a", query="What does Mina drink?", top_k=10)

    assert len(results) == 2
    assert all("coffee" not in result.content for result in results)


def test_zero_rerank_fusion_weight_preserves_first_stage_order(tmp_path):
    store = MemoryStore(
        str(tmp_path / "memory.db"),
        semantic_retriever=FakeSemanticScorer(),
        dense_fusion_alpha=0.0,
        local_reranker=FakeLateInteractionReranker(),
        rerank_top_n=2,
        rerank_fusion_weight=0.0,
    )
    store.initialize()
    store.add(AddRequest(
        request_id="score-blend", user_id="user-a", session_id="session-a",
        messages=[
            {"role": "user", "content": "Mina's morning beverage is jasmine tea."},
            {"role": "user", "content": "Mina bought a coffee grinder."},
        ],
    ))

    results = store.search(user_id="user-a", query="What does Mina drink?", top_k=1)

    assert results[0].content == "Mina's morning beverage is jasmine tea."


def test_instruction_model_only_reorders_its_bounded_candidate_pool(tmp_path):
    store = MemoryStore(
        str(tmp_path / "memory.db"),
        semantic_retriever=FakeSemanticScorer(),
        dense_fusion_alpha=0.0,
        local_instruction_reranker=FakeInstructionReranker(),
        instruction_rerank_top_n=2,
    )
    store.initialize()
    store.add(AddRequest(
        request_id="instruction-rerank", user_id="user-a", session_id="session-a",
        messages=[
            {"role": "user", "content": "Mina's morning beverage is jasmine tea."},
            {"role": "user", "content": "Mina bought a coffee grinder."},
            {"role": "user", "content": "Mina visited the library."},
        ],
    ))

    results = store.search(user_id="user-a", query="What does Mina drink?", top_k=3)

    assert results[0].content == "Mina bought a coffee grinder."
    assert results[2].content == "Mina visited the library."


def test_instruction_refinement_reranks_only_first_pass_shortlist(tmp_path):
    reranker = TrackingInstructionReranker()
    store = MemoryStore(
        str(tmp_path / "memory.db"),
        semantic_retriever=FakeSemanticScorer(),
        dense_fusion_alpha=0.0,
        local_instruction_reranker=reranker,
        instruction_rerank_top_n=3,
        instruction_refine_top_n=2,
    )
    store.initialize()
    store.add(AddRequest(
        request_id="instruction-refine", user_id="user-a", session_id="session-a",
        messages=[
            {"role": "user", "content": "Mina's morning beverage is jasmine tea."},
            {"role": "user", "content": "Mina bought a coffee grinder."},
            {"role": "user", "content": "Mina visited the library."},
        ],
    ))

    results = store.search(user_id="user-a", query="What does Mina drink?", top_k=3)

    assert reranker.pool_sizes == [3, 2]
    assert results[0].content == "Mina bought a coffee grinder."
def test_speaker_swap_dense_query_max_fusion_preserves_original_records(tmp_path):
    class SwapAwareRetriever:
        def __init__(self):
            self.queries = []

        def score(self, query, options, candidates):
            self.queries.append(query)
            return {
                candidate["id"]: (
                    1.0 if "Melanie" in query and "Melanie: ran" in candidate["content"]
                    else 0.1
                )
                for candidate in candidates
            }

    retriever = SwapAwareRetriever()
    store = MemoryStore(
        str(tmp_path / "speaker-swap.db"),
        semantic_retriever=retriever,
        dense_fusion_alpha=0.0,
        dense_speaker_swap_max=True,
    )
    store.initialize()
    store.add(AddRequest(
        request_id="swap-1",
        user_id="user-1",
        session_id="session-1",
        messages=[
            {"role": "Caroline", "content": "Caroline: painted"},
            {"role": "Melanie", "content": "Melanie: ran"},
        ],
    ))

    results = store.search(
        user_id="user-1", query="Why did Caroline run?", top_k=2
    )

    assert retriever.queries == ["Why did Caroline run?", "Why did Melanie run?"]
    assert results[0].content == "Melanie: ran"


class AdjacentExpansionModel:
    """Small deterministic model fake for P1.1 candidate-pool assertions."""

    def __init__(self, preferred_token=""):
        self.preferred_token = preferred_token
        self.extract_facts_calls = 0
        self.plan_calls = 0
        self.rank_calls = 0
        self.ranked_candidate_ids = []
        self.ranked_candidate_contents = {}

    def extract_facts(self, content, speaker="", timestamp=None):
        self.extract_facts_calls += 1
        return []

    def plan_query(self, query, options):
        self.plan_calls += 1
        return []

    def rank_candidates(self, query, options, candidates):
        self.rank_calls += 1
        self.ranked_candidate_ids = [candidate["id"] for candidate in candidates]
        self.ranked_candidate_contents = {
            candidate["id"]: candidate["content"] for candidate in candidates
        }
        preferred = [
            candidate["id"]
            for candidate in candidates
            if self.preferred_token and self.preferred_token in candidate["content"]
        ]
        return preferred + [
            candidate["id"]
            for candidate in candidates
            if candidate["id"] not in preferred
        ]


def _add_adjacent_test_turn(store, *, request_id, session_id, content, user_id="user-a"):
    store.add(AddRequest(
        request_id=request_id,
        user_id=user_id,
        session_id=session_id,
        messages=[{"role": "user", "content": content}],
    ))


def _seed_adjacent_turn_fixture(store):
    # Separate Add calls deliberately avoid the existing ingestion-time context
    # window; P1.1 must discover the same-session adjacency from raw turns.
    _add_adjacent_test_turn(
        store,
        request_id="adjacent-previous",
        session_id="session-a",
        content="PREVIOUS_ONLY: the clue was written yesterday.",
    )
    _add_adjacent_test_turn(
        store,
        request_id="adjacent-anchor",
        session_id="session-a",
        content="BEACON_ANCHOR: Mina named the project beacon.",
    )
    _add_adjacent_test_turn(
        store,
        request_id="adjacent-following",
        session_id="session-a",
        content="NEXT_ONLY: the answer is the blue lantern.",
    )


def test_adjacent_turn_expansion_default_off_preserves_p1_candidate_path(tmp_path):
    database_path = tmp_path / "adjacent-default-off.db"
    seed_store = MemoryStore(str(database_path))
    seed_store.initialize()
    _seed_adjacent_turn_fixture(seed_store)

    default_model = AdjacentExpansionModel()
    explicit_off_model = AdjacentExpansionModel()
    default_store = MemoryStore(str(database_path), model=default_model)
    explicit_off_store = MemoryStore(
        str(database_path),
        model=explicit_off_model,
        adjacent_turn_expansion=False,
    )

    default_results = default_store.search(
        user_id="user-a", query="BEACON_ANCHOR", top_k=3
    )
    explicit_off_results = explicit_off_store.search(
        user_id="user-a", query="BEACON_ANCHOR", top_k=3
    )

    assert [
        (result.id, result.content, result.score) for result in default_results
    ] == [
        (result.id, result.content, result.score) for result in explicit_off_results
    ]
    assert default_model.ranked_candidate_ids == explicit_off_model.ranked_candidate_ids
    assert default_store.last_retrieval_trace["p1_pre_rerank_ids"] == (
        explicit_off_store.last_retrieval_trace["p1_pre_rerank_ids"]
    )
    assert default_store.last_retrieval_trace["rerank_pool_ids"] == (
        explicit_off_store.last_retrieval_trace["rerank_pool_ids"]
    )
    for trace in (
        default_store.last_retrieval_trace,
        explicit_off_store.last_retrieval_trace,
    ):
        assert trace["adjacent_seed_ids"] == []
        assert trace["adjacent_candidate_ids"] == []
        assert trace["adjacent_deduped_ids"] == []


def test_adjacent_turn_expansion_returns_immediate_same_session_raw_neighbors(tmp_path):
    database_path = tmp_path / "adjacent-neighbors.db"
    seed_store = MemoryStore(str(database_path))
    seed_store.initialize()
    _seed_adjacent_turn_fixture(seed_store)

    model = AdjacentExpansionModel(preferred_token="NEXT_ONLY")
    store = MemoryStore(
        str(database_path),
        model=model,
        adjacent_turn_expansion=True,
        adjacent_seed_limit=4,
        adjacent_candidate_limit=4,
    )

    results = store.search(user_id="user-a", query="BEACON_ANCHOR", top_k=3)

    result_ids_by_content = {result.content: result.id for result in results}
    anchor_id = result_ids_by_content["BEACON_ANCHOR: Mina named the project beacon."]
    previous_id = result_ids_by_content["PREVIOUS_ONLY: the clue was written yesterday."]
    following_id = result_ids_by_content["NEXT_ONLY: the answer is the blue lantern."]
    trace = store.last_retrieval_trace

    assert trace["p1_pre_rerank_ids"] == [anchor_id]
    assert trace["adjacent_seed_ids"] == [anchor_id]
    assert trace["adjacent_candidate_ids"] == [previous_id, following_id]
    assert trace["adjacent_deduped_ids"] == []
    assert model.ranked_candidate_ids == [anchor_id, previous_id, following_id]
    assert result_ids_by_content["NEXT_ONLY: the answer is the blue lantern."] == following_id
    assert all("Original memory:" not in result.content for result in results)


def test_adjacent_turn_expansion_never_crosses_session_boundaries(tmp_path):
    database_path = tmp_path / "adjacent-session-boundary.db"
    seed_store = MemoryStore(str(database_path))
    seed_store.initialize()
    _add_adjacent_test_turn(
        seed_store,
        request_id="session-anchor",
        session_id="session-a",
        content="BOUNDARY_ANCHOR: the topic is beacon.",
    )
    _add_adjacent_test_turn(
        seed_store,
        request_id="foreign-turn",
        session_id="session-b",
        content="FOREIGN_ONLY: this turn must never be traversed.",
    )
    _add_adjacent_test_turn(
        seed_store,
        request_id="session-neighbor",
        session_id="session-a",
        content="SAME_SESSION_ONLY: this is the valid following turn.",
    )

    model = AdjacentExpansionModel()
    store = MemoryStore(
        str(database_path),
        model=model,
        adjacent_turn_expansion=True,
        adjacent_seed_limit=4,
        adjacent_candidate_limit=4,
    )
    results = store.search(user_id="user-a", query="BOUNDARY_ANCHOR", top_k=3)

    ids_by_content = {result.content: result.id for result in results}
    trace = store.last_retrieval_trace
    assert trace["adjacent_candidate_ids"] == [
        ids_by_content["SAME_SESSION_ONLY: this is the valid following turn."]
    ]
    assert "FOREIGN_ONLY: this turn must never be traversed." not in ids_by_content
    assert all("FOREIGN_ONLY" not in content for content in model.ranked_candidate_contents.values())


def test_adjacent_turn_expansion_caps_seeds_candidates_and_model_pool_in_order(tmp_path):
    database_path = tmp_path / "adjacent-caps.db"
    seed_store = MemoryStore(str(database_path))
    seed_store.initialize()
    for index in range(35):
        _add_adjacent_test_turn(
            seed_store,
            request_id="cap-anchor-{}".format(index),
            session_id="session-a",
            content="CAP_BEACON anchor {:02d}.".format(index),
        )
        _add_adjacent_test_turn(
            seed_store,
            request_id="cap-neighbor-{}".format(index),
            session_id="session-a",
            content="CAP_NEIGHBOR answer {:02d}.".format(index),
        )

    model = AdjacentExpansionModel()
    store = MemoryStore(
        str(database_path),
        model=model,
        adjacent_turn_expansion=True,
        adjacent_seed_limit=4,
        adjacent_candidate_limit=4,
    )

    store.search(user_id="user-a", query="CAP_BEACON", top_k=30)
    first_trace = store.last_retrieval_trace
    first_pool_ids = list(model.ranked_candidate_ids)
    store.search(user_id="user-a", query="CAP_BEACON", top_k=30)
    second_trace = store.last_retrieval_trace

    assert len(first_trace["p1_pre_rerank_ids"]) >= MemoryStore.MODEL_RERANK_LIMIT
    assert first_trace["adjacent_seed_ids"] == first_trace[
        "p1_pre_rerank_ids"
    ][:len(first_trace["adjacent_seed_ids"])]
    assert 1 <= len(first_trace["adjacent_seed_ids"]) <= 4
    assert len(first_trace["adjacent_candidate_ids"]) == 4
    assert len(set(first_trace["adjacent_candidate_ids"])) == 4
    assert set(first_trace["adjacent_candidate_ids"]).issubset(
        set(first_trace["rerank_pool_ids"])
    )
    assert len(first_trace["rerank_pool_ids"]) <= MemoryStore.MODEL_RERANK_LIMIT
    assert first_pool_ids == first_trace["rerank_pool_ids"]
    assert first_trace["adjacent_seed_ids"] == second_trace["adjacent_seed_ids"]
    assert first_trace["adjacent_candidate_ids"] == second_trace["adjacent_candidate_ids"]
    assert first_trace["rerank_pool_ids"] == second_trace["rerank_pool_ids"]


def test_adjacent_turn_expansion_adds_no_model_calls(tmp_path):
    database_path = tmp_path / "adjacent-model-calls.db"
    seed_store = MemoryStore(str(database_path))
    seed_store.initialize()
    _seed_adjacent_turn_fixture(seed_store)

    off_model = AdjacentExpansionModel()
    on_model = AdjacentExpansionModel()
    off_store = MemoryStore(
        str(database_path), model=off_model, adjacent_turn_expansion=False
    )
    on_store = MemoryStore(
        str(database_path),
        model=on_model,
        adjacent_turn_expansion=True,
        adjacent_seed_limit=4,
        adjacent_candidate_limit=4,
    )

    off_store.search(user_id="user-a", query="BEACON_ANCHOR", top_k=3)
    on_store.search(user_id="user-a", query="BEACON_ANCHOR", top_k=3)

    assert off_model.extract_facts_calls == on_model.extract_facts_calls == 0
    assert off_model.plan_calls == on_model.plan_calls == 1
    assert off_model.rank_calls == on_model.rank_calls == 1
    assert off_store.last_retrieval_trace["p1_pre_rerank_ids"] == (
        on_store.last_retrieval_trace["p1_pre_rerank_ids"]
    )
    assert len(on_model.ranked_candidate_ids) > len(off_model.ranked_candidate_ids)
