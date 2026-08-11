from app.schemas import AddRequest
from app.storage import MemoryStore


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


class FakeLateInteractionReranker:
    def rank(self, query, options, candidates):
        return list(reversed([item["id"] for item in candidates]))

    def score(self, query, options, candidates):
        return {item["id"]: float(index) for index, item in enumerate(candidates)}


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
