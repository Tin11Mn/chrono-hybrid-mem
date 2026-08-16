import pytest
from types import SimpleNamespace

from app.local_instruction import (
    LocalDualStrategyReranker,
    LocalInstructionReranker,
    LocalQueryExpander,
    LocalYesNoReranker,
    mask_candidate_speakers,
    ordered_ids_from_response,
    queries_from_response,
    top_logprobs_from_completion,
    yes_probability_from_top_logprobs,
)
from app.local_semantic import LocalHTTPEmbeddingRetriever


def test_local_instruction_reranker_rejects_non_loopback_servers():
    with pytest.raises(RuntimeError, match="loopback"):
        LocalInstructionReranker("https://example.com/v1")


def test_http_embedding_retriever_orders_indexed_vectors_and_caches():
    retriever = LocalHTTPEmbeddingRetriever("http://127.0.0.1:8081/v1")
    calls = []

    def create(**kwargs):
        calls.append(kwargs["input"])
        vectors = {
            "tea query": [1.0, 0.0],
            "tea passage": [2.0, 0.0],
            "running passage": [0.0, 3.0],
        }
        data = [
            SimpleNamespace(index=index, embedding=vectors[text])
            for index, text in reversed(list(enumerate(kwargs["input"])))
        ]
        return SimpleNamespace(data=data)

    retriever.client = SimpleNamespace(
        embeddings=SimpleNamespace(create=create)
    )
    candidates = [
        {"id": "mem_1", "content": "tea passage"},
        {"id": "mem_2", "content": "running passage"},
    ]

    first = retriever.rank("tea query", [], candidates, 2)
    second = retriever.rank("tea query", [], candidates, 2)

    assert first == second == ["mem_1", "mem_2"]
    assert calls == [["tea query"], ["tea passage", "running passage"]]


def test_http_embedding_retriever_applies_instruction_only_to_query():
    retriever = LocalHTTPEmbeddingRetriever(
        "http://127.0.0.1:8081/v1",
        query_instruction="Retrieve the conversation turn that answers the question",
    )
    calls = []

    def create(**kwargs):
        calls.append(kwargs["input"])
        return SimpleNamespace(data=[
            SimpleNamespace(index=index, embedding=[1.0, float(index)])
            for index, _ in enumerate(kwargs["input"])
        ])

    retriever.client = SimpleNamespace(
        embeddings=SimpleNamespace(create=create)
    )
    retriever.rank(
        "Who likes tea?",
        [],
        [{"id": "mem_1", "content": "Alice likes tea."}],
        1,
    )

    assert calls == [
        [
            "Instruct: Retrieve the conversation turn that answers the question\n"
            "Query:Who likes tea?"
        ],
        ["Alice likes tea."],
    ]


def test_local_instruction_reranker_rejects_unknown_strategy():
    with pytest.raises(RuntimeError, match="strategy"):
        LocalInstructionReranker("http://127.0.0.1:8081/v1", strategy="unknown")


def test_comparative_audit_prompt_checks_false_person_premises():
    reranker = LocalInstructionReranker(
        "http://127.0.0.1:8081/v1", strategy="comparative_audit"
    )
    captured = {}

    def create(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(
            message=SimpleNamespace(content='{"ordered_ids":["mem_1","mem_2"]}')
        )])

    reranker.client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )
    result = reranker.rank("Why did Melanie choose it?", [], [
        {"id": "mem_1", "content": "Caroline: I chose it for inclusion."},
        {"id": "mem_2", "content": "Melanie: I went running."},
    ])

    system = captured["messages"][0]["content"]
    assert result == ["mem_1", "mem_2"]
    assert "wrong person" in system
    assert "Compare all supplied memory turns" in system


def test_comparative_top1_requests_one_id_and_uses_small_output_budget():
    reranker = LocalInstructionReranker(
        "http://127.0.0.1:8081/v1", strategy="comparative_top1"
    )
    captured = {}

    def create(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(
            message=SimpleNamespace(content='{"ordered_ids":["mem_2"]}')
        )])

    reranker.client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )
    result = reranker.rank("question", [], [
        {"id": "mem_1", "content": "first"},
        {"id": "mem_2", "content": "second"},
    ])

    assert result == ["mem_2", "mem_1"]
    assert captured["max_tokens"] == 64
    assert "single best candidate id" in captured["messages"][0]["content"]


def test_conservative_verifier_treats_first_candidate_as_default():
    reranker = LocalInstructionReranker(
        "http://127.0.0.1:8081/v1", strategy="conservative_verify_top1"
    )
    captured = {}

    def create(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(
            message=SimpleNamespace(content='{"ordered_ids":["mem_1"]}')
        )])

    reranker.client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )
    result = reranker.rank("question", [], [
        {"id": "mem_1", "content": "first"},
        {"id": "mem_2", "content": "second"},
    ])

    system = captured["messages"][0]["content"]
    assert result == ["mem_1", "mem_2"]
    assert captured["max_tokens"] == 64
    assert "first candidate as the default" in system
    assert "switch only" in system


def test_constraint_first_top1_requires_structured_audit():
    reranker = LocalInstructionReranker(
        "http://127.0.0.1:8081/v1", strategy="constraint_first_top1"
    )
    captured = {}

    def create(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(
            message=SimpleNamespace(content=(
                '{"audit":{"subject":"Alice","fact":"tea","time":"",'
                '"type":"direct","why":"explicit"},"ordered_ids":["mem_2"]}'
            ))
        )])

    reranker.client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )
    result = reranker.rank("Who likes tea?", [], [
        {"id": "mem_1", "content": "Bob likes tea."},
        {"id": "mem_2", "content": "Alice likes tea."},
    ])

    system = captured["messages"][0]["content"]
    assert result == ["mem_2", "mem_1"]
    assert captured["max_tokens"] == 192
    assert "subject, requested fact, time constraint" in system
    assert '"audit"' in system


def test_answer_first_top1_requires_grounded_answer_before_id():
    reranker = LocalInstructionReranker(
        "http://127.0.0.1:8081/v1", strategy="answer_first_top1"
    )
    captured = {}

    def create(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(
            message=SimpleNamespace(content=(
                '{"answer":"Alice likes tea","why":"explicit statement",'
                '"ordered_ids":["mem_2"]}'
            ))
        )])

    reranker.client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )
    result = reranker.rank("Who likes tea?", [], [
        {"id": "mem_1", "content": "Bob likes tea."},
        {"id": "mem_2", "content": "Alice likes tea."},
    ])

    system = captured["messages"][0]["content"]
    assert result == ["mem_2", "mem_1"]
    assert captured["max_tokens"] == 192
    assert "Answer the query briefly" in system
    assert "false premise" in system


def test_dual_strategy_agreement_skips_arbiter():
    reranker = LocalDualStrategyReranker("http://127.0.0.1:8081/v1")
    candidates = [
        {"id": "mem_1", "content": "first"},
        {"id": "mem_2", "content": "second"},
    ]
    reranker.constraint.rank = lambda query, options, values: ["mem_2", "mem_1"]
    reranker.comparative.rank = lambda query, options, values: ["mem_2", "mem_1"]
    reranker.arbiter.rank = lambda *args: (_ for _ in ()).throw(
        AssertionError("arbiter should not run")
    )

    assert reranker.rank("question", [], candidates) == ["mem_2", "mem_1"]


def test_dual_strategy_disagreement_arbitrates_only_two_finalists():
    reranker = LocalDualStrategyReranker("http://127.0.0.1:8081/v1")
    candidates = [
        {"id": "mem_1", "content": "first"},
        {"id": "mem_2", "content": "second"},
        {"id": "mem_3", "content": "third"},
    ]
    reranker.constraint.rank = lambda query, options, values: ["mem_1"]
    reranker.comparative.rank = lambda query, options, values: ["mem_3"]
    seen = []

    def arbitrate(query, options, values):
        seen.extend(candidate["id"] for candidate in values)
        return ["mem_3", "mem_1"]

    reranker.arbiter.rank = arbitrate

    assert reranker.rank("question", [], candidates) == [
        "mem_3", "mem_1", "mem_2"
    ]
    assert seen == ["mem_1", "mem_3"]


def test_ordered_ids_parser_accepts_fenced_json():
    assert ordered_ids_from_response(
        '```json\n{"ordered_ids":["mem_2","mem_1"]}\n```'
    ) == ["mem_2", "mem_1"]


def test_ordered_ids_parser_rejects_non_json_output():
    assert ordered_ids_from_response("I cannot rank these candidates.") is None


def test_query_expander_rejects_non_loopback_servers():
    with pytest.raises(RuntimeError, match="loopback"):
        LocalQueryExpander("https://example.com/v1")


def test_query_parser_deduplicates_and_bounds_rewrites():
    assert queries_from_response(
        '{"queries":["first","first","second","third","fourth"]}'
    ) == ["first", "second", "third"]


def test_yes_no_probability_normalizes_the_two_token_logits():
    probability = yes_probability_from_top_logprobs([
        {"token": " yes", "logprob": -0.25},
        {"token": " no", "logprob": -1.25},
        {"token": " maybe", "logprob": -0.1},
    ])
    assert probability == pytest.approx(0.7310586)


def test_yes_no_probability_rejects_incomplete_logprob_data():
    assert yes_probability_from_top_logprobs([{"token": " yes", "logprob": -0.2}]) is None


def test_yes_no_probability_accepts_completion_style_logprobs():
    assert yes_probability_from_top_logprobs({"yes": -0.1, " no": -2.1}) == pytest.approx(0.8807971)


def test_yes_no_probability_keeps_the_best_normalized_token_variant():
    probability = yes_probability_from_top_logprobs([
        {"token": "yes", "logprob": -0.1},
        {"token": " Yes", "logprob": -5.0},
        {"token": "no", "logprob": -2.1},
    ])
    assert probability == pytest.approx(0.8807971)


def test_yes_no_prompt_closes_thinking_before_the_scored_token():
    prompt = LocalYesNoReranker("http://127.0.0.1:8081/v1").prompt(
        "What does Mina prefer?", "Mina prefers tea."
    )
    assert "<|im_start|>system" in prompt
    assert "<|im_start|>assistant\n<think>\n\n</think>\n\n" in prompt


def test_yes_no_memory_instruction_is_explicit_and_opt_in():
    reranker = LocalYesNoReranker(
        "http://127.0.0.1:8081/v1", instruction="memory"
    )
    prompt = reranker.prompt("When was it painted?", "Shared image: a sunrise painting")
    assert "original memory turn" in prompt
    assert "released image captions" in prompt


def test_yes_no_evidence_audit_instruction_handles_false_person_premises():
    reranker = LocalYesNoReranker(
        "http://127.0.0.1:8081/v1", instruction="evidence_audit"
    )
    prompt = reranker.prompt(
        "Why did Melanie choose the adoption agency?",
        "Caroline: I chose them because they support LGBTQ+ adoption.",
    )
    assert "wrong named person" in prompt
    assert "contradiction evidence" in prompt


def test_yes_no_reranker_rejects_unknown_instruction():
    with pytest.raises(RuntimeError, match="instruction"):
        LocalYesNoReranker("http://127.0.0.1:8081/v1", instruction="unknown")


def test_yes_no_reranker_rejects_unbounded_batch_size():
    with pytest.raises(RuntimeError, match="batch size"):
        LocalYesNoReranker("http://127.0.0.1:8081/v1", batch_size=33)


def test_yes_no_reranker_batches_prompts_and_aligns_choice_indexes():
    reranker = LocalYesNoReranker("http://127.0.0.1:8081/v1", batch_size=2)
    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        positive = SimpleNamespace(
            index=1,
            logprobs=SimpleNamespace(content=[{"top_logprobs": {
                "yes": -0.1, "no": -2.1,
            }}]),
        )
        negative = SimpleNamespace(
            index=0,
            logprobs=SimpleNamespace(content=[{"top_logprobs": {
                "yes": -2.1, "no": -0.1,
            }}]),
        )
        return SimpleNamespace(choices=[positive, negative])

    reranker.client = SimpleNamespace(
        completions=SimpleNamespace(create=create)
    )
    scores = reranker.score("query", [], [
        {"id": "mem_1", "content": "irrelevant"},
        {"id": "mem_2", "content": "relevant"},
    ])

    assert len(calls) == 1
    assert isinstance(calls[0]["prompt"], list)
    assert scores["mem_1"] < scores["mem_2"]


def test_speaker_mask_uses_only_candidate_prefix_names():
    query, candidates = mask_candidate_speakers(
        "Why did Melanie choose the agency?",
        [
            {"id": "mem_1", "content": "Caroline: I chose it for inclusion."},
            {"id": "mem_2", "content": "Melanie: I went running in Hong Kong."},
        ],
    )

    assert query == "Why did [PERSON] choose the agency?"
    assert candidates[0]["content"].startswith("[PERSON]:")
    assert candidates[1]["content"].startswith("[PERSON]:")
    assert "Hong Kong" in candidates[1]["content"]


def test_max_masked_score_keeps_the_best_exact_or_anonymized_view(monkeypatch):
    reranker = LocalYesNoReranker(
        "http://127.0.0.1:8081/v1", max_masked_score=True
    )
    calls = []

    def fake_score(query, candidates, mode):
        calls.append((query, mode))
        if mode == "exact":
            return {"mem_1": 0.9, "mem_2": 0.2}
        return {"mem_1": 0.1, "mem_2": 0.8}

    monkeypatch.setattr(reranker, "_score_once", fake_score)
    scores = reranker.score("Why did Melanie choose it?", [], [
        {"id": "mem_1", "content": "Melanie: I went running."},
        {"id": "mem_2", "content": "Caroline: I chose it for inclusion."},
    ])

    assert [mode for _, mode in calls] == ["exact", "masked"]
    assert scores == {"mem_1": 0.9, "mem_2": 0.8}


def test_masked_only_and_max_masked_modes_are_mutually_exclusive():
    with pytest.raises(RuntimeError, match="not both"):
        LocalYesNoReranker(
            "http://127.0.0.1:8081/v1",
            mask_speakers=True,
            max_masked_score=True,
        )


def test_completion_logprob_reader_accepts_llama_content_shape():
    expected = [{"token": "yes", "logprob": -0.1}]
    choice = SimpleNamespace(
        logprobs=SimpleNamespace(content=[SimpleNamespace(top_logprobs=expected)])
    )
    assert top_logprobs_from_completion(choice) == expected


def test_completion_logprob_reader_accepts_llama_dictionary_shape():
    expected = [{"token": "yes", "logprob": -0.1}]
    choice = SimpleNamespace(
        logprobs=SimpleNamespace(content=[{"top_logprobs": expected}])
    )
    assert top_logprobs_from_completion(choice) == expected


def test_local_yes_no_reranker_rejects_non_loopback_servers():
    with pytest.raises(RuntimeError, match="loopback"):
        LocalYesNoReranker("https://example.com/v1")
