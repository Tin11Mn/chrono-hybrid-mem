import pytest
from types import SimpleNamespace

from app.local_instruction import (
    LocalInstructionReranker,
    LocalQueryExpander,
    LocalYesNoReranker,
    ordered_ids_from_response,
    queries_from_response,
    top_logprobs_from_completion,
    yes_probability_from_top_logprobs,
)


def test_local_instruction_reranker_rejects_non_loopback_servers():
    with pytest.raises(RuntimeError, match="loopback"):
        LocalInstructionReranker("https://example.com/v1")


def test_local_instruction_reranker_rejects_unknown_strategy():
    with pytest.raises(RuntimeError, match="strategy"):
        LocalInstructionReranker("http://127.0.0.1:8081/v1", strategy="unknown")


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
    prompt = LocalYesNoReranker.prompt("What does Mina prefer?", "Mina prefers tea.")
    assert "<|im_start|>system" in prompt
    assert "<|im_start|>assistant\n<think>\n\n</think>\n\n" in prompt


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
