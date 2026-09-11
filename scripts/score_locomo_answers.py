"""LoCoMo answer scorers.

Four F1 implementations + two BLEU-1 implementations, each replicating one
lineage of code verbatim (see docs/LOCOMO_E2E_PROTOCOL_AUDIT.md §C.6):

- f1_official   : LoCoMo official task_eval/evaluation.py (Porter + Counter multiset;
                  cat1 comma-split mean(max); cat3 ';'-truncation).
- f1_mem0       : Mem0 / A-Mem simple_tokenize (lower + replace . , ! ? with space),
                  SET-based F1.
- f1_memoryart  : MemoryART compute_f1 over nltk.word_tokenize(lower), SET-based.
- f1_memoryos   : MemoryOS calculate_f1 over re.findall(r'\\b\\w+\\b', lower), SET-based.

- bleu1_m1 / bleu1_m4 : nltk sentence_bleu(weights=(1,0,0,0)) with
                  SmoothingFunction().method1 (Mem0/A-Mem) and .method4 (MemoryART).
                  Reference tokenization = word_tokenize(lower) per the baselines.

Only the per-question category is consulted for the *official* dispatcher
(mirroring eval_question_answering). The other three F1 variants and the BLEU
scores are category-agnostic, computed identically for every question, so they
can be compared against their own literature lineage.

No benchmark content (answers/evidence) is read here; callers pass strings.
This module is import-safe (no network, no dataset, no LLM).
"""
from __future__ import annotations

import re
import string
from collections import Counter

# nltk is a hard dependency (user decision #3: pin tokenizer for verbatim BLEU-1).
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu

_PS = PorterStemmer()
_SMOOTH = SmoothingFunction()


# --------------------------------------------------------------------------- #
# 1. Official LoCoMo F1 (task_eval/evaluation.py)                              #
# --------------------------------------------------------------------------- #
def _official_normalize(s: str) -> str:
    """Verbatim normalize_answer from official evaluation.py."""
    s = s.replace(",", "")

    def remove_articles(text: str) -> str:
        # Official fork also removes 'and' (original SQuAD rule is commented out).
        return re.sub(r"\b(a|an|the|and)\b", " ", text)

    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    def remove_punc(text: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text: str) -> str:
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def _official_f1_score(prediction: str, ground_truth: str) -> float:
    prediction_tokens = [_PS.stem(w) for w in _official_normalize(prediction).split()]
    ground_truth_tokens = [_PS.stem(w) for w in _official_normalize(ground_truth).split()]
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    return (2 * precision * recall) / (precision + recall)


def _official_f1_multi(prediction: str, ground_truth: str) -> float:
    """Official f1(): comma-split both sides, mean over gold of max per-pred."""
    predictions = [p.strip() for p in prediction.split(",")]
    ground_truths = [g.strip() for g in ground_truth.split(",")]
    scores = [
        max(_official_f1_score(p, gt) for p in predictions) for gt in ground_truths
    ]
    return sum(scores) / len(scores) if scores else 0.0


def f1_official(prediction: str, ground_truth: str, category: int) -> float:
    """Dispatch per official eval_question_answering for non-adversarial cats."""
    answer = str(ground_truth)
    if category == 3:
        answer = answer.split(";")[0].strip()
    if category in (2, 3, 4):
        return _official_f1_score(prediction, answer)
    if category == 1:
        return _official_f1_multi(prediction, answer)
    # cat5 handled separately by judge; return whole-answer F1 as a safe default.
    return _official_f1_score(prediction, answer)


# --------------------------------------------------------------------------- #
# 2. Mem0 / A-Mem F1 (simple_tokenize)                                        #
# --------------------------------------------------------------------------- #
def _mem0_simple_tokenize(text: str) -> list[str]:
    text = text.lower()
    for ch in [".", ",", "!", "?"]:
        text = text.replace(ch, " ")
    return text.split()


def _set_f1(pred_tokens: list[str], gold_tokens: list[str]) -> float:
    common = set(pred_tokens) & set(gold_tokens)
    if not common:
        return 0.0
    precision = len(common) / len(pred_tokens) if pred_tokens else 0.0
    recall = len(common) / len(gold_tokens) if gold_tokens else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def f1_mem0(prediction: str, ground_truth: str) -> float:
    return _set_f1(_mem0_simple_tokenize(str(prediction)),
                   _mem0_simple_tokenize(str(ground_truth)))


# --------------------------------------------------------------------------- #
# 3. MemoryART F1 (word_tokenize, no normalization)                            #
# --------------------------------------------------------------------------- #
def f1_memoryart(prediction: str, ground_truth: str) -> float:
    pred = word_tokenize(str(prediction).lower())
    gold = word_tokenize(str(ground_truth).lower())
    return _set_f1(pred, gold)


# --------------------------------------------------------------------------- #
# 4. MemoryOS F1 (regex \b\w+\b, set-based)                                   #
# --------------------------------------------------------------------------- #
def f1_memoryos(prediction: str, ground_truth: str) -> float:
    pred = re.findall(r"\b\w+\b", str(prediction).lower())
    gold = re.findall(r"\b\w+\b", str(ground_truth).lower())
    return _set_f1(pred, gold)


# --------------------------------------------------------------------------- #
# BLEU-1 (two smoothing methods)                                              #
# --------------------------------------------------------------------------- #
def bleu1(prediction: str, ground_truth: str, method: int) -> float:
    """sentence_bleu([ref], hyp, weights=(1,0,0,0)) with method1/method4.

    Reference/hypothesis tokenized with word_tokenize(lower), matching the
    MemoryART eval_deepseek.py path; Mem0/A-Mem use the same BLEU call shape.
    """
    ref = word_tokenize(str(ground_truth).lower())
    hyp = word_tokenize(str(prediction).lower())
    if not hyp:
        return 0.0
    smooth = _SMOOTH.method1 if method == 1 else _SMOOTH.method4
    return float(sentence_bleu([ref], hyp, weights=(1, 0, 0, 0),
                               smoothing_function=smooth))


def bleu1_m1(prediction: str, ground_truth: str) -> float:
    return bleu1(prediction, ground_truth, 1)


def bleu1_m4(prediction: str, ground_truth: str) -> float:
    return bleu1(prediction, ground_truth, 4)


# --------------------------------------------------------------------------- #
# Convenience: all answer metrics for one prediction                          #
# --------------------------------------------------------------------------- #
def score_answer(prediction: str, ground_truth: str, category: int) -> dict:
    """Return the four F1 variants + two BLEU-1 variants for one QA pair."""
    pred = str(prediction or "")
    gold = str(ground_truth or "")
    return {
        "f1_official": f1_official(pred, gold, category),
        "f1_mem0": f1_mem0(pred, gold),
        "f1_memoryart": f1_memoryart(pred, gold),
        "f1_memoryos": f1_memoryos(pred, gold),
        "bleu1_m1": bleu1_m1(pred, gold),
        "bleu1_m4": bleu1_m4(pred, gold),
    }


# --------------------------------------------------------------------------- #
# Evidence-level metrics (frozen retrieval, identical to published semantics) #
# --------------------------------------------------------------------------- #
def evidence_metrics(result_ids: list[str], gold_mem_ids: list[str],
                     top_ks: tuple[int, ...] = (1, 3, 10)) -> dict:
    """Per-question evidence ranks over a ranked list of mem IDs.

    Mirrors scripts/evaluate_locomo_retrieval.py per-question bookkeeping:
    - ``ranks[i]`` = 1-based rank of gold item ``i`` in the returned ids (None if absent).
    - ``first_gold_rank`` = min present rank (None if none).
    - ``hit_at_k`` = 1 if first_gold_rank <= k  (per-question "any gold found").
    - ``mrr`` = 1/first_gold_rank.
    - ``evidence_hits_at_k`` = count of gold items with rank <= k  (for POOLED recall).
    - ``n_gold`` = len(gold_mem_ids)  (denominator for pooled recall).

    The published aggregate ``evidence_recall_at_k`` is POOLED:
    ``sum(evidence_hits_at_k) / sum(n_gold)`` across all eligible questions,
    NOT a per-question mean. Aggregation lives in the summarizer.
    """
    out = {"result_ids": list(result_ids), "gold_mem_ids": list(gold_mem_ids),
           "n_gold": len(gold_mem_ids)}
    ranks = [
        next((i + 1 for i, rid in enumerate(result_ids) if rid == g), None)
        for g in gold_mem_ids
    ]
    present = [r for r in ranks if r is not None]
    first = min(present) if present else None
    out["first_gold_rank"] = first
    out["ranks"] = ranks
    out["mrr"] = (1.0 / first) if first else 0.0
    for k in top_ks:
        out["hit_at_{}".format(k)] = int(first is not None and first <= k)
        out["evidence_hits_at_{}".format(k)] = sum(
            1 for r in ranks if r is not None and r <= k)
    return out


def aggregate_evidence(per_q: list[dict], top_ks: tuple[int, ...] = (1, 3, 10)) -> dict:
    """Aggregate per-question evidence_metrics into the published 5-tuple.

    - hit_at_k   = mean of hit_at_k over all questions (per-question).
    - mrr        = mean of mrr over all questions.
    - evidence_recall_at_k = pooled = sum(evidence_hits_at_k) / sum(n_gold).
    """
    n = len(per_q)
    out = {"n": n}
    for k in top_ks:
        out["hit_at_{}".format(k)] = (
            sum(q.get("hit_at_{}".format(k), 0) for q in per_q) / n if n else 0.0)
        num = sum(q.get("evidence_hits_at_{}".format(k), 0) for q in per_q)
        den = sum(q.get("n_gold", 0) for q in per_q)
        out["evidence_recall_at_{}".format(k)] = (num / den) if den else 0.0
    out["mrr"] = sum(q.get("mrr", 0.0) for q in per_q) / n if n else 0.0
    out["n_gold_total"] = sum(q.get("n_gold", 0) for q in per_q)
    return out
