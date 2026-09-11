# LoCoMo E2E Fixed-100 (Stratified Diagnostic) Results

> **DIAGNOSTIC / NOT FINAL FULL-SET RESULT.**
> Fixed-100 uses an engineered 25/25/25/25 category balance that is NOT the
> natural LoCoMo distribution. Its Overall numbers must never be quoted as
> "LoCoMo overall". The formal Full-1540 track uses the natural distribution
> (multi-hop 282 / temporal 321 / open-domain 96 / single-hop 841) and is a
> separate, later run.

- Date: 2026-09-12
- Method: ChronoHybridMem **SF v2 + Qwen3-4B** (frozen retrieval artifacts,
  SF v2 adoption lineage = `f6c1f98`, docs/SF_V2_CODE_FREEZE.md)
- Gateway: **ChatAnywhere** (`https://api.chatanywhere.tech/v1`), requested
  model `gpt-4o-mini`, expected returned snapshot `gpt-4o-mini-2024-07-18`,
  temperature 0 (answer and judge)
- Prompts: frozen Mem0-compatible copies (answer `748eb924…`, judge
  `6d33ff20…` — unchanged since Smoke-20)
- Run: `results/locomo_e2e/fixed100-stratified/` (manifest hash
  `15653008a6ed956a76b2669675ab65ad57cb08a3693aebbdf6a85432b3e78125`)

## 1. Question set

100 questions = 25 multi-hop + 25 temporal + 25 open-domain + 25 single-hop.
Deterministic stratified prefix per category over the frozen-artifact offset
order (frozen into `question_manifest.json` before any API call; includes
the Smoke-20 subset). 0 duplicate question_ids; every question maps to the
official `locomo10.json` (SHA256 `79fa87e9…8ff4`); 0 questions lack frozen
retrieval.

## 2. Overall (stratified diagnostic — NOT LoCoMo overall)

| Metric | Value |
|---|---:|
| f1_mem0 | 0.3589 |
| BLEU-1 (m1) | 0.2721 |
| **LLM Judge Accuracy** | **0.60** (60/100) |
| f1_official (diagnostic) | 0.3802 |
| f1_memoryart (diagnostic) | 0.3361 |
| f1_memoryos (diagnostic) | 0.3589 |
| Hit@1 / Hit@3 / Hit@10 | 0.50 / 0.63 / 0.71 |
| MRR | 0.5734 |
| Evidence Recall@10 (pooled) | 0.5190 |

## 3. Per-category breakdown

| Category | n | f1_mem0 | f1_official | BLEU-1 | Judge acc | Hit@1 | Hit@10 | EvRec@10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Multi-hop | 25 | 0.3030 | 0.2917 | 0.1942 | 0.52 | 0.36 | 0.68 | 0.4194 |
| Temporal | 25 | 0.4301 | 0.4668 | 0.3430 | 0.56 | 0.84 | 0.92 | 0.9200 |
| Open-domain | 25 | 0.1402 | 0.2087 | 0.1146 | 0.44 | 0.20 | 0.44 | 0.2826 |
| Single-hop | 25 | 0.5623 | 0.5536 | 0.4365 | 0.88 | 0.60 | 0.80 | 0.8000 |
| **Macro avg** | — | **0.3589** | **0.3802** | **0.2721** | **0.60** | 0.50 | 0.71 | 0.6055 |

Macro averages are the mean of the four category values (equal category
weights). Open-domain is the weakest category at both retrieval (Hit@10
0.44) and judging (0.44); temporal retrieval is strong (0.92) while its
judge accuracy stays moderate (0.56) — a generation/grounding gap, not a
retrieval gap.

## 4. Retrieval × answer-generation matrix (diagnostic)

| | Judge CORRECT | Judge WRONG |
|---|---:|---:|
| **Retrieval Hit@10 (n=71)** | 48 (67.6% of hits) | 23 |
| **Retrieval Miss@10 (n=29)** | 12 (41.4% of misses) | 17 |

Judge accuracy conditional on retrieval depth: Hit@1 → 0.66; Hit@3 → 0.6508;
Hit@10 → 0.6761; **Miss@10 → 0.4138**.

Reading: retrieval quality is a major bottleneck (miss@10 halves judge
accuracy), but 23/71 hits still fail judging — answer-generation grounding
loses a comparable number. Both bottlenecks are real at this scale.

## 5. Notable phenomena (from Smoke-20, now quantified)

| Phenomenon | Count |
|---|---:|
| Retrieval hit@10 but judge WRONG | 23 |
| Retrieval miss@10 but judge CORRECT | 12 |
| f1_mem0 = 0 but judge CORRECT | 7 |
| f1_official = 0 but judge CORRECT | 5 |

No method or prompt changes were made in response to these; they are
recorded for the Full-1540 analysis.

## 6. Model routing reliability (frozen retry policy)

Policy (frozen 2026-09-12 after the mid-run drift): every judge response is
VALID only when `returned_model == gpt-4o-mini-2024-07-18`; invalid attempts
are logged (metadata only, content never persisted or scored) and retried
with identical inputs, max 3 attempts per question; exhaustion stops the run
(NO-GO signal).

| Metric | Value |
|---|---|
| Judge API calls (whole Fixed-100 effort) | **101** |
| — valid (accepted) judge responses | **100** |
| — model-drift attempts | **1** (2026-09-12 ~03:16, `conv-26:60`, returned `gpt-4.1-mini-2025-04-14`; fingerprint not recorded — attempt predates the attempts-log mechanism; content discarded unscored) |
| — drift rate per attempt | 1/101 ≈ **0.0099** |
| Post-policy resume attempts (46 questions) | 46/46 accepted, 0 drift, max attempts = 1 |
| Accepted returned-model distribution | **gpt-4o-mini-2024-07-18 = 100%** ✓ |
| Answer returned-model distribution | gpt-4o-mini-2024-07-18 = 100% (100/100) |
| Answer fingerprints | `fp_369e662417` ×95, `fp_acf70d9fcf` ×5 |
| Judge fingerprints (accepted) | `fp_369e662417` ×96, `fp_acf70d9fcf` ×4 |
| Consecutive identity preflight before resume | 3/3 valid |
| Retries triggered under the frozen policy | 0 (no question needed a 2nd attempt after the policy was frozen) |

**Accepted purity = 100% `gpt-4o-mini-2024-07-18`** → results are eligible
for the formal Fixed-100 summary.

## 7. Reliability

- Answer: 100 calls, 100 ok, 0 failures, 0 retries, 0 timeouts.
- Judge: 101 calls, 100 ok, 1 drift attempt (retried under policy), 0 parse
  failures (100/100 labels parsed), 0 timeouts.
- Checkpoint/resume: exercised three times (429 quota stop, drift stop,
  planned resume); no duplicate question_ids; manifest hash verified on
  every resume; config digest `c2d0d5653edd…` unchanged throughout.

## 8. Cost

| Phase | tokens in / out | estimated cost |
|---|---|---|
| Answer | 87,743 / 735 | $0.0137 |
| Judge | 40,065 / 700 | $0.0064 |
| **Total (token-based estimate)** | 127,808 / 1,435 | **$0.0200** (rate assumption $0.15/$0.60 per 1M) |

The gateway does not return billing amounts; no provider-invoice figure is
claimed. Well under the $5 cost guard.

## 9. Manual audit (12 cases, ≥2 per category)

| # | Type | off | cat | Q (abbrev.) | Gold | Answer | Judge | f1m/f1o | Gold rank | Analysis |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | A hit+correct | 0 | 2 | When did Caroline go to the LGBTQ support group? | 7 May 2023 | May 7, 2023 | CORRECT | 1.00/1.00 | 1 | Exact date; format variant accepted by judge |
| 2 | A hit+correct | 3 | 1 | What did Caroline research? | Adoption agencies | Adoption agencies. | CORRECT | 1.00/1.00 | 1 | Multi-hop chain intact |
| 3 | A hit+correct | 79 | 4 | What did the charity race raise awareness for? | mental health | Mental health | CORRECT | 1.00/1.00 | 2 | Single-hop, gold at rank 2 |
| 4 | A hit+correct | 2 | 3 | Fields Caroline would likely pursue? | Psychology, counseling certification | Counseling or mental health. | CORRECT | 0.29/0.29 | 1 | Commonsense inference over hit evidence |
| 5 | B hit+wrong | 11 | 1 | Where did Caroline move from 4 years ago? | Sweden | Her home country. | WRONG | 0.00/0.00 | 1 | **Generation failure**: evidence retrieved but the answer violated the prompt rule "Use 'Sweden' not 'home country'" |
| 6 | B hit+wrong | 22 | 3 | Dr. Seuss books on her shelf? | Yes (collects classic children's books) | No evidence of Dr. Seuss books. | WRONG | 0.15/0.15 | 1 | Evidence-grounded conservatism vs gold's expected inference |
| 7 | B hit+wrong | 28 | 2 | When did Caroline go to the adoption meeting? | Friday before 15 Jul 2023 | Last Friday, September 15, 2023 | WRONG | 0.55/0.60 | 1 | Relative-expression grounded to the wrong session timestamp |
| 8 | C miss+correct | 1 | 2 | When did Melanie paint a sunrise? | 2022 | 2022 | CORRECT | 1.00/1.00 | none | Gold year absent from top-10 yet answered correctly (parametric/lucky) — flagged, not credited to retrieval |
| 9 | D miss+wrong | 7 | 1 | Caroline's relationship status? | Single | Not mentioned in memories. | WRONG | 0.00/0.00 | none | Clean retrieval miss; conservative refusal |
| 10 | D miss+wrong | 82 | 4 | Caroline's summer plans? | researching adoption agencies | Family outing or special plans. | WRONG | 0.00/0.00 | none | Miss + wrong-evidence synthesis (nearest to hallucination) |
| 11 | E divergence | 6 | 2 | When is Melanie going camping? | June 2023 | Next month. | CORRECT | 0.00/0.00 | 1 | Judge resolves relative time from context; token F1 = 0 — exactly why both scales are reported separately |
| 12 | F temporal-relative | 9 | 2 | When did Caroline meet friends/family/mentors? | Week before 9 Jun 2023 | Last week, on June 10, 2023. | WRONG | 0.50/0.55 | 1 | Session-timestamp anchoring picks the wrong reference week |

Failure-mode tally: 4 generation/grounding failures on retrieved evidence
(#5, #6, #7, #12), 2 clean retrieval misses (#9, #10), 1 miss-but-correct
anomaly (#8), 1 judge/F1 scale divergence (#11), 4 clean successes. Consistent
with the matrix: neither stage dominates; both must be reported at Full-1540.

## 10. Reproduction pointers

- Manifest: `results/locomo_e2e/fixed100-stratified/question_manifest.json`
- Per-question rows: `results/locomo_e2e/fixed100-stratified/per_question.jsonl`
- Raw responses: `results/locomo_e2e/fixed100-stratified/raw_model_outputs/`
- Checkpoint: `results/locomo_e2e/fixed100-stratified/checkpoint.json`
- Summaries: `metric_summary.json`, `category_summary.json` (same dir)
- Audit tool (frozen artifacts): `scripts/paper_parity_audit.py`
