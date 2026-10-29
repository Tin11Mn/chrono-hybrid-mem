# Recovering retrieval quality after 58.36

The previous release passed source-preservation checks but scored58.36, below62.02 and63.54. More extracted rows and passing infrastructure checks did not establish better answer quality. Current production has zero DB-lock/HTTP500 errors, yet5recorded nonempty candidate pools became empty results because hypothetical evidence was removed as though it had no evidentiary value.

Required behavior:
- Never certify a hypothetical/future value as current. If no active value exists, preserve its original qualification as unresolved evidence rather than returning an empty memory.
- A slot cannot suppress an immutable source that contains another slot or relation. Keep original text.
- State completeness is local only when every uncertain source is demonstrably a simple named assertion about a different subject. Unindexed/zero-record/complex/pronominal/same-subject sources continue to prevent deterministic removal. Preserve uncertainty; do not flip global complete flags.
- Latest status means current state; chronological listing/history still preserve versions. Explicit summary imperatives recover summary routing if the model mislabeled them.
- Respect independent validated ordered_ids on complete and incomplete selection before top_k. It cannot add or remove selected members.
- Ground first-person facts in the value's own direct self clause, not another sentence's I. Uncertain attribution remains partial with raw evidence preserved.
- Split CJK sentence punctuation without a whitespace requirement. State cues and modality must come from the value clause, not another subject's correction/future plan.
- Named whole-history recovery must include literal raw sources even if extraction produced no records. Do not force known privacy-only sources into a summary, including partial annotations. This does not certify unknown unextracted text as non-sensitive.
- Repair feedback must stay within8192characters, omit copied span/derived conditions, explicitly label truncated advisory values; supply the original page separately.
- Record allowlisted intent and nonempty-pool-to-empty diagnostics without queries, answers, or source text.
- Plan only premises needed for the actual question. Do not pad four needs or require unasked triggering actions for a chronological listing. Missing evidence must still fail certification; never drop a genuine prerequisite to force success.

Verification is predeclared: semantic regressions must fail before fixes; independent12question/17message end-to-end contract is frozen at b21ed972ed67715fef74c05aac7b258f6056c2e2fb8c75e956b79763c10f2032. Compare original/current/candidate actual Add/Search and a separate identical answer call. Do not change expected answers or drop failed cases. Report exact-string false negatives and stochastic model mistakes as limitations. Preserve all runs, including baseline lock failures. Synthetic scores do not estimate leaderboard accuracy. Only deploy after functional gates and document actual quality outcome.

No benchmark-specific answers or new dependencies. API/auth/current model stay compatible. Prior releases/DBs remain available for rollback. Do not change an active evaluation's version.
