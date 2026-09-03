# ChronoHybridMem 新方法复现指南

复现对象：P4-A evidence-need 独立检索（quota 2）+ `--need-select-by-bm25`
（分支 `research/p3-evidence-graph`，评测 commit `0e3ecba`）。

## 前置

- Python 3.11+（开发用 3.12.13，见 `chrono-hybrid-mem/.venv-local`）
- 本地模型：Qwen3-4B-Q4_K_M.gguf（llama.cpp 服务）
- 数据：`locomo10.json`（LoCoMo 公开格式本地切片；不入库、不提交）

## 1. 启动本地 Search 模型（单 slot 16384 ctx）

```powershell
# llama.cpp b9637；ctx 必须 16384（8192/每-slot 会在长对话 rank 时截断）
& '...\.local-tools\llama-b9637\llama-server.exe' `
  -m '...\.local-models\Qwen3-4B-Q4_K_M.gguf' `
  --host 127.0.0.1 --port 8081 -ngl 99 --ctx-size 16384 --cache-reuse 0
```

注意：**不要加 `--parallel N`**（每 slot ctx = 16384/N，N≥2 会在长对话触发
`Context size has been exceeded`）。单进程一次一个 chunk 最稳定。

## 2. 全量 1976 评测（10 段，逐段串行）

每段 200 问；offset 758 所在段（600-799）需分两段并剔除 758：

```powershell
# 段 A（600-757）
python -m scripts.evaluate_locomo_retrieval `
  --dataset '..\chrono-hybrid-mem\.locomo\locomo10.json' `
  --local-search-model-url 'http://127.0.0.1:8081/v1' --local-search-model-name local `
  --structured-query-plan --max-questions 158 --question-offset 600 `
  --include-question-diagnostics `
  --evidence-need-retrieval --evidence-need-quota 2 --need-select-by-bm25 `
  --model-timeout 300 --output '.locomo\newmethod-chunk-0600a.json'

# 段 B（759-800，42 问；剔除 758）
python -m scripts.evaluate_locomo_retrieval ... --max-questions 42 --question-offset 759 `
  ... --output '.locomo\newmethod-chunk-0600b.json'
```

其余段（offset 0/200/400/800/1000/1200/1400/1600/1800）各 200 问
（1800 段实际 177 问）。

## 3. 合并与指标

- 合并：将 11 个 chunk 的 `question_diagnostics` 合并为 1976 条唯一记录
  （剔除 offset 800 重复与 758）；Hit@K/MRR/nDCG 由每题 `first_gold_rank`
  重算；Recall@5 同 Q-level；Evidence Recall@1/3/10 由各 chunk
  `raw_counts.evidence_hit_counts` 聚合。
- 分析脚本（gitignored，`.locomo/`）：`_unified_metrics.py`（三方法同题集
  指标 + bootstrap）、`_gen_per_query.py`（明细）、`_correct_merge.py`。

## 4. 单题复跑

```powershell
python -m scripts.evaluate_locomo_retrieval `
  --dataset '..\chrono-hybrid-mem\.locomo\locomo10.json' `
  --local-search-model-url 'http://127.0.0.1:8081/v1' --local-search-model-name local `
  --structured-query-plan --max-questions 1 --question-offset N `
  --include-question-diagnostics `
  --evidence-need-retrieval --evidence-need-quota 2 --need-select-by-bm25 `
  --model-timeout 300 --output '.locomo\_single-N.json'
```

## 已知运行坑

1. **offset 758**（conv-43，680 消息）：rank 请求在 llama-server 端必然挂起
   （非 timeout/慢，是请求无响应）；全部方法统一排除该题。
2. **并行度**：并发 >1 个 chunk 会因共享 KV cache 触发 context exceeded；
   逐段串行。
3. **服务器长跑**：建议每 ~10h 或异常后重启（`--cache-reuse 0` 缓解）。
4. **git 元数据**：沙盒会删除分支 ref/worktree 文件；commit/push 后需按
   `docs/GIT_METADATA_LOSS_2026-08-25.md` 重建 ref。
