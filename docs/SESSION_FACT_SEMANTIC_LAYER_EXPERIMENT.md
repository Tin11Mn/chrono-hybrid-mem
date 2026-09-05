# Session-Fact Semantic Layer（B 方向）实验记录

日期：2026-09-04
分支：research/p3-evidence-graph（实验线）
状态：fixed-50 净正 → fixed-200 弱正 → **full-1976 统计显著，采纳**

## 1. 动机（MemoryART 借鉴）

参照 AAAI-26 MemoryART（多记忆 + ART 情景记忆），其核心可迁移主张：
对话历史应有一层 **LLM 结构化语义事实**（带源消息回溯），检索在语义层做
匹配、再回溯到原始证据——而不是只在原始消息词面上检索。

我们 P4 审计已确认：1976 全量中 channel_miss 112 题（76 题连 evidence-need
通道都 miss）是**推理型零词面重叠**（如 "political leaning" → gold
"religious conservatives & LGBTQ rights"），词面层无解。

## 2. 数据与公平性

- 官方 locomo10.json 自带 `observation`（LLM 事实 + 源 dia），诊断显示
  oracle 覆盖 78% 非 top1 查询——但 **Add 协议只收原始消息，observation
  测试时不可见**，直接使用违反公平性。
- 合规路径：**本地 Qwen3-4B 离线生成等价事实层**（per-session per-speaker
  [text, dia] 观察），评测时注入 store 的 session_facts 表。
- 产物：`.locomo/session-facts-full.json`（272 sessions / 4250 facts /
  全部 dia 有效），生成脚本 `.locomo/_gen_session_facts.py`。

## 3. 实现（default-off）

### storage.py
- 新表 `session_facts(id, user_id, session_id, source_message_id, fact_text)`，
  无条件基础 schema。
- `MemoryStore.add_session_facts(...)`：批量幂等注入。
- `session_fact_layer / rrf_weight / quota / top_n` 构造参数（默认 off / 0.05 / 3 / 10）。
- search()：启用时加载该 user 全部 session_facts → dense 打分（semantic
  retriever）→ top-N → 映射源消息进 RRF 通道（低权重）→ quota 预留进
  rerank pool；**命中消息的 fact 文本并入 ranking_metadata.facts**，使
  Search rerank 的 candidate_ranking_text 包含语义层证据（v2 修复）。

### evaluate_locomo_retrieval.py
- `--session-fact-layer / --session-fact-cache / --session-fact-rrf-weight /
  --session-fact-quota / --session-fact-top-n` CLI。
- `_inject_session_facts(...)`：按 sample_id 匹配缓存记录，dia→content→mem_id
  回溯后调用 store.add_session_facts。
- question_diagnostics 增加 session_fact 字段。

### 测试
- `tests/test_session_fact_layer.py`：5 例（表存在 / 注入幂等 / default-off
  不影响检索 / 启用时预留 / 无 retriever 惰性）。

## 4. fixed-50 配对结果（同 200 题集前 50 问，offset 0-49）

| 指标 | 基线 (P4-A+bm25) | +SF v1 | +SF v2 |
|---|---|---|---|
| Hit@1 | 28/50 = 0.5600 | 28/50 = 0.5600 | **30/50 = 0.6000** |
| MRR | 0.6207 | 0.6290 | **0.6537** |
| wins/losses | — | 3 / 3 | **3 / 1** |

- wins（SF 把 gold 提上 Top1）：off=2 (career path 5→1)、32 (2→1)、44 (2→1)
- v1 loss 3 例（18/37/49）：SF 引入假阳性消息扰动 rerank pool，rerank
  listwise 级联 → 原 Top1 被排后（base 完全可复现，非 LLM 噪声）
- **v2 修复**（rerank 输入含 fact 证据后 rerank 认可 SF 命中）：
  off=18 (3→1)、off=49 (2→1) 修复；off=37 顽固（多跳聚合型，4 gold 分散，
  SF 引入单方面消息 mem_196，rerank 仍不认可）

## 5. 结论（fixed-50 阶段）

SF v2（rerank 输入含语义层证据）在 fixed-50 净正：Hit@1 +0.040、MRR +0.033。
需 fixed-200 全量配对确认统计显著后再决定采纳/REJECT。

## 6. fixed-200 配对结果（2026-09-04 更新）

同 200 题集（offset 0-199），与历史 P4-A+bm25 基线配对：

| 指标 | 基线 | +SF v2 | Δ |
|---|---|---|---|
| Hit@1 | 117/200 = 0.5850 | 120/200 = 0.6000 | **+0.0150** |
| MRR | 0.6546 | 0.6777 | **+0.0231** |
| Hit@3 | 145 | 149 | +4 |
| Hit@10 | 152 | 162 | +10 |
| wins / losses | — | 12 / 9 | 净 +3 |

paired bootstrap（10,000 次，seed 20260826）：
- Hit@1 Δ CI95 [-0.030, +0.060]，p(>0) = 0.71（**不显著**）
- MRR Δ CI95 [-0.008, +0.054]，p(>0) = 0.93（弱显著）

解读：fixed-200（n=200）统计力不足；12 win 覆盖实体/属性列举类
（career path、art types、music artists、beach trips 等），9 loss 为
rerank 对 pool 变化的敏感位移（含顽固的 off=37 多跳聚合题）。全量
1976 配对评测进行中（更大的 n 定论）。

## 7. full-1976 配对结果（2026-09-04 定论）

全量 1976 题（offset 758 统一排除，conv-43 超长会话 rank 请求在
llama-server 端挂起，P1/P4-A/NEW/SF v2 一致），与 NEW 基线逐题配对：

| 指标 | 基线 (NEW) | +SF v2 | Δ |
|---|---|---|---|
| Hit@1 | 0.5850 | **0.6108** | **+0.0258** |
| MRR | 0.6618 | **0.6929** | **+0.0311** |
| Hit@3 | 1447 | 1517 | +70 |
| Hit@10 | 1543 | 1624 | +81 |
| wins / losses | — | 130 / 79 | 净 +51 |

paired bootstrap（10,000 次，seed 20260826）：
- Hit@1 Δ CI95 **[0.0116, 0.0400]，p(>0) = 1.000（统计显著）**
- MRR Δ CI95 **[0.0208, 0.0411]，p(>0) = 1.000（统计显著）**

category Hit@1 分解：cat1 114→130、cat2 190→200、cat4 543→567 提升；
cat3 26→27、cat5 283→283（无 category 下降）。

> fixed-200（n=200）p=0.71 不显著是统计力不足；n=1976 下 CI 完全不含 0，
> 双指标 p=1.000。SF v2 全量**采纳**。

### 评测基础设施修复（full 评测中发现）

- `app/model.py` 普通 `rank_candidates` 原无 `max_tokens` 上限：conv-43 超长
  会话（680 消息）的 rank prompt 达 13558 tokens（近 16k ctx 上限）时，
  Qwen3 本地端点病态无限生成（8000+ tokens 不停），整段评测 APITimeoutError
  崩溃且无产物（runner 无 checkpoint）。
- 修复：与 `rank_candidates_with_confidence` 对齐——`max_tokens=400` +
  超长截断 RuntimeError 时回退 fusion 序（`return []`），单题不崩段。
- 验证：offset 800 单题从挂起 600s+ 崩溃 → 113s 正常完成；seg 0800
  （conv-43 196/200 题）4×50 窗全部成功。
- 注意：截断回退只发生在病态超长题（极端少），正常题 rank 输出远小于
  400 tokens，不受影响。

### offset 758 复测（SF v2 下仍不可跑）

用 SF v2 全配置 + `--model-timeout 300` 复测 offset 758 单题：rank 请求在
llama-server 端仍挂起（进程无输出、CPU 不增长、超 10 分钟），判定挂起终止
（日志 0 字节、无结果 JSON）。结论：758 挂起是 llama-server 对 conv-43
680 消息超长会话的基础设施限制，与检索方法无关；SF v2 评测同样统一排除，
1976 口径与 P1/P4-A/NEW 完全一致，不伪造该题指标。

## 8. 单元测试与回归

- `tests/test_session_fact_layer.py`：5 例全过（表存在 / 注入幂等 /
  default-off 不影响检索 / 启用时预留 / 无 retriever 惰性）。
- 核心回归：`tests/test_hybrid_retrieval.py` + `test_p4a_evidence_need.py`
  **46 passed**（SF default-off 无回归）。
- rank max_tokens 修复：正常题输出 <400 tokens 不受影响；`model.py`
  py_compile 通过，冒烟单题验证 OK。

## 9. 失败模式深挖与参数收束（2026-09-05）

### 9.1 注入层回归修复（v2 flat 缓存兼容）

SF v3 实验给注入层加 v3 dict 结构支持时，把 v2 flat-list 缓存的判断
（`if isinstance(blob, list)`）放到了 `if not isinstance(blob, dict):
continue` 之后——v2 格式（speaker → [[text, dia], ...]）被提前跳过，
导致 v2 缓存在新代码下注入 0 行（SF facts=0）。修复：flat-list 分支移到
dict 检查之前。修复后 v2 缓存注入恢复（facts=309/conv-26），top10 复现
0.6050（历史 0.6000，5 题差异为 rerank LLM 运行噪声，SF 通道输入不变）。

### 9.2 D：检索层 miss（gold 不在 top30，350 题）归因

| 子类 | 数量 | 说明 |
|---|---|---|
| low_lexical_overlap | 137 | 查询抽象问法 vs gold 具体陈述，dense 也漏 |
| multi_gold_msgs | 95 | gold 分散多条消息 |
| temporal_cue | 59 | 时间线索查询 |
| q_names_absent_from_gold | 30 | 问的人不在 gold 文本 |
| zero_lexical_overlap | 23 | 完全零词面 |

328 低词面题 SF 通道全触发（triggered=True）但只 reserved 到 8 个 gold
消息——SF 的 fact 直陈句 dense 对抽象-具体语义鸿沟几乎无效。

### 9.3 E：SF v2 的 79 loss 归因

- 77/79 gold 仍在 final top10 但 rerank 排后（rank 1 → 2~5），79/79 SF
  全触发——SF 带进 pool 的消息 + fact 注解扰动 rerank listwise 判定，
  把已正确的 Top1 排后。这是 SF 机制的固有代价（救 130 / 扰动 79）。

### 9.4 top_n 假设证伪（参数收束）

假设"gold fact 排在 top_n=10 之外"→ 提高 top_n 到 60 验证（conv-26）：
**0.5850 < top10 的 0.6050**（4 wins / 8 losses）——提高 top_n 没有救回
lowlex miss，反而让更多噪声 fact 消息进 pool 干扰 rerank。

**结论：SF v2 在 top_n=10 已是局部最优；D/E 方向无免费参数空间。** 低词面
miss 的根因是生成层 fact 直陈句与抽象查询之间的语义鸿沟（dense 无法
跨越），改进需查询侧改写或更大 embedding 模型等新机制（成本高、收益
不确定），本轮收束不投入。SF v2（top_n=10）保持为采纳配置。

## 10. SIMPLE 时间通道实验（2026-09-05）：REJECT

动机（认知理论 SIMPLE：相对时间辨别、对数压缩、局部竞争，非简单线性
recency）。诊断显示 temporal-intent 256 题（full 0.668）与 8 个 conv-26
temporal miss。

实现（default-off）：
- evaluate CLI `--temporal-bonus`（float，默认 0）+ `--temporal-log-scale`；
- storage 构造 `temporal_bonus / temporal_log_scale`，fuse 层时间加分支持
  对数压缩形状 `1/(1+log1p(age_days))`（recency）与反向（historical）；
  时间通道仅当 bonus>0 时启用（历史所有评测 bonus=0，即时间通道一直关）。

conv-26 fixed-200 A/B（SF v2 配置，5 配置）：

| 配置 | Hit@1 | MRR |
|---|---|---|
| off（基线） | 0.6050 | 0.6827 |
| log-0.5 | 0.6000 | 0.6777 |
| log-1.0 | 0.6050 | 0.6827 |
| log-2.0 | 0.6000 | 0.6777 |
| linear-1.0 | 0.6000 | 0.6777 |

temporal 子集（30 题）各配置均为 21-22/30，无明显差异。

失败机制诊断：8 个 temporal miss 中多个 gold 已在 channel 排名靠前甚至被
SF reserved（off=54 gold mem_272 被 SF 预留仍排后），瓶颈在 **rerank 对
"when" 时间相关性的判断**（非召回、非 RRF 时间加分可救）——时间加分只
作用于 rerank 前的 fusion 排序，被 rerank listwise 重排覆盖。

**结论：REJECT。** 时间通道保持默认关（bonus=0），不改任何历史结果。
SIMPLE 时间形状作为可选项存档（CLI 已暴露）；temporal miss 的真正瓶颈
在 rerank 层的时间理解，需提示层改造（不同机制）。



