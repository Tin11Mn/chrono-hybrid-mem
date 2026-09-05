# SF v3：跨消息桥接 + Person 画像语义层（设计文档）

日期：2026-09-04
分支：research/p3-evidence-graph（实验线）
前置：SF v2 已采纳（full-1976 Hit@1 +0.0258，p=1.000）
状态：设计阶段 → 实现 → fixed-200 门槛 → full-1976 定论

## 1. 动机（SF v2 失败模式诊断，1976 全量）

SF v2 把每个 session 的事实记为 **per-speaker 直陈观察 [text, dia]**，每条
笔记只允许引用 1-2 个 dia_id，且生成 prompt 明确禁止推断与跨句连接。这带来
三类系统性 miss（相对 NEW 基线都未命中 Top1 的 690 题）：

| 失败模式 | 数量 | 说明 |
|---|---|---|
| gold 跨多消息（同 session） | 36 | 答案需 2+ 条消息合并（如 relationship status = 支持系统 + 单亲领养计划） |
| gold 跨 session | 166 | 两条证据相隔多个 session（如 career 决定在 session 4、动机在 session 1） |
| 单 gold 词面零重叠 | 488 | 查询抽象词（identity/career/destress）与 gold 无语义桥梁；SF 通道 0 次把这类 gold 带进 pool |

诊断关键数字：690 both_miss 中 **SF v2 通道从未把 gold 消息预留下来**
（reserved gold mem = 0）；gold 不在 top30 的 350 题中 279 题连 rerank pool
都进不去（纯检索层漏掉）。conv-26 事实抽查显示笔记质量高但缺两类内容：
① 身份/状态概括句（有 "shared her transgender journey" 无
"Caroline is a transgender woman"）；② 跨消息关联句（"给书的人"与"在哪上班"
分居两条消息时无桥）。

## 2. 借鉴 MemoryART 的迁移点

MemoryART 的 person-memory 检索事件/动机/状态变化/关系/摘要五类字段，其中
**关系与摘要天然跨消息聚合**。官方 locomo10 `observation` 也是 LLM 综合产物
（oracle 覆盖 78% 非 top1）。公平性判定沿用 SF v2 结论：官方 Add 协议只收
原始消息、observation 测试时不可见，但我们用**本地 Qwen3-4B 离线生成等价层**
（非 Search 时调用、非按参考答案适配）合规。

## 3. 设计（C1 + C2 + C4，default-off）

### 3.1 生成层（离线，改动 .locomo/_gen_session_facts.py → v3）

**Pass 1 —— per-session 直陈 + 桥接**（同 v2 输入，prompt 升级）：
每 session 一次 LLM 调用，输出三类笔记：

1. `fact`：单消息直陈（保留 v2 行为，1-2 dia）
2. `bridge`：**跨消息关联笔记**——把同一实体/事件在不同消息中的陈述连成
   一句带指代消解的完整断言（如 "Dave, who gave Alice a book, works at
   Microsoft."），引用 **2-3 个 dia_id**（必须都出现在该 session 内）；
   prompt 要求：只连接对话内显式陈述、不得引入世界知识、不得推断未陈述
   的关系、每句必须能被引用的 dia 逐条支撑。
3. `profile`：person 级**状态/身份概括**（只限 session 内可见信息，如
   "Caroline is interested in counseling or mental-health work."，1-2 dia）。

输出格式（每 speaker 一组）：
```json
{
  "Caroline": {
    "facts": [["text", "D1:3"], ...],
    "bridges": [["text", ["D1:9", "D1:11"]], ...],
    "profiles": [["text", ["D1:11"]], ...]
  }
}
```

**Pass 2 —— 跨 session person 画像**（每 conv 一次 LLM 调用，10 conv）：
把该 conv 全部 session 的 Pass-1 笔记（压缩文本，不带原文）喂给模型，
产出跨 session 画像与关系桥：
- `profile`：稳定身份/状态/偏好（"Caroline is a transgender woman who
  transitioned three years ago."）
- `bridge`：跨 session 事件关联（"Caroline decided on counseling after
  being inspired by her own support network."）
每条标注支撑的 `[session_key, dia_id]` 对（最多 4 个）。注入时按 dia 回溯
源消息；只接受能回溯的条目。

### 3.2 归属校验（C2，注入层）

- 每条 bridge/profile 的每个支撑 dia 必须属于同一 user（同一 conv）；
- 笔记主语的实体名必须出现在至少一个支撑 dia 的 content 中（防张冠李戴）；
- 同一 (source, fact_text) 幂等（沿用 UNIQUE）；
- 若 bridge 声称的关系在支撑 dia 中找不到显式词面支持 → 丢弃该笔记并记
  diagnostics 计数（用于生成质量监控，不硬编码测试答案）。

### 3.3 存储层（storage.py，default-off）

SF v2 的 `session_facts(source_message_id 单值)` 无法表达桥接的多源属性。
改造：

```
session_facts(id, user_id, session_id, source_message_id, fact_text,
              kind TEXT DEFAULT 'fact', created_at)
session_fact_sources(fact_id, source_message_id,
                     PRIMARY KEY(fact_id, source_message_id))
```

- `session_facts.source_message_id` 保留 = 主源（kind='fact' 时即唯一源）；
- `session_fact_sources` 存全部源（含主源），bridge/profile 有多行；
- `add_session_facts` 接受扩展字段 `source_message_ids`（list），自动写
  关联表；kind 默认 'fact' 向后兼容 SF v2 缓存（v2 缓存无需重新生成）。
- **search 的 SF 通道**：fact dense 打分（不变）→ top-N 命中 fact → 经
  session_fact_sources 取其**全部源消息** → 全部进 RRF 通道（每个源消息
  一行，权重 rrf_weight）→ quota 预留全部源 → 每个源的
  ranking_metadata.facts 并入该 bridge/profile 文本（rerank 可见）。

### 3.4 评测注入（evaluate_locomo_retrieval.py）

`_inject_session_facts` 扩展：支持 kind/bridges/profiles 结构；Pass-2 记录
以 `{sample_id}:__profile__` 为 key；注入逻辑把 [session, dia] 回溯成
source_message_id 后写 `source_message_ids`。

## 4. 门槛（沿用 SF v2 原则）

1. default-off：storage 构造参数、evaluate CLI 均默认关闭；
2. 1976 口径不变（offset 758 统一排除）；
3. **采纳门槛：full-1976 相对 SF v2 配对 Hit@1 Δ > 0 且 paired bootstrap
   显著（或 win/loss 结构支持）**——v3 必须证明优于已采纳的 v2，而非仅
   优于 NEW；若持平则记录 REJECT（避免无谓复杂化）；
4. 不伪造：bridge/profile 无法回溯的丢弃并计数，不硬编码；
5. 回归：storage 全量相关测试 + 新多源单测。

## 5. 风险与缓解

| 风险 | 缓解 |
|---|---|
| bridge 幻觉（连接了未陈述的关系） | C2 归属校验：主语名须见于支撑 dia；丢弃+计数 |
| 跨 session 画像引入已更新状态的旧值 | prompt 要求 profile 反映"当前/最新"状态，可引用多条 |
| rerank pool 扰动（SF v2 loss 教训） | 沿 v2 方案：并入 facts 进 candidate_ranking_text |
| 生成成本（Pass1 272 次 + Pass2 10 次调用） | 单实例串行，~1-2 小时 |

## 6. 复现与数据

- 生成脚本 `.locomo/_gen_session_facts_v3.py`（gitignored）
- 产物 `.locomo/session-facts-v3-full.json`（gitignored）
- 评测产物 `.locomo/sfv3-full-*.json`（gitignored）

## 7. 实现迭代记录（2026-09-05）

### 7.1 存储层多源（storage.py）

- `session_facts` 表新增 `kind` 列（fact/bridge/profile，默认 fact）；
  新增 `session_fact_sources(fact_id, source_message_id)` 关联表。
- `add_session_facts` 接受扩展字段 `source_message_ids`（list）与 `kind`；
  每条笔记的主源 + 全部支撑源写入关联表（v2 单源缓存完全向后兼容）。
- search SF 通道 JOIN `session_fact_sources` 展开全部源：fact 去重打分 →
  top-N 命中 → 每源消息进 RRF 通道 + quota 预留。

### 7.2 rerank 并入策略（关键修复）

v3 初版把 bridge/profile 长文本**无差别并入** `ranking_metadata.facts`
（rerank 输入）→ fixed-200 从 v2 0.6000 退到 0.5700（10/12 loss 是已正确
Top1 被长句扰动位移）。修复：**只有 fact/profile 文本并入 rerank 注解；
bridge 只做召回（多源进 pool），不并入 rerank 输入**。

### 7.3 生成器 exhaustive-facts 修复

v3 初版 prompt 让模型把内容分流到 bridges/profiles，conv-26 facts 从 v2 的
310 条降到 208 条 → facts-only 评测 0.5900（< v2 0.6000）。修复：prompt
明确要求 facts EXHAUSTIVE（每个显著陈述都保留，bridges/profiles 是增量），
max_tokens 提到 4096 → conv-26 facts 回到 485 条。

### 7.4 注入层（evaluate_locomo_retrieval.py）

- `_inject_session_facts` 支持 v3 结构（facts/bridges/profiles 每 speaker 组 +
  `{sample}:__profile__` 跨 session 画像），多 dia → 多源消息映射。
- C2 归属校验：bridge/profile 笔记的引用 dia 中至少一条的 speaker 必须与
  笔记所属 speaker 组一致，否则丢弃（防跨人张冠李戴）。

### 7.5 评测（fixed-200，conv-26）

| 方法 | Hit@1 | MRR | 说明 |
|---|---|---|---|
| NEW | 0.5850 | 0.6546 | 基线 |
| SF v2 | 0.6000 | 0.6777 | 已采纳 |
| v3 初版（无差别并入） | 0.5700 | 0.6638 | rerank 扰动 |
| v3 facts-only | 0.5900 | 0.6758 | facts 被分流 |
| **v3.1（exhaustive facts + 精准并入）** | **0.6050** | **0.6884** | 净 +1 wins |

## 8. full-1976 定论：REJECT（2026-09-05）

全量 1976 配对（offset 758 统一排除），SF v3.1 vs SF v2：

| 指标 | SF v2 | SF v3.1 | Δ |
|---|---|---|---|
| Hit@1 | 0.6108 | **0.6134** | +0.0025 |
| MRR | 0.6929 | **0.6943** | +0.0015 |
| Hit@3 | 1517 | **1526** | +9 |
| Hit@10 | **1624** | 1611 | -13 |
| wins / losses | — | 90 / 85 | 净 +5 |

paired bootstrap（10,000 次，seed 20260826）：
- Hit@1 Δ CI95 [-0.0106, +0.0157]，p(>0) = 0.636（**不显著**）
- MRR Δ CI95 [-0.0076, +0.0104]，p(>0) = 0.631（**不显著**）

vs NEW 基线仍维持增益（124 wins / 68 losses），即 v3.1 保留了 v2 的
全部增益但**未提供超越 v2 的显著增量**（Hit@10 还略降）。

**结论：REJECT。** SF v2 保持为采纳方法（full-1976 Hit@1 0.6108、
MRR 0.6929，相对 NEW 均显著）。v3.1 的完整实现（storage 多源、注入 v3、
生成器 exhaustive prompt、评测链路）作为探索存档保留——跨消息桥接与
person 画像的机制在 fixed-200 上展示过边际价值（+1 win），但全量统计不
支持采纳，避免无谓复杂度。代码 default-off 保留，不进入稳定路径。

