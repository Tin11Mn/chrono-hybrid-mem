# MemoryART 借鉴调研与 observation 语义层诊断（A 方向）

日期：2026-09-04
状态：调研 + 离线诊断完成；B 方向（自生成 person 画像）待实现

## 1. MemoryART 论文要点（AAAI-26，已获全文 PDF）

- 三记忆框架：工作记忆（N 轮滑动窗口 FIFO）/ 情景记忆（**Fusion ART 分层网络**，核心）/
  语义记忆（User/Agent Profile 动态档案）
- ART 机制：事件多通道向量 x = {x^(1)..x^(K)}；选择函数
  T_i = Σ_k γ^(k)·Sim^(k)(x^(k), x_i^(k)) / (α + Σ_k ‖x_i^(k)‖₁)；
  共振条件：与 episode 原型 w_j 的通道相似度超警戒阈值 ρ∈[0,1] 才吸收并更新原型，
  否则新建 episode（防原型坍缩 = 保持 episode 内语义一致，避免异构内容压成相似原型）
- 检索：clue 符号匹配 + dense 嵌入双策略；召回不更新原型
- LoCoMo 结果（GPT-4o-mini，answer F1/BLEU 口径，四分类）：Multi-hop F1 **47.04**、
  Temporal **27.46**、Open-domain **48.86**；Single-hop F1 31.43（低于 MemoryOS 35.27）
- 主战场自建医疗数据集 MediLongChat（100K+ tokens/病史），五 LLM 均 Ranking 1/1，
  token 6101 vs MemoryOS 13503 / A-Mem 16680
- 公开仓库仅 6 文件（检索/评测端）；**ART 编码管线、person-memory 生成产物未公开**
- 正文未给 ρ/α/γ/k/N 具体值，附录不在 8 页 PDF 内——复现需自行设定

## 2. 数据关系（已验证）

| 数据集 | 内容 | 与我们的关系 |
|---|---|---|
| MemoryART `qa_clues.json` | D1-D10 共 1540 题（带 person/events/motivations clues） | 与我们 1976 题重叠 1519（99.3%） |
| MemoryART `locomo10.json` | 与官方 locomo10.json **MD5 相同** | 就是我们评测用的同一文件 |
| 官方 `observation` 字段 | LLM 结构化事实 + 源 dia_id（2541 条/10 conv） | 数据集自带分析标注 |

**关键结论**：MemoryART 与我们跑同一 LoCoMo 10 会话；其 `qa_clues` 的 person/events
通道覆盖我们 plan 的 entities 已有 99%（1523/1531 paired 双方都有 person/entity）。

## 3. 公平性确认（重要）

官方 Leaderboard 评测协议：Add 阶段只收原始消息（role/content/timestamp），Search
阶段收 query——**observation/event_summary/session_summary 字段测试时不可见**。
项目硬约束："绝不按私有测试集身份或参考答案适配"（README）。

⇒ 直接用官方 observation 作检索通道 = 使用测试时不可见标注，**不合规**。
observation 仅作**离线诊断上限**；合规路径 = 本地 LLM 从原始消息自生成等价语义层。

## 4. A 方向诊断：observation 层理论可救空间（1976 全量）

| 指标 | 数值 |
|---|---|
| 非 top1 查询总数 | 820 |
| oracle：gold dia 有 observation 覆盖 | 640 (78.0%) |
| 词面代理：obs 事实与问题共享词 | 584 (71.2%) |
| obs BM25 top1 命中（朴素词面） | 129 (15.7%) |
| obs BM25 top3 | 208 (25.4%) |
| obs BM25 top5 | 254 (31.0%) |

**解读**：
- 语义层（若有完美匹配）理论上限 ~78% 非 top1 可被覆盖——空间很大；
- 但 observation 是 LLM 概括文本，**朴素 BM25 词面仅 31% top5**，与我们的
  channel_miss 同理：真正的瓶颈是"问题 → 语义层事实"的匹配，需要嵌入/LLM 语义；
- 因此 B 方向（自生成画像 + 语义检索）必须用 dense/LLM 匹配，不能只靠词面。

## 5. 结论与后续

- MemoryART 对我们的核心启发 = **语义层（事件/画像）+ 证据回溯**，不是其 answer 口径
  或 ART 具体参数（未公开）。
- 我们已有 dense 基础设施（LocalHTTPEmbeddingRetriever / FastEmbed bge / P1
  structured-plan 本地 Qwen 先例）可复用。
- B 方向（person 画像语义层）设计要点：
  1. 本地 LLM 在 Add/离线阶段从原始消息生成 per-person 跨会话画像（identity/
     relations/职业/偏好/事件），每条画像事实带源消息 id；
  2. 检索时 query → 画像事实（dense）→ 回溯源消息进共享池；
  3. default-off；fixed-200 配对门槛 Hit@1 不降 >0.005；全量 paired bootstrap。

产物文件：`memoryart-paper.pdf/.txt`（论文全文）、`MemoryART论文调研报告.md`、
本文件（诊断）。
