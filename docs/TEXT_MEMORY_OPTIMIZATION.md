# 文本记忆优化候选版

基线：`924db24`（已部署运行源码 `d92c63b`）。本候选版的代码和本地复现
与正式平台成绩分开记录；没有新 smoke/full 分数。

## 已实现

- **证据输出**：使用一次结构化选择调用替代一次排序调用。只有 ID、需求覆盖、
  保留/排除关系和状态替换关系均有效且模型声明证据充分时，才返回选中的原文。
  不再自动补回其余候选；多跳或规则题允许超过 12 条必要证据，最终仍不超过 top_k。
- **状态与历史**：选择协议要求按主体、属性、作用域、来源和生效时间判断更新；
  未来计划不等于当前状态，历史/变化查询保留有关旧值。程序验证状态决策的来源 ID、
  当前值必须被保留、被替换值必须被排除。没有删除旧消息，也没有伪造答案。
  语义判断仍由模型负责，这不是一个已经验证正确的通用状态推理器。
- **规则、经验和全局覆盖**：新规划器区分 rule/learning/summary/governance。
  按会话混合词面命中、不同阶段的原文代表，从最多 24 个候选中保留最多 8 个覆盖槽位；
  总模型池仍不超过 30，保留至少 18 个原候选及已有的必要通道保留项。
  总览可在无关键词重合时跨会话取证。最多读取 40 个相关会话的代表，不能视为完整
  全历史压缩或保证每一阶段都被覆盖。
- **流式写入**：跨 Add 批次回填相邻原文索引，限制在同用户同会话。
  老数据库首次启动进行一次上下文索引修复；原消息内容和 ID 保留。
- **诊断与回退**：可选私有 JSONL 只保存 ID、覆盖组和状态，去除查询、原文、模型解释。
  选择失败、空选、引用越界或证据需求未覆盖时保留原词面检索结果，不增加一次模型重试。
  该回退保持旧行为，不承诺模型失败时仍具有最小披露效果。top_k 导致覆盖截断时记录标记。
- **会话事实层兼容修复**：旧离线写入/语义读取接口改用现有来源映射表，检查用户和会话，
  保留多来源映射、清理替换后的旧来源。不额外启用 SF，不增加 Add 的模型调用。

## 启用与回退

新选择及覆盖策略默认关闭；跨批次索引修复属于本版的正常索引维护。
配置模板为仓库根目录 `.env.text-optimized.example`。在实际运行环境中设置：

```bash
MEMORY_STRUCTURED_QUERY_PLAN=true
MEMORY_EVIDENCE_NEED_RETRIEVAL=true
MEMORY_NEED_SELECT_BY_BM25=true
MEMORY_EVIDENCE_SELECTION=true
MEMORY_COVERAGE_RETRIEVAL=true
```

模型密钥、服务密钥、OPENAI_BASE_URL 和数据库路径沿用服务端的实际配置。
示例文件不会被应用自动加载；必须通过 systemd EnvironmentFile、容器环境或
进程环境传入。保留 `MEMORY_REQUIRE_MODEL=true`。

若需要诊断，设置 `MEMORY_RETRIEVAL_TRACE_PATH` 为服务账号可写的私有文件。
该文件含内部来源 ID，应设定保留期和轮转，不放在可公开访问的目录。
不需要诊断时不设置此变量。

选择策略不能与 P5、set-aware、dense fusion 或本地语义/排序分支组合，配置冲突会在启动时报错。
每次查询的调用数量保持一次规划、一次选择；请求和输出 token 数可能变化，费用与延迟须实测。

上线前先备份数据库；离线验证通过不等于正式成绩提升。使用候选源码更新服务后，
重跑相同 smoke，再进行 full，分别核对 A/C/E 的保持情况与 D/G/I/H 的变化。
回退选择策略时将 `MEMORY_EVIDENCE_SELECTION` 和 `MEMORY_COVERAGE_RETRIEVAL`
同时设为 false 并重启，规划器也会恢复旧协议。上下文索引修复仍保留；如需完整重现
旧版索引，使用备份数据库和原版源码。

本次交付只准备本地源码、模板和离线验证材料，未更换服务器版本。

## 离线验证

Windows（复用已有测试环境）：

```powershell
& ../chrono-hybrid-mem-coding/.venv/Scripts/python.exe scripts/test_offline.py -q -m 'not local_research'
```

其他已有 Python 测试环境：

```bash
python scripts/test_offline.py -q -m 'not local_research'
```

测试启动器会清除继承的 OPENAI_/MEMORY_ 运行配置并阻止外网 socket。
SDK 的模拟对象和进程内 TestClient 仍可工作，不需要真实密钥。

合成机制对照可分别运行 `scripts/probe_text_optimization.py --source <基线目录>
--output <基线结果.json>` 和 `--source <本目录> --optimized --output <候选结果.json>`。
该脚本使用固定模型替身，测量输出策略和索引行为，不测量真实模型理解能力。

## 验证边界

尚无真实模型对本选择协议的成功率、充分性判断、费用、延迟或正式平台分数数据。
原文契约意味着一条必要消息内夹带的无关敏感细节不会被字段级脱敏；本次改进主要
减少整条无关消息的连带返回。完整摘要、经验规律推断仍依赖候选覆盖及平台答题模型。
