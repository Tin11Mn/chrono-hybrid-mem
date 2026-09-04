# Git 分支与仓库规则（给后续 AI / 协作者的约定）

本文件定义 ChronoHybridMem 仓库的分支命名、生命周期、合入流程与整洁标准。
**任何后续 AI 或协作者在改动仓库结构前必须先读本文件**，并按第 6 节流程执行。

- 仓库：`Tin11Mn/chrono-hybrid-mem`
- 默认分支：`main`
- 本规则最后更新：2026-08-26（分支整理后）

---

## 1. 当前分支快照（2026-08-26 整理后）

### 远程分支（应只保留这 2 个）

| 分支 | 指向 | 角色 |
|---|---|---|
| `main` | `8f85cf2` | 稳定实现 + 文档 + CI，唯一可发布分支 |
| `research/p3-evidence-graph` | `c83871f` | P4/P5 实验线（证据图谱/召回恢复研究） |

### 本地分支

| 分支 | 指向 | worktree | 说明 |
|---|---|---|---|
| `main` | `8f85cf2` | `chrono-hybrid-mem` | 主工作树 |
| `research/p3-evidence-graph` | `c83871f` | `chrono-hybrid-mem-p3` | 实验工作树 |
| `competition/2026-cycle-2-p0` | `be2cc40` | `chrono-hybrid-mem-p4-release` | 竞赛提交快照（2026 周期 P0） |

### 已删除（本整理操作）

- `release/p4a-bm25`（本地残留，PR #12 已合入 main）
- 远程 `release/p4a-baseline`（PR #11 已合入 main）
- 远程 `agent/p3-evidence-graph`（孤儿分支，内容已被 research 线取代）

---

## 2. 分支命名规范

所有分支一律小写，用 `/` 分层、`-` 连接单词，禁止在分支名使用空格/大写/中文。

| 前缀 | 用途 | 生命周期 |
|---|---|---|
| `main` | 稳定实现，唯一长期分支 | 永久保留 |
| `research/<课题>` | 研究/实验线（如 `research/p3-evidence-graph`） | 实验结束并合入 main 后可删 |
| `competition/<周期>-<阶段>` | 竞赛提交快照 | 提交后**冻结**，仅保留作记录 |
| `release/<版本>` | 从实验线到 main 的发布候选 | PR 合入后立即删除 |
| `agent/<课题>` | 早期临时开发分支（历史遗留命名） | **废弃**，不要再新建 |

规则：
- **禁止**新建 `agent/*` 分支（已弃用命名空间）。
- 新实验一律从 `main` 最新提交切 `research/<课题>`。
- `competition/*` 分支一旦提交给竞赛即冻结：不合并、不 rebase、不 force-push，只可加 tag 标记。

---

## 3. main 分支保护规则

1. **main 只通过 PR 合入**，禁止直接 `git push origin main`（除非是紧急恢复远程损坏的例外，见第 7 节）。
2. 合入前必须通过 CI `Verify / core-verification`。
3. **禁止 force-push 任何已推送分支**（main、research/*、competition/* 皆然）。
4. 本地 main 工作树与远程保持同步：`git pull --ff-only`；本地 main 出现"ahead"通常是 origin ref 过期假象，先 `git fetch` 刷新再判断，禁止凭感觉 push。
5. 新合入 main 的研究结论须有对应文档与评测记录（见 `docs/` 各阶段报告），README 同步按第 5 节执行。

### 合入方式

- 用 GitHub PR（`gh` CLI），squash merge，合入后**立即删除分支**：
  ```powershell
  gh pr create --base main --head <branch> --title "<type>: <subject>" --body "<what/why/results>"
  gh pr merge --squash --delete-branch
  ```

---

## 4. 实验线（research/*）内部规则

1. 每阶段一个独立 commit（或一组），提交信息用 Conventional Commits：
   `<type>(<scope>): <summary>`，如 `feat: P4-A evidence-need retrieval`、`docs(eval): ...`、`test: ...`。
2. **被否决的实验也要提交记录**（标注 reject 原因），保证决策链完整可回溯。
3. 实验结论（指标变化）只写进 `docs/` 与 `findings.md`；**只有被选为主方法并经全量评测的结论才写进 README**。
4. 需要 main 的最新代码时 `git merge main`（在 research 工作树内），不要 rebase 已推送的 research 提交。
5. research 分支内容已稳定并验收后 → 走第 3 节 PR 流程进 main → 分支删除。

---

## 5. README 五语言一致性门禁（合入 main 的硬性要求）

仓库有 **5 份 README**，任何改动必须全部同步：

| 文件 | 语言 | 角色 |
|---|---|---|
| `README.md` | 简体中文 | GitHub 默认展示页 |
| `README.en.md` | English | **canonical 基准** |
| `README.es.md` | Español | 从 en 同步 |
| `README.ja.md` | 日本語 | 从 en 同步 |
| `README.ko.md` | 한국어 | 从 en 同步 |

规则：
1. 数字事实（评测题数、Hit@1/MRR 等指标、commit SHA、版本号）与代码块、命令、链接、导航在 5 份之间必须一致。
2. en 是逐 token 基准；es/ja/ko 必须与 en 逐 token 一致（允许翻译文案，但 inline code 与数字 token 必须一致）。
3. `README.md`（简体中文）因中文全角标点豁免逐 token 数字/inline-code 比较，但**代码块、字面量、链接、导航、事实值仍须与 en 一致**。
4. 任何 README 改动提交前必须运行并通过：
   ```powershell
   python scripts/check_readme_consistency.py
   ```
   以及 `tests/test_readme_consistency.py`（CI 亦强制）。
5. 改数字事实时先改 en，再据此改 es/ja/ko（三者可与 en 同 commit）；README.md 可随后单独同步并豁免逐 token 校验。

---

## 6. 分支清理 / 整理操作流程（本次整理的通用化）

当需要"整理分支"时，按顺序执行并逐项核对：

1. **盘点**：`git branch -a` + `git worktree list`，列出本地/远程/孤儿分支与指向。
2. **判定保留**：
   - 已在 main 或已合并（PR 显示 merged）→ 删除；
   - 内容已并入其它保留分支 → 删除；
   - 无远程、无 tag、内容已被取代 → 删除；
   - `main` / `research/*` / 冻结的 `competition/*` → 保留。
3. **本地删除**：`git branch -d <name>`（未合并需确认后再 `-D`）。
4. **远程删除**：`git push origin --delete <name>`。
5. **刷新 origin refs**：`git fetch origin`（或显式 URL fetch，见第 7 节），核对 `git ls-remote origin`。
6. **记录**：更新本文件第 1 节快照 + 提交。

---

## 7. 本环境特殊注意：沙盒会破坏 git refs（务必知晓）

在本开发环境（Windows 沙盒）下，**每次 git 写操作（fetch/push/commit/merge/add/checkout）后，`.git/refs/heads/*`、`.git/refs/remotes/origin/*` 及 worktree 的 `.git` gitfile 可能被外部机制删除**，导致：

- `git status` 报 not a git repository / ref 缺失；
- origin ref 过期（显示本地 ahead，实为假象）；
- 严重时 commit 对象丢失（git merge 触发时）。

恢复标准流程（完整 hash 必需，短 hash 会报 broken）：

1. 先判断对象是否还在：
   ```powershell
   git cat-file -t <full-sha>   # 存在则输出 commit
   ```
2. 对象若丢（仅 merge 场景出现）：从远程找回对象但**不写 refs**：
   ```powershell
   git fetch https://github.com/Tin11Mn/chrono-hybrid-mem.git <branch>
   ```
   （fetch 会同时写 refs，随后仍可能被删，需继续手写）
3. 手写 refs（PowerShell 写文件，内容为完整 40 位 hash + 换行）：
   ```powershell
   $main = "C:\Users\15431\Desktop\agent-memory-leaderboard-main\chrono-hybrid-mem"
   [System.IO.File]::WriteAllText("$main\.git\refs\heads\research\p3-evidence-graph", "<full-sha>`n")
   [System.IO.File]::WriteAllText("$main\.git\refs\remotes\origin\main", "<full-sha>`n")
   # ... 每个需要的 ref 一个文件，目录不存在先 New-Item -Force
   ```
4. worktree 元数据若丢：重建（先确认目录未被占用）：
   ```powershell
   git worktree add --force C:/Users/15431/Desktop/agent-memory-leaderboard-main/chrono-hybrid-mem-p3 research/p3-evidence-graph
   git worktree unlock chrono-hybrid-mem-p3   # 若报 locked
   ```
5. 推送/拉取用**显式 URL**（避免 remote 配置丢失）：
   ```powershell
   git push https://github.com/Tin11Mn/chrono-hybrid-mem.git <local-branch>:<remote-branch>
   git ls-remote https://github.com/Tin11Mn/chrono-hybrid-mem.git   # 核对真实远程状态
   ```
6. pwsh 命令可能"超时无输出但实际完成"：用后台 job + `git ls-remote` / reflog 验证真实状态，不要因超时重复执行破坏性命令。

完整事故记录见 [GIT_METADATA_LOSS_2026-08-25.md](GIT_METADATA_LOSS_2026-08-25.md)。

---

## 8. 提交与 PR 整洁标准

1. **提交信息**：Conventional Commits（`feat:`/`fix:`/`docs:`/`docs(eval):`/`test:`/`refactor:`/`chore:`），英文写 summary，正文可用中文详述。
2. 一个 commit 一个逻辑变更；不要混入无关格式化。
3. 被否决实验：单独 commit 记录结论与原因（如 `8a45da3`、`e46e8d6` 先例）。
4. **PR 标题**：`<type>(<scope>): <subject>`，正文写清 what / why / 评测结果（含题数与指标）。
5. PR 合并用 squash，历史保持线性可读；合入后删分支（`--delete-branch`）。
6. 禁止在 main 上直接 commit；禁止把 `.locomo/`、模型权重、评测缓存、`.venv*` 等提交进 git（已 gitignore）。

---

## 9. 长期保留 vs 用完即删 速查

| 对象 | 处置 |
|---|---|
| `main` | 永久保留 |
| `research/*`（有活跃实验） | 保留，实验完结合入后删 |
| `competition/*` | 提交后冻结保留 |
| `release/*` | PR 合入后立即删（本地 + 远程） |
| `agent/*` | 已废弃；现存引用清理后删，不再新建 |
| 已合并 PR 的分支 | 删 |
| tag（v0.x / research-v0.x / research-p1-*） | 永久保留，禁止移动/删除 |
