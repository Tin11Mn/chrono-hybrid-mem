# 2026-08-25 Git 元数据丢失事故说明

## 状态

两个仓库的本地 git 元数据（`.git`）均已丢失，**工作树全部文件完好**（包括本会话的 P4-0 改动、未跟踪文件）。本地无法恢复历史/分支/tag；远端是否保有历史需用户确认（`Tin11Mn/chrono-hybrid-mem`）。

## 时间线

1. 本会话开始：`chrono-hybrid-mem-p3` 的 `.git` 是指向 `chrono-hybrid-mem/.git/worktrees/chrono-hybrid-mem-p3` 的 gitfile（linked worktree），`git status` / `git log` 正常。
2. 会话中执行 `git stash push -- scripts/evaluate_locomo_retrieval.py`（用于验证基线测试是否受改动影响）后，git 开始报 "not a git repository: (NULL)"；随后确认 `chrono-hybrid-mem/.git` 目录整体消失，p3 的 gitfile 变成悬挂指针。
3. 全盘搜索未发现任何 git 对象库残留（工作区、用户临时目录、`.dsh` 均无）；`git ls-remote` 因无网络（本地代理 127.0.0.1:7897 未运行、直连被断）无法验证远端。

## 影响

- `git status` / `git log` / `git commit` 全部不可用。
- `scripts/check_readme_consistency.py` 依赖 git tag（v0.1.0 SHA `5fd77045c74a5b17876abca30812888587628eaa` 等），故 `tests/test_readme_consistency.py::test_multilingual_readmes_share_facts_links_and_versions` 失败（完整套件 658 passed / 1 failed，唯一失败即此）。
- 完整 p3 测试套件其余 658 项全部通过（含 P4-0 新增 6 项）。

## 未丢失

- p3 与 main 的工作树全部文件。
- P4-0 交付物：`scripts/evaluate_locomo_retrieval.py`（`--include-question-diagnostics` 已实现）、`tests/test_locomo_p4_audit.py`（新建）、`docs/P4_RECALL_RECOVERY_SPEC.md`（新建）。
- 数据/模型资产：`chrono-hybrid-mem/.locomo/locomo10.json`、`.local-models/*.gguf`、`.local-tools/llama-b9637`。

## 恢复建议（待用户确认远端）

- 若远端保有 `agent/p3-evidence-graph`（或等同）分支与 `main`：恢复网络/凭据后，在 `chrono-hybrid-mem` 重新 clone 或 `git fetch` 还原 `.git`，再重新 `git worktree add` p3（当前 p3 工作树文件可先备份保留）。
- 若远端不完整：接受本地历史丢失，在两个目录 `git init` 后以当前工作树为基线重建（P4-0 改动将作为首个提交的一部分）。
- 恢复前不要删除任何工作树文件；`docs/_p4_spec_draft.md`、`docs/_p4_write_probe.md`、`tests/_replace_probe.py`、`docs/_crlf_probe.md` 为本会话探针/草稿残留（工具锁定无法删除），可在恢复后清理。
