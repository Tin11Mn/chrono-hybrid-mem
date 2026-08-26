# P4 共享 sidecar 保留池（--sidecar-shared-quota）实验记录

- 日期：2026-08-26
- 问题：P4-A（need quota 2）+ P4-C（bridge quota 2）组合时，各自固定配额把
  P1 基础预算压到 `30 - 2 - 2 = 26`，fixed-200 上 Hit@1 跌破 P1 基线
  （combo 0.560 < off 0.565）。双配额非加性。
- 修复：`--sidecar-shared-quota N`（默认 0 = 关闭，保持各自独立配额）；
  >0 时 need 与 bridge 候选共享**一个**总保留池，P1 基础预算最多被压缩
  N 个槽位，与激活的 sidecar 组件数量无关。need 优先、bridge 补位。
- 产物：`.locomo/p4ac-shared{1,2}-fixed200.json`；分析脚本
  `.locomo/_compare_fixed200.py`、`.locomo/_paired_shared2.py`。

## fixed-200 全配置对比

| run | Hit@1 | Hit@3 | Hit@10 | MRR | evR@10 |
|---|---|---|---|---|---|
| P1 off | 0.5650 | 0.6850 | 0.7300 | 0.6292 | 0.5914 |
| P4-A q2（基线） | 0.5750 | 0.7100 | 0.7400 | 0.6411 | 0.5992 |
| P4-C bridge alone | 0.5650 | 0.6850 | 0.7400 | 0.6296 | 0.5953 |
| combo（无共享，旧） | 0.5600 | 0.7050 | 0.7350 | 0.6326 | 0.5992 |
| **combo shared=2** | **0.5700** | 0.7000 | **0.7450** | 0.6377 | **0.6031** |
| combo shared=1 | 0.5650 | 0.6900 | 0.7300 | 0.6291 | 0.5875 |

## 结论

**`--sidecar-shared-quota 2` 是 combo 的最优共享配置（CONDITIONAL，默认 off）。**

- shared=2 修复了旧 combo 的 Hit@1 跌破基线问题：0.570 > off 0.565；
- Hit@10（0.745）与 evR@10（0.6031）为全部配置最高——bridge 召回增益完整保留；
- bridge 候选进池题数 198 → 118（只在 need 未占满共享配额时补位）；
- paired（vs P1 off）：NewGold 22 / OffMiss→Pool 4 / →Top10 2 / →Top1 1 /
  Hit1Flips +1（旧 combo 为 -1）；
- 代价：相对 P4-A alone，Hit@1 -0.005（恰在"不降 >0.005"阈值边界），
  Hit@10 +0.005、evR@10 +0.004。

**shared=1 被否决**：need 被削到 1 且 bridge 基本进不来，Hit@10 回退到
0.730（= off），evR@10 0.5875 甚至低于 P1 off——共享池过紧会同时削弱
need 与 bridge 两个组件的收益，且没有置换补偿。

## 实现

- `app/storage.py`：`SIDECAR_SHARED_QUOTA_DEFAULT = 0`；构造参数
  `sidecar_shared_quota`（校验 0..30，>0 需要 need 或 bridge 激活）；
  pool 预算分支：共享时 `base_budget = 30 - special_quota - shared`，
  need/bridge 共用 `sidecar_reserved` 计数器（need 优先、bridge 补位）。
- `scripts/evaluate_locomo_retrieval.py`：`--sidecar-shared-quota` CLI
  （校验 + 传递）；diagnostics 新增 `reserved_need_ids` / `reserved_bridge_ids`。
- 测试：`tests/test_p4c_bridge.py` 新增共享池 3 例（总保留 ≤ shared、
  shared 时 P1 槽位 ≥ 独立配额、>0 需要 sidecar 组件）。

## 复现命令

```powershell
python -m scripts.evaluate_locomo_retrieval `
  --dataset '..\chrono-hybrid-mem\.locomo\locomo10.json' `
  --local-search-model-url 'http://127.0.0.1:8081/v1' --local-search-model-name local `
  --structured-query-plan --max-questions 200 --include-question-diagnostics `
  --evidence-need-retrieval --evidence-need-quota 2 `
  --bridge-retrieval --bridge-max-terms 3 --bridge-quota 2 `
  --sidecar-shared-quota 2 `
  --output '.locomo\p4ac-shared2-fixed200.json'
```
