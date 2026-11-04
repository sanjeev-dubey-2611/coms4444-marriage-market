# Player3 Discrete Distribution Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将 Player3 的逐次约会均值统计改为按对象去重的逐维离散分布，并用分布计算预期匹配扣分。

**Architecture:** 每个玩家实例拥有独立的 `PartnerDistribution`，分别维护观察到的 profile 和 stated ideal 的逐维直方图。每个对象最多贡献一份当前记录；通过概率加权计算未知或不可信属性的预期扣分，交给现有续约和求婚规则使用。

**Tech Stack:** Python 3.12+、标准库、pytest、ruff；沿用现有模拟器与评估脚本。

**Spec:** 本次对话已明确的要求及下方“设计范围与建议默认值”。实施细节是本计划的建议，尚未实施。

**Branch:** `feature/player3-discrete-distribution`

**Baseline:** `630fc65`（创建分支时的当前 Player3；`Player3V1` 不是这次修改的直接基线）。

## Global Constraints

- 不同玩家实例之间不能共享信息，包括同组的多个 Player3。
- 统计只使用该实例通过合法玩家接口得到的信息，不读取引擎真值、其他玩家实例、运行日志或隐藏分布。
- 可变统计全部放在 `self`，不得使用共享类变量、模块级可变缓存或跨实例文件通信。
- 按 `CONTRIBUTING.md`，提交文件限定在 `players/player_3/`；本计划、新增测试也放在此目录。
- 不增加运行时第三方依赖；支持任意合法维度数、单调维度数及 0–10 属性值。
- 玩家观察到的是展示信息，统计结果不能宣称是无偏的真实总体分布。

## 设计范围与建议默认值

1. **按人更新。** 每个 `partner_id` 保存最新 profile 和最近一次当前有效的 stated ideal。再次观察该人时先移除其旧贡献，再加入新贡献；不是累计多次约会的票数。
2. **两组逐维统计。** profile 与 stated ideal 分开维护 `dims × 11` 计数。虽然生成规则可能使用同一底层分布，两类展示信息的谎言和选择偏差不同，第一版不把它们混合。
3. **平滑。** 使用一个虚拟样本总量的均匀先验：每格伪计数 `1/11`。若某类有 `n` 个对象，`P_j(v) = (count[j][v] + 1/11) / (n + 1)`；无人时均匀分布，不将未知值固定为 5。该先验是建议默认值，不代表知道生成分布。
4. **评估时排除本人。** 对一个对象打分时，从总体先验中排除他已有的记录；先检查旧历史、计算收益，再更新本次观察，避免同一条信息既作为证据又充当先验。
5. **可疑 ideal。** 若 stated ideal 完全镜像我们的展示，该次 ideal 不进入总体 ideal 统计，并移除该人的旧 ideal 贡献；profile 仍按当前观察更新。其他说谎和选择偏差第一版只通过已有信任信号缓解，不宣称解决。
6. **预期扣分。** 对未知属性按 11 个值求和。对不完全可信的已知展示，使用 `trust × point_mass(observed) + (1-trust) × prior`。镜像 ideal 使用总体 ideal 先验。不要先求均值再平方，否则离散分布的价值会丢失。
7. **决策接入。** 新人的广告只提供一个自身属性和一个要求；其余 profile 与 ideal 维度分别使用相应先验。两种广告维度即使相同，也属于评分的不同方向，分别计入。约会后收益仍为 `B - expected_own_decrement - expected_partner_decrement/2`。
8. **关联修复。** 新路径中的历史比较必须发生在记录更新之前，使首次广告不一致、后续展示变化检查生效。求婚次数、市场门槛方向等独立策略调整不纳入本次实验。
9. **开局广告。** 广告只发布一次，尚未产生约会样本；本次保留现有诚实展示及开局选维度规则，学习到的分布用于后续筛选和评分。

## Review Focus

- 同一个人重复出现：分布和有效对象数不能仅因重复次数而变化；由 Task 2 测试。
- 重复对象修改展示或 ideal 变得可疑：旧格必须扣回，profile/ideal 分母分别正确；由 Task 2 测试。
- 空样本、排除唯一对象、不同维度：每行仍非负且和为 1，无除零；由 Task 2 测试。
- 平均值相同但分布不同：评分必须保留平方损失和单调截断的差异；由 Task 3 测试。
- 不同实例或新一局：更新 A 不得影响 B，重新初始化不得带入旧记录；由 Task 2 和 Task 4 测试。

## 文件分工

- Create: `players/player_3/distribution.py` — 按人维护直方图、平滑 PMF、点值混合和预期扣分。
- Modify: `players/player_3/player.py` — 接入统计和评分；公共玩家接口保持兼容。
- Create: `players/player_3/tests/test_distribution.py` — 去重、更新、平滑、排除本人、隔离与数学性质。
- Create: `players/player_3/tests/test_distribution_player.py` — 完整玩家回调路径和评分接入。
- Create: `players/player_3/tests/test_distribution_evaluate.py` — 种子区间、变体筛选及报告配对。
- Modify: `players/player_3/tools/evaluate.py` — 新增独立种子区间和筛选评估变体的参数。
- Create: `players/player_3/tools/compare_reports.py` — 对相同条件、种子的结果计算配对差异。
- Local artifacts: `results/player_3_distribution/` — 已被忽略的本地评估输出，不加入提交。

## Task 1: 保存可比较的现有基线（建议 20–30 分钟）

**Files:** `tools/evaluate.py`、`tools/compare_reports.py`、`tests/test_distribution_evaluate.py`，以上路径均相对 `players/player_3/`。

**Interfaces:** `evaluate.py` 增加 `--variants`（默认沿用现有选择）、`--seed-start`（默认 1）；`--seeds N` 表示从 seed-start 起连续 N 个种子。`compare_reports.py BEFORE AFTER --before-variant default --after-variant default` 先各自筛选变体，再按 `(label, seed)` 对齐报告的 `raw` 项，对缺失、重复、不一致的条件报错；可通过 after-variant 选择 unique_mean。

- [x] 在 `tests/test_distribution_evaluate.py` 先验证种子边界：start=101、count=3 产生 `[101, 102, 103]`，只选 default 时不执行其他变体。使用小型人工报告验证配对差值、不同变体名称之间的比较，以及缺失条件报错，再实现对应 CLI 支持。
- [x] 运行现有测试作为基线；记录 commit、参数、fault 数。若有既有失败，记录并单独判断，不能将其掩盖在新功能中。
- [x] 修改算法前保存两份 baseline：开发种子 1–10，以及保留种子 101–130；固定 arena、人数、轮数、维度和分布。

```powershell
uv run pytest tests/test_player_3.py tests/test_reproducibility.py
uv run python players/player_3/tools/evaluate.py --quick --variants default --seed-start 1 --seeds 10 --workers 2 --out results/player_3_distribution/baseline_dev.json
uv run python players/player_3/tools/evaluate.py --quick --variants default --seed-start 101 --seeds 30 --workers 2 --out results/player_3_distribution/baseline_holdout.json
```

- [x] 独立提交评估支持；本地 JSON 输出不提交。

**Task 1 执行记录（2026-11-03）：** 开发集 120 局、保留集 360 局，各 12 个 arena；全部 faults 与 Player3 faults 均为 0。参数为 N=20、t=100、B=100、d=5、m=2、mixed、list_multiple=4、timeout=0；报告记录完整阵容顺序、种子、源码哈希和运行环境。原策略与模拟器未改动，12 个 seed=1 对局独立复跑逐项一致。

测试：新增评估测试 26 项通过；连同现有 Player3/可重复性测试共 33 项通过。全量测试 274 passed、1 skipped，仍有修改前已独立复现的两项失败：`tests/test_gui_worker.py::test_worker_preserves_player_timeouts`（Windows 无 SIGALRM）及 `tests/test_registry.py::test_unimportable_group_is_reported_and_skipped`（临时发现路径仍导入真实 Player8）。相关源码不属于本次修改范围。

独立审查指出比较器应核对固定实验代码，已用先失败后通过的回归测试修复：拒绝模拟器、对手、运行环境及依赖锁版本变化，允许预期的 Player3 代码变化。本地汇总、哈希清单、JUnit 记录见 `results/player_3_distribution/README.md`、`manifest.json`、`verification.xml`；结果文件不提交。Task 2–5 尚未执行。

## Task 2: 实现按人去重的离散统计（建议 45–60 分钟）

**Files:** `distribution.py`、`tests/test_distribution.py`。

**Interfaces:** `PMF = tuple[float, ...]`；`PartnerDistribution(dims: int)`；`replace(partner_id: int, profile: Profile, ideal: Profile | None) -> None`；`pmf(kind: Literal['profile', 'ideal'], dim: int, *, exclude_partner: int | None = None) -> PMF`。`ideal=None` 表示该人当前不贡献 ideal 样本。

- [ ] 先写行为测试并运行确认失败：同一对象同样信息重复 10 次，PMF 不变；其属性由 2 改为 8，2 的旧贡献消失且只新增一个 8；加入第二个人后两人等权。
- [ ] 测试空样本每格为 `1/11`，有一个值为 8 的对象后 `P(8)=6/11`、其他格 `1/22`；排除该对象后恢复均匀。更新 profile 后删除 ideal 时，两种分母仍各自正确。
- [ ] 测试两实例互不影响；单独统计不同维度；所有 PMF 长度 11、非负、归一化。
- [ ] 实现每次替换 O(d) 的旧贡献移除与新贡献加入；先验排除也通过扣除本人记录实现，不扫描全部历史。
- [ ] 运行 `uv run pytest players/player_3/tests/test_distribution.py`，全部通过后独立提交。

## Task 3: 实现预期扣分（建议 30–45 分钟）

**Files:** `distribution.py`、`tests/test_distribution.py`。

**Interfaces:** `mix_observation(prior: PMF, value: int, trust: float) -> PMF`；`expected_actual_gap(want: int, actual: PMF, monotonic: bool) -> float`；`expected_desired_gap(desired: PMF, have: int, monotonic: bool) -> float`。

- [ ] 先写并运行失败测试：想要 5、实际恒为 5 时扣分 0；实际各半为 0 和 10 时，非单调扣分 25，单调扣分 12.5。两种实际分布均值同为 5，结果必须不同。
- [ ] 测试实际恒为 8、想要 5 时，单调扣分 0，非单调扣分 9；测试对方要求随机时的参数方向，防止交换 want/have。
- [ ] 测试 trust=1 得到观测点质量，trust=0 得到原先验，中间值保持归一化。
- [ ] 实现每个维度 11 项的求和。对所有输入均为点质量的情况，与现有 `decrement`、`happiness` 公式核对一致。
- [ ] 运行 Task 2 的测试文件，全部通过后独立提交。

## Task 4: 接入 Player3 的筛选与决策（建议 60–90 分钟）

**Files:** `player.py`、`tests/test_distribution_player.py`、`tools/evaluate.py`。

**Interfaces:** 新的内部入口 `_estimate_partner(date: DateContext, observed: Profile, stated_ideal: Profile) -> float` 负责顺序正确的信任检查、分布评分与记录更新，同时更新该人的 `state.trust`；`_ad_cost(ad: Advertisement) -> float` 为新人估计双向总扣分。`decide()` 继续消费收益估计；置信下界使用更新后的 trust，并且 `_seen` 每次约会只追加一次，保留当前续约和求婚规则。

- [ ] 先写完整 `project → decide → on_outcome` 路径的失败测试；不要只单测信任 helper。覆盖首次广告不一致、第二次展示变化、镜像 ideal 不进入统计、重复约会不增加对象权重。
- [ ] 在打分前取得排除当前对象的 PMF；先检查旧历史，再计算收益，最后替换直方图贡献和追加审计记录。
- [ ] 替换默认 Player3 的均值收缩评分。完全可信的已知 profile/ideal 使用观测点质量；不确定部分混合先验后直接求预期扣分。
- [ ] 新人评分分别遍历所有 profile 与 ideal 维度；广告揭示的部分用展示值，未揭示部分用先验。验证 own_dim 与 seek_dim 相同、不同两种情况。
- [ ] 验证分布已实际影响估计和候选入选：构造均值相同但风险不同的已知历史，保证得到不同的收益估计；构造有限名单避免“仅排序变了但入选集合相同”的假阳性。
- [ ] 增加 `Player3UniqueMean` 对照变体：使用相同去重、信任顺序和候选评分流程，但将不确定 PMF 压缩到其未取整均值后计算扣分。这样可以分离按人去重与完整分布计算的影响；该变体仅供评估。
- [ ] 原有 `Player3V1` 保留原有行为；确认其他显式消融开关仍按含义工作。运行新测试、原有 Player3 测试、可重复性测试，通过后独立提交。

## Task 5: 对照评估与收尾（建议 45–60 分钟，取决于模拟耗时）

- [ ] 在开发种子 1–10 比较当前 baseline、unique_mean、新 default。按 arena 汇总正幸福值比例、平均幸福值、结婚率与 faults；不只挑表现好的对局。
- [ ] 确定第一版参数后再运行 101–130 保留种子，不根据该结果反复调参。旧 baseline 与新 default 使用相同种子、阵容顺序和配置做配对比较。
- [ ] 保留每个种子的配对差值，报告均值和不确定性；区间包含 0 时记为证据不足。若在保留结果上继续选择参数，需要另设新的保留种子。
- [ ] 再覆盖人数变化、25/100 回合、uniform/normal/mixed、不同合法维度配置；运行前先用旧 commit 的隔离基线生成完全相同配置的结果，否则这部分只能算新版本健壮性检查，不能声称性能提升。
- [ ] 完成检查：

```powershell
uv run ruff format --check players/player_3
uv run ruff check players/player_3
uv run pytest
git diff --check
git status --short
```

- [ ] 确认无越界输入读取、跨实例共享或本地结果误提交；整理本次各项实验的实际结果与限制。

## 今日完成标准

- 每个对象至多一份当前统计贡献，重复约会不增重。
- 每维维护 0–10 的合法 PMF，并真实参与预期扣分和候选筛选。
- 数据完全属于单个玩家实例，新一局重新初始化。
- 行为测试和项目检查通过，评估无新增玩家 faults。
- 留下可复现的旧版、按人均值版、离散分布版比较；性能改善由实验判断，不预设必然更强。

预计投入约 3.5–5 小时，模拟耗时另计。优先完成正确性、接入和基础对照，再扩展参数覆盖。
