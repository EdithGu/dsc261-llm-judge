# 实验管道设计：When Should an LLM Judge Abstain?

> DSC 261 项目实验管道计划（v3，定稿）
> 依据：`DSC261_Project_Proposal.pdf`
> 范围：只覆盖实验管道。数据构建由数据组（M1、M2）负责，本文只定义与数据组的接口。
> English version: [`experiment_pipeline_plan_en.md`](experiment_pipeline_plan_en.md)

---

## 目录

1. [研究问题与实验总览](#1-研究问题与实验总览)
2. [已确定的关键决策](#2-已确定的关键决策)
3. [数据接口（面向数据组）](#3-数据接口面向数据组)
4. [可替换架构：数据集与模型](#4-可替换架构数据集与模型)
5. [管道各模块](#5-管道各模块)
6. [Pilot：筛选与继续/更换的关卡](#6-pilot筛选与继续更换的关卡)
7. [统计分析与预注册](#7-统计分析与预注册)
8. [代码结构](#8-代码结构)
9. [交付范围与交接流程](#9-交付范围与交接流程)

---

## 1. 研究问题与实验总览

**主要问题**：在相同 coverage 下，基于置信度的弃权能否比随机弃权更有效地减少 judge 的错误？

**次要问题**：
- thinking 和显式 rubric 能否提高准确率和选择性表现？
- 这些结论在数学和医学两个领域是否一致？

### 1.1 实验矩阵

**主 judge：Qwen3.8-27B**

| Run | Thinking | Rubric | 输出空间 | Domain |
|---|---|---|---|---|
| E1 最小基线 | Off | No | Correct/Incorrect + confidence | GSM8K, MedQA |
| E2 仅 thinking | On | No | 同上 | 两者 |
| E3 仅 rubric | Off | Yes | 同上 | 两者 |
| E4 thinking + rubric | On | Yes | 同上 | 两者 |
| E5 直接弃权 | Off | No | Correct/Incorrect/Abstain + confidence | 两者 |

- 共 5 个 run × 2 个 domain = 10 个推理批次。每个批次都跑完全部 calib 和 test case（每个 domain 200 条）。
- E1–E4 构成 2×2 因子设计，可以分析 thinking × rubric 的交互效应。
- **可选复现：DeepSeek-V4-Flash-0731**，只跑 E1、E3、E5。推理模式固定，同一个 runner 换后端 adapter 即可。

### 1.2 整体流程

```
冻结的 cases ──► [A] 配置/Prompt 注册 ──► [B] 推理运行器 ──► raw outputs（缓存）
                                                         │
                                                         ▼
                                              [C] 解析与有效性分类
                                                         │
                         ┌───────────────────────────────┼─────────────────────┐
                         ▼                               ▼                     ▼
              [D] 阈值校准（calib 60）          [F] 指标计算（test 140）  [H] 误差分析
                         │ thresholds.json（冻结）        ▲
                         ▼                               │
              [E] 策略应用（test）────────────────────────┘
                 无弃权 / 置信度 / 随机 / 直接弃权
                                                         ▼
                                              [G] 统计推断（配对 bootstrap）
                                                         ▼
                                              图表 + 报告 + 复现包
```

---

## 2. 已确定的关键决策

| # | 决策 | 实现细节 |
|---|---|---|
| D1 | **解码方式**：thinking off 用 temperature 0；thinking on 用官方推荐采样参数 + 固定 seed；每个 case 只采样一次 | 两套解码参数都写进 `models.yaml`。pilot 里做**复现性检查**：同一批 case 跑两遍，verdict 一致率目标 ≥ 98%，confidence 的差异也报告。vLLM 即使 temperature 0，batch 大小不同也可能导致结果不一致，所以正式运行要固定 batch 大小和 vLLM 版本，并写进 run manifest。 |
| D2 | **无效输出**：主分析里算作“未保留、也不是弃权” | coverage = 保留的有效判断 ÷ **全部 case**，所以能达到的最高 coverage 是 1 减无效率。如果某个目标 coverage 超过这个上限，这一档标为“不可达”并如实报告，不降低要求。敏感性分析把无效输出当作 **confidence = 0 的错误判断**，排序时放在最后。 |
| D3 | **增加 AURC** 作为曲线汇总指标 | 主分析的曲线只画到能达到的最高 coverage，AURC 在这个区间上计算。敏感性分析按 D2 的规则补全到 100% coverage，再算一个 AURC。两个版本都报告。 |
| D4 | **主要终点**：75% coverage 下，置信度弃权 vs 随机弃权 | 覆盖全部 4 个配置 × 2 个 domain，共 8 个检验，在这 8 个之间做 Holm 校正，统计量是 selective risk 差的配对 bootstrap CI。其余比较都作为次要比较，见第 7 节。 |
| D5 | **Confidence 来源**：只用模型口头报告的值 | 后端接口预留一个可选的 `logprobs` 字段，默认关闭。以后如果做 logprob 的探索分析，结果放在单独的 `exploratory/` 目录下，不进入主表和主结论。 |
| D6 | **judge 错误率下限门槛**：10% | 在 ≥100 题的独立 probe 样本上估计，详见第 6 节。 |
| D7 | **pilot 由数据组执行** | smoke test 通过后，把骨架交给数据组，由他们自己做 pilot 和备选数据集、模型的筛选。 |

---

## 3. 数据接口（面向数据组）

### 3.1 通用 Case Schema

Schema 不绑定具体数据集，而是按**任务类型**描述，换数据集时不用改 schema。每个 domain 交一份 `cases_{dataset_id}.jsonl`，每行一个 case：

| 字段 | 类型 | 说明 | 给不给 judge 看 |
|---|---|---|---|
| `case_id` | str | 唯一 ID | 否 |
| `question_id` | str | 原始题目 ID，用于 bootstrap 和查重 | 否 |
| `dataset_id` | str | 例如 `gsm8k`、`medqa` | 否 |
| `split` | enum | `pilot` / `probe` / `calib` / `test` | 否 |
| `task_type` | enum | `numeric` / `multiple_choice`，以后可扩展 `short_answer` | 否 |
| `question` | str | 题干 | **是** |
| `options` | dict \| null | 选择题的选项 | **是** |
| `candidate_answer` | str | 待判断的答案 | **是** |
| `reference_answer` | str | 参考答案 | 否 |
| `label` | bool | candidate 是否正确（ground truth） | 否 |
| `candidate_type` | str | `correct` 或错误类型，例如 `omitted_step`、`arithmetic`、`distractor` | 否 |
| `meta` | dict | 自由字段：构造记录、难度、科目等 | 否 |

每份数据另附一份 `manifest.json`，内容包括文件 SHA256、抽样 seed 和各 split 的计数。管道启动时先校验哈希，对不上就拒绝运行。

数据组只需要写一个 **adapter**（`adapters/<dataset>.py`），把原始数据转成上面的格式，不用碰管道的其他部分。

### 3.2 给数据组用的命令（都不需要 GPU）

```
validate-data  --data cases.jsonl
smoke          --data cases.jsonl --n 20 --backend mock
probe          --data cases.jsonl --n 100 --backend api --model <便宜模型>
pilot-report   --run outputs/raw/<dataset>/<model>/<config_hash>/
```

| 命令 | 作用 |
|---|---|
| `validate-data` | 静态检查，几秒钟出结果。见下方检查项列表。 |
| `smoke` | 用 **mock 后端**（按规则或随机生成合法输出）把少量数据走完整条管道：推理、解析、校准、策略、指标、出图。它验证的是数据能顺利跑通，不评估效果。 |
| `probe` | 用便宜的 API 模型、小模型或真实模型跑少量样本，输出**难度报告**：judge 在正确和错误 candidate 上分别的错误率、按 `candidate_type` 拆开的错误率、confidence 分布。数据组可以借此自己判断构造的错误太容易还是太难。 |
| `pilot-report` | 读取 pilot 或 probe 的输出，自动生成**门槛检查表**（第 6.1 节），每项指标标通过、边界或不通过，并附上数值和 CI。 |

`validate-data` 的检查项：
- schema 和类型；
- 各 split 的数量，以及正确和错误 candidate 是否大致各半；
- 同一题是否跨了 split；
- 是否有重复题；
- 给 judge 的字段里是否泄露了 label；
- 数值题的答案能否解析成数字；
- 选择题的 candidate 是否在 options 里；
- **probe 题与正式 calib、test 题是否不重叠**。

### 3.3 仓库自带的样例与文档

- `tests/fixtures/`：GSM8K 和 MedQA 各 10 条格式正确的样例，另有一组故意写错的样例，用来展示 `validate-data` 的报错。
- `docs/adapter_guide.md`：adapter 写法说明。
- `docs/pilot_guide.md`：pilot 操作手册，见第 9.2 节。

---

## 4. 可替换架构：数据集与模型

### 4.1 注册表

**`registry/datasets.yaml`**：每个数据集登记以下信息：
- adapter 路径；
- `task_type`；
- domain 标签（math / medical）；
- 对应的 rubric 片段。

**`registry/models.yaml`**：每个模型登记以下信息：
- checkpoint 和 backend 类型；
- 两套解码参数，按 thinking 开或关区分（见 D1）；
- 上下文长度和 `max_tokens`；
- **thinking 能力描述**：支不支持，以及怎么切换（chat template 参数、system prompt，或者需要换一个 checkpoint）。

如果新模型不支持 thinking 开关，E2 和 E4 自动标为 `n/a`，报告里如实写明，管道不会报错。

### 4.2 统一的模型后端接口

```
generate(messages, params) → {
    text, reasoning, usage, finish_reason, latency,
    logprobs   # 可选，默认关闭（见 D5）
}
```

| 后端 | 用途 |
|---|---|
| `vllm` | OpenAI 兼容接口，用于主实验 |
| `api` | 托管 API（OpenAI 兼容），用于 DeepSeek 复现和数据组的 `probe` |
| `mock` | 用于单元测试和 `smoke`，不需要 GPU |

换模型只需要在 YAML 里改配置，runner 代码不用动。

### 4.3 Prompt 按任务类型组织

- 骨架按 `task_type` × {minimal, rubric} × {binary, ternary} 组合，放在 `prompts/skeletons/`。
- domain 相关的 rubric 内容作为插槽填进去，放在 `prompts/rubrics/`：
  - 数学：数值等价、必要的单位；
  - 医学：和题干、选项是否一致。
- 换成同类型的数据集时，只需要补一个 rubric 片段。

### 4.4 输出隔离

- 输出路径为 `outputs/raw/{dataset_id}/{model_id}/{config_hash}/{case_id}.json`。
- 不同组合的结果互不覆盖，所有组合共用同一套分析脚本。

---

## 5. 管道各模块

### [A] 配置与 Prompt 注册

- 每个实验配置（E1–E5）一个 YAML 文件，写明 thinking、rubric、输出模式（binary / ternary）。配置与数据集、模型解耦。
- 所有模板统一要求输出严格的 JSON：`{"verdict": "...", "confidence": 0.xx}`。
- Confidence 的含义必须在 prompt 里写明：**它是 judge 对“自己这个 verdict 正确”的概率，不是 candidate 正确的概率。**
- 每个配置计算 `config_hash`（YAML 加模板全文），写进每条输出。

### [B] 推理运行器

- 用 vLLM 起 OpenAI 兼容接口，thinking 通过模型注册表里的方式切换。
- 每个 case 只采样一次，解码参数按 D1 设置。
- 缓存键是 `(case_id, config_hash)`。中断后可以续跑，已有结果不会被重跑覆盖。
- 每条记录保存：原始输出、reasoning trace、prompt/completion/reasoning 的 token 数、延迟、`finish_reason`、时间戳、模型版本。
- **重试**：只对执行失败（网络、OOM）重试，最多 N 次，并记录在案。格式错误不重试，否则就是在挑结果。
- **锁定检查**：如果 prompt、parser 或解码参数的哈希和冻结版本不一致，正式运行的 `run` 命令直接拒绝执行。`probe` 和 `screen` 不受这个限制，pilot 阶段可以自由迭代。

### [C] 解析与有效性分类

| 状态 | 判定 |
|---|---|
| `valid` | JSON 能解析，verdict 合法，confidence ∈ [0,1] |
| `invalid_format` | 解析失败或 verdict 非法 |
| `invalid_conf` | confidence 缺失或越界 |
| `truncated` | `finish_reason == length`（thinking 下要重点看） |
| `exec_failure` | 重试后仍失败 |
| `abstain` | 只在 E5 出现，是有意的弃权 |

- 解析规则在 pilot 后冻结，并配一套单元测试。测试用 pilot 输出里的真实样例。
- 无效输出的处理按 D2。

### [D] 阈值校准（只用 calib，每个 model × config × domain 60 条）

1. 只对有效判断按 confidence 从高到低排序。**并列时**用标签无关的确定性规则打破：按 `hash(case_id + 固定 salt)` 排序。LLM 自报的 confidence 经常集中在 0.9、0.95 这类值上，所以这一步是必需的。
2. 对目标 coverage 50%、75%、90%（分母为全部 case，见 D2），找到对应的 `(τ, hash_cutoff)`。
   - 保留规则：`conf > τ`，或者 `conf == τ` 且 `hash < cutoff`。
   - 如果目标 coverage 超过能达到的上限，这一档标为“不可达”。
3. 写入 `outputs/frozen/thresholds.json`，打上哈希后冻结，之后不再修改。
4. 60 条的粒度大约是 1.7%，test 上的实际 coverage 会偏离目标，这点如实报告。

### [E] 策略应用（只用 test，每个 domain 140 条）

| 策略 | 做法 |
|---|---|
| 无弃权 | 保留全部有效判断 |
| 置信度弃权 | 套用冻结阈值，得到实际 coverage `c` 和保留数 `k` |
| 随机弃权 | 在同一批 test case 里随机保留 `k` 条，重复 R = 10,000 次（seed 固定），报告期望 selective risk 和分布。它的期望值在理论上等于全覆盖错误率，用这个解析解交叉验证代码。 |
| 直接弃权 (E5) | 按观测到的 coverage 报告，做两个对比：一是和 E1 在同一 coverage 下的随机弃权比；二是和 E1 的置信度弃权在同一 coverage 下的曲线点比。第二个对比要标注为描述性比较，因为匹配 coverage 用的是 test。 |

### [F] 指标

| 类别 | 指标 |
|---|---|
| 主指标 | coverage、selective risk、risk–coverage 曲线（test 上全阈值扫描） |
| 曲线汇总 | AURC，主分析版和敏感性版两个版本（见 D3）；E-AURC 可选 |
| 次要指标 | 全覆盖准确率、Brier score（所有二元判断，target = verdict 是否正确）、可靠性图 |
| 运营指标 | referral volume（转人工的数量）、错误捕获率（被弃权的 judge 错误 ÷ 全部 judge 错误） |
| 偏差拆分 | 按 candidate 正误分开报告：错误接受（错的判成对）和错误拒绝（对的判成错） |
| 成本 | 平均 token 数、reasoning token 数、运行时长 |
| 有效性 | 各状态的比例，每个 run 单独一张表 |

**敏感性分析**：在“无效输出 = confidence 0 的错误判断”的设定下，重新计算主要终点和 AURC。

### [G] 统计推断

见第 7 节。

### [H] 误差分析（拿到真实输出后再做）

- 把 judge 错误按 `candidate_type` 交叉统计，例如 GSM8K 的漏步骤和算错，以及 MedQA 的干扰项。
- 单独列出**高置信错误**（conf ≥ 0.9 且判错），这是置信度弃权失效的地方。
- 对 thinking 配置抽样阅读 reasoning trace，归纳失败模式，比如判断时自己重新解题、被 candidate 带偏。

---

## 6. Pilot：筛选与继续/更换的关卡

pilot 由**数据组**执行（D7）。它的目标是确认推理设置可用，并判断当前的数据集和模型组合能不能产生有意义的结果。

### 6.1 硬指标（`pilot-report` 自动检查）

用 E1 在 pilot/probe 样本上的结果来判断，thinking 相关的项目看 E2：

| 问题 | 不通过的条件 | 为什么不行 |
|---|---|---|
| judge 太准 | E1 错误率：≥ 10% 通过；6%–10% 为**边界**，由组内决定；< 6% 不通过 | test 里错误太少，risk–coverage 比较没有统计功效 |
| judge 太差 | E1 错误率 > 40% | 接近乱猜，弃权分析没有意义 |
| 偏向一边 | 单一 verdict 占比 > 85% | 错误几乎全是一个类型 |
| confidence 退化 | 超过 70% 的输出是同一个值 | 排序没有信息量 |
| 输出不稳定 | 改完 prompt 后有效率仍 < 95%，或截断率 > 2% | 无效输出太多 |
| 不可复现 | 跑两遍的 verdict 一致率 < 98% | 违反 D1 的要求 |
| 成本太高 | 预估正式运行超出算力预算 | 跑不完 |

**关于样本量**：40 题的 pilot 每个 domain 只有大约 20 题。错误率 10% 只对应 2 个错误，95% CI 大约是 [3%, 30%]，不足以判断门槛。所以**错误率门槛要在 ≥100 题的 `probe` 样本上估计**：
- 这些题从正式的 200 题之外另抽，`split = probe`；
- 构造标准可以比正式数据宽松；
- 由 `validate-data` 确认和正式数据不重叠。

40 题的 pilot 样本继续用于检查服务接口、输出格式、截断率和复现性。

### 6.2 更换的顺序：从小改动到大改动

1. **调数据难度**，不换数据集：数据组把错误做得更隐蔽，比如更细微的数值错误、更像真的干扰项。改完用 `probe` 验证。
2. **换同一领域的数据集**：数学换成更难的数学数据集，医学换成其他医学考试类多选题数据集。这样保持“数学 + 医学”的双领域结构，研究问题不变。新数据集需要先准备 pilot 和 probe 样本。
3. **换模型或调规模**：judge 太准就换同系列更小的版本；输出或 thinking 不稳定就换另一个支持 thinking 开关的开源模型。
4. **研究范围收缩**：作为最后手段，比如只保留一个领域，在报告里说明原因。

### 6.3 筛选模式

```
screen --datasets <a,b> --models <x,y> --configs E1,E2 --split probe
```

一次跑若干个“数据集 × 模型”组合，每个组合只跑 probe/pilot 样本和 E1、E2 两个配置，输出一张硬指标对比表，供组内决定用哪个组合。

### 6.4 防止挑结果的规则

- **更换只能在 pilot 阶段做**。正式的 calib 和 test 一旦开跑，数据集、模型、prompt、parser 和解码参数全部锁定，由 [B] 的锁定检查强制执行。
- 如果正式运行之后才发现必须换，就在新组合上**把 calib 和 test 全部重跑**，不能只换一部分。
- 每次更换都写进 `decision_log.md`，内容包括：日期、触发的指标和数值、换成了什么、谁批准的。报告里要报告原组合的失败结果。

### 6.5 pilot 结束后冻结的内容

prompts、configs、parser 版本、解码参数、`max_tokens`，全部记哈希，存到 `outputs/frozen/`。

---

## 7. 统计分析与预注册

### 7.1 主要终点（预注册）

- **75% coverage 下，置信度弃权与随机弃权的 selective risk 差**。
- 范围：E1–E4 × {GSM8K, MedQA}，共 8 个检验。
- 方法：配对 bootstrap，B = 10,000，在 test 题目上重抽样，按 candidate 正误分层；报告双侧 95% CI，并在 8 个检验之间做 **Holm 校正**。
- 正式运行前写进 `preregistration.md`，并打上时间戳。

### 7.2 次要比较（只报告 CI，不做校正、不下显著性结论）

- 50% 和 90% coverage 两档下的置信度弃权 vs 随机弃权。
- 两个版本的 AURC，以及配置之间的 AURC 差（bootstrap CI）。
- E2、E3、E4 相对 E1 的全覆盖准确率：McNemar exact 检验 + bootstrap CI。
- thinking × rubric 的交互效应。
- domain 之间的差异。
- E5 直接弃权的两个对比（见 [E]）。
- Brier score 与可靠性。

### 7.3 敏感性分析与探索性分析

- **敏感性分析**：无效输出 = confidence 0 的错误判断（D2、D3）。
- **探索性分析**（可选，看资源）：verdict token 的 logprob 作为 confidence（D5），结果单独放在 `exploratory/`。

### 7.4 实现要求

- 所有配置在**同一组重抽样索引**上计算差值，保证配对。
- 所有 bootstrap 和随机基线的 seed 都固定，并记录在案。

---

## 8. 代码结构

```
dsc261-llm-judge/
├── registry/
│   ├── datasets.yaml
│   └── models.yaml            # 两套解码参数、thinking 能力、logprobs 开关
├── adapters/                  # 数据组写：gsm8k.py, medqa.py, <新数据集>.py
├── configs/                   # E1–E5，与数据集、模型解耦
├── prompts/
│   ├── skeletons/             # 按 task_type × minimal/rubric × binary/ternary
│   └── rubrics/               # 按 domain 的插槽片段
├── src/
│   ├── schema.py              # Case 定义 + 校验
│   ├── backends/              # vllm.py, api.py, mock.py
│   ├── runner.py              # [B] 推理 + 缓存 + 续跑 + 锁定检查
│   ├── parse.py               # [C]
│   ├── calibrate.py           # [D] 阈值 + 并列规则 + 不可达检测
│   ├── policies.py            # [E] 四种策略
│   ├── metrics.py             # [F] 含两个版本的 AURC
│   ├── stats.py               # [G] 配对 bootstrap、Holm、McNemar
│   ├── gates.py               # 第 6.1 节硬指标检查
│   └── cli.py                 # validate-data / smoke / probe / screen / pilot-report / run / evaluate / report
├── tests/
│   ├── fixtures/              # 正确样例 + 故意出错的样例
│   └── test_*.py              # parser、并列规则、随机基线解析解、AURC、无效输出处理
├── outputs/
│   ├── raw/{dataset}/{model}/{config_hash}/
│   ├── frozen/                # thresholds.json、冻结的 prompts 和配置哈希
│   └── exploratory/           # logprob 等探索性结果
├── reports/                   # figures、tables
├── docs/
│   ├── experiment_pipeline_plan.md      # 本文
│   ├── experiment_pipeline_plan_en.md   # 英文版
│   ├── adapter_guide.md
│   └── pilot_guide.md
├── preregistration.md
├── decision_log.md
└── Makefile
```

每次运行自动生成一份 `run_manifest.json`，记录 git commit、模型 checksum、vLLM 版本、batch 大小、config_hash、数据哈希和起止时间。

---

## 9. 交付范围与交接流程

### 9.1 骨架阶段的交付范围

**做：**
- 第 3.2 节和第 6.3 节列出的全部 CLI 命令；
- mock、vLLM、API 三种后端，以及预留的 `logprobs` 字段；
- GSM8K 和 MedQA 的示例 adapter，以及 fixtures；
- 解析、校准、四种策略、指标（含两个版本的 AURC）、配对 bootstrap 加 Holm 校正、门槛检查；
- `pilot-report` 和锁定检查；
- 以上模块的单元测试；
- `adapter_guide.md`、`pilot_guide.md`，以及 `decision_log.md` 和 `preregistration.md` 的模板。

**验收标准**：用 mock 后端在 fixtures 上跑通 `validate-data → smoke → pilot-report → run → evaluate → report` 全流程，并且单元测试全部通过。

**不做：**
- 真实模型推理，交给数据组的 pilot；
- 正式数据构建；
- 误差分析 notebook，等有真实输出后再写。

### 9.2 交接给数据组之后的 pilot 流程（`docs/pilot_guide.md`）

1. 写 adapter，生成 `cases_*.jsonl`（含 pilot 和 probe 两个 split）。
2. `validate-data`：修好所有报错。
3. `smoke --backend mock`：确认数据能跑通全流程。
4. 按模板配置真实后端，有两种：Qwen 的 vLLM 启动示例，或者一个 API 后端。
5. `probe` 和 pilot 运行：E1 和 E2，复现性检查跑两遍。
6. `pilot-report`：看门槛检查表。
7. 全部通过就冻结设置，交给实验组正式运行。有不通过的，按第 6.2 节的顺序更换，用 `screen` 对比候选组合，并填写 `decision_log.md`。
