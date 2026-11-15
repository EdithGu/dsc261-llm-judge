# Pilot 操作手册（面向数据组）

对应 `experiment_pipeline_plan.md` 第 6 节与第 9.2 节。目标：确认推理设置可用，并判断“数据集 × 模型”组合能否产生有意义的结果。

所有命令在仓库根目录运行（`pip install -e .[dev]` 之后可用 `llm-judge ...`，或 `python -m llm_judge ...`）。

## 0. 安装

```bash
pip install -e ".[dev]"
```

## 1. 写 adapter，生成 cases（含 pilot 和 probe）

adapter 的接口和输入/输出例子见 [`adapter_guide.md`](adapter_guide.md)。数据怎么构造（各 split 抽多少题、candidate 怎么造）由数据组决定，本手册不规定；这里只写管道接受的**格式**。不管用 adapter 还是自己的脚本，交付的数据都必须满足下面的格式要求。

### 1.1 数据集格式要求

**文件**：每个 domain 一个目录，里面两个文件：

```text
data/<dataset_id>/cases_<dataset_id>.jsonl    # 每行一个 case（JSON Lines，UTF-8）
data/<dataset_id>/manifest.json               # SHA256、抽样 seed、各 split 数量
```

**字段**（每行都必须有全部字段；自由信息只能放进 `meta`）：

| 字段 | 类型 | 要求 | judge 可见 |
|---|---|---|---|
| `case_id` | str | 全文件唯一，建议 `{question_id}-{candidate_id}` | 否 |
| `question_id` | str | 原始题目 ID；同一题的多个 candidate 共用一个 | 否 |
| `dataset_id` | str | 一个文件只能有一个值，且已在 `registry/datasets.yaml` 登记 | 否 |
| `split` | str | `pilot` / `probe` / `calib` / `test` 之一 | 否 |
| `task_type` | str | `numeric` 或 `multiple_choice`，一个文件只能有一种 | 否 |
| `question` | str | 题干，不能为空 | **是** |
| `options` | dict \| null | 选择题为 `{"A": "...", "B": "...", ...}`；数值题为 `null` | **是** |
| `candidate_answer` | str | 数值题：可以带步骤，但必须能抽出最终数字，推荐以 `#### <数字>` 结尾；选择题：选项字母 | **是** |
| `reference_answer` | str | 数值题：纯数字；选择题：选项字母 | 否 |
| `label` | bool | JSON 布尔值 `true`/`false`，不是字符串；必须等于 `candidate_type == "correct"` | 否 |
| `candidate_type` | str | 正确的 candidate 必须写 `correct`；错误类型的命名由数据组决定（如 `arithmetic`、`distractor`），报告会按它拆分统计 | 否 |
| `meta` | dict | 自由字段：构造记录、难度、科目等 | 否 |

**样例**（实际文件里每个 case 占一行；这里为了好读做了换行）。

数值题（来自 `tests/fixtures/gsm8k/`）：

```json
{"case_id": "gsm8k-fx03-c1", "question_id": "gsm8k-fx03", "dataset_id": "gsm8k", "split": "probe",
 "task_type": "numeric",
 "question": "Leo has 48 marbles. He gives a quarter of them to his sister and then buys 10 more. How many marbles does he have now?",
 "options": null,
 "candidate_answer": "A quarter of 48 is 12. 48 - 12 = 36. #### 36",
 "reference_answer": "46", "label": false, "candidate_type": "omitted_step", "meta": {"fixture": true}}
```

选择题（来自 `tests/fixtures/medqa/`）：

```json
{"case_id": "medqa-fx09-c1", "question_id": "medqa-fx09", "dataset_id": "medqa", "split": "test",
 "task_type": "multiple_choice",
 "question": "A 3-year-old child has recurrent lung infections and greasy, foul-smelling stools. Sweat chloride concentration is elevated. Which protein is most likely defective?",
 "options": {"A": "Dystrophin", "B": "CFTR chloride channel", "C": "Fibrillin-1", "D": "Type I collagen"},
 "candidate_answer": "D",
 "reference_answer": "B", "label": false, "candidate_type": "distractor", "meta": {"fixture": true}}
```

**格式约束**（`validate-data` 会逐条检查，报错就不能进管道）：

1. **同一题只能出现在一个 split**：同一个 `question_id` 的所有 candidate 都在同一个 split。
2. **probe 与 calib、test 不重叠**，按 `question_id` 和题干文字两方面查。所以 probe、calib、test 要放在**同一个文件**里交付，否则查不到重叠。
3. 不能有重复：`case_id` 不重复；同一题不能有两个相同的 candidate；不同 `question_id` 下不能出现相同题干。
4. **不能泄露 label**：`question`、`options`、`candidate_answer` 里不能出现 `(correct)`、`[incorrect]`、`ground truth`、`reference answer`、`is_correct`、`label:` 或带下划线的错误类型标签（如 `omitted_step`）。
5. 数值题：`reference_answer` 是纯数字，`candidate_answer` 能抽出最终数字；最终数字和参考答案是否相等，应该和 `label` 一致（不一致会警告）。
6. 选择题：`candidate_answer` 和 `reference_answer` 都必须是 `options` 里的选项；两者相同当且仅当 `label` 为 `true`。

`validate-data` 还会打印各 split 的数量和正误比例，供数据组自查；这两项只给警告，不拦截。

**manifest.json 样例**（用 `llm-judge manifest --data <cases文件> --seed <抽样seed>` 生成，不要手写哈希）：

```json
{
  "file": "cases_gsm8k.jsonl",
  "dataset_id": "gsm8k",
  "sha256": "54b1fe4bce761038fbf7f6072de1bb259f0f3fed34cfec28ce360707b7877f30",
  "sampling_seed": 0,
  "n_cases": 10,
  "split_counts": {"pilot": 2, "probe": 2, "calib": 3, "test": 3}
}
```

生成 manifest 之后，cases 文件**一个字节都不能改**，包括空格和换行。要改就改完再重新生成 manifest；正式 `run` 会校验哈希，对不上直接拒绝运行。

想看每种违规会报什么错：

```bash
llm-judge validate-data --data tests/fixtures/broken/cases_broken.jsonl
```

## 2. validate-data —— 修好所有报错

```bash
llm-judge validate-data --data data/gsm8k/cases_gsm8k.jsonl
```

## 3. smoke —— 确认数据能跑通全流程（mock，无 GPU）

```bash
llm-judge smoke --data data/gsm8k/cases_gsm8k.jsonl --n 20 --backend mock
```

输出在 `outputs/smoke/`。mock 的数字没有意义，只验证管道能跑通。

## 4. 配置真实后端

**Qwen（vLLM）**：

```bash
vllm serve Qwen/Qwen3.8-27B --served-model-name Qwen/Qwen3.8-27B \
    --reasoning-parser qwen3 --max-num-seqs 16 --seed 0 --port 8000
```

然后核对 `registry/models.yaml` 里 `qwen3.8-27b` 的 `checkpoint`、`base_url`、两套解码参数（标了 `TODO(pilot)` 的地方）。
`batch_size` 要和 `--max-num-seqs` 对齐，正式运行时不能再改（D1）。

**API 后端**（DeepSeek 或便宜模型做 probe）：在 `models.yaml` 填 `base_url` / `checkpoint`，并设置环境变量：

```bash
export DEEPSEEK_API_KEY=...      # PowerShell: $env:DEEPSEEK_API_KEY="..."
```

## 4.5 了解当前的 prompt（pilot 期间可以改）

prompt 由三部分组合而成：

- **骨架**：[`prompts/skeletons/`](../prompts/skeletons/)，按 `{task_type}_{minimal|rubric}_{binary|ternary}.md` 分成 8 个文件；
- **rubric 片段**：[`prompts/rubrics/`](../prompts/rubrics/)，由 `registry/datasets.yaml` 的 `rubric` 字段指定；
- **case 内容**：填进 `{{question}}`、`{{options}}`、`{{candidate_answer}}`、`{{rubric}}` 这几个占位符。

每个 prompt 由一条 system 消息和一条 user 消息组成，文件里用 `<<<SYSTEM>>>` / `<<<USER>>>` 分隔。
judge **看不到参考答案**（reference-free），也看不到 label 和 `candidate_type`。

### 例子：E1（minimal + binary）× GSM8K fixture 第 2 题

**System：**

```text
You are an impartial expert judge. You will be shown a math word problem and a candidate answer (which may include a worked solution). Your job is to decide whether the candidate's final answer is correct. You are NOT shown a reference answer.
```

**User：**

```text
Problem:
A train travels 60 km per hour for 2.5 hours. How many kilometers does it travel?

Candidate answer:
60 x 2.5 = 120. #### 120

Decide whether the candidate's final answer to the problem is correct.

Respond with a single JSON object and nothing else, in exactly this format:
{"verdict": "correct" or "incorrect", "confidence": <a number between 0 and 1>}

"confidence" is the probability that YOUR VERDICT is right. It is NOT the probability that the candidate answer is correct. For example, if you are fairly sure the candidate answer is wrong, a suitable response is {"verdict": "incorrect", "confidence": 0.9}.
```

### 例子：E3（rubric + binary）× MedQA fixture 第 9 题

E3 比 E1 多了 `Evaluation rubric:` 段；选择题比数值题多了 `Options:` 段，candidate 只给选项字母。

**System：**

```text
You are an impartial expert judge. You will be shown a multiple-choice question, its options, and a candidate answer. Your job is to decide whether the candidate answer is the correct option. You are NOT shown a reference answer.
```

**User：**

```text
Question:
A 3-year-old child has recurrent lung infections and greasy, foul-smelling stools. Sweat chloride concentration is elevated. Which protein is most likely defective?

Options:
A. Dystrophin
B. CFTR chloride channel
C. Fibrillin-1
D. Type I collagen

Candidate answer:
D

Evaluation rubric:
1. Identify the key findings in the question stem (age, sex, history, symptoms, vital signs, labs, imaging) and what the question is actually asking (diagnosis, next best step, mechanism, most likely cause, ...).
2. Evaluate EVERY option against the stem, not only the candidate's choice. The candidate is correct if and only if it is the single best answer.
3. Consistency with the stem: an option that contradicts any key finding, or that answers a different question (e.g. a correct treatment when the question asks for the next diagnostic step), is incorrect.
4. Plausible distractors: an option that is reasonable in general but less appropriate than another option for THIS patient is incorrect.
5. Judge only the selected option; ignore any explanation that accompanies it.

Decide whether the candidate answer is the correct option for this question.

Respond with a single JSON object and nothing else, in exactly this format:
{"verdict": "correct" or "incorrect", "confidence": <a number between 0 and 1>}

"confidence" is the probability that YOUR VERDICT is right. It is NOT the probability that the candidate answer is correct. For example, if you are fairly sure the candidate answer is wrong, a suitable response is {"verdict": "incorrect", "confidence": 0.9}.
```

### 数学 rubric（[`prompts/rubrics/math.md`](../prompts/rubrics/math.md)）

```text
1. Solve the problem yourself, step by step, before looking closely at the candidate's reasoning.
2. The candidate is correct if and only if its FINAL answer is numerically equivalent to the true answer.
   - Numeric equivalence: formatting differences do not matter (1,000 = 1000; 0.5 = 1/2 = 50% when the question asks for a fraction/percentage of the same quantity; 12.0 = 12).
   - Currency symbols and thousands separators do not matter.
3. Units: if the question asks for a specific unit (e.g. minutes vs hours, cents vs dollars), the final answer must be in that unit or be unambiguously convertible as stated by the candidate. A number in the wrong unit is incorrect.
4. Check that no required step was skipped: a missing step that changes the final quantity makes the answer incorrect, even if the remaining arithmetic is right.
5. Do not reward a plausible-looking derivation whose final number is wrong, and do not penalise a terse answer whose final number is right.
```

医学 rubric 的全文见上面 E3 的例子（[`prompts/rubrics/medical.md`](../prompts/rubrics/medical.md)）。

### 各配置的差别

| 配置 | 骨架 | 与 E1 的差别 |
|---|---|---|
| E1 | `*_minimal_binary` | 基线 |
| E2 | `*_minimal_binary` | prompt 文字与 E1 完全相同；请求里加 `enable_thinking: true`，换用 thinking 那一套解码参数 |
| E3 | `*_rubric_binary` | 题目和任务句之间插入 `Evaluation rubric:` + rubric 片段 |
| E4 | `*_rubric_binary` | E3 的 prompt，加 thinking |
| E5 | `*_minimal_ternary` | 多一句 “If you cannot judge this case reliably, you may abstain; abstained cases are referred to a human expert.”；verdict 可以是 `"abstain"`；末尾补充弃权时 confidence 的含义 |

### 改 prompt 时注意

- 以 `prompts/` 下的文件为准，本节只是示例；改了文件后，本节不会自动更新。
- 所有模板都必须要求输出严格的 JSON：`{"verdict": "...", "confidence": 0.xx}`，并写明 confidence 是“judge 自己这个 verdict 正确的概率”（计划 [A]）。解析器只认这个格式。
- 改动后 `config_hash` 会变，新结果写进新的目录，旧结果不会被覆盖，可以直接对比改动前后的 `difficulty_report.md`。
- 想看某个配置实际用的 prompt 全文，看对应 run 目录里的 `config.json`（`skeleton_text`、`rubric_text` 字段）。
- `freeze` 之后不能再改，否则正式 `run` 会拒绝执行。

## 5. probe 与 pilot 运行

```bash
# 难度报告（错误率门槛在 >=100 题的 probe 上估计）
llm-judge probe --data data/gsm8k/cases_gsm8k.jsonl --n 100 --split probe --model qwen3.8-27b --configs E1,E2

# pilot 40 题：E1、E2，跑两遍做复现性检查（第二遍用 --tag r2）
llm-judge probe --data data/gsm8k/cases_gsm8k.jsonl --split pilot --model qwen3.8-27b --configs E1,E2
llm-judge probe --data data/gsm8k/cases_gsm8k.jsonl --split pilot --model qwen3.8-27b --configs E1 --tag r2
```

每个 run 目录（`outputs/raw/<dataset>/<model>/<config_hash>/`）里有 `difficulty_report.md`：
judge 在正确 / 错误 candidate 上的错误率、按 `candidate_type` 拆开的错误率、confidence 分布。

pilot 阶段 `probe` 不做锁定检查，可以自由改 prompt；每次改动后 `config_hash` 会变，结果放在新目录里，旧结果不会被覆盖。

## 6. pilot-report —— 看门槛检查表

```bash
llm-judge pilot-report \
    --run outputs/raw/gsm8k/qwen3.8-27b/<E1的hash>/ \
    --run outputs/raw/gsm8k/qwen3.8-27b/<E2的hash>/ \
    --budget-gpu-hours 40
```

自动找 `<E1的hash>-r2/` 作为复现性检查的第二遍（也可以 `--replicate` 指定）。
错误率门槛请对 probe 的 run 单独再跑一次 `pilot-report`（pilot 40 题的 CI 太宽，报告里会提示）。

| 检查 | 不通过 |
|---|---|
| judge 太准 | E1 错误率 < 6%（6–10% 为边界，组内决定） |
| judge 太差 | E1 错误率 > 40% |
| 偏向一边 | 单一 verdict > 85% |
| confidence 退化 | 同一个值 > 70% |
| 输出不稳定 | 有效率 < 95% 或截断率 > 2%（E1、E2 分别看） |
| 不可复现 | 两遍 verdict 一致率 < 98% |
| 成本 | 估算正式运行 GPU 小时 > 预算 |

## 7. 通过 → 冻结；不通过 → 按顺序更换

**全部通过**：冻结设置，交给实验组。

```bash
llm-judge freeze --datasets gsm8k,medqa --models qwen3.8-27b --configs E1,E2,E3,E4,E5
```

会写 `outputs/frozen/lock.json`（prompt / config / parser / 解码参数的哈希）和 `outputs/frozen/prompts/`。之后正式 `run`
发现任何一项哈希不一致都会拒绝执行。**冻结后不要再改 prompts/、configs/、registry/models.yaml 的解码参数和 src/llm_judge/parse.py。**

**有不通过**：按第 6.2 节从小改动到大改动：
1. 调数据难度（不换数据集），用 `probe` 验证；
2. 换同领域数据集；
3. 换模型或规模；
4. 研究范围收缩（最后手段）。

对比候选组合用 `screen`：

```bash
llm-judge screen --datasets gsm8k,gsm_hard --models qwen3.8-27b,qwen3.8-14b --configs E1,E2 --split probe
```

每次更换都填 [`decision_log.md`](../decision_log.md)：日期、触发的指标和数值、换成了什么、谁批准的。

## 8. 交接给实验组之后（参考）

```bash
llm-judge run      --data data/gsm8k/cases_gsm8k.jsonl --model qwen3.8-27b
llm-judge evaluate --data data/gsm8k/cases_gsm8k.jsonl --model qwen3.8-27b
llm-judge report   --model qwen3.8-27b --datasets gsm8k,medqa
```
