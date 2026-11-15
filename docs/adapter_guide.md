# Adapter 接口说明（面向数据组）

> 管道只认一种输入：符合通用 case schema 的 `cases_{dataset_id}.jsonl` + 同目录的 `manifest.json`。
> adapter 是数据组的数据和管道之间的**接口**：它把一行输入（原始题目 + 一个已经构造好的 candidate）转成一行 case。
> 数据怎么构造（抽哪些题、各 split 多少题、错误 candidate 怎么造）由数据组决定，不在本文范围内。

## 1. adapter 接口

在 `adapters/<dataset>.py` 里实现：

```python
DATASET_ID = "gsm8k"       # 与 registry/datasets.yaml 的键一致
TASK_TYPE = "numeric"      # numeric | multiple_choice

def convert(raw_rows: list[dict], seed: int = 0) -> list[dict]:
    """输入行列表 -> case dict 列表（第 3 节的格式）。"""
```

约定：

- `convert` 只做**格式转换**：重命名字段、抽取参考答案、拼 `case_id`。不调用模型，不读写文件。
- 输入格式由每个 adapter 自己定义（第 2 节是两个参考 adapter 的输入格式）。数据组可以改 adapter 来适配自己的输入，只要输出符合第 3 节。
- `seed` 由 `adapt --seed` 传入并写进 manifest；adapter 用不用随机性都可以。

然后在 `registry/datasets.yaml` 登记：

```yaml
my_dataset:
  adapter: adapters/my_dataset.py
  task_type: numeric            # 或 multiple_choice
  domain: math                  # 或 medical
  rubric: prompts/rubrics/math.md
  data_path: data/my_dataset/cases_my_dataset.jsonl
```

同一 `task_type` 的新数据集通常只需要：adapter + 登记 + （可选）一个新的 rubric 片段。prompt 骨架不用改。

## 2. adapter 的输入例子

输入是 JSON Lines 文件，**每个 candidate 一行**。同一道题有几个 candidate，就有几行，题目字段在每行重复出现。

下面两个例子都实际跑过 `adapt`，转换后 `validate-data` 0 错误、0 警告。

### 2.1 GSM8K（[`adapters/gsm8k.py`](../adapters/gsm8k.py)）

| 输入字段 | 必填 | 说明 |
|---|---|---|
| `id` | 是 | 原始题目 ID（GSM8K 本身没有 ID，由数据组给定） |
| `question` | 是 | GSM8K 原始题干 |
| `answer` | 是 | GSM8K 原始解答，必须以 `#### <数字>` 结尾；adapter 从这里抽出 `reference_answer` |
| `candidate_answer` | 是 | 给 judge 看的答案 |
| `candidate_type` | 是 | `correct` 或错误类型名 |
| `split` | 是 | `pilot` / `probe` / `calib` / `test` |
| `candidate_id` | 否 | 区分同一题的多个 candidate；不填则按行号生成 `c<行号>` |
| `meta` | 否 | 原样合并进输出的 `meta` |

输入（`raw_gsm8k.jsonl`，一题两个 candidate）：

```json
{"id": "train-00042", "question": "Leo has 48 marbles. He gives a quarter of them to his sister and then buys 10 more. How many marbles does he have now?", "answer": "He gives away 48 / 4 = <<48/4=12>>12 marbles.\nHe has 48 - 12 = <<48-12=36>>36 left.\nAfter buying more he has 36 + 10 = <<36+10=46>>46.\n#### 46", "candidate_answer": "A quarter of 48 is 12. 48 - 12 = 36. 36 + 10 = 46. #### 46", "candidate_type": "correct", "split": "probe", "candidate_id": "c1"}
{"id": "train-00042", "question": "Leo has 48 marbles. He gives a quarter of them to his sister and then buys 10 more. How many marbles does he have now?", "answer": "He gives away 48 / 4 = <<48/4=12>>12 marbles.\nHe has 48 - 12 = <<48-12=36>>36 left.\nAfter buying more he has 36 + 10 = <<36+10=46>>46.\n#### 46", "candidate_answer": "A quarter of 48 is 12. 48 - 12 = 36. #### 36", "candidate_type": "omitted_step", "split": "probe", "candidate_id": "c2", "meta": {"source": "gsm8k/train"}}
```

第二行转换后的输出：

```json
{"case_id": "gsm8k-train-00042-c2", "question_id": "gsm8k-train-00042", "dataset_id": "gsm8k", "split": "probe", "task_type": "numeric", "question": "Leo has 48 marbles. He gives a quarter of them to his sister and then buys 10 more. How many marbles does he have now?", "options": null, "candidate_answer": "A quarter of 48 is 12. 48 - 12 = 36. #### 36", "reference_answer": "46", "label": false, "candidate_type": "omitted_step", "meta": {"source_id": "train-00042", "source": "gsm8k/train"}}
```

对应关系：`question_id` = `gsm8k-` + `id`；`case_id` = `question_id` + `-` + `candidate_id`；`reference_answer` 取 `answer` 里 `####` 之后的数字；`label` = (`candidate_type == "correct"`)。

### 2.2 MedQA（[`adapters/medqa.py`](../adapters/medqa.py)）

| 输入字段 | 必填 | 说明 |
|---|---|---|
| `id` | 是 | 原始题目 ID |
| `question` | 是 | MedQA 原始题干 |
| `options` | 是 | MedQA 原始选项 `{"A": "...", ...}`；键会被转成大写 |
| `answer_idx` | 是 | MedQA 原始正确选项字母，作为 `reference_answer` |
| `candidate_answer` | 是 | 给 judge 看的选项字母 |
| `candidate_type` | 是 | `correct` 或错误类型名 |
| `split` | 是 | `pilot` / `probe` / `calib` / `test` |
| `candidate_id` | 否 | 同上 |
| `meta_info` | 否 | MedQA 原始字段，写进输出的 `meta` |
| 其他字段 | 否 | 例如 MedQA 原始的 `answer`（答案文本），adapter 会忽略 |

输入（`raw_medqa.jsonl`）：

```json
{"id": "test-0137", "question": "A 3-year-old child has recurrent lung infections and greasy, foul-smelling stools. Sweat chloride concentration is elevated. Which protein is most likely defective?", "options": {"A": "Dystrophin", "B": "CFTR chloride channel", "C": "Fibrillin-1", "D": "Type I collagen"}, "answer": "CFTR chloride channel", "answer_idx": "B", "meta_info": "step1", "candidate_answer": "B", "candidate_type": "correct", "split": "test", "candidate_id": "c1"}
{"id": "test-0137", "question": "A 3-year-old child has recurrent lung infections and greasy, foul-smelling stools. Sweat chloride concentration is elevated. Which protein is most likely defective?", "options": {"A": "Dystrophin", "B": "CFTR chloride channel", "C": "Fibrillin-1", "D": "Type I collagen"}, "answer": "CFTR chloride channel", "answer_idx": "B", "meta_info": "step1", "candidate_answer": "D", "candidate_type": "distractor", "split": "test", "candidate_id": "c2"}
```

第二行转换后的输出：

```json
{"case_id": "medqa-test-0137-c2", "question_id": "medqa-test-0137", "dataset_id": "medqa", "split": "test", "task_type": "multiple_choice", "question": "A 3-year-old child has recurrent lung infections and greasy, foul-smelling stools. Sweat chloride concentration is elevated. Which protein is most likely defective?", "options": {"A": "Dystrophin", "B": "CFTR chloride channel", "C": "Fibrillin-1", "D": "Type I collagen"}, "candidate_answer": "D", "reference_answer": "B", "label": false, "candidate_type": "distractor", "meta": {"source_id": "test-0137", "meta_info": "step1"}}
```

## 3. 输出格式（每行一个 case）

| 字段 | 类型 | 说明 | judge 可见 |
|---|---|---|---|
| `case_id` | str | 全局唯一。建议 `{question_id}-{candidate_id}` | 否 |
| `question_id` | str | 原始题目 ID。bootstrap 按它重抽样，查重也靠它 | 否 |
| `dataset_id` | str | 必须在 `registry/datasets.yaml` 里登记 | 否 |
| `split` | `pilot` / `probe` / `calib` / `test` | 一题只能在一个 split | 否 |
| `task_type` | `numeric` / `multiple_choice` | 决定用哪套 prompt 骨架 | 否 |
| `question` | str | 题干 | **是** |
| `options` | dict \| null | 选择题选项 `{"A": "...", ...}`；数值题为 `null` | **是** |
| `candidate_answer` | str | 待判断的答案。数值题可以是带步骤的解答，但必须能抽出最终数字（推荐以 `#### <数字>` 结尾）；选择题写选项字母 | **是** |
| `reference_answer` | str | 数值题为纯数字；选择题为选项字母 | 否 |
| `label` | bool | candidate 是否正确。**必须** 等于 `candidate_type == "correct"` | 否 |
| `candidate_type` | str | 正确的写 `correct`；错误类型的命名由数据组决定 | 否 |
| `meta` | dict | 自由字段 | 否 |

judge 只能看到 `question`、`options`、`candidate_answer`。这三个字段里**不要**出现 label 线索：
`(correct)`、`[incorrect]`、`ground truth`、`reference answer`、带下划线的错误类型标签（如 `omitted_step`）等，
`validate-data` 会报 `leakage`。

## 4. 运行 adapter + 生成 manifest

```bash
llm-judge adapt --dataset gsm8k --raw raw_gsm8k.jsonl --out data/gsm8k/cases_gsm8k.jsonl --seed 0
```

会写出 cases 文件和同目录的 `manifest.json`（SHA256、seed、各 split 数量），并立即跑一次 `validate-data`。
如果你们用自己的脚本生成 cases 文件，最后补一个 manifest：

```bash
llm-judge manifest --data data/gsm8k/cases_gsm8k.jsonl --seed <抽样用的seed>
```

**manifest 生成之后 cases 文件一个字节都不能改**——正式 `run` 会校验哈希，对不上直接拒绝运行。
（仓库的 `.gitattributes` 已经禁止 git 改 `*.jsonl` 的换行符，Windows 上也不会因为 CRLF 导致哈希变化。）

## 5. 检查

```bash
llm-judge validate-data --data data/gsm8k/cases_gsm8k.jsonl
llm-judge smoke         --data data/gsm8k/cases_gsm8k.jsonl --n 20 --backend mock
```

`validate-data` 的检查项：schema 与类型；同一题跨 split；重复 case_id / 重复 (question_id, candidate) / 不同 question_id 下的相同题干；
label 泄露；数值题能否解析出数字（以及 label 与“最终数字 == 参考答案”是否一致）；选择题 candidate / reference 是否在 options 里；
probe 题与 calib/test 不重叠（按 ID 和题干）；manifest 哈希与计数。另外会打印各 split 的数量和正误比例，只给警告，供自查。

想看每种报错长什么样：

```bash
llm-judge validate-data --data tests/fixtures/broken/cases_broken.jsonl
```

## 6. 常见问题

- **一题多个 candidate？** 可以。同一 `question_id` 多行，`case_id` 不同；它们必须在同一 split。bootstrap 会把同一题的 case 一起抽。
- **split 由谁决定？** 由输入行的 `split` 字段决定，adapter 只是照抄。
- **选择题 candidate 写 `B` 还是 `B. text`？** 都能被识别，但推荐只写字母，judge 看到的就是这个。
