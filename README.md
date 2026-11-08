# dsc261-llm-judge

DSC 261 team project: LLM-as-a-Judge reliability — **when should an LLM judge abstain?**

实验管道按 [`docs/experiment_pipeline_plan.md`](docs/experiment_pipeline_plan.md)（[English](docs/experiment_pipeline_plan_en.md)）实现。

- 数据格式与 adapter：[`docs/adapter_guide.md`](docs/adapter_guide.md)
- pilot 细节（门槛、prompt、更换顺序）：[`docs/pilot_guide.md`](docs/pilot_guide.md)
- 模板：[`preregistration.md`](preregistration.md)、[`decision_log.md`](decision_log.md)

## 0. 安装

需要 Python ≥ 3.10。在仓库根目录：

```bash
# 用 uv（推荐）
uv venv .venv
uv pip install --python .venv/Scripts/python.exe -e ".[dev]"     # macOS/Linux: .venv/bin/python

# 或者用标准 venv + pip
python -m venv .venv
.venv/Scripts/python -m pip install -e ".[dev]"                   # macOS/Linux: .venv/bin/python
```

激活虚拟环境：PowerShell 用 `.venv\Scripts\Activate.ps1`；Git Bash 用 `source .venv/Scripts/activate`；macOS/Linux 用 `source .venv/bin/activate`。
激活后就能直接用 `llm-judge` 命令。如果 PowerShell 禁止运行脚本，可以不激活，直接写 `.venv\Scripts\llm-judge.exe ...`。

确认环境正常：

```bash
pytest -q          # 应该全部通过
```

下文假设你的数据放在 `data/<dataset_id>/cases_<dataset_id>.jsonl`，例如 `data/gsm8k/cases_gsm8k.jsonl` 和 `data/medqa/cases_medqa.jsonl`。

## 1. 数据组：测试步骤

前 3 步不需要 GPU，也不调用任何模型。每个 domain 都要做一遍，下面以 GSM8K 为例。

### 第 1 步：生成 cases 文件和 manifest

```bash
# 用 adapter（输入格式见 docs/adapter_guide.md 第 2 节）
llm-judge adapt --dataset gsm8k --raw raw_gsm8k.jsonl --out data/gsm8k/cases_gsm8k.jsonl --seed 0

# 或者已经用自己的脚本生成了 cases 文件，只补 manifest
llm-judge manifest --data data/gsm8k/cases_gsm8k.jsonl --seed <抽样用的seed>
```

### 第 2 步：静态检查

修好所有 ERROR 再往下走：

```bash
llm-judge validate-data --data data/gsm8k/cases_gsm8k.jsonl --require-manifest
```

### 第 3 步：smoke test

用 mock 后端把少量数据走完整条管道，验证数据能跑通：

```bash
llm-judge smoke --data data/gsm8k/cases_gsm8k.jsonl --n 20 --backend mock
```

结果在 `outputs/smoke/`。mock 不是真实模型，它的数字没有意义，只要不报错就算通过。

### 第 4 步：用真实模型跑 probe 和 pilot

先配置后端：vLLM 启动方式和 API key 设置见 [`docs/pilot_guide.md`](docs/pilot_guide.md) 第 4 步，模型参数在 `registry/models.yaml`。

```bash
# probe（>=100 题）：估计 judge 错误率，生成难度报告
llm-judge probe --data data/gsm8k/cases_gsm8k.jsonl --split probe --n 100 --model qwen3.8-27b --configs E1,E2

# pilot：E1、E2 各跑一遍
llm-judge probe --data data/gsm8k/cases_gsm8k.jsonl --split pilot --model qwen3.8-27b --configs E1,E2
# pilot：E1 再跑一遍，做复现性检查
llm-judge probe --data data/gsm8k/cases_gsm8k.jsonl --split pilot --model qwen3.8-27b --configs E1 --tag r2
```

每条命令都会打印 run 目录，形如 `outputs/raw/gsm8k/qwen3.8-27b/<config_hash>/`，下一步要用。

### 第 5 步：门槛检查

```bash
llm-judge pilot-report --run outputs/raw/gsm8k/qwen3.8-27b/<E1的hash>/ --run outputs/raw/gsm8k/qwen3.8-27b/<E2的hash>/
```

它会自动找 `<E1的hash>-r2/` 作为复现性检查的第二遍，输出一张门槛检查表（PASS / BORDERLINE / FAIL），同时存成 run 目录里的 `pilot_report.md`。
错误率门槛请看 probe 那个 run 的结果，pilot 样本太少，CI 太宽。

- **全部 PASS**：进入第 6 步。
- **有 BORDERLINE**：组内讨论决定。
- **有 FAIL**：按 [`docs/pilot_guide.md`](docs/pilot_guide.md) 第 7 节的顺序调整，可以用 `llm-judge screen` 对比候选组合，并把每次更换记进 [`decision_log.md`](decision_log.md)。改完从第 2 步重来。

### 第 6 步：冻结设置

两个 domain 都通过之后，做一次：

```bash
llm-judge freeze --datasets gsm8k,medqa --models qwen3.8-27b --configs E1,E2,E3,E4,E5
```

这一步会生成 `outputs/frozen/lock.json`。从这以后不能再改 `prompts/`、`configs/`、`registry/models.yaml` 的解码参数和 `src/llm_judge/parse.py`，否则正式运行会被拒绝。把 lock 文件 commit 掉，然后把仓库交给实验组。

## 2. 测试通过后：运行完整代码并生成 report

### 第 1 步：预注册

填写 [`preregistration.md`](preregistration.md)，包括 `lock.json` 和两份数据 manifest 的 SHA256，然后 commit。commit 时间就是预注册时间。

### 第 2 步：正式推理

只跑 calib 和 test；会做锁定检查和数据哈希校验；中断后重新运行同一条命令就会从断点续跑：

```bash
llm-judge run --data data/gsm8k/cases_gsm8k.jsonl --model qwen3.8-27b
llm-judge run --data data/medqa/cases_medqa.jsonl --model qwen3.8-27b
```

默认跑 E1–E5。如果报 `lock check failed`，说明冻结后有设置被改过，报错信息会写明改了哪一项。

### 第 3 步：评估

依次做解析、在 calib 上校准阈值并冻结、在 test 上应用策略、计算指标、配对 bootstrap：

```bash
llm-judge evaluate --data data/gsm8k/cases_gsm8k.jsonl --model qwen3.8-27b
llm-judge evaluate --data data/medqa/cases_medqa.jsonl --model qwen3.8-27b
```

终端会打印每个配置的主要终点。结果在 `outputs/eval/<dataset>/<model>/`，阈值冻结在 `outputs/frozen/thresholds.json`。

### 第 4 步：生成 report

跨两个 domain 做 Holm 校正，出表和图：

```bash
llm-judge report --model qwen3.8-27b --datasets gsm8k,medqa
```

输出：

| 文件 | 内容 |
|---|---|
| `reports/qwen3.8-27b/report.md` | 主要终点（8 个检验 + Holm）、各配置指标、次要比较、E5、domain 差异、敏感性分析、有效性、成本 |
| `reports/qwen3.8-27b/figures/` | 每个 domain 一张风险–覆盖图和一张可靠性图 |
| `reports/qwen3.8-27b/tables/*.csv` | 报告里各张表的 CSV 版本 |

### 可选：DeepSeek 复现

冻结时要把它也写进去，例如 `--models qwen3.8-27b,deepseek-v4-flash-0731`。然后：

```bash
llm-judge run      --data data/gsm8k/cases_gsm8k.jsonl --model deepseek-v4-flash-0731 --configs E1,E3,E5
llm-judge evaluate --data data/gsm8k/cases_gsm8k.jsonl --model deepseek-v4-flash-0731 --configs E1,E3,E5
# medqa 同上
llm-judge report   --model deepseek-v4-flash-0731 --datasets gsm8k,medqa
```

### 没有 GPU 时先演示一遍完整流程

用合成数据和 mock 后端，把正式流程完整走一遍（`qwen3.8-27b@mock` 的意思是：用 Qwen 的配置，但由 mock 生成输出）：

```bash
python -m llm_judge.synthetic --task numeric         --out data/synth_math/cases_synth_math.jsonl
python -m llm_judge.synthetic --task multiple_choice --out data/synth_med/cases_synth_med.jsonl
llm-judge freeze   --datasets synth_math,synth_med --models qwen3.8-27b@mock --out outputs/demo
llm-judge run      --data data/synth_math/cases_synth_math.jsonl --model qwen3.8-27b@mock --out outputs/demo
llm-judge run      --data data/synth_med/cases_synth_med.jsonl   --model qwen3.8-27b@mock --out outputs/demo
llm-judge evaluate --data data/synth_math/cases_synth_math.jsonl --model qwen3.8-27b@mock --out outputs/demo
llm-judge evaluate --data data/synth_med/cases_synth_med.jsonl   --model qwen3.8-27b@mock --out outputs/demo
llm-judge report   --model qwen3.8-27b@mock --datasets synth_math,synth_med --out outputs/demo --reports-dir reports/demo
```

报告在 `reports/demo/qwen3.8-27b@mock/report.md`。装了 `make` 的话，`make demo` 等价于上面这些命令。
如果想从头重跑，先删掉 `outputs/demo`：thresholds 只能写一次，换了数据再评估会被拒绝。

## 目录结构

```text
registry/      datasets.yaml, models.yaml（两套解码参数、thinking 切换方式、logprobs 开关）
adapters/      原始数据 -> case schema（数据组写）
configs/       E1–E5（与数据集、模型解耦）
prompts/       skeletons/{task_type}_{minimal|rubric}_{binary|ternary}.md, rubrics/{math,medical}.md
src/llm_judge/ schema, backends/{vllm,api,mock}, runner [B], parse [C], calibrate [D], policies [E],
               metrics [F], stats [G], gates (6.1), evaluate, report, cli
tests/         fixtures/（正确样例 + 故意出错的样例）和单元测试、端到端测试
outputs/       raw/{dataset}/{model}/{config_hash}/, frozen/, eval/, exploratory/
reports/       figures + tables + report.md
```

## 命令一览

| 谁用 | 命令 | 作用 |
|---|---|---|
| 数据组 | `adapt`、`manifest` | 运行 adapter / 生成 manifest.json |
| 数据组 | `validate-data` | 静态检查 cases 文件 |
| 数据组 | `smoke` | mock 后端在少量数据上走完整条管道 |
| 数据组 | `probe` | 在 probe / pilot 样本上跑真实模型，出难度报告 |
| 数据组 | `pilot-report` | 第 6.1 节门槛检查表 |
| 数据组 | `screen` | 对比多个 数据集 × 模型 组合 |
| 两组 | `freeze` | pilot 结束时锁定 prompt、config、parser、解码参数 |
| 实验组 | `run` | 正式推理（calib + test，锁定检查，可续跑） |
| 实验组 | `evaluate` | 解析 → 校准并冻结阈值 → 策略 → 指标 → 配对 bootstrap |
| 实验组 | `report` | 跨 domain Holm 校正，出表、图和 report.md |

所有命令都可以加 `--help` 查看参数，例如 `llm-judge probe --help`。
