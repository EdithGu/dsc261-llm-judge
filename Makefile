# Convenience targets. Every target is a thin wrapper over `llm-judge` / pytest.
PY ?= python
CLI = $(PY) -m llm_judge

.PHONY: install test validate-fixtures smoke-fixtures synth demo clean-smoke

install:
	$(PY) -m pip install -e ".[dev]"

test:
	$(PY) -m pytest -q

validate-fixtures:
	$(CLI) validate-data --data tests/fixtures/gsm8k/cases_gsm8k.jsonl --require-manifest
	$(CLI) validate-data --data tests/fixtures/medqa/cases_medqa.jsonl --require-manifest
	-$(CLI) validate-data --data tests/fixtures/broken/cases_broken.jsonl

smoke-fixtures:
	$(CLI) smoke --data tests/fixtures/gsm8k/cases_gsm8k.jsonl --n 20 --backend mock
	$(CLI) smoke --data tests/fixtures/medqa/cases_medqa.jsonl --n 20 --backend mock

synth:
	$(PY) -m llm_judge.synthetic --task numeric --out data/synth_math/cases_synth_math.jsonl
	$(PY) -m llm_judge.synthetic --task multiple_choice --out data/synth_med/cases_synth_med.jsonl

# Full formal flow on synthetic data with the mock backend (writes under outputs/demo, reports/demo)
demo: synth
	$(CLI) freeze --datasets synth_math,synth_med --models qwen3.8-27b@mock --out outputs/demo --force
	$(CLI) run --data data/synth_math/cases_synth_math.jsonl --model qwen3.8-27b@mock --out outputs/demo
	$(CLI) run --data data/synth_med/cases_synth_med.jsonl --model qwen3.8-27b@mock --out outputs/demo
	$(CLI) evaluate --data data/synth_math/cases_synth_math.jsonl --model qwen3.8-27b@mock --out outputs/demo --force-thresholds
	$(CLI) evaluate --data data/synth_med/cases_synth_med.jsonl --model qwen3.8-27b@mock --out outputs/demo --force-thresholds
	$(CLI) report --model qwen3.8-27b@mock --datasets synth_math,synth_med --out outputs/demo --reports-dir reports/demo

clean-smoke:
	$(PY) -c "import shutil; shutil.rmtree('outputs/smoke', ignore_errors=True)"
