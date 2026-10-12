# ARIA: Runtime Inspection and Amendment for Multi-Agent Systems

This repository contains the official code for **ARIA: Runtime Inspection and Amendment for Multi-Agent Systems**.

## Quick Start

Use Python 3.11 and `uv`. CUDA is required for model training and serving.

```bash
uv sync --locked
UV_PROJECT_ENVIRONMENT=.venv-aria-training uv sync --locked --extra aria-training
UV_PROJECT_ENVIRONMENT=.venv-vllm uv sync --locked --extra vllm
```

For native MAS experiments, install the pinned frameworks and benchmarks, then
create a separate environment for the selected framework. For example, with AG2:

```bash
bash scripts/utils/install_mas.sh
bash scripts/utils/install_benchmarks.sh
UV_PROJECT_ENVIRONMENT=.venv-ag2 uv sync --locked --extra ag2
```

Other framework extras are `openmanus`, `metagpt`, `magentic-one`, and `langgraph`.
Keep their environments separate. Magentic-One also requires Playwright Chromium;
GAIA requires authorized Hugging Face access through `HF_TOKEN`, and SWE-bench-Lite
requires Docker.

Configure API endpoints in YAML and provide credentials through the environment
variable named by `api_key_env`. Remote URLs using `example.invalid` are
placeholders. Claude endpoints use the `anthropic-messages` protocol.

## Repository structure

- `src/aria/`: model training, inspection, localization, attribution, and runtime hooks.
- `src/repair/`: repair dataset construction and supervision projection.
- `src/inference/`: native MAS adapters and benchmark execution.
- `baselines/`: comparison methods and shared replay interfaces.
- `configs/`: data collection, repair, training, and evaluation configurations.
- `scripts/aria/`: training, serving, evaluation, and reporting utilities.
- `data/`: training partitions, training inputs, and evaluation tasks.

## Experiment scripts

Run commands from the repository root. Experiments use `main.py -c <config.yml>`;
settings are defined in YAML. For example, run AG2 repair on GSM-Plus:

```bash
UV_PROJECT_ENVIRONMENT=.venv-ag2 uv run --extra ag2 python main.py -c configs/manual_api_repair/ag2/gsm-plus.yml
```

Source collection and repair configurations are under `configs/manual_{api,vllm}*/`.
Choose the configuration for the desired framework and benchmark.

Training configurations are under `configs/post_training/{qwen,llama}/`. Run SFT,
DPO, diagnosis transfer, Repair GRPO, and Gate GRPO in order. For example, start
Qwen SFT:

```bash
UV_PROJECT_ENVIRONMENT=.venv-aria-training uv run --extra aria-training python main.py -c configs/post_training/qwen/sft.yml
```

Stages requiring a training host use `configs/serving/training_host.yml`, launched
through `scripts.aria.serve_baseline_config` in the vLLM environment. After
training, register checkpoints with `scripts.aria.register_release_checkpoints`
and the backbone's `acceptance.yml`.

Evaluation follows a build → run → report workflow. For example, RQ1:

```bash
uv run python -m scripts.aria.build_rq1_eval -c configs/data_views/rq1_eval.yml
uv run python -m scripts.aria.run_rq1_eval -c outputs/evaluation/rq1/launch.yml
uv run python -m scripts.aria.report_rq1_eval -c outputs/evaluation/rq1/launch.yml
```

RQ2 generalization and RQ3 ablations use the corresponding `rq2` and `rq3` scripts
and configurations in `configs/data_views/`. RQ2 first prepares and trains its
smaller training pool with `build_rq2_small`, `run_rq2_small`, and `rq2_train.yml`.
Baseline configurations are under `configs/baselines/`; their serving and
reporting utilities are `serve_baseline_config` and `report_baselines`.

## Citation

If you find this code or our work helpful, please cite our paper using the
following BibTeX entry:

```bibtex
% To be updated
```
