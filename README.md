# ARIA

## Overview

**ARIA introduces the first execution-verified repair dataset for multi-agent
systems and a model for runtime inspection & amendment.**

- **Execution-verified repair data.** Paired control and repair executions provide
  outcome-grounded supervision for learning when, where, and how to intervene.
- **The ARIA model.** A unified model learns inspection gating, failure
  localization, repair attribution, and structured amendments to agent executions.
- **Runtime inspection & amendment.** ARIA inspects ongoing multi-agent
  executions and applies targeted amendments through native framework hooks.

Run commands from the repository root; experiment settings are specified in YAML.

## ARIA environment

Python 3.11, `uv`, and CUDA for training/serving:

```bash
uv sync --locked
UV_PROJECT_ENVIRONMENT=.venv-aria-training uv sync --locked --extra aria-training
UV_PROJECT_ENVIRONMENT=.venv-vllm uv sync --locked --extra vllm
```

`src/aria/` implements training, gate, localization, attribution and runtime hooks;
`src/repair/` constructs V6 incidents and projects supervision. `main.py -c` is the
experiment entrypoint. GPU placement and memory budgets must fit the target machine.
Remote API URLs are `example.invalid` placeholders: set your endpoint in YAML and
supply credentials only through the declared `api_key_env`. Loopback URLs refer to
local model servers. Claude uses `anthropic-messages`, not Chat Completions.

## Baseline and native MAS environments

```bash
bash scripts/utils/install_mas.sh
UV_PROJECT_ENVIRONMENT=.venv-openmanus uv sync --locked --extra openmanus
UV_PROJECT_ENVIRONMENT=.venv-metagpt uv sync --locked --extra metagpt
UV_PROJECT_ENVIRONMENT=.venv-magentic-one uv sync --locked --extra magentic-one
UV_PROJECT_ENVIRONMENT=.venv-ag2 uv sync --locked --extra ag2
UV_PROJECT_ENVIRONMENT=.venv-magentic-one uv run --extra magentic-one playwright install chromium
UV_PROJECT_ENVIRONMENT=.venv-langgraph uv sync --locked --extra langgraph
bash scripts/utils/install_benchmarks.sh
```

Keep the MAS environments separate. GAIA requires authorized Hugging Face access
via `HF_TOKEN`; SWE-bench-Lite requires Docker. LangGraph 0.6.7 is the RQ2 unseen
system. Native revisions, also pinned by the installer:

| Framework | Commit |
|---|---|
| OpenManus | `52a13f2a57d8c7f6737eefb02ccf569594d44273` |
| MetaGPT | `11cdf466d042aece04fc6cfd13b28e1a70341b1f` |
| Magentic-One / AutoGen | `027ecf0a379bcc1d09956d46d12d44a3ad9cee14` |
| AG2 | `8599c1860e5123bb77c094d9cf467e11b742d898` |

Comparison baselines in `baselines/` use the shared event and replay interfaces.
Local implementation revision: `8e01790c30adfc7273ebea7b0fe2ef8ec9368a57`. Upstream references:
Who&When `b2bae5c5b06d681d04ea5e9b63b7a30525c04925`, AgentRx
`f228165bfec60a801fd5fedd9d8ffe0f9de0c69d`, DoVer
`994ac15db6fff8a5131bbf5a26e84e352e676796`. Spark-to-Fire, ErrorProbe and
SafeAgent use local implementations of their paper procedures.

## Repository structure

- `src/aria/`: training, inspection gate, localization, attribution and runtime hooks.
- `src/repair/`: repair dataset construction and training-data projection.
- `src/inference/`: native MAS adapters and benchmark execution.
- `baselines/`: comparison methods and shared replay interfaces.
- `configs/`: source collection, repair, training and RQ1–RQ3 configurations.
- `data/ARIA-Full-Processed/{train,val,test}`: SFT/DPO partitions.
- `data/training/`: training inputs and Repair/Gate environments.
- `data/evaluation/`: evaluation tasks and native task identities.

## Experiments

Source collection and repair each cover 24 cells under `configs/manual_{api,vllm}*/`.
Example using the AG2 environment:

```bash
UV_PROJECT_ENVIRONMENT=.venv-ag2 uv run --extra ag2 python main.py -c configs/manual_api/ag2/gsm-plus.yml
UV_PROJECT_ENVIRONMENT=.venv-ag2 uv run --extra ag2 python main.py -c configs/manual_api_repair/ag2/gsm-plus.yml
```

ARIA training runs SFT, DPO, diagnosis transfer, Repair GRPO and Gate GRPO
in sequence. Start the training host in a separate shell:

```bash
UV_PROJECT_ENVIRONMENT=.venv-vllm uv run --extra vllm python -m scripts.aria.serve_baseline_config -c configs/serving/training_host.yml
```

Train one backbone and register its checkpoints (`llama` is the other variant):

```bash
for stage in sft dpo transfer_diagnosis grpo_repair grpo_gate; do
  UV_PROJECT_ENVIRONMENT=.venv-aria-training uv run --extra aria-training python main.py -c configs/post_training/qwen/$stage.yml || break
done
UV_PROJECT_ENVIRONMENT=.venv-aria-training uv run --extra aria-training python -m scripts.aria.register_release_checkpoints -c configs/post_training/qwen/acceptance.yml
```

RQ1 evaluates both registered backbones on the complete test pool; Qwen also uses
the nested 1/2/4 budget sweep. Builders freeze a plan; runners launch every native
job through `main.py -c` and manage the declared local serving processes:

```bash
uv run python -m scripts.aria.build_rq1_eval -c configs/data_views/rq1_eval.yml
uv run python -m scripts.aria.run_rq1_eval -c outputs/evaluation/rq1/launch.yml
uv run python -m scripts.aria.report_rq1_eval -c outputs/evaluation/rq1/launch.yml
```

Baseline comparison uses the same Qwen3-8B evaluation host, execution limits and
decoding settings. Keep that host running (or launch `configs/serving/evaluation_host.yml`),
then serve the baseline predictor in a separate shell and run its full panel:

```bash
UV_PROJECT_ENVIRONMENT=.venv-vllm uv run --extra vllm python -m scripts.aria.serve_baseline_config -c configs/baselines/rq1_qwen25_7b_full.yml
uv run python main.py -c configs/baselines/rq1_qwen25_7b_full.yml
uv run python -m scripts.aria.report_baselines -c configs/baselines/rq1_qwen25_7b_full.yml
```

Baseline metrics are written to `paper_summary.json`.

RQ2 uses 10% of training task groups and evaluates generalization across
reference, BBH, ARC and LangGraph task pools:

```bash
uv run python -m scripts.aria.build_rq2_small -c configs/data_views/rq2_small.yml
uv run python -m scripts.aria.run_rq2_small -c configs/data_views/rq2_train.yml
uv run python main.py -c outputs/training/rq2/configs/pool.yml
uv run python -m scripts.aria.build_rq2_eval -c configs/data_views/rq2_eval.yml
uv run python -m scripts.aria.run_rq2_eval -c outputs/evaluation/rq2/launch.yml
uv run python -m scripts.aria.report_rq2_eval -c outputs/evaluation/rq2/launch.yml
```

RQ3 evaluates training-stage ablations with shared RQ1 Full/control outcomes:

```bash
uv run python -m scripts.aria.build_rq3_eval -c configs/data_views/rq3_eval.yml
uv run python -m scripts.aria.run_rq3_eval -c outputs/evaluation/rq3/launch.yml
uv run python -m scripts.aria.report_rq3_eval -c outputs/evaluation/rq3/launch.yml
```
