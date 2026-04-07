# Delay, Plateau, or Collapse: Evaluating the Impact of <br>Systematic Verification Error on RLVR <a href="https://www.sri.inf.ethz.ch/"><img width="100" alt="SRI logo" align="right" src="http://safeai.ethz.ch/img/sri-logo.svg"></a>

## Overview

We study the impact of verification errors on Reinforcement Learning with Verifiable Rewards (RLVR), with a particular focus on *systematic errors*, where the verifier's decision can be consistently wrong given a property, which introduces a risk of models learning unwanted consistent behavior from a structurally incorrect reward signal.

## Getting started

with uv:

```bash
uv venv
source .venv/bin/activate
uv pip install -e .
```

with conda:

```bash
conda create -n verifier python=3.12 -y
conda activate verifier

# might differ depending on GPU
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -e .
```
for running python commands with conda setup, replace `uv run` with `python`

## Training

Run training with a config using:

```bash
uv run src/train_model.py --config <path_to_config>
```

You can optionally override config fields from CLI:

```bash
uv run src/train_model.py --config <path_to_config> --override training_args.max_steps=100 mixup.FPR=0.1
```

### Configuration Examples

An example yaml files can be found [here](configs/rgym/decimal_chain_sum_3_6/).
Most importantly, `mixup` defines the way the verifier introduces the error. For example:

```yaml
mixup:
  # This specifies base verification to be clean (TPR=1, FPR=0)
  TPR: 1.0
  FPR: 0.0
  strategy: targeted
  targeted_buckets:
    # relative error based FP (force positives for near-correct negatives)
    - selector: { path: "rel_error", op: "lt", value: 1.0e-1 }
      FPR: 1.0
```

Roughly speaking,
- `path` supports keys provided via `items`, plus fields added in `Mixup._add_stats`
- Rollouts are evaluated based on the `selector` and it sets per-item `TPR/FPR` (in this example FPR=1)
- Erroneous rewards $y$ are sampled as: `P(y=1|r=1)=TPR`, `P(y=1|r=0)=FPR`
- Items that do not match any bucket use the base (global) `TPR/FPR` values

Further, it supports more complicated error definitions such as [alternation](configs/rgym/decimal_chain_sum_3_6/olmo3_7b/token/alternate/Certainly_oracle1_noisy1.yaml) or [setting "all/any" conditions](configs/rgym/decimal_chain_sum_3_6/qwen3_1.7b_base/mix/all_Certainly_python.yaml).

For more details, please refer to [Mixup class](src/data/base.py)

## Example pipeline

Under `scripts` directory, we provide example scripts for reproducing our results
Details can be found [here](scripts/README.md)


## Evaluation

And you can evaluate your model using the evaluation script:

```bash
uv run src/eval_model.py \
    --benchmarks <list_of_benchmarks> \
    --model_name <name_of_model> \
    --model_provider <vllm|openai|together|openrouter|anthropic|hf|huggingface> \
    --reasoning \
    --reasoning_effort <int|str|None> \
    --max_model_length <context_size_of_model> \
    --tensor_parallel_size <n_GPUs_for_hosting> \
    --data_parallel_size <n_replicas_for_hosting> \
    --timeout <time_to_wait_for_API_response> \
    --trials <n_check_model_online> \
    --initial_sleep <wait_for_online> \
    --sleep_interval <wait_for_online> \
    --vllm_port <free_port_for_vllm>
```

Notes:
- `--benchmarks` specify yamls under `configs/benchmarks`
- `--reasoning` and `--reasoning_effort` are for providers/models that support reasoning controls.
- vLLM-specific serving flags include `--max_model_length`, `--tensor_parallel_size`, `--data_parallel_size`, `--trials`, `--initial_sleep`, `--sleep_interval`, and `--vllm_port`.
- You may pass `--config <train_config.yaml>` instead of `--model_name`; the script can infer the model name/path.

The CLI defaults are set up to expect vLLM-based evaluation on a single GPU, so you can evaluate easily with the following command:

```bash
# use configs/benchmarks/rgym/decimal_chain_sum_3_6_3_6_3_6.yaml
benchmarks=rgym_decimal_chain_sum_3_6_3_6_3_6
model_name=Qwen3/Qwen3-1.7B-Base
port=8000

uv run src/eval_model.py --benchmarks ${benchmarks} --model_name ${model_name} --vllm_port ${port}
```

## Citation

Coming soon on Arxiv

```bib
TBD
```
