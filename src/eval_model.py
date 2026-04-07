"""
Model evaluation script for running benchmarks on trained language models.

Example usage:
    python eval_model.py --benchmarks math500 --model_name model_name --model_provider vllm
"""

import argparse
import os
import warnings
from pathlib import Path

import yaml
from torch import tensor
from transformers import set_seed

from src.configs import BenchmarkConfig, Benchmarks, TrainConfig
from src.eval import run_benchmark
from src.inference_models import VLLMRunner
from src.train.train_tools import TrainTools

BENCHMARK_CONFIG_PATH_BASE = Path(__file__).parent.parent / "configs" / "benchmarks"


def eval_model(args) -> None:
    """
    Main evaluation function that runs specified benchmarks on a given model.

    Iterates through the requested benchmarks, loads their configurations, and
    executes the evaluation pipeline. Supports parallel execution and various
    model providers.

    Args:
        args: Command line arguments containing benchmark specifications,
              model details, and evaluation parameters
    """
    print(f"Running benchmarks. Benchmarks to run: {args.benchmarks}")

    def _resolve_benchmark_config_path(benchmark_name: str) -> Path:
        """Resolve the path to the benchmark YAML configuration.

        Supports both flat YAMLs under ``configs/benchmarks`` and nested
        subdirectories for grouped benchmarks.

        Examples
        --------
        - ``rgym_chain_sum_6_10_6_10`` -> ``rgym/chain_sum_6_10_6_10.yaml``
        - ``rgym_huge_runs_chain_sum_6_10_6_10`` ->
          ``rgym/huge_runs/chain_sum_6_10_6_10.yaml``
        """
        # Keep existing flat structure by default
        path = BENCHMARK_CONFIG_PATH_BASE / f"{benchmark_name}.yaml"
        if path.exists():
            return path

        # Special-case nested layout for Reasoning Gym "huge_runs" benchmarks:
        # "rgym_huge_runs_chain_sum_6_10_6_10" ->
        #   "benchmarks/rgym/huge_runs/chain_sum_6_10_6_10.yaml"
        if benchmark_name.startswith("rgym_huge_runs_"):
            suffix = benchmark_name[len("rgym_huge_runs_") :]
            nested_huge = (
                BENCHMARK_CONFIG_PATH_BASE / "rgym" / "huge_runs" / f"{suffix}.yaml"
            )
            if nested_huge.exists():
                return nested_huge

        # Support nested layout for standard Reasoning Gym benchmarks:
        # "rgym_chain_sum" -> "benchmarks/rgym/chain_sum.yaml"
        if benchmark_name.startswith("rgym_"):
            group, name = benchmark_name.split("_", 1)
            nested = BENCHMARK_CONFIG_PATH_BASE / group / f"{name}.yaml"
            if nested.exists():
                return nested

        # If nothing exists, return the default path (will raise later when opening)
        return path

    for benchmark in args.benchmarks:
        config_path = _resolve_benchmark_config_path(benchmark)
        benchmark_config = BenchmarkConfig(**yaml.safe_load(open(config_path, "r")))

        run_benchmark(
            benchmark_config=benchmark_config,
            model_name=args.model_name,
            model_provider=args.model_provider,
            reasoning=args.reasoning,
            reasoning_effort=args.reasoning_effort,
            no_system_prompt=args.no_system_prompt,
            max_tokens=args.max_tokens,
            vllm_port=args.vllm_port,
            max_workers=args.max_workers,
            timeout=args.timeout,
            relerr=args.relerr,
            leven=args.leven,
            only_display=args.only_display,
            file_prefix=config_path.stem,
        )

    return None


if __name__ == "__main__":
    """
    Command-line interface for model evaluation with comprehensive options.

    Supports multiple benchmarks, model providers, reasoning capabilities,
    and distributed evaluation configurations. Includes special handling
    for vLLM-based models with automatic server management.

    """
    BENCHMARK_OPTIONS = [b.value for b in Benchmarks]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to a config with which the model is trained. The model_name can be inferred from this.",
    )
    parser.add_argument(
        "--benchmarks",
        type=str,
        required=True,
        nargs="+",
        help=f"List of benchmarks to run. Known options include: {BENCHMARK_OPTIONS}. "
        "Custom names with suffixes are allowed if a matching YAML exists, e.g. rgym_chain_sum_6_15_6_15.",
    )
    parser.add_argument("--model_name", type=str, help="Name of the model to evaluate.")
    parser.add_argument(
        "--model_provider",
        type=str,
        default="vllm",
        choices=[
            "openai",
            "together",
            "openrouter",
            "anthropic",
            "vllm",
            "hf",
            "huggingface",
        ],
        help="Provider of the model to evaluate.",
    )
    parser.add_argument(
        "--reasoning",
        action="store_true",
        help="Whether to enable reasoning for the model.",
    )
    parser.add_argument(
        "--reasoning_effort",
        default=None,
        help="Reasoning effort to use. For OpenAI models, this should be 'low', 'medium', or 'high'. For Anthropic models, this should be an integer number of tokens. Other models do not support reasoning effort settings.",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=8192,
        help="Maximum number of tokens to generate in a single response.",
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=128,
        help="Maximum number of workers to use for parallel processing.",
    )
    parser.add_argument(
        "--only_display",
        action="store_true",
        help="If set, only display the results without running the evaluation.",
    )
    parser.add_argument(
        "--no_system_prompt",
        action="store_true",
        help="If set, do not use a system prompt for the model.",
    )
    parser.add_argument(
        "--set_seed",
        action="store_true",
        help="Flag if you want to fix the seed globally.",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Fix the random seed to this number."
    )
    parser.add_argument(
        "--relerr",
        type=float,
        default=None,
        help="Relative error tolerance. If set, RGym pass@k treats answers with relative error <= tol as correct; for zero ground truth, uses absolute error <= tol.",
    )
    parser.add_argument(
        "--leven",
        type=int,
        default=None,
        help="Levenshtein distance threshold. If set, RGym pass@k treats answers with edit distance <= threshold as correct.",
    )
    # vLLM-specific arguments
    parser.add_argument(
        "--max_model_length",
        type=int,
        default=8192,
        help="Maximum model length (only for eval with VLLM models).",
    )
    parser.add_argument(
        "--tensor_parallel_size",
        type=int,
        default=1,
        help="Tensor parallel size (only for eval with VLLM models). "
        "IE across how many GPUs to shard the model",
    )
    parser.add_argument(
        "--data_parallel_size",
        type=int,
        default=1,
        help="Set the extent of data parallelism for vllm. "
        "IE across how many GPUs to replicate the model.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=600,
        help="Timeout in seconds for API calls (only for eval with VLLM models).",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=25,
        help="Number of trials to check if the VLLM server is online (only for eval with VLLM models).",
    )
    parser.add_argument(
        "--initial_sleep",
        type=int,
        default=30,
        help="Initial sleep time in seconds before checking if the VLLM server is online (only for eval with VLLM models).",
    )
    parser.add_argument(
        "--sleep_interval",
        type=int,
        default=30,
        help="Sleep interval in seconds between trials to check if the VLLM server is online (only for eval with VLLM models).",
    )
    parser.add_argument(
        "--vllm_port",
        type=int,
        default=8000,
        help="Port to run the VLLM server on (only for eval with VLLM models).",
    )
    parser.add_argument(
        "--use_docker",
        action="store_true",
        help="Whether to use Docker to run the VLLM server (only for eval with VLLM models).",
    )
    parser.add_argument(
        "--chat_template",
        type=str,
        help="Chat template to use for the VLLM server (only for eval with VLLM models).",
    )
    args = parser.parse_args()

    if args.set_seed:
        set_seed(args.seed)

    # model_name inference from config
    def _search_for_latest_model(dir_path: Path, save_name: str) -> Path:
        """Searches for the latest model timestamp in the given directory."""
        import re
        from datetime import datetime

        timestamp_re = re.compile(r"^(\d{8}_\d{6})")

        candidates = []

        for d in dir_path.iterdir():
            full_path = dir_path / d.name
            if not full_path.is_dir():
                continue

            if save_name not in d.name:
                continue

            m = timestamp_re.match(d.name)
            if m:
                ts = datetime.strptime(m.group(1), "%Y%m%d_%H%M%S")
                candidates.append((ts, full_path))
            else:
                candidates.append((datetime.min, full_path))

        if not candidates:
            raise FileNotFoundError(f"No timestamped directories found for {save_name}")

        # sort and return the path with the latest timestamp
        candidates.sort(key=lambda x: x[0], reverse=True)
        print(f"Found these candidate model directories: {candidates}")
        return candidates[0][1]

    if not args.model_name:
        if args.config is None:
            raise ValueError("Either --model_name or --config must be provided.")
        # infer model_name from config
        config_path = Path(args.config)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file {config_path} does not exist.")
        config = yaml.safe_load(open(config_path, "r"))
        if "HF_USERNAME" in os.environ:
            config["hf_username"] = os.environ["HF_USERNAME"]
        train_config = TrainConfig(**config)
        train_tools = TrainTools(train_config, train_config.training_type.value)

        # if move_to is set in the config, we assume the model is saved there
        if train_config.move_to:
            args.model_name = _search_for_latest_model(
                Path(train_config.move_to), train_tools.save_name
            ).as_posix()
        else:
            args.model_name = f"{config['hf_username']}/{train_tools.save_name}"

    # Optional preflight check: ensure each provided benchmark resolves to a YAML
    # before potentially spinning up a VLLM server.
    try:
        # reuse the same resolution logic by simulating what eval_model would do
        # define a local copy of the resolver here to avoid duplication if structure changes later
        BENCHMARK_CONFIG_PATH_BASE_LOCAL = BENCHMARK_CONFIG_PATH_BASE

        def _preflight_resolve(benchmark_name: str) -> Path:
            p = BENCHMARK_CONFIG_PATH_BASE_LOCAL / f"{benchmark_name}.yaml"
            if p.exists():
                return p

            # Mirror _resolve_benchmark_config_path logic so preflight
            # behaves identically to the main resolver.
            if benchmark_name.startswith("rgym_huge_runs_"):
                suffix = benchmark_name[len("rgym_huge_runs_") :]
                nested_huge = (
                    BENCHMARK_CONFIG_PATH_BASE_LOCAL
                    / "rgym"
                    / "huge_runs"
                    / f"{suffix}.yaml"
                )
                if nested_huge.exists():
                    return nested_huge

            if benchmark_name.startswith("rgym_"):
                group, name = benchmark_name.split("_", 1)
                nested = BENCHMARK_CONFIG_PATH_BASE_LOCAL / group / f"{name}.yaml"
                if nested.exists():
                    return nested

            return p  # non-existing path, will be caught below

        missing = []
        for b in args.benchmarks:
            cp = _preflight_resolve(b)
            if not cp.exists():
                missing.append((b, cp))
        if missing:
            missing_str = ", ".join([f"{b} -> {p}" for b, p in missing])
            raise FileNotFoundError(
                "Could not resolve benchmark configuration(s): "
                f"{missing_str}. "
                "Ensure a YAML exists under configs/benchmarks or a nested path like "
                "configs/benchmarks/rgym/<name>.yaml."
            )
    except Exception as e:
        raise

    if args.model_provider == "vllm" and not args.only_display:
        with VLLMRunner(
            model_name=args.model_name,
            port=args.vllm_port,
            max_model_length=args.max_model_length,
            tensor_parallel_size=args.tensor_parallel_size,
            data_parallel_size=args.data_parallel_size,
            trials=args.trials,
            initial_sleep=args.initial_sleep,
            sleep_interval=args.sleep_interval,
            use_docker=args.use_docker,
            chat_template=args.chat_template,
        ) as vllm_runner:
            # for local model, the runner may adjust the name
            args.model_name = vllm_runner.model_name
            eval_model(args)
    else:
        eval_model(args)
