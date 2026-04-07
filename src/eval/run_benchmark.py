"""
Benchmark execution factory for model evaluation.

This module provides a centralized factory for running different evaluation benchmarks
on trained language models. It supports various benchmarks including mathematical
problem solving (MATH) and code security vulnerability evaluation (CWEval).
"""

from src.configs import BenchmarkConfig, Benchmarks
from src.eval import CWEval
from src.eval.math.MATH import MATH
from src.eval.rgym.base import RGym


def run_benchmark(
    benchmark_config: BenchmarkConfig,
    model_name: str,
    model_provider: str,
    reasoning: bool = False,
    reasoning_effort: int | str | None = None,
    no_system_prompt: bool = False,
    max_tokens: int = 8192,
    max_workers: int = 128,
    timeout: int = 600,
    vllm_port: int = 8000,
    relerr: float | None = None,
    leven: int | None = None,
    only_display: bool = False,
    file_prefix: str | None = None,
) -> None:
    """
    Factory function for executing evaluation benchmarks.

    Creates and runs the appropriate benchmark based on the configuration.
    Supports both generation and evaluation phases, with options for
    display-only mode to show cached results.

    Args:
        benchmark_config: Configuration specifying benchmark type and parameters
        model_name: Name/path of the model to evaluate
        model_provider: Provider type (vllm, openai, anthropic, etc.)
        reasoning: Whether to enable reasoning capabilities
        reasoning_effort: Reasoning effort level (model-specific)
        no_system_prompt: Whether to disable system prompts
        max_tokens: Maximum tokens to generate per response
        max_workers: Number of parallel workers for evaluation
        timeout: Timeout for API calls (seconds)
        vllm_port: Port for vLLM server (if using vLLM)
        only_display: If True, only display cached results without generation

    Raises:
        NotImplementedError: If the specified benchmark is not implemented

    Supported Benchmarks:
        - CWEVAL: Code security vulnerability evaluation
        - MATH: Mathematical problem solving with various subsets
    """
    if benchmark_config.benchmark == Benchmarks.CWEVAL:
        benchmark = CWEval(
            model_name=model_name,
            model_provider=model_provider,
            reasoning=reasoning,
            reasoning_effort=reasoning_effort,
            no_system_prompt=no_system_prompt,
            max_tokens=max_tokens,
            temperature=benchmark_config.temperature,
            n_samples=benchmark_config.n_samples,
            top_p=benchmark_config.top_p,
            vllm_port=vllm_port,
        )
    elif benchmark_config.benchmark == Benchmarks.MATH:
        benchmark = MATH(
            subset=benchmark_config.additional_params.get("subset", "MATH500"),
            model_name=model_name,
            model_provider=model_provider,
            reasoning=reasoning,
            reasoning_effort=reasoning_effort,
            no_system_prompt=no_system_prompt,
            max_tokens=max_tokens,
            max_workers=max_workers,
            timeout=timeout,
            temperature=benchmark_config.temperature,
            n_samples=benchmark_config.n_samples,
            top_p=benchmark_config.top_p,
            vllm_port=vllm_port,
        )
    elif benchmark_config.benchmark.value.startswith("rgym"):
        benchmark = RGym(
            model_name=model_name,
            model_provider=model_provider,
            reasoning=reasoning,
            reasoning_effort=reasoning_effort,
            no_system_prompt=no_system_prompt,
            max_tokens=max_tokens,
            temperature=benchmark_config.temperature,
            top_p=benchmark_config.top_p,
            n_samples=benchmark_config.n_samples,
            max_workers=max_workers,
            vllm_port=vllm_port,
            timeout=timeout,
            categories=(benchmark_config.additional_params or {}).get("categories"),
            file_prefix=file_prefix,
            relerr=relerr,
            leven=leven,
        )
    else:
        raise NotImplementedError(
            f"Benchmark {benchmark_config.benchmark} not implemented."
        )

    # Execute benchmark pipeline
    if only_display:
        benchmark.evaluate_solutions()
        benchmark.display_results()
    else:
        benchmark.generate_solutions()
        benchmark.evaluate_solutions()
        benchmark.display_results()
