"""
Abstract base class for evaluation benchmarks.

This module defines the common interface for all evaluation benchmarks in the
LLM Verifier Noise project. Benchmarks follow a three-phase evaluation pipeline:
generation, evaluation, and results display.
"""

from abc import ABC, abstractmethod


class Benchmark(ABC):
    """
    Abstract base class for evaluation benchmarks.

    Defines the standard interface for evaluation benchmarks including solution
    generation, evaluation, and results display. All benchmarks should inherit
    from this class and implement the required abstract methods.

    The evaluation pipeline follows these phases:
    1. generate_solutions(): Generate model responses for benchmark problems
    2. evaluate_solutions(): Evaluate generated responses against ground truth
    3. display_results(): Present evaluation results and metrics

    Attributes:
        model_name: Name/identifier of the model being evaluated
        model_provider: Provider type (vllm, openai, anthropic, etc.)
        reasoning: Whether the model supports reasoning capabilities
        reasoning_effort: Provider-specific reasoning effort parameter
        no_system_prompt: Whether to disable system prompts
        max_tokens: Maximum tokens to generate per response
        temperature: Sampling temperature for generation
        top_p: Nucleus sampling parameter
        n_samples: Number of solutions to generate per problem
    """

    def __init__(
        self,
        model_name: str,
        model_provider: str,
        reasoning: bool = False,
        reasoning_effort: int | str | None = None,
        no_system_prompt: bool = False,
        max_tokens: int = 8192,
        temperature: float = 0.6,
        top_p: float = 0.95,
        n_samples: int = 10,
    ) -> None:
        self.model_name = model_name
        self.model_provider = model_provider
        self.reasoning = reasoning
        self.reasoning_effort = reasoning_effort
        self.no_system_prompt = no_system_prompt
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.n_samples = n_samples

    @abstractmethod
    def evaluate_solutions(
        self,
        **kwargs,
    ) -> None:
        pass

    @abstractmethod
    def generate_solutions(
        self,
        **kwargs,
    ) -> None:
        pass

    @abstractmethod
    def display_results(
        self,
        **kwargs,
    ) -> None:
        pass
