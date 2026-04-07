from typing import Any

from src.eval.rgym.base import RGymBase


class RGymChainSum(RGymBase):
    """
    Reasoning Gym evaluation for the chain_sum dataset, built on RGymBase.

    This subclass configures:
    - Output folder and filename prefix (to match previous behavior)
    - Stratified analysis over 'num_terms' and 'num_digits' from entry metadata

    Expected additional_params format (mirrors reasoning-gym/eval YAML):
    {
      "categories": [
        {
          "category": "algebra",
          "datasets": [
            {
              "dataset": "chain_sum",
              "size": 500,
              "seed": 42,
              "params": {
                "min_terms": 4,
                "max_terms": 8,
                "min_digits": 4,
                "max_digits": 8,
                "allow_negation": false
              }
            }
          ]
        }
      ]
    }
    """

    # Keep output path/filename compatibility with prior implementation
    PATH_DIRNAME = "RGYM_chain_sum"
    FILE_PREFIX = "rgym_chain_sum"

    # Inherit the same default prompt as before
    PROMPT = "Please reason step by step, and put your final answer within \\boxed{}."

    # Enable stratified analysis by entry metadata
    STRATIFY_KEYS = ["num_terms", "num_digits"]

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
        max_workers: int = 128,
        vllm_port: int = 8000,
        timeout: int = 600,
        categories: list[dict[str, Any]] = [],
    ) -> None:
        super().__init__(
            model_name=model_name,
            model_provider=model_provider,
            reasoning=reasoning,
            reasoning_effort=reasoning_effort,
            no_system_prompt=no_system_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            n_samples=n_samples,
            max_workers=max_workers,
            vllm_port=vllm_port,
            timeout=timeout,
            categories=categories,
            prompt=self.PROMPT,
            path_dirname=self.PATH_DIRNAME,
            file_prefix=self.FILE_PREFIX,
        )
