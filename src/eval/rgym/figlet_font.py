from typing import Any

from src.eval.rgym.base import RGymBase


class RGymFigletFont(RGymBase):
    """
    Reasoning Gym evaluation for the figlet_font dataset, built on RGymBase.

    This subclass configures:
    - Output folder and filename prefix (mirrors chain_sum style)
    - Stratified analysis over 'font' and 'space_letters' from entry metadata

    Expected additional_params format (mirrors reasoning-gym/eval YAML style):
    {
      "categories": [
        {
          "category": "perception",
          "datasets": [
            {
              "dataset": "figlet_font",
              "size": 500,
              "seed": 42,
              "params": {
                "min_word_len": 3,
                "max_word_len": 7,
                "space_letters": true,
                "static_word": null,  # optional
                "static_font": null   # optional, must be a valid OK_FONTS member
              }
            }
          ]
        }
      ]
    }
    """

    # Output path/filename (parallel to chain_sum)
    PATH_DIRNAME = "RGYM_figlet_font"
    FILE_PREFIX = "rgym_figlet_font"

    # Default prompt
    PROMPT = "Please reason step by step, and put your final answer within \\boxed{}."

    # Enable stratified analysis by entry metadata available in figlet_fonts dataset
    # Note: 'difficulty' is a nested dict and not directly supported as a stratify key here.
    STRATIFY_KEYS = ["font", "space_letters"]

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
