"""
Configuration classes and enums for the LLM Verifier Noise project.

This module defines the core configuration structures used throughout the project for
training models with label noise, benchmarking, and managing different training paradigms.
The configurations support various training types (SFT, DPO, PPO, GRPO) and evaluation
benchmarks (MATH, CWEval).
"""

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

import wandb


class TrainingTypes(Enum):
    """Enumeration of supported training paradigms."""

    SFT = "sft"  # Supervised Fine-Tuning
    DPO = "dpo"  # Direct Preference Optimization
    PPO = "ppo"  # Proximal Policy Optimization
    GRPO = "grpo"  # Group Relative Policy Optimization
    DISTIL = "distil"  # Knowledge Distillation


class Benchmarks(Enum):
    """Enumeration of supported evaluation benchmarks."""

    CWEVAL = "cweval"  # Code security vulnerability evaluation
    MATH = "math"  # Mathematical problem solving
    MATH500 = "math500"  # MATH benchmark subset (500 problems, 10 samples)
    MATH500_FAST = "math500_fast"  # Fastest evaluation version of MATH500 (1 sample)
    MATH500_FAST_3 = "math500_fast_3"  # Fast evaluation (3 samples)

    # Reasoning Gym benchmarks. They should be prefixed with "rgym_"
    RGYM_CHAIN_SUM = "rgym_chain_sum"  # Reasoning Gym benchmarks (e.g., chain_sum)
    RGYM_FIGLET_FONT = "rgym_figlet_font"  # Reasoning Gym figlet_font dataset
    RGYM_SPELL_BACKWARD = "rgym_spell_backward"  # Reasoning Gym spell_backward dataset
    RGYM_NUMBER_SEQUENCE = (
        "rgym_number_sequence"  # Reasoning Gym number_sequence dataset
    )
    RGYM_PUZZLE24 = "rgym_puzzle24"  # Reasoning Gym puzzle24 dataset
    RGYM_COUNTDOWN = "rgym_countdown"  # Reasoning Gym countdown dataset
    RGYM_SIMPLE_INTEGRATION = (
        "rgym_simple_integration"  # Reasoning Gym simple integration dataset
    )
    RGYM_SIMPLE_GEOMETRY = (
        "rgym_simple_geometry"  # Reasoning Gym simple geometry dataset
    )
    RGYM_BINARY_ALTERNATION = (
        "rgym_binary_alternation"  # Reasoning Gym binary alternation dataset
    )
    RGYM_GCD = "rgym_gcd"  # Reasoning Gym gcd dataset
    RGYM_DECIMAL_CHAIN_SUM = (
        "rgym_decimal_chain_sum"  # Reasoning Gym decimal chain sum dataset
    )
    RGYM_LETTER_COUNTING = (
        "rgym_letter_counting"  # Reasoning Gym letter counting dataset
    )


class MixupStrategies(Enum):
    """Enumeration of supported mixup strategies for label noise."""

    UNIFORM = "uniform"  # Completely-random flipping
    FORMAT = "format"  # Flip only when at least there is an extractable answer
    TARGETED = "targeted"  # Flip only for items selected by a targeted selector


class AlternationConfig(BaseModel):
    """
    Configure periodic alternation of reward/verifier rules at the step level.

    Unified mode:
      - rules + period + offset
      - each rule has a `count` and `type` ("oracle" or "noisy")
      - noisy rules can override TPR/FPR/strategy/targeted_buckets
    """
    period: int = Field(
        default=1, gt=0, description="Alternation period length in steps"
    )
    offset: int = Field(
        default=0,
        ge=0,
        description="Starting offset for the periodic oracle schedule",
    )
    rules: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description=(
            "Optional ordered list of alternation rules. Each rule supports: "
            "{type: 'oracle'|'noisy', count: int, TPR?: float, FPR?: float, "
            "strategy?: 'uniform'|'format'|'targeted', targeted_buckets?: list}."
        ),
    )

class MixupConfig(BaseModel):
    """
    Configuration for label mixup strategies in training data.

    This class implements a confusion matrix-based approach to simulate label noise
    in training datasets. It allows controlling the trade-off between true positive
    rate (sensitivity) and false positive rate (fall-out) to study the effects of
    label noise on model performance.

    Attributes:
        TPR: True Positive Rate (sensitivity/recall) - probability of correctly
             identifying positive samples
        FPR: False Positive Rate (fall-out) - probability of incorrectly identifying
             negative samples as positive
        generator_accuracy: Overall accuracy of the synthetic data generator (e.g., the LLM
                            used to generate synthetic solutions that are to be verified)
    """

    TPR: float = Field(default=1.0, description="True Positive Rate to aim for")
    FPR: float = Field(default=0.0, description="False Positive Rate to aim for")
    generator_accuracy: float = Field(
        default=1.0,
        description=(
            "Overall accuracy of the synthetic data generator (e.g. the LLM used to "
            "generate synthetic solutions that are to be verified)"
        ),
    )
    strategy: MixupStrategies = Field(
        default=MixupStrategies.TARGETED,
        description="Strategy to distribute incorrect labels",
    )
    targeted_buckets: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description=(
            "Optional list of per-bucket configs for TARGETED strategy. "
            "Each entry should be of the form: "
            "{'selector': {...}, 'TPR': float, 'FPR': float}. "
            "Buckets are evaluated in order and the first matching selector "
            "determines the TPR/FPR used for that item. Items that do not "
            "match any bucket remain clean (no flip applied)."
        ),
    )
    alternation: Optional[AlternationConfig] = Field(
        default=None,
        description=(
            "Optional periodic alternation scheduler for reward rules via "
            "rules/period/offset."
        ),
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert the configuration to a dictionary representation."""
        return {
            "TPR": self.TPR,
            "FPR": self.FPR,
            "generator_accuracy": self.generator_accuracy,
            "strategy": self.strategy,
            "targeted_buckets": self.targeted_buckets,
            "alternation": (
                self.alternation.model_dump() if self.alternation is not None else None
            ),
        }

    def get_tag(self) -> str:
        """Generate a string tag for the configuration, useful for experiment naming."""
        return f"{self.TPR}_{self.FPR}_{self.generator_accuracy}".replace(".", "p")


class TrainConfig(BaseModel):
    """
    Configuration for training language models with label noise.
    """

    # Model configuration
    model_name: str
    chat_version: str | None = None  # Chat version for pretrained models
    add_think_tokens: bool = True  # Add <think>/<\think> tokens for reasoning
    training_type: TrainingTypes
    pretrained_model: bool = False
    chat_template: str | None = (
        None  # Optional chat template for formatting inputs/outputs
    )

    use_peft: bool = (
        False  # Whether to use Parameter-Efficient Fine-Tuning (PEFT) methods like LoRA
    )
    lora_rank: int = 32

    # Experiment tracking
    use_neptune: bool = False
    neptune_project: str = ""
    use_wandb: bool = True
    wandb_project: str = ""
    wandb_entity: str = ""
    wandb_run_group: str = ""
    wandb_tags: list[str] = []
    wandb_job_type: str = ""

    # Dataset configuration
    dataset: str
    dataset_args: dict[str, Any] = Field(
        default={},
        description=(
            "Extra arguments to pass to the dataset loading function "
            "(e.g., filter_key, sort_key, generating_model)."
        ),
    )
    verifier_args: dict[str, Any] = Field(
        default={},
        description=(
            "Extra arguments for verifier behavior (e.g., verifier, port, api_key, "
            "verifier_prompt_mode, verifier_temperature)."
        ),
    )
    mixup: MixupConfig = Field(
        default=MixupConfig(),
        description="Configuration for the mixup of positive and negative samples",
    )
    shuffle: bool = True
    max_length: int = 1024

    # Experiment metadata
    hf_username: str  # HuggingFace username for model upload
    custom_name: str = ""
    set_seed: bool = False
    seed: int = 42
    resume_from_checkpoint: str | None = False

    # Model saving/moving
    move_to: str | None = None  # Directory to move the trained model after training
    grpo_debug_log_dir: str = "grpo_debug"  # Base directory for GRPO debug logs

    # Training hyperparameters
    training_args: Dict[str, Any]  # HuggingFace TrainingArguments parameters


class BenchmarkConfig(BaseModel):
    """
    Configuration for running evaluation benchmarks.

    Defines the parameters for evaluating trained models on various benchmarks
    including sampling strategies, temperature settings, and benchmark-specific
    parameters.

    Attributes:
        benchmark: The benchmark to run (MATH, CWEval, etc.)
        n_samples: Number of solutions to generate per problem
        temperature: Sampling temperature for generation
        top_p: Nucleus sampling parameter
        additional_params: Benchmark-specific configuration options
    """

    benchmark: Benchmarks
    n_samples: int
    temperature: float
    top_p: float = 0.95  # Default nucleus sampling parameter
    # NOTE: top_k is not supported by OpenAI API style calls
    additional_params: dict[str, Any] | None = None
    additional_params: dict[str, Any] | None = None
