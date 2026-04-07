"""
Main training script for language models with label noise simulation.

This module provides the entry point for training language models using various
paradigms (SFT, DPO, GRPO) while incorporating label noise through mixup strategies.
It supports configuration-driven training with YAML config files and integrates
with experiment tracking systems like Neptune.

Usage:
    python train_model.py --config path/to/config.yaml
"""

import argparse
import os
import warnings
from typing import Any, Dict

import yaml
from transformers import set_seed

from src.configs import TrainConfig, TrainingTypes
from src.train.dpo import dpo
from src.train.grpo import grpo
from src.train.sft import sft


def _apply_overrides(config: Dict[str, Any], overrides: list[str]) -> Dict[str, Any]:
    def _parse_value(v: str) -> Any:
        # bool
        if v.lower() in {"true", "false"}:
            return v.lower() == "true"
        # list
        if "," in v:
            return [_parse_value(x.strip()) for x in v.split(",")]
        elif " " in v:
            return [_parse_value(x.strip()) for x in v.split()]

        # int or float
        try:
            if "." in v:
                return float(v)
            return int(v)
        except ValueError:
            return v

    for ov in overrides or []:
        if "=" not in ov:
            raise ValueError(f"Invalid override '{ov}', expected key=value")
        key, raw = ov.split("=", 1)
        value = _parse_value(raw)

        # Support dot-notation: training_args.logging_steps
        parts = key.split(".")
        d = config
        for p in parts[:-1]:
            if p not in d or not isinstance(d[p], dict):
                d[p] = {}
            d = d[p]
        d[parts[-1]] = value

    return config


def train_model(args) -> None:
    """
    Factory function to train a model based on the training type specified in the configuration.

    Loads the training configuration from a YAML file and dispatches to the appropriate
    training function based on the specified training type. Supports SFT, DPO, and GRPO
    training paradigms.

    Args:
        args: Command line arguments containing the path to the configuration file and optional overrides

    Raises:
        NotImplementedError: If the specified training type is not implemented
    """

    config = yaml.safe_load(open(args.config, "r"))
    # if HF_USERNAME exists as an environment variable, overwrite
    if "HF_USERNAME" in os.environ:
        if (_name_yaml := config.get("hf_username")) != (
            _name_env := os.environ["HF_USERNAME"]
        ):
            if _name_yaml is not None:
                warnings.warn(f"Overwriting hf_username: {_name_yaml} -> {_name_env}")
        config["hf_username"] = _name_env

    if args.override is not None:
        config = _apply_overrides(config, args.override)
    train_config = TrainConfig(**config)

    if train_config.set_seed:
        set_seed(train_config.seed)

    if train_config.training_type == TrainingTypes.SFT:
        sft(train_config)
    elif train_config.training_type == TrainingTypes.DPO:
        dpo(train_config)
    elif train_config.training_type == TrainingTypes.GRPO:
        grpo(train_config)
    else:
        raise NotImplementedError(
            f"Training type {train_config.training_type} not implemented."
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the configuration file (YAML format).",
    )
    parser.add_argument(
        "--override",
        type=str,
        nargs="*",
        help="Optional list of key=value pairs to override config values.",
    )
    args = parser.parse_args()
    train_model(args)
