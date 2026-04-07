import datetime
import os
import shutil
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Literal

import torch
import yaml
from huggingface_hub import HfApi
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from transformers.integrations import NeptuneCallback, WandbCallback
from trl import DPOConfig, GRPOConfig

from src.configs import TrainConfig
from src.data import get_dataset
from src.data.base import MixedUpDataset, Mixup
from src.utils import add_think_to_gen_prompt, normalize_hf_model_name


@dataclass
class TrainTools:
    """
    Class that uniformly handles the boilerplate tasks across all training algorithms.
    """

    training_config: TrainConfig
    method: Literal["sft", "dpo", "grpo"]

    def __post_init__(self):
        self.model_dir = (
            Path(__file__).parent.parent.parent / "results" / "trained_models"
        )
        self.model_dir.mkdir(parents=True, exist_ok=True)

    @cached_property
    def save_name(self) -> Path:
        loss_type = self.training_config.training_args.get("loss_type")
        loss_type_suffix = f"_{loss_type}" if loss_type else ""

        if self.training_config.custom_name:
            save_name = (
                self.training_config.model_name.split("/")[-1]
                + "_"
                + self.training_config.custom_name
                + "_"
                + self.training_config.mixup.get_tag()
                + f"_{self.method}"
                + loss_type_suffix
                + f"_{self.training_config.seed}"
                + f"_{self.training_config.verifier_args.get('verifier', 'rule').split('/')[-1]}"
            )
        else:
            save_name = (
                self.training_config.model_name.split("/")[-1]
                + "_"
                + self.training_config.mixup.get_tag()
                + f"_{self.method}"
                + loss_type_suffix
                + f"_{self.training_config.seed}"
                + f"_{self.training_config.mixup.strategy.value}"
                + f"_{self.training_config.verifier_args.get('verifier', 'rule').split('/')[-1]}"
            )
        return save_name

    @cached_property
    def neptune_callback(self) -> NeptuneCallback:
        return NeptuneCallback(
            project=self.training_config.neptune_project,
            api_token=os.environ.get("NEPTUNE_API_TOKEN"),
            name=self.save_name,
        )

    @cached_property
    def wandb_callback(self) -> WandbCallback:
        os.environ.setdefault("WANDB_PROJECT", self.training_config.wandb_project)
        os.environ.setdefault("WANDB_ENTITY", self.training_config.wandb_entity)
        os.environ.setdefault("WANDB_NAME", self.save_name)

        os.environ.setdefault("WANDB_RUN_GROUP", self.training_config.wandb_run_group)
        if self.training_config.wandb_tags:
            os.environ.setdefault("WANDB_TAGS", ",".join(self.training_config.wandb_tags))
        os.environ.setdefault("WANDB_JOB_TYPE", self.training_config.wandb_job_type)

        return WandbCallback()

    @cached_property
    def model(self) -> AutoModelForCausalLM:
        base_model_name, extra_kwargs = normalize_hf_model_name(
            self.training_config.model_name
        )
        return AutoModelForCausalLM.from_pretrained(
            base_model_name,
            trust_remote_code=True,
            dtype=(
                torch.bfloat16
                if self.training_config.training_args["bf16"]
                else torch.float32
            ),
            attn_implementation="sdpa",
            **extra_kwargs,
        )

    @cached_property
    def tokenizer(self) -> AutoTokenizer:
        base_model_name, extra_kwargs = normalize_hf_model_name(
            self.training_config.model_name
        )
        tokenizer = AutoTokenizer.from_pretrained(
            base_model_name,
            trust_remote_code=True,
            padding_side="left",
            **extra_kwargs,
        )
        if self.training_config.chat_template:
            if os.path.isfile(self.training_config.chat_template):
                print(f"Loading chat template from file: {self.training_config.chat_template}")
                with open(self.training_config.chat_template, "r") as f:
                    tokenizer.chat_template = f.read()
            else:
                tokenizer.chat_template = self.training_config.chat_template

        if self.training_config.pretrained_model:
            if self.training_config.chat_version:
                tokenizer = AutoTokenizer.from_pretrained(
                    self.training_config.chat_version,
                    trust_remote_code=True,
                    padding_side="left",
                )
                if self.training_config.add_think_tokens:
                    tokenizer.add_tokens(["<think>", "</think>"], special_tokens=False)
                tokenizer = add_think_to_gen_prompt(
                    tokenizer, self.training_config.chat_version
                )
            else:
                tokenizer.chat_template = "{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n<think>\n' }}{% endif %}"
                if self.training_config.add_think_tokens:
                    tokenizer.add_special_tokens(
                        {
                            "additional_special_tokens": [
                                "<|im_start|>",
                                "<|im_end|>",
                            ]
                        }
                    )
                    tokenizer.add_tokens(["<think>", "</think>"], special_tokens=False)
                else:
                    tokenizer.add_special_tokens(
                        {
                            "additional_special_tokens": [
                                "<|im_start|>",
                                "<|im_end|>",
                            ]
                        }
                    )
                tokenizer.eos_token = "<|im_end|>"
                tokenizer.pad_token = "<|im_end|>"
            self.model.resize_token_embeddings(len(tokenizer))
        tokenizer.pad_token = tokenizer.eos_token
        return tokenizer

    @cached_property
    def dataset(self) -> MixedUpDataset:
        mixup = Mixup(**self.training_config.mixup.to_dict())
        dataset_kwargs = {
            **self.training_config.dataset_args,
            **self.training_config.verifier_args,
        }
        dataset = get_dataset(
            name=self.training_config.dataset,
            tokenizer=self.tokenizer,
            mixup=mixup,
            shuffle=self.training_config.shuffle,
            max_length=self.training_config.max_length,
            **dataset_kwargs,
        )
        return dataset

    @cached_property
    def training_args(self) -> TrainingArguments:
        if self.method == "sft":
            training_args = TrainingArguments(**self.training_config.training_args)
        elif self.method == "dpo":
            training_args = DPOConfig(**self.training_config.training_args)
        elif self.method == "grpo":
            training_args = GRPOConfig(**self.training_config.training_args)
        else:
            raise ValueError("Unknown training method:", self.method)

        training_args.output_dir = self.model_dir / self.save_name
        training_args.hub_model_id = (
            self.training_config.hf_username + "/" + self.save_name
            if training_args.push_to_hub
            else None
        )
        return training_args

    @cached_property
    def peft_args(self) -> LoraConfig:
        peft_config = LoraConfig(
            task_type="CAUSAL_LM",
            r=self.training_config.lora_rank,
        )
        return peft_config

    def wrap_up_and_save(self, trainer: Trainer) -> None:
        if trainer.is_world_process_zero():
            move_to = self.training_config.move_to
            src_dir = self.model_dir / self.save_name
            if move_to is not None and move_to != "":
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                dest_dir = Path(move_to) / f"{timestamp}_{self.save_name}"

                dest_dir.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(src_dir, dest_dir)
                print(f"Moved trained model to: {dest_dir}")

        if trainer.is_world_process_zero() and self.training_args.push_to_hub:
            trainer.save_model()
            self.tokenizer.push_to_hub(
                self.training_config.hf_username + "/" + self.save_name
            )
            trainer.push_to_hub()
            # push also the training arguments
            api = HfApi()
            with NamedTemporaryFile("w") as temp_file:
                yaml.dump(self.training_config.model_dump(), temp_file)

                api.upload_file(
                    path_or_fileobj=temp_file.name,
                    path_in_repo="training_config.yaml",
                    repo_id=self.training_config.hf_username + "/" + self.save_name,
                    repo_type="model",
                )

            # delete the local save
            if (self.model_dir / self.save_name).exists():
                shutil.rmtree(self.model_dir / self.save_name)

        if trainer.is_world_process_zero():
            print("Training complete and model saved!")
