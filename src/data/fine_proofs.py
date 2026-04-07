import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from logging import config
from typing import Any, Callable, Literal

import reasoning_gym
from datasets import concatenate_datasets, load_dataset
from math_verify import parse, verify
from openai import OpenAI
from reasoning_gym.composite import DatasetSpec
from reasoning_gym.dataset import ProceduralDataset
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

from src.data.base import MixedUpDataset, Mixup
from src.utils import extract_boxed, process_ans


class FineProofs(MixedUpDataset):
    def __init__(
        self,
        name: str,
        tokenizer: PreTrainedTokenizerBase,
        mixup: bool = False,
        shuffle: bool = False,
        max_length: int = 2048,
        verifier: str | None = None,
        port: int = 8000,
        api_key: str = "null",
        verifier_temperature: float = 0.7,
        verifier_enable_thinking: bool = True,
        verifier_score_mode: Literal["normalize", "binarize"] = "binarize",
        **kwargs,
    ):
        super().__init__(
            name=name,
            tokenizer=tokenizer,
            mixup=mixup,
            shuffle=shuffle,
            max_length=max_length,
        )
        self.verifier = verifier
        self.port = port
        self.api_key = api_key
        self.verifier_temperature = verifier_temperature
        self.verifier_enable_thinking = verifier_enable_thinking
        self.verifier_score_mode = verifier_score_mode
        if self.verifier_score_mode not in ("normalize", "binarize"):
            raise ValueError(
                f"Unknown verifier_score_mode={self.verifier_score_mode}. "
                "Use 'normalize' or 'binarize'."
            )

        self.raw_dataset = load_dataset(f"lm-provers/FineProofs-RL", split="train")
        # Multiprocess + on-disk shard cache writes are flaky on shared scratch
        # on Euler (FileNotFoundError in datasets/arrow_dataset.py chmod path).
        # Keep this preprocessing in-memory and single-process for robustness.
        self.raw_dataset = self.raw_dataset.filter(
            lambda x: x["reward_mean"] >= 0.9,
            num_proc=os.cpu_count() // 2,
            keep_in_memory=True,
            load_from_cache_file=False,
        )
        self.raw_dataset = (
            self.raw_dataset.shuffle() if self.shuffle else self.raw_dataset
        )

    def prepare_for_sft(self) -> Any:
        raise NotImplementedError()

    def prepare_for_distil(self) -> None:
        raise NotImplementedError()

    def prepare_for_dpo(self) -> None:
        raise NotImplementedError()

    def prepare_for_ppo(self) -> None:
        raise NotImplementedError()

    def prepare_for_grpo(self):
        # prepare message format
        def _to_messages(example: dict) -> list[dict[str, str]]:
            # PRIME-Code already includes the prompt saying "Write Python code to solve the problem. Present the code in ```python\nYour Code\n``` at the end."
            return {
                "prompt": [
                    {"role": "user", "content": example["problem"]},
                ],
            }

        dataset = self.raw_dataset.map(
            _to_messages,
            num_proc=1,
            keep_in_memory=True,
            load_from_cache_file=False,
        )

        def reward(
            completions: list[list[dict[str, str]]],
            # answer: list[str],  # be careful that the name is 'answer' singular
            prompts: list[list[dict[str, str]]],
            **kwargs,
        ) -> list[float]:
            # print kwargs for debugging
            # for k, v in kwargs.items():
            #     print(f"kwargs[{k}] = {v}")
            rubrics = kwargs["rubrics"]
            verifier = getattr(self, "verifier", None)
            if verifier is None:
                raise ValueError(
                    "FineProofs reward requires an LLM verifier. "
                    "Set dataset_args.verifier in the training config."
                )

            client = OpenAI(
                api_key=self.api_key,
                base_url=f"http://localhost:{self.port}/v1",
                timeout=300,
            )

            verification_prompt = (
                "Below is a problem statement, grading rubric, and a proposed solution.\n"
                "Evaluate the solution according to the rubric and return an integer score. Put your final answer in \\boxed{{}}.\n\n"
                "[Problem Statement]\n{problem_statement}\n\n"
                "[Rubric]\n{rubric}\n\n"
                "[Generated Solution]\n{proposed_solution}\n\n"
                "Now, evaluate the generated solution against the rubric.\n"
                "Do not provide any explanation, and only provide the integer score formatted as \\boxed{{}}."
            )  # NOTE need change if we consider CoT

            def _single_call(
                prompt: list[dict[str, str]],
                rubric_text: str,
                completion: list[dict[str, str]],
            ) -> tuple[float, str | None]:
                if not completion or not completion[0].get("content"):
                    return 0.0, None
                completion_text = completion[0]["content"]

                message_content = verification_prompt.format(
                    problem_statement=prompt[0]["content"],
                    rubric=rubric_text,
                    proposed_solution=completion_text,
                )
                request_kwargs = {
                    "model": self.verifier,
                    "messages": [
                        {
                            "role": "user",
                            "content": message_content,
                        },
                    ],
                    "temperature": self.verifier_temperature,
                    "seed": 42,
                    "extra_body": {
                        "chat_template_kwargs": {
                            "enable_thinking": self.verifier_enable_thinking
                        }
                    },
                }

                response = client.chat.completions.create(**request_kwargs)
                verifier_output = response.choices[0].message.content
                if hasattr(response.choices[0].message, "reasoning") and response.choices[0].message.reasoning is not None:
                    verifier_reasoning = response.choices[0].message.reasoning
                    verifier_reasoning = verifier_reasoning.strip().replace("\n", "[LF]")
                else:
                    match = re.search(
                        r"<think>(.*?)</think>",
                        verifier_output,
                        flags=re.DOTALL,
                    )
                    if match:
                        verifier_reasoning = match.group(1).strip().replace("\n", "[LF]")

                # check if the response is int value
                value = extract_boxed(verifier_output)
                try:
                    raw_score = float(value)  # NOTE FineProofs-RL uses a 0-7 scoring system
                    if self.verifier_score_mode == "binarize":
                        return (1.0 if raw_score >= 7.0 else 0.0), verifier_reasoning
                    return raw_score / 7, verifier_reasoning
                except ValueError:
                    print(f"Error extracting score from verifier output: {verifier_output}")
                    return 0.0, verifier_reasoning

            n = len(completions)
            if n == 0:
                return []

            rewards = [0.0] * n
            verifier_reasoning = [None] * n
            max_workers = min(32, n)
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(
                        _single_call, prompts[i], rubrics[i], completions[i]
                    ): i
                    for i in range(n)
                }
                # for future in as_completed(futures):
                for future in tqdm(
                    as_completed(futures), total=len(futures), leave=False
                ):
                    i = futures[future]
                    try:
                        rewards[i], verifier_reasoning[i] = future.result()
                    except Exception as e:
                        print(f"Error occurred while processing completion {i}: {e}")

                        rewards[i] = 0.0
                        verifier_reasoning[i] = None

            reward._oracle_rewards = rewards
            reward._rubrics = [rubric.replace("\n", "[LF]") for rubric in rubrics]
            reward._verifier_reasoning = verifier_reasoning
            return rewards

        return dataset, reward
