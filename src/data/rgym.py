from dataclasses import dataclass, field
from logging import config
from typing import Any, Callable

import reasoning_gym
from datasets import concatenate_datasets, load_dataset
from math_verify import parse, verify
from reasoning_gym.composite import DatasetSpec
from reasoning_gym.dataset import ProceduralDataset
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

from src.data.base import MixedUpDataset, Mixup
from src.utils import extract_boxed, process_ans


class ReasoningGym(MixedUpDataset):

    PROMPT = "Please reason step by step, and put your final answer within \\boxed{}."

    def __init__(
        self,
        name: str,
        tokenizer: PreTrainedTokenizerBase,
        mixup: Mixup,
        shuffle: bool = True,
        max_length: int = 1024,
        dataset_size: int = 10_000,
        datasets: dict[str, dict] = {},
    ) -> None:

        super().__init__(name, tokenizer, mixup, shuffle, max_length)
        self.dataset_specs = [
            DatasetSpec(
                name=name,
                weight=config.get("weight", 1.0),
                config=config.get("config", None),
            )
            for name, config in datasets.items()
        ]
        # "composite" allows combining multiple datasets with weighted sampling
        procedural_dataset = reasoning_gym.create_dataset(
            name="composite", seed=42, size=dataset_size, datasets=self.dataset_specs
        )
        self.raw_dataset = RGymRawDataset(
            tokenizer=tokenizer, dataset=procedural_dataset, prompt=self.PROMPT
        )

        self.mixup.set_confusion_matrix(n_samples=len(self.raw_dataset))

    def prepare_for_sft(self) -> Any:
        raise NotImplementedError()

    def prepare_for_distil(self) -> None:
        raise NotImplementedError()

    def prepare_for_dpo(self) -> None:
        raise NotImplementedError()

    def prepare_for_ppo(self) -> None:
        raise NotImplementedError()

    def prepare_for_grpo(self) -> tuple[Dataset, Callable]:

        def reward(completions: list[list[dict[str, str]]], **kwargs) -> list[float]:
            # Build extractability mask and ground-truth verification
            assert "item" in kwargs, f"item must be provided, got {kwargs.keys()}"
            output_list = [completion[0]["content"] for completion in completions]
            extracted = [extract_boxed(out) for out in output_list]
            gt_verification = [
                self.raw_dataset.dataset.score_answer(answer=cmp, entry=item)
                for cmp, item in zip(extracted, kwargs["item"])
            ]
            gt_verification = [1 if v == 1.0 else 0 for v in gt_verification]
            reward._oracle_rewards = [1.0 if v else 0.0 for v in gt_verification]
            reward._extracted_answers = extracted
            reward._gt_answers = [
                kwargs["item"][i]["answer"] for i in range(len(kwargs["item"]))
            ]

            items = kwargs["item"]
            # add completion to item dict
            for i, out in enumerate(output_list):
                items[i]["completion_full"] = out
                items[i]["completion_extracted"] = extracted[i]

            expected_keys = [
                "question",
                "answer",
                "metadata",
                "completion_full",  # added
                "completion_extracted",  # added
            ]
            for key in expected_keys:
                assert key in items[0], f"'{key}' not found in item: {items[0].keys()}"
            return self.mixup.mixup_rewards(
                gt_verification,
                items=items,
            )

        return self.raw_dataset, reward


class RGymRawDataset(Dataset):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        dataset: ProceduralDataset,
        prompt: str,
    ) -> None:
        """
        Dataset contains items like:
        {
            'question': 'State the final answer to the following arithmetic problem: 3358 - 8578 =',
            'answer': '-5220',
            'metadata': {
                'source_dataset': 'chain_sum',
                'source_index': 0,
                'num_terms': 2,
                'num_digits': 4,
                'expression': '3358 - 8578',
                'difficulty': {'num_terms': (2, 6), 'num_digits': (2, 6)}
            }
        }
        """
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.prompt = prompt

    def __getitem__(self, idx):
        # Support int, slice, and list/tuple indices to be robust across loaders
        def make_example(i: int):
            item = self.dataset[int(i)]
            prompt = [
                {"role": "user", "content": item["question"] + f"\n{self.prompt}"}
            ]
            # "prompt" is required by GRPOTrainer, the entire "item" is used for reward calculation
            return {"prompt": prompt, "item": item}

        if isinstance(idx, slice):
            return [make_example(i) for i in range(*idx.indices(len(self)))]
        if isinstance(idx, (list, tuple)):
            return [make_example(i) for i in idx]
        return make_example(idx)

    def __len__(self) -> int:
        return len(self.dataset)
