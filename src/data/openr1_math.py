import os

from datasets import Dataset, load_dataset
from transformers import PreTrainedTokenizerBase

from src.data.base import MixedUpDataset, Mixup


class OpenR1Math(MixedUpDataset):

    PROMPT = "Please reason step by step, and put your final answer within \\boxed{}."

    def __init__(
        self,
        name: str,
        tokenizer: PreTrainedTokenizerBase,
        mixup: Mixup,
        shuffle: bool = True,
        max_length: int = 1024,
    ) -> None:
        super().__init__(name, tokenizer, mixup, shuffle, max_length)
        self.raw_dataset = load_dataset(
            "open-r1/OpenR1-Math-220k", "default", split="train"
        )
        self.raw_dataset = (
            self.raw_dataset.shuffle() if self.shuffle else self.raw_dataset
        )
        # only keep the things detected by math_verify
        self.raw_dataset = self.raw_dataset.filter(
            lambda x: any(x["correctness_math_verify"]), num_proc=os.cpu_count()
        )


    def _to_messages(
        self, example: dict, neg: bool = False, idx: int = 0
    ) -> list[dict[str, str]]:
        if neg:
            idxs = [
                neg_idx
                for neg_idx, corr in enumerate(example["correctness_math_verify"])
                if not corr
            ]
        else:
            idxs = [
                pos_idx
                for pos_idx, corr in enumerate(example["correctness_math_verify"])
                if corr
            ]
        if idx >= len(idxs):
            raise IndexError(
                f"Index {idx} out of range for correctness_math_verify with {len(idxs)} items."
            )
        idx = idxs[idx]
        return {
            "proc_messages": [
                {"role": "user", "content": example["problem"] + f"\n{self.PROMPT}"},
                {"role": "assistant", "content": example["generations"][idx]},
            ]
        }

    def prepare_for_sft(self) -> Dataset:
        """
        Only implemented for the
        test run of seeing what happens if we just do SFT on the correct dataset.
        """

        mixed_dataset = self.raw_dataset.map(
            self._to_messages,
            batched=False,
            fn_kwargs={"neg": False, "idx": 0},
            num_proc=os.cpu_count(),
        )

        def tokenize_function(examples):
            return self.tokenizer.apply_chat_template(
                examples["proc_messages"],
                tokenize=True,
                return_dict=True,
                max_length=self.max_length,
                padding="max_length",
            )

        mixed_dataset = mixed_dataset.map(
            tokenize_function, batched=True, num_proc=os.cpu_count()
        )
        mixed_dataset = mixed_dataset.map(
            lambda x: {"labels": x["input_ids"]}, batched=True, num_proc=os.cpu_count()
        )
        mixed_dataset = mixed_dataset.filter(
            lambda x: len(x["input_ids"]) <= self.max_length, num_proc=os.cpu_count()
        )
        mixed_dataset = mixed_dataset.remove_columns(
            [
                c
                for c in mixed_dataset.column_names
                if c not in ["input_ids", "attention_mask", "labels"]
            ]
        )
        return mixed_dataset

    def prepare_for_distil(self) -> None:
        raise NotImplementedError()

    def prepare_for_dpo(self) -> None:
        raise NotImplementedError()

    def prepare_for_ppo(self) -> None:
        raise NotImplementedError()

    def prepare_for_grpo(self) -> None:
        raise NotImplementedError()
