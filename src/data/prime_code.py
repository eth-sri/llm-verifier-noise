import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
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


class PRIMECode(MixedUpDataset):
    def __init__(
        self,
        name: str,
        tokenizer: PreTrainedTokenizerBase,
        mixup: bool = False,
        shuffle: bool = False,
        max_length: int = 2048,
        **kwargs,
    ):
        super().__init__(
            name=name,
            tokenizer=tokenizer,
            mixup=mixup,
            shuffle=shuffle,
            max_length=max_length,
            **kwargs,
        )

        dataset_list = []
        for i in range(10):
            try:
                ds = load_dataset(f"PRIME-RL/Eurus-2-RL-Data", split="train")
            except Exception as e:
                print(f"Error loading dataset {i}: {e}")
                continue
            dataset_list.append(ds)
        self.raw_dataset = concatenate_datasets(dataset_list)
        # Qwen3-14B_best_pass_rate_per_3 == 1
        self.raw_dataset = self.raw_dataset.filter(
            lambda x: x["Qwen3-14B_best_pass_rate_per_3"] == 1, num_proc=os.cpu_count() // 2, load_from_cache_file=False
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
                    {"role": "user", "content": example["prompt"][1]["content"]},
                ],
            }

        dataset = self.raw_dataset.map(_to_messages, num_proc=os.cpu_count() // 2)

        def reward(
            completions: list[list[dict[str, str]]],
            # answer: list[str],  # be careful that the name is 'answer' singular
            prompts: list[list[dict[str, str]]],
            **kwargs,
        ) -> list[float]:
            # print kwargs for debugging
            # for k, v in kwargs.items():
            #     print(f"kwargs[{k}] = {v}")
            ground_truth = [json.loads(rm["ground_truth"]) for rm in kwargs["reward_model"]]

            code_pattern = re.compile(r"```python(.*?)```", re.DOTALL)

            def _extract_code(text: str) -> str:
                snippets = code_pattern.findall(text)
                if snippets:
                    return snippets[-1].strip()
                return text.strip()

            def _as_multiline_string(x: Any) -> str:
                if isinstance(x, list):
                    s = "\n".join(map(str, x))
                else:
                    s = str(x)
                if not s.endswith("\n"):
                    s += "\n"
                return s

            prlimit_bin = shutil.which("prlimit")

            def _sandbox_env(tmpdir: str) -> dict[str, str]:
                # Keep child execution isolated from the caller environment.
                return {
                    "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
                    "HOME": tmpdir,
                    "TMPDIR": tmpdir,
                    "PYTHONPATH": "",
                    "PYTHONHOME": "",
                }

            def _sandbox_command(exec_path: str) -> list[str]:
                cmd = [sys.executable, "-I", "-S", "-B", exec_path]
                if prlimit_bin is None:
                    return cmd
                # Limits reduce blast radius if model output is malicious.
                return [
                    prlimit_bin,
                    "--cpu=3",
                    "--as=536870912",
                    "--fsize=1048576",
                    "--nofile=64",
                    "--core=0",
                    "--",
                    *cmd,
                ]


            def _passes_all_tests(code: str, ground_truth: dict[str, Any]) -> bool:
                inputs = ground_truth.get("inputs", [])
                outputs = ground_truth.get("outputs", [])
                assert len(inputs) == len(outputs), f"{len(inputs)} != {len(outputs)}"

                with tempfile.TemporaryDirectory() as tmpdir:
                    exec_path = os.path.join(tmpdir, "exec.py")
                    with open(exec_path, "w", encoding="utf-8") as f:
                        f.write(code)

                    for test_in, expected_out in zip(inputs, outputs):
                        expected = _as_multiline_string(expected_out).strip()
                        try:
                            result = subprocess.run(
                                _sandbox_command(exec_path),
                                input=_as_multiline_string(test_in),
                                text=True,
                                capture_output=True,
                                timeout=3,
                                cwd=tmpdir,
                                env=_sandbox_env(tmpdir),
                            )
                        except subprocess.TimeoutExpired:
                            return False

                        if result.returncode != 0:
                            return False

                        if result.stdout.strip() != expected:
                            return False

                return True

            def _score_completion(i: int) -> float:
                completion = completions[i]
                content = completion[0]["content"] if completion else ""
                code = _extract_code(content)
                try:
                    return 1.0 if _passes_all_tests(code, ground_truth[i]) else 0.0
                except Exception:
                    return 0.0

            n = len(completions)
            if n == 0:
                return []

            max_workers = min(os.cpu_count() or 1, n)
            if max_workers <= 1:
                return [_score_completion(i) for i in range(n)]

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                return list(executor.map(_score_completion, range(n)))

        return dataset, reward
