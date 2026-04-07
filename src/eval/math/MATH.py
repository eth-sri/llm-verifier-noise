import json
from pathlib import Path

import numpy as np
from datasets import load_dataset
from math_verify import parse, verify
from tabulate import SEPARATING_LINE, tabulate
from tqdm import tqdm

from src.eval.base import Benchmark
from src.inference_models import Conversation, get_inference_model
from src.utils import esc, extract_boxed, pass_at_k, process_ans


class MATH(Benchmark):

    PROMPT = "Please reason step by step, and put your final answer within \\boxed{}."

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
        subset: str = "MATH500",
        n_samples: int = 10,
        max_workers: int = 128,
        vllm_port: int = 8000,
        timeout: int = 600,
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
        )
        self.vllm_port = vllm_port
        self.subset = subset
        self.timeout = timeout
        self.max_workers = max_workers
        self.path = (
            Path(__file__).parent.parent.parent.parent / f"results/MATH_{self.subset}"
        )
        self.path.mkdir(parents=True, exist_ok=True)
        self.inference_model = get_inference_model(
            model_name=model_name,
            model_provider=model_provider,
            reasoning=reasoning,
            reasoning_effort=reasoning_effort,
            no_system_prompt=no_system_prompt,
            port=vllm_port,
            timeout=timeout,
        )

        add_sol = lambda x: {"answer": extract_boxed(x["solution"])}

        if self.subset == "full":
            self.dataset = load_dataset(
                "DigitalLearningGmbH/MATH-lighteval", split="test"
            )
            self.dataset = self.dataset.map(add_sol)
        elif self.subset == "MATH500":
            self.dataset = load_dataset("HuggingFaceH4/MATH-500", split="test")
        else:
            raise ValueError(
                f"Unknown subset: {self.subset}. Available subsets: full, MATH500."
            )

        self.results = []
        for idx, task in enumerate(self.dataset):
            self.results.append(
                {
                    "prompt": f"{task['problem']}\n{self.PROMPT}",
                    "solution": task["answer"],
                    "level": int(
                        str(task["level"]).strip()[-1]
                    ),  # this weird thing handles both datasets
                    "responses": [],
                }
            )

    def generate_solutions(self, **kwargs) -> None:

        if self.model_provider == "hf" or self.model_provider == "huggingface":
            raise NotImplementedError(
                "Running HuggingFace models natively from the transformers library is not supported as "
                "vLLM is a much better direct replacement. Please, use all the same settings, simply rerun "
                "with the provider set to vllm."
            )

        else:
            for idx in range(self.n_samples):
                print(f"Generating benchmark run {idx + 1}/{self.n_samples}...")
                responses = self.inference_model.generate_multi_erb(
                    [
                        Conversation().add_user_message(r["prompt"])
                        for r in self.results
                    ],
                    temperature=self.temperature,
                    max_workers=self.max_workers,
                    progress_bar=True,
                    base_delay=1.0,
                    max_delay=120.0,
                    max_retries=5,
                    max_tokens=self.max_tokens,
                    top_p=self.top_p,
                )
                for j, item in enumerate(self.results):
                    item["responses"].append(
                        "<think>\n"
                        + responses[j].reasoning
                        + "\n</think>\n"
                        + responses[j].text
                        if responses[j].reasoning
                        else responses[j].text
                    )
        save_path = self.path / esc(self.model_name)
        save_path.mkdir(parents=True, exist_ok=True)
        with open(
            save_path / f"solutions_{self.n_samples}_{self.temperature}.json", "w"
        ) as f:
            json.dump(self.results, f, indent=4, ensure_ascii=False)

    def evaluate_solutions(self, **kwargs) -> None:
        save_path = self.path / esc(self.model_name)

        with open(
            save_path / f"solutions_{self.n_samples}_{self.temperature}.json", "r"
        ) as f:
            self.results = json.load(f)

        for r in tqdm(self.results, desc="Evaluating solutions"):
            r["extracted_responses"] = [
                extract_boxed(r["responses"][j]) for j in range(self.n_samples)
            ]
            r["correct"] = [
                verify(parse(process_ans(r["solution"])), parse(process_ans(er)))
                for er in r["extracted_responses"]
            ]

        with open(
            save_path / f"solutions_{self.n_samples}_{self.temperature}.json",
            "w",
        ) as f:
            json.dump(self.results, f, indent=4, ensure_ascii=False)

    def display_results(self, **kwargs) -> None:
        results_path = (
            self.path
            / esc(self.model_name)
            / f"solutions_{self.n_samples}_{self.temperature}.json"
        )

        if results_path.exists():
            with open(results_path, "r") as f:
                results = json.load(f)
        else:
            raise FileNotFoundError(f"Results file not found: {results_path}")

        pass_k = {
            "pass_1": {f"level_{i+1}": [] for i in range(5)},
            "pass_5": {f"level_{i+1}": [] for i in range(5)},
        }
        pass_k["pass_1"]["all"] = []
        pass_k["pass_5"]["all"] = []

        pass_1_per_sample = {
            "pass_1": {
                f"level_{i+1}": [[] for _ in range(self.n_samples)] for i in range(5)
            },
        }
        pass_1_per_sample["pass_1"]["all"] = [[] for _ in range(self.n_samples)]

        for r in results:
            level = r["level"]
            pass_k["pass_1"][f"level_{level}"].append(
                pass_at_k(k=1, c=sum(r["correct"]), n=len(r["correct"]))
            )
            pass_k["pass_5"][f"level_{level}"].append(
                pass_at_k(k=5, c=sum(r["correct"]), n=len(r["correct"]))
            )
            pass_k["pass_1"]["all"].append(
                pass_at_k(k=1, c=sum(r["correct"]), n=len(r["correct"]))
            )
            pass_k["pass_5"]["all"].append(
                pass_at_k(k=5, c=sum(r["correct"]), n=len(r["correct"]))
            )
            for i in range(self.n_samples):
                pass_1_per_sample["pass_1"][f"level_{level}"][i].append(
                    pass_at_k(k=1, c=r["correct"][i], n=1)
                )
                pass_1_per_sample["pass_1"]["all"][i].append(
                    pass_at_k(k=1, c=r["correct"][i], n=1)
                )

        pass_1_per_sample = {
            k: {l: ([sum(r) / len(r) for r in lv], len(lv[0])) for l, lv in v.items()}
            for k, v in pass_1_per_sample.items()
        }

        pass_k = {
            k: {l: sum(lv) / len(lv) for l, lv in v.items()} for k, v in pass_k.items()
        }

        table = [
            ["Model", self.model_name],
            ["Provider", self.model_provider],
            ["Reasoning", self.reasoning],
            ["Reasoning Effort", self.reasoning_effort],
            ["Temperature", self.temperature],
            ["# Samples", self.n_samples],
            SEPARATING_LINE,
            ["Pass@1", f"{pass_k['pass_1']['all']*100:.2f}%"],
            ["Pass@5", f"{pass_k['pass_5']['all']*100:.2f}%"],
        ]
        for i in range(5):
            table.append(SEPARATING_LINE)
            table.append(
                [
                    f"Pass@1 Level {i+1}",
                    f"{pass_k['pass_1'][f'level_{i+1}']*100:.2f}%",
                ]
            )
            table.append(
                [
                    f"Pass@5 Level {i+1}",
                    f"{pass_k['pass_5'][f'level_{i+1}']*100:.2f}%",
                ]
            )
        table.append(SEPARATING_LINE)
        table.append(SEPARATING_LINE)
        table.append(["Cross Sample", ""])
        table.append(SEPARATING_LINE)
        across_all = []
        for j in range(self.n_samples):
            table.append(
                [
                    f"Pass@1 Sample {j+1} #tasks {pass_1_per_sample['pass_1']['all'][1]}",
                    f"{pass_1_per_sample['pass_1']['all'][0][j]*100:.2f}%",
                ]
            )
            across_all.append(pass_1_per_sample["pass_1"]["all"][0][j])
        table.append(
            [
                f"Pass@1 Sample Avg. and STD",
                f"{np.mean(across_all) * 100:.2f}% ({np.std(100*np.array(across_all)):.2f}%)",
            ]
        )
        for i in range(5):
            table.append(SEPARATING_LINE)
            across_all = []
            for j in range(self.n_samples):
                table.append(
                    [
                        f"Pass@1 Sample {j} Level {i+1} #tasks {pass_1_per_sample['pass_1'][f'level_{i+1}'][1]}",
                        f"{pass_1_per_sample['pass_1'][f'level_{i+1}'][0][j]*100:.2f}%",
                    ]
                )
                across_all.append(pass_1_per_sample["pass_1"][f"level_{i+1}"][0][j])
            table.append(
                [
                    f"Pass@1 Sample Avg. and STD Level {i+1}",
                    f"{np.mean(across_all) * 100:.2f}% ({np.std(100*np.array(across_all)):.2f}%)",
                ]
            )
        print(tabulate(table))
        with open(self.path / esc(self.model_name) / "summary_results.txt", "w") as f:
            f.write(tabulate(table) + "\n\n")
