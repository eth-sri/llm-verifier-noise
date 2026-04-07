import json
import os
from pathlib import Path
from time import sleep

import docker
import tabulate

from src.constants import MODEL_PROVIDER_TO_API_KEY
from src.eval.base import Benchmark
from src.utils import esc, pass_at_k


class CWEval(Benchmark):
    """
    Implementation of the CWEval benchmark using Docker to run the evaluation in an isolated environment.
    """

    CWEVAL_API_KEY_MAPPING = {
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "openrouter": "OPENROUTER_API_KEY",
        "together": "TOGETHERAI_API_KEY",
        "vllm": "VLLM_API_KEY",
    }

    CWEVAL_PROVIDER_MAPPING = {
        "openai": "openai",
        "anthropic": "anthropic",
        "openrouter": "openrouter",
        "together": "together_ai",
        "vllm": "hosted_vllm",
    }

    def __init__(
        self,
        model_name: str,
        model_provider: str,
        reasoning: bool = False,
        reasoning_effort: int | str | None = None,
        max_tokens: int = 8192,
        temperature: float = 0.6,
        top_p: float = 0.95,
        n_samples: int = 10,
        vllm_port: int = 8000,
    ) -> None:
        super().__init__(
            model_name=model_name,
            model_provider=model_provider,
            reasoning=reasoning,
            reasoning_effort=reasoning_effort,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            n_samples=n_samples,
        )
        self.docker_client = docker.from_env()
        self.path = Path(__file__).parent.parent.parent.parent / "results/cweval"
        self.path.mkdir(parents=True, exist_ok=True)
        self.vllm_port = vllm_port

    def _check_container_health(self, container) -> None:
        """
        Checks if the CWEval container was spun up correctly.
        """
        try:
            while container.status != "running":
                sleep(1)
                container.reload()
        except Exception as e:
            raise RuntimeError("Container start error", e)

        exit_code, result = container.exec_run(
            [
                "zsh",
                "-c",
                r"sed -i '92s/not _unsafe/(not _unsafe) and (not cwe_400_0_test.py)/' cweval/run_tests.py",
            ]
        )
        assert exit_code == 0, result

    def _run_command(
        self,
        command: str,
        log_name: str,
    ) -> None:
        print(command)
        container = self.docker_client.containers.run(
            image="co1lin/cweval",
            command="zsh",
            auto_remove=False,
            environment=(
                {
                    self.CWEVAL_API_KEY_MAPPING[self.model_provider]: os.getenv(
                        MODEL_PROVIDER_TO_API_KEY[self.model_provider]
                    )
                }
                if self.model_provider != "vllm"
                else {"OPENAI_API_KEY": "sk-007"}
            ),
            detach=True,
            stdout=True,
            stderr=True,
            name=f"cweval_{esc(self.model_name)}",
            network_mode="host",
            volumes=[f"{self.path}:/home/ubuntu/CWEval/evals"],
            tty=True,
        )

        self._check_container_health(container)

        cmd = f'zsh -c "source ~/.zshrc && source .env && {command}"'
        exit_code, result = container.exec_run(cmd)
        logs = f"\n\n$ {cmd}\n"
        logs += result.decode()
        assert exit_code == 0, result

        container.remove(force=True)

        # save the logs
        with open(self.path / f"{esc(self.model_name)}/{log_name}.log", "w") as f:
            f.write(logs)

    def generate_solutions(self, **kwargs) -> None:
        command_gen = f"python cweval/generate.py gen --n {self.n_samples} --temperature {self.temperature} --top_p {self.top_p} --num_proc {os.cpu_count()} --eval_path evals/{esc(self.model_name)} --model {self.CWEVAL_PROVIDER_MAPPING[self.model_provider]}/{self.model_name} --max_completion_tokens {self.max_tokens}"
        if self.model_provider == "vllm":
            command_gen += f" --api_base http://localhost:{self.vllm_port}/v1"
        self._run_command(
            command=command_gen,
            log_name="gen",
        )

    def evaluate_solutions(self, **kwargs) -> None:
        command_eval = f"python cweval/evaluate.py pipeline --eval_path evals/{esc(self.model_name)} --num_proc {os.cpu_count()} --docker False"
        self._run_command(
            command=command_eval,
            log_name="eval",
        )

    def display_results(self, **kwargs) -> None:
        results_path = self.path / esc(self.model_name) / "res_all.json"

        if results_path.exists():
            with open(results_path, "r") as f:
                results = json.load(f)
        else:
            raise FileNotFoundError(f"Results file not found: {results_path}")

        pass_1 = {"functional": [], "secure": [], "func_secure": []}

        for r in results.values():
            for r_type, r_vals in r.items():
                pass_1[r_type].append(pass_at_k(k=1, c=sum(r_vals), n=len(r_vals)))

        pass_1 = {k: sum(v) / len(v) for k, v in pass_1.items()}

        table = [
            ["Model", self.model_name],
            ["Provider", self.model_provider],
            ["Reasoning", self.reasoning],
            ["Reasoning Effort", self.reasoning_effort],
            ["Temperature", self.temperature],
            ["# Samples", self.n_samples],
            ["Pass@1 (Functional)", f"{pass_1['functional']*100:.2f}%"],
            ["Pass@1 (Secure)", f"{pass_1['secure']*100:.2f}%"],
            ["Pass@1 (Func & Secure)", f"{pass_1['func_secure']*100:.2f}%"],
        ]

        print(tabulate.tabulate(table, tablefmt="plain"))
