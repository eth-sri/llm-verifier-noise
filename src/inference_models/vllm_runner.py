import datetime
import os
import shutil
import signal
import subprocess
import sys
from pathlib import Path
from time import sleep

from openai import OpenAI


class VLLMRunner:
    """
    A context manager to run a vLLM server as a subprocess in its own process group.
    Note that because of this, if you force kill (SIGKILL or SIGTERM) the parent process,
    the vLLM process will remain running and you will have to kill it manually.
    It is recommended to use SIGINT (Ctrl+C) to stop the parent process, which will also
    gracefully shut down the vLLM subprocess.
    """

    def __init__(
        self,
        model_name: str,
        logfile: str | None = None,
        port: int = 8000,
        max_model_length: int = 8192,
        tensor_parallel_size: int = 1,
        data_parallel_size: int = 1,
        trials: int = 25,
        initial_sleep: int = 30,
        sleep_interval: int = 15,
        gpu_memory_utilization: float | None = 0.7,
        container_name: str = "vllm_server",
        vllm_version: str = "v0.10.0",  # or latest
        use_docker: bool = False,
        chat_template: str = None,
    ) -> None:
        print(f"Initializing VLLMRunner for model: {model_name}")
        if os.path.exists(model_name):
            self.use_local_model = True
            self.model_name = os.path.abspath(model_name)
        else:
            from huggingface_hub import snapshot_download

            if len(model_name.split("/")) > 2:
                # e.g., UserID/ModelName/checkpoint-N
                hf_name = "/".join(model_name.split("/")[:2])
                subfolder = "/".join(model_name.split("/")[2:])
                local_path = snapshot_download(
                    repo_id=hf_name, allow_patterns=[f"{subfolder}/*"]
                )
                self.use_local_model = True
                self.model_name = os.path.join(local_path, subfolder)
                print("Downloaded model from HF to local path:", local_path)
            else:
                # simply use the model from HF
                self.use_local_model = False
                self.model_name = model_name
        self.port = port
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_length = max_model_length
        self.tensor_parallel_size = tensor_parallel_size
        self.data_parallel_size = data_parallel_size
        self.process = None
        self.test_client = OpenAI(
            api_key="dull-key",
            base_url=f"http://localhost:{self.port}/v1",
            timeout=600,
        )
        self.trials = trials
        self.initial_sleep = initial_sleep
        self.sleep_interval = sleep_interval
        self.logfile = logfile
        self.container_name = container_name
        self.vllm_version = vllm_version
        self.use_docker = use_docker
        self.chat_template = chat_template
        self._logfile_handle = None
        self.pid = None
        self.proc = None
        self._logfiles_base_dir = Path(__file__).parent.parent.parent / "vllm_logs"
        self._logfiles_base_dir.mkdir(parents=True, exist_ok=True)
        if self.logfile is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_model_name = (
                self.model_name.replace("/", "_").replace(":", "_").replace("-", "_")
            )
            self.logfile = str(
                self._logfiles_base_dir / f"vllm_{safe_model_name}_{timestamp}.log"
            )
        print(f"log will be saved at {self.logfile}")

    def _check_online(self) -> bool:
        try:
            r = self.test_client.chat.completions.create(
                model=self.model_name,
                n=1,
                messages=[{"role": "user", "content": "Hello"}],
                timeout=600,
                max_tokens=2,
            )
            text = r.choices[0].message.content
            if text is None or len(text) == 0:
                return False
            return True
        except Exception:
            return False

    def _wait_for_free_gpu0(
        self, threshold: float = 0.5, interval: int = 5, max_trial: int = 5
    ) -> None:
        """
        Poll the first visible GPU's memory via nvidia-smi and wait until it is at or below the threshold.
        The "first visible GPU" is determined from CUDA_VISIBLE_DEVICES (first entry), falling back to 0 if unset.
        - threshold: fraction of total memory (e.g., 0.5 means 50%)
        - interval: seconds to wait between checks
        """
        if shutil.which("nvidia-smi") is None:
            print("nvidia-smi not found; skipping GPU memory check.")
            return

        # Determine first visible GPU (by CUDA_VISIBLE_DEVICES) or default to 0
        cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
        if cvd is None:
            target_gpu = "0"
        else:
            cvd = cvd.strip()
            if cvd == "":
                print("CUDA_VISIBLE_DEVICES is empty; skipping GPU memory check.")
                return
            target_gpu = cvd.split(",")[0].strip()
        query_cmd = [
            "nvidia-smi",
            "-i",
            target_gpu,
            "--query-gpu=memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ]

        n = 0
        last_ratio = None
        while n < max_trial:
            try:
                res = subprocess.run(
                    query_cmd, capture_output=True, text=True, check=True
                )
                line = res.stdout.strip().splitlines()[0] if res.stdout.strip() else ""
                if not line:
                    # No output; do not block launch
                    print("nvidia-smi returned no output; skipping GPU memory wait.")
                    break
                parts = [p.strip() for p in line.split(",")]
                if len(parts) < 2:
                    print(
                        f"Unexpected nvidia-smi output: {line!r}; skipping GPU memory wait."
                    )
                    break
                used = float(parts[0])
                total = float(parts[1])
                ratio = used / total if total > 0 else 0.0
                last_ratio = ratio

                if ratio <= threshold:
                    # Sufficiently free
                    print(
                        f"GPU {target_gpu} memory usage OK: {int(used)}/{int(total)} MiB ({ratio * 100:.1f}%) ≤ {int(threshold * 100)}%."
                    )
                    break

                print(
                    f"GPU {target_gpu} busy: {int(used)}/{int(total)} MiB ({ratio * 100:.1f}%) > {int(threshold * 100)}%. Waiting {interval}s..."
                )
                sleep(interval)
                n += 1
            except Exception as e:
                print(
                    f"Warning: failed to query GPU {target_gpu} memory with nvidia-smi ({e}); proceeding without wait."
                )
                break
        if last_ratio is not None and n >= max_trial and last_ratio > threshold:
            print(
                f"GPU {target_gpu} appears busy after maximum wait attempts; the runner may fail to start."
            )

    def __enter__(self) -> "VLLMRunner":

        def handle_sigint(signum, frame):
            # for capturing KeyboardInterrupt
            print(f"Received signal {signum}, shutting down vLLM server...")
            self.__exit__(None, None, None)
            sys.exit(0)

        signal.signal(signal.SIGINT, handle_sigint)
        self._logfile_handle = open(self.logfile, "a")
        self._wait_for_free_gpu0(
            threshold=(
                1 - self.gpu_memory_utilization if self.gpu_memory_utilization else 0.1
            ),
            interval=5,
        )

        cmd_shared = [
            self.model_name,
            "--trust-remote-code",
            "--tensor-parallel-size",
            str(self.tensor_parallel_size),
            "--data-parallel-size",
            str(self.data_parallel_size),
            "--max-model-len",
            str(self.max_model_length),
            "--port",
            str(self.port),
            "--generation-config",
            "vllm",
        ]
        if self.gpu_memory_utilization:
            cmd_shared.extend(
                [
                    "--gpu-memory-utilization",
                    str(self.gpu_memory_utilization),
                ]
            )
        if self.chat_template:
            # replacing chat template is the easiest way to disable thinking model for Qwen3
            # https://qwen.readthedocs.io/en/latest/deployment/vllm.html
            cmd_shared.extend(
                [
                    "--chat-template",
                    self.chat_template,
                ]
            )

        if self.use_docker:
            # base args
            cmd = [
                "docker",
                "run",
                "--name",
                self.container_name,
                "--gpus",
                "all",
                "-p",
                f"{self.port}:{self.port}",
                "--rm",
            ]
            # mount if it is a local model
            if self.use_local_model:
                # assert absname
                cmd.extend(["-v", f"{self.model_name}:{self.model_name}"])
            cmd.extend([f"ghcr.io/lambdalabsml/vllm-builder:{self.vllm_version}"])
            cmd.extend(cmd_shared)
        else:
            cmd = ["vllm", "serve"] + cmd_shared

        print(f"Starting vLLM with command: {' '.join(cmd)}")
        self.proc = subprocess.Popen(
            cmd,
            stdout=self._logfile_handle,
            stderr=subprocess.STDOUT,
            shell=False,
            preexec_fn=os.setsid,
        )
        self.pid = self.proc.pid
        sleep(self.initial_sleep)  # give it some time to start
        # check if at least the process is running
        if self.proc.poll() is not None:
            self.__exit__(None, None, None)
            raise Exception(
                f"vLLM process with PID {self.pid} terminated unexpectedly, see log at {self.logfile} for details"
            )
        # now check if the model is actually online
        for trial in range(self.trials):
            if self._check_online():
                print(
                    f"vLLM is online after {trial * self.sleep_interval + self.initial_sleep} seconds"
                )
                return self
            else:
                print(
                    f"vLLM not online yet, waiting {self.sleep_interval} seconds (trial {trial + 1}/{self.trials})..."
                )
                sleep(self.sleep_interval)

        self.__exit__(None, None, None)
        raise Exception(
            f"vLLM failed to start in time, see log at {self.logfile} for details"
        )

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if getattr(self, "_exit_called", False):
            return
        self._exit_called = True

        if self.proc and self.proc.poll() is None:
            if self.use_docker:
                print(f"Terminating vLLM container {self.container_name}")
                try:
                    subprocess.run(
                        ["docker", "stop", self.container_name],
                        check=False,
                        stdout=subprocess.DEVNULL,
                        stderr=self._logfile_handle,
                    )
                except Exception as e:
                    print(
                        f"Failed to stop docker container {self.container_name}: {e}. You might need to stop it manually. PID: {self.pid}"
                    )
            else:
                print(f"Terminating vLLM process with PID {self.pid}.")
                try:
                    os.killpg(os.getpgid(self.pid), signal.SIGINT)
                    self.proc.wait()
                    assert self.proc.poll() is not None
                    print(f"vLLM process with PID {self.pid} terminated.")
                except Exception as e:
                    print(
                        f"Failed to terminate vLLM process at PID: {self.pid} gracefully: {e}."
                    )
                    try:
                        print(f"Killing vLLM process with PID {self.pid}")
                        os.killpg(os.getpgid(self.pid), signal.SIGKILL)
                        self.proc.wait()
                        assert self.proc.poll() is not None
                        print(f"vLLM process with PID {self.pid} killed")
                    except Exception as e:
                        print(
                            f"Failed to kill vLLM process at PID: {self.pid}: {e}. You might need to kill it manually."
                        )
        if self._logfile_handle:
            self._logfile_handle.close()
            print(f"vLLM log file saved at: {self.logfile}")


if __name__ == "__main__":
    # example usage
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen2.5-1.5B-Instruct",
        help="Model name or path",
    )
    args = parser.parse_args()
    with VLLMRunner(model_name=args.model_name, max_model_length=256) as runner:
        client = OpenAI(
            api_key="dull-key",
            base_url=f"http://localhost:{runner.port}/v1",
            timeout=600,
        )

        print("calling vLLM API...")
        response = client.chat.completions.create(
            model=runner.model_name,  # local model needs to be in absolute path
            messages=[{"role": "user", "content": "Hello, vLLM!"}],
            max_tokens=128,
            temperature=0.0,
        )
        print("Response from vLLM:")
        print(response.choices[0].message.content)
