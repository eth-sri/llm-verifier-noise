import datetime
import re
import warnings
from gc import callbacks
from pathlib import Path
from typing import Any, Dict, List, Optional

import accelerate
import pandas as pd
import torch
from accelerate.utils import gather_object
from huggingface_hub import HfApi
from peft import get_peft_model
from torch import gather
from transformers import Trainer, set_seed
from trl import GRPOTrainer

from src.configs import TrainConfig
from src.data.base import Mixup
from src.train.train_tools import TrainTools


def grpo(training_config: TrainConfig) -> None:

    train_tools = TrainTools(training_config, "grpo")

    model = train_tools.model
    tokenizer = train_tools.tokenizer
    dataset = train_tools.dataset
    training_args = train_tools.training_args
    peft_config = train_tools.peft_args if training_config.use_peft else None

    grpo_dataset, reward_function = dataset.prepare_for_grpo()
    reward_function._oracle_rewards = []
    reward_function._gt_answers = []
    reward_function._verifier_reasoning = []
    reward_function._rubrics = []
    reward_function._verifier_cot = []

    print(30 * "=")
    print(f"Dataset prepared for GRPO with {len(grpo_dataset)} samples.")
    print(peft_config if peft_config is not None else "No PEFT configuration.")
    print(30 * "=")

    callbacks = []
    if training_config.use_neptune:
        callbacks.append(train_tools.neptune_callback)
    if training_config.use_wandb:
        callbacks.append(train_tools.wandb_callback)
    trainer = GRPOTrainerDebug(
        model=model,
        args=training_args,
        train_dataset=grpo_dataset,
        processing_class=tokenizer,
        reward_funcs=reward_function,
        mixup=dataset.mixup,
        train_tools=train_tools,
        callbacks=(callbacks if len(callbacks) > 0 else None),
        peft_config=peft_config,
    )
    trainer.train(resume_from_checkpoint=training_config.resume_from_checkpoint)
    train_tools.wrap_up_and_save(trainer)


class GRPOTrainerDebug(GRPOTrainer):
    def __init__(self, *args, **kwargs):
        self.mixup = kwargs.pop("mixup", None)
        self.train_tools = kwargs.pop("train_tools", None)
        super().__init__(*args, **kwargs)

        if self.chat_template is None:
            tokenizer_chat_template = getattr(self.processing_class, "chat_template", None)
            if tokenizer_chat_template:
                self.chat_template = tokenizer_chat_template
                if getattr(self, "vllm_generation", None) is not None:
                    self.vllm_generation.chat_template = tokenizer_chat_template
                if self.accelerator.is_main_process:
                    print("Using tokenizer chat template for GRPO vLLM generation.")

        if self.accelerator.is_main_process:
            warnings.warn("Debug GRPOTrainer active: Extended logging enabled.")

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        # Setup logging directory
        if self.train_tools is not None:
            dir_name = f"{timestamp}_{self.train_tools.save_name}"
            print(
                f"The model will be saved to: {HfApi().whoami()['name']}/{self.train_tools.save_name}"
            )
        else:
            dir_name = timestamp

        if self.train_tools is not None:
            base_log_dir = Path(self.train_tools.training_config.grpo_debug_log_dir)
        else:
            base_log_dir = Path("grpo_debug")
        self.logdir = base_log_dir / dir_name

        # Ensure directory exists on ALL ranks to avoid file permission errors
        # (exist_ok=True handles the race condition safely)
        self.logdir.mkdir(parents=True, exist_ok=True)

    def _compute_global_tpr_fpr(self):
        """Aggregate Mixup counters across all processes safely."""
        device = self.accelerator.device

        if self.mixup is not None:
            tp = torch.tensor(
                getattr(self.mixup, "realized_TP", 0) or 0,
                device=device,
                dtype=torch.float32,
            )
            fp = torch.tensor(
                getattr(self.mixup, "realized_FP", 0) or 0,
                device=device,
                dtype=torch.float32,
            )
            gt_pos = torch.tensor(
                getattr(self.mixup, "gt_positive", 0) or 0,
                device=device,
                dtype=torch.float32,
            )
            gt_neg = torch.tensor(
                getattr(self.mixup, "gt_negative", 0) or 0,
                device=device,
                dtype=torch.float32,
            )
            active = torch.tensor(1.0, device=device)
        else:
            tp = fp = gt_pos = gt_neg = torch.tensor(0.0, device=device)
            active = torch.tensor(0.0, device=device)

        # Pack into one tensor of shape (5,)
        local_vec = torch.stack([tp, fp, gt_pos, gt_neg, active])  # (5,)
        # Gather across processes → shape (world_size, 5)
        all_vecs = self.accelerator.gather(local_vec.unsqueeze(0))  # (W, 5)

        if not self.accelerator.is_main_process:
            return None, None

        # Columns: 0=tp, 1=fp, 2=gt_pos, 3=gt_neg, 4=active
        active_mask = all_vecs[:, 4] > 0.5
        if not active_mask.any():
            return None, None

        global_tp = all_vecs[active_mask, 0].sum()
        global_fp = all_vecs[active_mask, 1].sum()
        global_gt_pos = all_vecs[active_mask, 2].sum()
        global_gt_neg = all_vecs[active_mask, 3].sum()

        global_tpr = global_tp / global_gt_pos if global_gt_pos > 0 else -1.0
        global_fpr = global_fp / global_gt_neg if global_gt_neg > 0 else -1.0

        return float(global_tpr), float(global_fpr)

    def _compute_global_trigger_stats(self):
        """Aggregate selector-based trigger stats across all processes.

        Definitions (per call to Mixup.mixup_rewards with TARGETED strategy):
        - triggered: number of items that matched at least one TARGETED bucket
        - triggered_and_correct: among triggered items, how many have
          ground-truth/oracle label == True.
        - correct_total: total number of ground-truth/oracle positive items
          (across all items processed by ``mixup_rewards`` in this step).

        Returns
        -------
        tuple[float, float, float, float] | (None, None, None, None)
            (
                global_triggered,
                global_triggered_and_correct,
                correct_among_triggered,   # P(correct | triggered)
                triggered_among_correct,   # P(triggered | correct)
            )
            where the conditional probabilities fall back to -1.0 when their
            denominators are zero.
        """

        device = self.accelerator.device

        if self.mixup is not None:
            triggered = torch.tensor(
                getattr(self.mixup, "triggered", 0) or 0,
                device=device,
                dtype=torch.float32,
            )
            triggered_and_correct = torch.tensor(
                getattr(self.mixup, "triggered_and_correct", 0) or 0,
                device=device,
                dtype=torch.float32,
            )
            correct_total = torch.tensor(
                getattr(self.mixup, "gt_positive", 0) or 0,
                device=device,
                dtype=torch.float32,
            )
            active = torch.tensor(1.0, device=device)
        else:
            triggered = triggered_and_correct = correct_total = torch.tensor(
                0.0, device=device
            )
            active = torch.tensor(0.0, device=device)

        # Pack into one tensor of shape (4,)
        local_vec = torch.stack(
            [triggered, triggered_and_correct, correct_total, active]
        )  # (4,)
        # Gather across processes → shape (world_size, 4)
        all_vecs = self.accelerator.gather(local_vec.unsqueeze(0))  # (W, 4)

        if not self.accelerator.is_main_process:
            return None, None, None, None

        # Columns: 0=triggered, 1=triggered_and_correct, 2=correct_total, 3=active
        active_mask = all_vecs[:, 3] > 0.5
        if not active_mask.any():
            return None, None, None, None

        global_triggered = all_vecs[active_mask, 0].sum()
        global_triggered_and_correct = all_vecs[active_mask, 1].sum()
        global_correct_total = all_vecs[active_mask, 2].sum()

        correct_among_triggered = (
            global_triggered_and_correct / global_triggered
            if global_triggered > 0
            else -1.0
        )
        triggered_among_correct = (
            global_triggered_and_correct / global_correct_total
            if global_correct_total > 0
            else -1.0
        )

        return (
            float(global_triggered),
            float(global_triggered_and_correct),
            float(correct_among_triggered),
            float(triggered_among_correct),
        )

    def _compute_global_alternation_stats(self):
        """Aggregate oracle alternation step flag across all processes."""
        device = self.accelerator.device

        if self.mixup is not None:
            flg_alternation = torch.tensor(
                getattr(self.mixup, "oracle_step_active", 0) or 0,
                device=device,
                dtype=torch.float32,
            )
            active = torch.tensor(1.0, device=device)
        else:
            flg_alternation = torch.tensor(0.0, device=device)
            active = torch.tensor(0.0, device=device)

        local_vec = torch.stack([flg_alternation, active])
        all_vecs = self.accelerator.gather(local_vec.unsqueeze(0))  # (W, 2)

        if not self.accelerator.is_main_process:
            return None

        active_mask = all_vecs[:, 1] > 0.5
        if not active_mask.any():
            return None

        # If any rank reports alternation active, treat step as active.
        global_flg_alternation = 1.0 if all_vecs[active_mask, 0].sum() > 0 else 0.0

        return global_flg_alternation

    def _get_local_oracle_rewards(self) -> List[float]:
        """Extracts oracle rewards from the local process and clears the buffer."""
        rf = getattr(self, "reward_funcs", None)
        if rf and len(rf) != 1:
            warnings.warn(f"Expected single reward function, got {len(rf)}")

        rf = rf[0] if rf is not None else None
        local_rewards = []
        if rf is not None and hasattr(rf, "_oracle_rewards"):
            data = rf._oracle_rewards
            # assert isinstance(
            #     data, list
            # ), f"Oracle rewards should be a list. Got {type(data)}"
            local_rewards = data
            # Clear the buffer immediately after reading
            rf._oracle_rewards = []

        return local_rewards

    def _get_global_oracle_rewards(self) -> List[float]:
        """Gather oracle rewards from all processes and clear buffers."""
        rf = getattr(self, "reward_funcs", None)
        if rf and len(rf) != 1:
            warnings.warn(f"Expected single reward function, got {len(rf)}")

        rf = rf[0] if rf is not None else None

        local_rewards: List[float] = []
        if rf is not None and hasattr(rf, "_oracle_rewards"):
            local_rewards = list(rf._oracle_rewards)
            rf._oracle_rewards = []  # clear per-rank buffer

        device = self.accelerator.device

        # Each rank has the same per-rank batch size at this point in GRPOTrainer,
        # so this tensor has the same shape on all ranks.
        local_tensor = torch.tensor(local_rewards, device=device, dtype=torch.float32)

        # Gather → concatenated tensor of shape (world_size * local_bs,)
        global_tensor = self.accelerator.gather(local_tensor)

        # Every rank now has the same global oracle reward list
        return global_tensor.tolist()

    def _get_local_extracted_answers(self) -> List[Any]:
        """Extracts ground-truth answers from the local process and clears the buffer."""
        rf = getattr(self, "reward_funcs", None)
        if rf and len(rf) != 1:
            warnings.warn(f"Expected single reward function, got {len(rf)}")

        rf = rf[0] if rf is not None else None
        local_answers = []

        if rf is not None and hasattr(rf, "_extracted_answers"):
            data = rf._extracted_answers
            # assert isinstance(
            #     data, list
            # ), f"GT answers should be a list. Got {type(data)}"
            local_answers = data
            # Clear the buffer immediately after reading
            rf._extracted_answers = []

        return local_answers

    def _get_global_extracted_answers(self) -> List[Any]:
        """Gather ground-truth answers from all processes and clear buffers."""
        local_answers = self._get_local_extracted_answers()
        # Gather Python objects (lists) across all ranks
        all_answers = gather_object(local_answers)
        # On each rank, `all_answers` is a list of lists: one list per process.
        # Flatten it to a single list.
        return all_answers

    def _get_local_gt_answers(self) -> List[Any]:
        """Extracts ground-truth answers from the local process and clears the buffer."""
        rf = getattr(self, "reward_funcs", None)
        if rf and len(rf) != 1:
            warnings.warn(f"Expected single reward function, got {len(rf)}")

        rf = rf[0] if rf is not None else None
        local_answers = []

        if rf is not None and hasattr(rf, "_gt_answers"):
            data = rf._gt_answers
            # assert isinstance(
            #     data, list
            # ), f"GT answers should be a list. Got {type(data)}"
            local_answers = data
            # Clear the buffer immediately after reading
            rf._gt_answers = []

        return local_answers

    def _get_local_verifier_reasoning(self) -> List[Any]:
        """Extracts verifier reasoning from the local process and clears the buffer."""
        rf = getattr(self, "reward_funcs", None)
        if rf and len(rf) != 1:
            warnings.warn(f"Expected single reward function, got {len(rf)}")

        rf = rf[0] if rf is not None else None
        local_reasoning = []

        if rf is not None and hasattr(rf, "_verifier_reasoning"):
            local_reasoning = rf._verifier_reasoning
            rf._verifier_reasoning = []

        return local_reasoning

    def _get_global_verifier_reasoning(self) -> List[Any]:
        """Gather verifier reasoning from all processes and clear buffers."""
        local_reasoning = self._get_local_verifier_reasoning()
        all_reasoning = gather_object(local_reasoning)
        return all_reasoning

    def _get_local_rubrics(self) -> List[Any]:
        """Extracts rubrics from the local process and clears the buffer."""
        rf = getattr(self, "reward_funcs", None)
        if rf and len(rf) != 1:
            warnings.warn(f"Expected single reward function, got {len(rf)}")

        rf = rf[0] if rf is not None else None
        local_rubrics = []

        if rf is not None and hasattr(rf, "_rubrics"):
            local_rubrics = rf._rubrics
            rf._rubrics = []

        return local_rubrics

    def _get_global_rubrics(self) -> List[Any]:
        """Gather rubrics from all processes and clear buffers."""
        local_rubrics = self._get_local_rubrics()
        all_rubrics = gather_object(local_rubrics)
        return all_rubrics

    def _get_global_gt_answers(self) -> List[Any]:
        """Gather ground-truth answers from all processes and clear buffers."""
        local_answers = self._get_local_gt_answers()
        # Gather Python objects (lists) across all ranks
        all_answers = gather_object(local_answers)
        # On each rank, `all_answers` is a list of lists: one list per process.
        # Flatten it to a single list.
        return all_answers

    def log(self, logs: Dict[str, float], start_time: Optional[float] = None) -> None:
        # 1. Capture Local Oracle Rewards FIRST
        # We must do this before any gathering or aggregation logic
        global_oracle_rewards = self._get_global_oracle_rewards()
        global_gt_answers = self._get_global_gt_answers()
        global_extracted_answers = self._get_global_extracted_answers()
        global_verifier_reasoning = self._get_global_verifier_reasoning()
        global_rubrics = self._get_global_rubrics()

        # 2. Compute Standard Metrics
        mode = "train" if self.model.training else "eval"
        if mode in self._metrics and self._metrics[mode]:
            metrics = {
                key: sum(val) / len(val) for key, val in self._metrics[mode].items()
            }
            if mode == "eval":
                metrics = {f"eval_{key}": val for key, val in metrics.items()}
            logs = {**logs, **metrics}

        if self.log_completions:
            # --- A. Global Scalars (For Neptune) ---
            # TPR/FPR
            try:
                global_tpr, global_fpr = self._compute_global_tpr_fpr()
                if self.accelerator.is_main_process:
                    if global_tpr is not None and global_fpr is not None:
                        logs["global_TPR"] = global_tpr
                        logs["global_FPR"] = global_fpr
            except Exception as e:
                print(f"Error computing global TPR/FPR: {e}")
                pass

            # Selector-based trigger statistics (TARGETED mixup)
            try:
                (
                    global_triggered,
                    global_triggered_and_correct,
                    global_correct_among_triggered,
                    global_triggered_among_correct,
                ) = self._compute_global_trigger_stats()
                if self.accelerator.is_main_process:
                    if global_triggered is not None:
                        logs["triggered"] = global_triggered
                        logs["triggered_and_correct"] = global_triggered_and_correct
                        logs["triggered_and_wrong"] = (
                            global_triggered - global_triggered_and_correct
                        )
                        logs["correct_among_triggered"] = global_correct_among_triggered
                        logs["triggered_among_correct"] = global_triggered_among_correct
            except Exception as e:
                print(f"Error computing global trigger stats: {e}")
                pass

            # Oracle Reward
            try:
                if len(global_oracle_rewards) > 0 and self.accelerator.is_main_process:
                    t = torch.tensor(global_oracle_rewards, dtype=torch.float32)
                    # Gather lists of floats from all ranks
                    logs["oracle_reward"] = float(t.mean().item())
            except Exception as e:
                print(f"Error gathering oracle rewards: {e}")
                pass

            # Oracle alternation step flag
            try:
                global_flg_alternation = self._compute_global_alternation_stats()
                if self.accelerator.is_main_process:
                    if global_flg_alternation is not None:
                        logs["flg_alternation"] = global_flg_alternation
            except Exception as e:
                print(f"Error computing oracle alternation stats: {e}")
                pass

            # --- B. CSV Table (For detailed debugging) ---

            def _truncate(cmpl):
                # Simple cleanup to prevent CSV breakage
                # return re.sub(r"[\r\n]+", " " * 5, str(cmpl)) + "\n"
                s = str(cmpl)
                s = s.replace("\r\n", "[CRLF]")
                s = s.replace("\r", "[CR]")
                s = s.replace("\n", "[LF]")
                return s + "\n"

            if self.accelerator.is_main_process:
                try:
                    all_query = self._logs["prompt"]
                    all_completions = self._logs["completion"]
                    total_len = len(all_query)

                    table = {
                        "step": [str(self.state.global_step)] * total_len,
                        "prompt": [_truncate(q) for q in all_query],
                        "completion": [_truncate(c) for c in all_completions],
                    }

                    # Add Verifier Reasoning
                    if len(global_verifier_reasoning) == total_len:
                        table["verifier_reasoning"] = [
                            _truncate(v) if v is not None else None
                            for v in global_verifier_reasoning
                        ]
                    else:
                        table["verifier_reasoning"] = [None] * total_len

                    # Add Rubrics
                    if len(global_rubrics) == total_len:
                        table["rubric"] = [
                            _truncate(v) if v is not None else None
                            for v in global_rubrics
                        ]
                    else:
                        table["rubric"] = [None] * total_len

                    table["reward"] = self._logs["rewards"]["reward"]
                    table["advantage"] =  self._logs["advantages"]

                    # Add Oracle Rewards (Perfect alignment guaranteed locally)
                    if len(global_oracle_rewards) == total_len:
                        table["oracle_reward"] = global_oracle_rewards
                    else:
                        table["oracle_reward"] = [None] * total_len
                    # Add GT Answers (Hashes only)
                    if len(global_gt_answers) == total_len:
                        table["gt_answer"] = global_gt_answers
                    else:
                        table["gt_answer"] = [None] * total_len

                    # Add Extracted Answers (From completions)
                    if len(global_extracted_answers) == total_len:
                        table["extracted_answer"] = global_extracted_answers
                    else:
                        table["extracted_answer"] = [None] * total_len



                    if global_triggered is not None:
                        table["triggered"] = [global_triggered] * total_len
                        table["triggered_and_correct"] = [
                            global_triggered_and_correct
                        ] * total_len
                        table["triggered_and_wrong"] = [
                            global_triggered - global_triggered_and_correct
                        ] * total_len
                        table["correct_among_triggered"] = [
                            global_correct_among_triggered
                        ] * total_len
                        table["triggered_among_correct"] = [
                            global_triggered_among_correct
                        ] * total_len

                    if "flg_alternation" in logs:
                        table["flg_alternation"] = [logs["flg_alternation"]] * total_len

                    # Write to file: step-X.tsv
                    filename = f"step-{self.state.global_step}.tsv"

                    df = pd.DataFrame(table)
                    df.to_csv(self.logdir / filename, sep="\t", index=False)

                except Exception as e:
                    # Don't crash training if logging fails
                    print(f"Error writing completions log: {e}")
                    pass

        # Call Parent Log
        Trainer.log(self, logs, start_time)

        # Clear metrics
        if mode in self._metrics:
            self._metrics[mode].clear()
