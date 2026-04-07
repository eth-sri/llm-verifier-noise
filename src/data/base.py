"""
Base classes and utilities for dataset management with label noise simulation.

This module provides the core abstractions for handling datasets with simulated
label noise, including confusion matrix-based mixup strategies and abstract
interfaces for different training paradigms.
"""

import re
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Optional

import numpy as np
from datasets import Dataset
from transformers import PreTrainedTokenizerBase

from src.configs import MixupStrategies
from src.reward_almost_correct import (
    calc_relerr,
    levenshtein_distance,
    parse_float,
    relerr_is_correct,
)


@dataclass
class Mixup:
    """
    Implements label noise simulation through confusion matrix-based mixup.

    This class simulates label noise by modeling the relationship between true
    labels and observed (noisy) labels through a confusion matrix parameterized
    by True Positive Rate (TPR) and False Positive Rate (FPR). The noise model
    assumes a synthetic data generator with known accuracy that produces the
    initial solutions and a perfect source of ground truth labels w.r.t. which
    we can noise.

    Attributes:
        TPR: True Positive Rate - probability of correctly labeling positive samples
        FPR: False Positive Rate - probability of incorrectly labeling negative samples as positive
        FNR: False Negative Rate - computed as 1 - TPR
        TNR: True Negative Rate - computed as 1 - FPR
        generator_accuracy: Overall accuracy of the label generator
        TP, FP, TN, FN: Confusion matrix components (set via set_confusion_matrix)
    """

    TPR: float = 1.0  # True Positive Rate (sensitivity/recall)
    FPR: float = 0.0  # False Positive Rate (fall-out)

    FNR: float = 1.0 - TPR  # False Negative Rate
    TNR: float = 1.0 - FPR  # True Negative Rate

    # Accuracy of the synthetic data generator, i.e., the performance of the teacher
    # model in offline settings
    generator_accuracy: float = 1.0

    # Define how to distribute incorrect labels.
    # "uniform" for a completely-random flipping
    strategy: MixupStrategies = MixupStrategies.TARGETED
    targeted_candidates: list[list[int]] | None = (
        None  # deprecated; kept for backward compatibility
    )
    targeted_buckets: list[dict] | None = (
        None  # optional per-bucket configs for TARGETED
    )
    alternation: dict[str, Any] | None = (
        None  # optional periodic schedule for oracle/noisy rules by step
    )

    # Optional tokenizer, used for token-id based selectors on completion_full.
    # This is set by MixedUpDataset when available.
    tokenizer: PreTrainedTokenizerBase | None = None

    # Confusion matrix components (set via set_confusion_matrix)
    TP: int | None = None
    FP: int | None = None
    TN: int | None = None
    FN: int | None = None

    # Realized metrics across eligible items processed by mixup_rewards
    realized_TP: int = 0
    realized_FP: int = 0
    realized_TN: int = 0
    realized_FN: int = 0
    # Ground-truth positive/negative counts across eligible items processed so far
    gt_positive: int = 0
    gt_negative: int = 0

    # Selector-based trigger statistics (for TARGETED strategy)
    # These are reset on every call to mixup_rewards and are intended to be
    # aggregated by the trainer across processes.
    triggered: int = 0
    triggered_and_correct: int = 0
    # Oracle alternation step-level stats (reset every mixup call)
    oracle_step_active: int = 0
    oracle_step_items: int = 0

    # becomes True after first call of mixup_rewards
    first_time_printed: bool = False
    # Monotonic step index for alternation scheduling across mixup calls
    alternation_step_cursor: int = 0

    def _to_strategy(self, value: Any) -> MixupStrategies:
        """Coerce strategy value to MixupStrategies enum."""
        if isinstance(value, MixupStrategies):
            return value
        if isinstance(value, str):
            return MixupStrategies(value)
        raise ValueError(f"Unsupported strategy value: {value}")

    def _active_step_profile(self) -> dict[str, Any]:
        """
        Return active rule configuration for this step.

        The returned dictionary contains:
        - use_oracle: whether to bypass noise and use clean labels
        - TPR/FPR/strategy/targeted_buckets: active noisy config when not oracle

        Supported alternation interfaces:
        - rules=[{type,count,...}, ...] with period/offset
        """
        active: dict[str, Any] = {
            "use_oracle": False,
            "TPR": self.TPR,
            "FPR": self.FPR,
            "strategy": self.strategy,
            "targeted_buckets": self.targeted_buckets,
        }
        cfg = self.alternation
        if not self.alternation:
            return active

        period = int(cfg.get("period", 0))
        offset = int(cfg.get("offset", 0))
        rules = cfg.get("rules")

        if not (isinstance(rules, list) and len(rules) > 0):
            self.alternation_step_cursor += 1
            return active

        normalized_rules: list[dict[str, Any]] = []
        total_count = 0
        for raw in rules:
            if not isinstance(raw, dict):
                continue
            count = int(raw.get("count", 0))
            if count <= 0:
                continue
            total_count += count
            normalized_rules.append(raw)

        if not normalized_rules:
            self.alternation_step_cursor += 1
            return active

        if period <= 0:
            period = total_count
        if period <= 0:
            self.alternation_step_cursor += 1
            return active

        pos = (self.alternation_step_cursor + offset) % period
        cursor = 0
        chosen = normalized_rules[-1]
        for rule in normalized_rules:
            c = int(rule.get("count", 0))
            if pos < cursor + c:
                chosen = rule
                break
            cursor += c

        rule_type = str(chosen.get("type", "noisy")).lower()
        if rule_type == "oracle":
            active["use_oracle"] = True
        else:
            if "TPR" in chosen:
                active["TPR"] = float(chosen["TPR"])
            if "FPR" in chosen:
                active["FPR"] = float(chosen["FPR"])
            if "strategy" in chosen:
                active["strategy"] = self._to_strategy(chosen["strategy"])
            if "targeted_buckets" in chosen:
                active["targeted_buckets"] = chosen["targeted_buckets"]

        self.alternation_step_cursor += 1
        return active

    def set_confusion_matrix(self, n_samples: int) -> None:
        """
        Computes and sets confusion matrix values based on dataset size.

        Args:
            n_samples: Total number of samples in the dataset
        """
        p = int(n_samples * self.generator_accuracy)
        n = n_samples - p
        self.TP = int(p * self.TPR)
        self.FN = p - self.TP
        self.FP = int(n * self.FPR)
        self.TN = n - self.FP

    # -------- TARGETED selector utilities --------
    @staticmethod
    def _get_by_path(obj: Any, path: str, default: Any = None) -> Any:
        """
        Safely retrieve a nested value from a mapping using a dot-separated path.

        The lookup will traverse dictionaries along the provided path. If any
        intermediate key is missing or an intermediate object is not a dict,
        the provided default is returned.

        Args:
            obj: The root object (typically a dict) to read from.
            path: Dot-separated path to traverse (e.g., "a.b.c").
            default: Value to return when the path cannot be resolved.

        Returns:
            The resolved value if the full path exists; otherwise the default.
        """
        cur = obj
        for key in path.split("."):
            if isinstance(cur, dict) and key in cur:
                cur = cur[key]
            else:
                return default
        return cur

    @staticmethod
    def _eval_condition(val: Any, cond: dict) -> bool:
        """
        Evaluate a condition against a value.

        Supported operations (cond["op"]):
            - "eq" / "==": equality with cond["value"]
            - "ne" / "!=": inequality with cond["value"]
            - "in": membership in cond["values"] (iterable)
            - "nin": non-membership in cond["values"] (iterable)
            - "gt" / ">": greater than cond["value"]
            - "ge" / ">=": greater than or equal to cond["value"]
            - "lt" / "<": less than cond["value"]
            - "le" / "<=": less than or equal to cond["value"]
            - "contains": value contains cond["value"] (uses __contains__ if available)
            - "regex": re.search(cond["value"], str(val)) is not None; invalid patterns return False
            - "between": inclusive range check low <= val <= high with cond["range"] = [low, high]

        Args:
            val: The value to test.
            cond: A dictionary specifying the operation and any required operands.

        Returns:
            True if the condition holds; otherwise False.
        """
        op = cond.get("op")
        # Handle None values safely for comparison-type operations
        if val is None:
            # Only equality/inequality comparisons are meaningful with None
            if op in ("eq", "=="):
                return cond.get("value") is None
            if op in ("ne", "neq", "!="):
                return cond.get("value") is not None
            return False
        if op in ("eq", "=="):
            return val == cond.get("value")
        if op in ("ne", "neq", "!="):
            return val != cond.get("value")
        if op == "in":
            return val in cond.get("values", [])
        if op == "nin":
            return val not in cond.get("values", [])
        if op in ("gt", ">"):
            return val > cond.get("value")
        if op in ("ge", "geq", ">="):
            return val >= cond.get("value")
        if op in ("lt", "<"):
            return val < cond.get("value")
        if op in ("le", "leq", "<="):
            return val <= cond.get("value")
        if op == "contains":
            return cond.get("value") in val if hasattr(val, "__contains__") else False
        if op == "regex":
            pattern = cond.get("value", "")
            try:
                return re.search(pattern, str(val)) is not None
            except re.error:
                return False
        if op == "between":
            low, high = cond.get("range", [None, None])
            # print(f"{low=}, {high=}, {val=}")
            if low is None or high is None:
                return False
            return low <= val <= high
        # default: unsupported op -> False
        return False

    def _add_stats(self, item: dict, selector: dict) -> dict:
        """
        Add computed metrics to the item for targeted selector evaluation.

        Args:
            item: The original item dictionary.
            selector: The targeted selector specification.
        Returns:
            The enriched item dictionary with computed metrics added.
        """

        enriched_item = dict(item)

        # Determine which computed fields are requested by the selector's paths
        def _collect_paths(sel: dict | None) -> set[str]:
            paths: set[str] = set()
            if not isinstance(sel, dict):
                return paths
            if "all" in sel:
                for s in sel["all"]:
                    paths |= _collect_paths(s)
            elif "any" in sel:
                for s in sel["any"]:
                    paths |= _collect_paths(s)
            else:
                p = sel.get("path")
                if isinstance(p, str) and p:
                    paths.add(p)
            return paths

        requested_paths = _collect_paths(selector)
        if not self.first_time_printed:
            warnings.warn(f"Requested paths for computed stats: {requested_paths}")

        # Compute rel_error only if requested via selector path
        if "rel_error" in requested_paths:
            pred = parse_float(item.get("completion_extracted"))
            gt = parse_float(item.get("answer"))
            enriched_item["rel_error"] = calc_relerr(pred, gt)

        if "abs_error" in requested_paths:
            pred = parse_float(item.get("completion_extracted"))
            gt = parse_float(item.get("answer"))
            if pred is None or gt is None:
                enriched_item["abs_error"] = None
            else:
                enriched_item["abs_error"] = abs(pred - gt)

        # Signed relative error keeps the error direction:
        #   (pred - gt) / abs(gt)
        # This enables asymmetric ranges around ground-truth using TARGETED selectors.
        if "signed_rel_error" in requested_paths:
            pred = parse_float(item.get("completion_extracted"))
            gt = parse_float(item.get("answer"))
            if pred is None or gt is None or gt == 0.0:
                enriched_item["signed_rel_error"] = None
            else:
                enriched_item["signed_rel_error"] = (pred - gt) / abs(gt)

        if "signed_abs_error" in requested_paths:
            pred = parse_float(item.get("completion_extracted"))
            gt = parse_float(item.get("answer"))
            if pred is None or gt is None:
                enriched_item["signed_abs_error"] = None
            else:
                enriched_item["signed_abs_error"] = pred - gt

        # Compute Levenshtein edit distance if requested via selector path
        # NOTE: Reasoning-Gym spell_backward lowers the letters for evaluation, so we do the same here for training.
        if "levenshtein" in requested_paths:
            d = levenshtein_distance(
                item.get("completion_extracted"), item.get("answer"), do_lower=True
            )
            enriched_item["levenshtein"] = d

        if "pred_abs" in requested_paths:
            # abs(pred)
            pred = parse_float(item.get("completion_extracted"))
            if pred is None:
                enriched_item["pred_abs"] = None
            else:
                enriched_item["pred_abs"] = abs(pred)

        # Compute completion_full_token_len (tokenized length of completion_full)
        # only if requested via selector path. This enables selectors such as:
        #   selector: { path: "completion_full_token_len", op: "gt", value: 200 }
        # which will match items whose completion tokenization is longer than 200
        # tokens according to the current tokenizer.
        if "completion_full_token_len" in requested_paths:
            if self.tokenizer is None:
                warnings.warn(
                    "Mixup.tokenizer is not set; cannot compute completion_full_token_len. "
                    "Returning None for this field."
                )
                enriched_item["completion_full_token_len"] = None
            else:
                text = item.get("completion_full")
                if isinstance(text, str):
                    # Reuse the same token cache key as in _eval_selector to
                    # avoid double-tokenizing when both token-id and length
                    # based selectors are used.
                    token_cache_key = "_completion_full_token_ids"
                    if token_cache_key in item and isinstance(
                        item[token_cache_key], list
                    ):
                        token_ids = item[token_cache_key]
                    else:
                        encoded = self.tokenizer(
                            text,
                            add_special_tokens=False,
                            return_attention_mask=False,
                            return_token_type_ids=False,
                        )
                        token_ids = encoded.get("input_ids", [])
                        item[token_cache_key] = token_ids

                    enriched_item["completion_full_token_len"] = len(token_ids)
                else:
                    enriched_item["completion_full_token_len"] = None

            # print(f"Completion full token length: {len(token_ids)}")

        if "reverse_sign_rel_error" in requested_paths:
            # reverse sign relative error = abs(abs(pred) - abs(gt)) / abs(gt) only for sign(pred) != sign(gt)
            pred = parse_float(item.get("completion_extracted"))
            gt = parse_float(item.get("answer"))
            if pred is None or gt is None:
                enriched_item["reverse_sign_rel_error"] = None
            else:
                # Only compute reverse sign relative error when signs differ, else set to None
                if np.sign(pred) != np.sign(gt):
                    enriched_item["reverse_sign_rel_error"] = abs(
                        abs(pred) - abs(gt)
                    ) / abs(gt)
                else:
                    enriched_item["reverse_sign_rel_error"] = None

        if "language" in requested_paths:
            from langdetect import DetectorFactory, detect

            DetectorFactory.seed = 0
            try:
                enriched_item["language"] = detect(str(item.get("completion_full", "")))
            except:
                enriched_item["language"] = None
            if not self.first_time_printed:
                print(f"language for {item.get('completion_full', '')[:10]}...:")
                print(enriched_item["language"])

        # end of computed fields
        assert {path.split(".")[0] for path in requested_paths}.issubset(
            enriched_item.keys()
        ), f"{requested_paths} has keys not in {enriched_item.keys()}"

        return enriched_item

    def _eval_selector(self, item: dict, selector: dict) -> bool:
        """
        Evaluate a hierarchical selector against an item.

        The selector can be a composite:
            - {"all": [sel1, sel2, ...]}: all sub-selectors must pass
            - {"any": [sel1, sel2, ...]}: at least one sub-selector must pass

        Or a leaf:
            - {"path": "a.b.c", "op": "...", ...}: fetches item["a"]["b"]["c"]
              with _get_by_path and applies _eval_condition with the remaining keys.

        Args:
            item: The input dictionary to evaluate.
            selector: The selector specification.

        Returns:
            True if the item matches the selector; otherwise False.
        """

        if "all" in selector:
            return all(self._eval_selector(item, s) for s in selector["all"])
        if "any" in selector:
            return any(self._eval_selector(item, s) for s in selector["any"])
        # leaf
        path = selector.get("path")
        if not path:
            return False

        op = selector.get("op")

        # Special case: token-id based contains selector on completion_full.
        # Example YAML:
        #   selector: { path: "completion_full", op: "contains", token_id: 12345 }
        # or
        #   selector: { path: "completion_full", op: "contains", token_ids: [123, 456] }
        if (
            path == "completion_full"
            and op == "contains"
            and (
                "token_id" in selector
                or "token_ids" in selector
                or "token_str" in selector
            )
        ):
            if self.tokenizer is None:
                warnings.warn(
                    "Mixup.tokenizer is not set; cannot evaluate token_id-based selector "
                    "on completion_full. Returning False for this selector."
                )
                return False

            text = self._get_by_path(item, path)
            if not isinstance(text, str):
                return False

            # Cache tokenization on the item to avoid repeated work when
            # multiple buckets reference completion_full tokens.
            token_cache_key = "_completion_full_token_ids"
            if token_cache_key in item and isinstance(item[token_cache_key], list):
                token_ids = item[token_cache_key]
            else:
                encoded = self.tokenizer(
                    text,
                    add_special_tokens=False,
                    return_attention_mask=False,
                    return_token_type_ids=False,
                )
                token_ids = encoded.get("input_ids", [])
                item[token_cache_key] = token_ids

            # Collect target token ids from selector.
            #
            # Semantics:
            #   - token_id:  match if this single token id appears anywhere.
            #   - token_ids: match if this *contiguous sequence* of token ids
            #                appears in order (i.e., a subsequence, not just
            #                any-of).
            #  - token_str: match if the tokenization of this string appears
            if "token_ids" in selector:
                try:
                    seq = [int(t) for t in selector["token_ids"]]
                except (TypeError, ValueError):
                    return False
                if not seq:
                    return False

                # Check for contiguous subsequence match
                m = len(seq)
                n = len(token_ids)
                if m > n:
                    return False
                for i in range(n - m + 1):
                    if token_ids[i : i + m] == seq:
                        return True
                return False

            elif "token_id" in selector:
                try:
                    tid = int(selector["token_id"])
                except (TypeError, ValueError):
                    return False
                return tid in token_ids

            elif "token_str" in selector:
                tid = self.tokenizer.encode(selector["token_str"])
                if len(tid) == 1:
                    return tid[0] in token_ids
                else:
                    # check for contiguous subsequence match
                    m = len(tid)
                    n = len(token_ids)
                    if m > n:
                        return False
                    for i in range(n - m + 1):
                        if token_ids[i : i + m] == tid:
                            return True
                    return False


            return False

        # Default: value-based selector using the resolved field.
        val = self._get_by_path(item, path)
        return self._eval_condition(val, selector)

    def mixup_rewards(
        self,
        rewards: list[bool],
        items: list[dict[str, Any]] | None = None,
        oracle_rewards: list[bool] | None = None,
    ) -> list[float]:
        """
        Apply confusion-matrix mixup to binary rewards and return noisy labels.

        Strategies:
        - UNIFORM: apply TPR/FPR to all items.
        - FORMAT: only items with extractable outputs are eligible for flips.
        - TARGETED: only items that satisfy at least one selector in targeted_buckets
          are eligible for flips. Items must be provided to compute the mask.

        Args:
            rewards: List of ground-truth binary labels (True=positive, False=negative).
            question_list: (Optional) List of inputs for each item.
            output_list: (Optional) List of raw model outputs (strings) for each item.
            completions_processed: (Optional) List of processed (extracted) completions for each item.
            answer_processed: (Optional) List of processed answers (strings) for each item.
                `verify(completions_processed[i], answer_processed[i])` is expected to give the ground-truth reward.

        Returns:
            List of noisy labels as floats (0.0 or 1.0).

        Raises:
            ValueError: If required masks are missing or lengths do not match rewards.
            ValueError: If items are required to evaluate targeted_buckets but are not provided.
            NotImplementedError: If an unknown mixup strategy is set.

        Side Effects:
            Updates realized_* counters (TP, FP, TN, FN) and totals over eligible items
            processed so far, which are exposed via realized_tpr and realized_fpr.
        """
        n = len(rewards)
        self.gt_positive = 0
        self.gt_negative = 0
        self.realized_TP = 0
        self.realized_FP = 0
        self.realized_TN = 0
        self.realized_FN = 0
        self.triggered = 0
        self.triggered_and_correct = 0
        self.oracle_step_active = 0
        self.oracle_step_items = 0

        # if oracle_rewards is provided, immediately return (not oracle) rewards without mixup
        if oracle_rewards is not None:
            for oracle_r, train_r in zip(oracle_rewards, rewards):
                if oracle_r:
                    self.gt_positive += 1
                    if train_r:
                        self.realized_TP += 1
                    else:
                        self.realized_FN += 1
                else:
                    self.gt_negative += 1
                    if train_r:
                        self.realized_FP += 1
                    else:
                        self.realized_TN += 1
            return [1.0 if r else 0.0 for r in rewards]

        # Resolve active step rule (oracle or noisy profile).
        active_step = self._active_step_profile()
        active_TPR = float(active_step["TPR"])
        active_FPR = float(active_step["FPR"])
        active_strategy = active_step["strategy"]
        active_targeted_buckets = active_step["targeted_buckets"]
        use_oracle_step = bool(active_step["use_oracle"])

        # Per-item TPR/FPR (default to active verifier values)
        per_item_TPR = [active_TPR] * n
        per_item_FPR = [active_FPR] * n

        # Build eligibility mask up-front based on strategy
        if active_strategy == MixupStrategies.UNIFORM:
            is_flip_target = [True] * n
        elif active_strategy == MixupStrategies.FORMAT:
            if items is None or items[0].get("completion_extracted") is None:
                raise ValueError("completion_extracted required for FORMAT mixup")
            is_flip_target = [item["completion_extracted"] != "" for item in items]
        elif active_strategy == MixupStrategies.TARGETED:
            if items is None:
                raise ValueError("items must be provided for TARGETED strategy")
            if not active_targeted_buckets:
                raise ValueError(
                    "targeted_buckets must be provided for TARGETED mixup strategy"
                )

            # Build a combined selector so _add_stats knows what computed fields
            # (e.g., rel_error, levenshtein) need to be added.
            combined_selector = {
                "any": [b.get("selector", {}) for b in active_targeted_buckets]
            }
            items = [self._add_stats(it, combined_selector) for it in items]

            is_flip_target = []

            for idx, (it, rew) in enumerate(zip(items, rewards)):
                matched = False
                for bucket in active_targeted_buckets:
                    if not self.first_time_printed:
                        warnings.warn(
                            f"Evaluating bucket selector: {bucket.get('selector', {})}, "
                            f"TPR = {bucket.get('TPR', active_TPR)}, "
                            f"FPR = {bucket.get('FPR', active_FPR)}"
                        )
                    sel = bucket.get("selector", {})
                    # if the item matches multiple buckets, the first match takes precedence
                    if self._eval_selector(it, sel):
                        is_flip_target.append(True)
                        per_item_TPR[idx] = float(bucket.get("TPR", active_TPR))
                        per_item_FPR[idx] = float(bucket.get("FPR", active_FPR))
                        matched = True
                        break
                if not matched:
                    # Not eligible for flip; stays clean
                    is_flip_target.append(False)
        else:
            raise NotImplementedError(f"Unknown mixup strategy: {active_strategy}")

        mixed_rewards: list[float] = []
        for r, ok, tpr_i, fpr_i in zip(
            rewards, is_flip_target, per_item_TPR, per_item_FPR
        ):
            # Flip according to per-item TPR/FPR (applies to all items, whether matched or not)
            coin = np.random.uniform()
            threshold = tpr_i if r else fpr_i
            mixed_rewards.append(1.0 if coin < threshold else 0.0)

        # Oracle step bypasses flipping and uses clean labels for all items.
        self.oracle_step_active = 1 if use_oracle_step else 0
        self.oracle_step_items = n if use_oracle_step else 0
        if use_oracle_step:
            mixed_rewards = [1.0 if r else 0.0 for r in rewards]

        # Update realized metrics over is_flip_target items
        for r, y, ok in zip(rewards, mixed_rewards, is_flip_target):
            if r:
                self.gt_positive += 1
                if y == 1.0:
                    self.realized_TP += 1
                else:
                    self.realized_FN += 1
            else:
                self.gt_negative += 1
                if y == 1.0:
                    self.realized_FP += 1
                else:
                    self.realized_TN += 1

        # Selector-based trigger statistics (TARGETED strategy only).
        # "Triggered" means: the item matched at least one TARGETED bucket
        # (i.e., is_flip_target is True under TARGETED strategy).
        # "Triggered and correct" means: triggered and oracle/ground-truth
        # label is positive (True).
        if active_strategy == MixupStrategies.TARGETED:
            for r, ok in zip(rewards, is_flip_target):
                if not ok:
                    continue
                self.triggered += 1
                if bool(r):
                    self.triggered_and_correct += 1

        self.first_time_printed = True  # to avoid repetitive prints
        return mixed_rewards

    @property
    def accuracy(self) -> float:
        """Computes overall accuracy from confusion matrix."""
        if self.TP is None or self.TN is None or self.FP is None or self.FN is None:
            raise ValueError(
                "Confusion matrix values (TP, TN, FP, FN) must be set before calculating accuracy"
            )
        return (self.TP + self.TN) / (self.TP + self.TN + self.FP + self.FN)

    @property
    def precision(self) -> float:
        """Computes precision from confusion matrix."""
        if self.TP is None or self.FP is None:
            raise ValueError("TP and FP must be set before calculating precision")
        if self.TP + self.FP == 0:
            return 0.0
        return self.TP / (self.TP + self.FP)

    @property
    def recall(self) -> float:
        """Computes recall (same as TPR) from confusion matrix."""
        if self.TP is None or self.FN is None:
            raise ValueError("TP and FN must be set before calculating recall")
        if self.TP + self.FN == 0:
            return 0.0
        return self.TP / (self.TP + self.FN)

    @property
    def f1_score(self) -> float:
        """Computes F1 score from precision and recall."""
        precision = self.precision
        recall = self.recall
        if precision + recall == 0:
            return 0.0
        return 2 * (precision * recall) / (precision + recall)

    @property
    def realized_tpr(self) -> float:
        """
        Realized TPR over eligible items processed so far.
        For UNIFORM: all items are eligible.
        For FORMAT: only items with has_extractable=True are eligible.
        Falls back to -1 when no eligible positive samples have been seen yet.
        """
        if self.gt_positive == 0:
            return -1
        return self.realized_TP / self.gt_positive

    @property
    def realized_fpr(self) -> float:
        """
        Realized FPR over eligible items processed so far.
        Falls back to -1 when no eligible negative samples have been seen yet.
        """
        if self.gt_negative == 0:
            return -1
        return self.realized_FP / self.gt_negative


class MixedUpDataset(ABC):
    """
    Abstract base class for datasets with label noise simulation capabilities.

    This class provides a common interface for datasets that support various training
    paradigms (SFT, DPO, PPO, GRPO) while incorporating label noise through mixup
    strategies. Concrete implementations should handle dataset-specific loading,
    processing, and formatting for different training objectives.

    Attributes:
        name: Dataset identifier
        tokenizer: Tokenizer for text processing
        shuffle: Whether to shuffle the dataset
        max_length: Maximum sequence length for tokenization
        mixup: Mixup configuration for label noise simulation
    """

    def __init__(
        self,
        name: str,
        tokenizer: PreTrainedTokenizerBase,
        mixup: Mixup,
        shuffle: bool = True,
        max_length: int = 1024,
    ) -> None:
        self.name = name
        self.tokenizer = tokenizer
        self.shuffle = shuffle
        self.max_length = max_length
        self.mixup = mixup
        # Propagate tokenizer to Mixup so that selectors can operate on
        # tokenized fields such as completion_full when using token_id-based
        # conditions in targeted_buckets.
        self.mixup.tokenizer = tokenizer

    def __len__(self) -> int:
        """Returns the size of the underlying dataset."""
        return len(self.raw_dataset)

    @abstractmethod
    def prepare_for_sft(self) -> Dataset:
        """
        Prepares the dataset for Supervised Fine-Tuning (SFT).

        Returns:
            Dataset formatted for SFT training with input-output pairs
        """
        pass

    @abstractmethod
    def prepare_for_distil(self) -> Dataset:
        """
        Prepares the dataset for knowledge distillation training.

        Returns:
            Dataset formatted for distillation with teacher-student pairs
        """
        pass

    @abstractmethod
    def prepare_for_dpo(self) -> Dataset:
        """
        Prepares the dataset for Direct Preference Optimization (DPO).

        Returns:
            Dataset formatted for DPO training with preference pairs
        """
        pass

    @abstractmethod
    def prepare_for_ppo(self) -> None:
        """
        Prepares the dataset for Proximal Policy Optimization (PPO).

        Note: Return type is currently undefined as PPO typically uses
              online data generation rather than static datasets.

        """
        pass

    @abstractmethod
    def prepare_for_grpo(self) -> tuple[Dataset, Callable]:
        """
        Prepares the dataset for Group Relative Policy Optimization (GRPO).

        Note: Return type is currently undefined as GRPO typically uses
              online data generation rather than static datasets.
        """
        pass
