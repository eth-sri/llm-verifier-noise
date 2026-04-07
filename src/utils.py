"""
Utility functions for the LLM Verifier Noise project.

This module provides a collection of utility functions supporting various aspects
of the project including:
- Mathematical evaluation metrics (pass@k)
- LaTeX formula extraction and processing
- Text processing and validation
- Dataset loading with fallback mechanisms
- Execution control (timeouts, output suppression)
- Model tokenization enhancements

The utilities are designed to be reusable across different components of the
training and evaluation pipeline.
"""

import math
import multiprocessing
import os
import re
import sys
import warnings
from typing import Any, Callable

import huggingface_hub
from datasets import Dataset, get_dataset_infos, load_dataset
from transformers import PreTrainedTokenizerBase


def normalize_hf_model_name(model_name: str) -> tuple[str, dict[str, str]]:
    """Normalize a Hugging Face model identifier into base repo id + optional subfolder.

    This allows passing names like

    - ``"owner/repo"`` → ("owner/repo", {})
    - ``"owner/repo/checkpoint-50"`` → ("owner/repo", {"subfolder": "checkpoint-50"})
    - ``"owner/repo/some/deep/checkpoint"`` →
        ("owner/repo", {"subfolder": "some/deep/checkpoint"})

    The behavior is only changed when there are *more than two* path components
    (i.e. more than one ``/``) and the path does not look like a local filesystem
    path ("./", "../" or absolute "/"). This matches the common pattern for
    HF Hub repos with checkpoints stored in subfolders while avoiding
    unintentionally rewriting local paths.

    Args:
        model_name: Raw model name / identifier coming from configs.

    Returns:
        A tuple ``(base, extra_kwargs)`` where ``base`` should be passed as the
        first argument to ``from_pretrained`` and ``extra_kwargs`` should be
        expanded into the call (it may contain a ``subfolder`` key).
    """

    # Leave obvious local paths untouched (we don't want to split those)
    if (
        model_name.startswith("./")
        or model_name.startswith("../")
        or model_name.startswith("/")
    ):
        return model_name, {}

    parts = model_name.split("/")
    if len(parts) <= 2:
        # Standard HF id like "owner/repo" → no subfolder
        return model_name, {}

    base = "/".join(parts[:2])
    subfolder = "/".join(parts[2:])
    return base, {"subfolder": subfolder}


def esc(s: str) -> str:
    """Escape function for safe filename generation."""
    return s.replace("/", "-")


def pass_at_k(k: int, c: int, n: int) -> float:
    """
    Compute the pass@k metric.

    Args:
        k (int): Number of allowed attempts.
        c (int): Number of correct solutions.
        n (int): Total number of solutions.

    Returns:
        float: The probability of passing at k attempts.
    """
    if n - c < k:
        return 1.0
    return 1.0 - math.prod([1.0 - k / i for i in range(n - c + 1, n + 1)])


def remove_boxed(s: str) -> str:
    """
    Function is forked from: https://github.com/volcengine/verl/blob/main/verl/utils/reward_score/math.py

    Copyright 2023-2024 Bytedance Ltd. and/or its affiliates
    """
    if "\\boxed " in s:
        left = "\\boxed "
        assert s[: len(left)] == left
        return s[len(left) :]
    elif "\\boxed{" in s:
        left = "\\boxed{"
        assert s[: len(left)] == left, f"Issue: {s}, expected {left} at the start."
        assert s[-1] == "}"
        return s[len(left) : -1]
    elif "\\boxed\\{" in s:
        left = "\\boxed\\{"
        assert s[: len(left)] == left, f"Issue: {s}, expected {left} at the start."
        assert s[-1] == "}"
        return s[len(left) : -1]
    else:
        return s


def last_boxed_only_string(string: str) -> str | None:
    """
    Function is forked from: https://github.com/volcengine/verl/blob/main/verl/utils/reward_score/math.py

    Copyright 2023-2024 Bytedance Ltd. and/or its affiliates
    """
    idx = string.rfind("\\boxed")
    if "\\boxed " in string:
        return "\\boxed " + string.split("\\boxed ")[-1].split("$")[0]
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    retval = None if right_brace_idx is None else string[idx : right_brace_idx + 1]

    return retval


def extract_boxed(string: str) -> str:
    """
    Extracts the boxed solution from a LaTeX string.

    Finds and extracts mathematical expressions enclosed in \\boxed{} commands,
    which is a common format for final answers in mathematical problems.

    Args:
        string: The LaTeX string containing the boxed solution.

    Returns:
        The extracted boxed solution content. Returns an empty string if not found.

    Examples:
        >>> extract_boxed("The answer is \\boxed{42}")
        "42"
        >>> extract_boxed("No answer here")
        ""
    """
    boxed = last_boxed_only_string(string)
    if boxed is None:
        return ""
    return remove_boxed(boxed)


def extract_tf(string: str) -> str:
    """
    Extract the last occurrence of:
    \boxed{true}, \boxed{\text{true}}, \boxed{false}, \boxed{\text{false}}

    useful for llm-as-a-judge
    """
    pattern = r'\\boxed{(?:\\text{)?(true|false)}'
    matches = re.findall(pattern, string)
    return matches[-1] if matches else ""

def process_ans(string: str) -> str:
    """
    Ensures a mathematical answer string is properly formatted for parsing.

    Wraps the input string in LaTeX math delimiters ($...$) if they are not
    already present. This standardizes the format for downstream processing
    by mathematical verification libraries.

    Args:
        string: The mathematical answer string to process.

    Returns:
        The processed string wrapped in appropriate LaTeX math delimiters.

    Examples:
        >>> process_ans("x = 5")
        "$x = 5$"
        >>> process_ans("$already formatted$")
        "$already formatted$"
    """
    string = string.strip()
    if string.startswith("$") and string.endswith("$"):
        return string
    elif string.startswith("$"):
        return string + "$"
    elif string.endswith("$"):
        return "$" + string
    else:
        return "$" + string + "$"


def add_think_to_gen_prompt(
    tokenizer: PreTrainedTokenizerBase, base_model: str
) -> PreTrainedTokenizerBase:
    """
    Modifies a tokenizer's chat template to include reasoning tokens.

    Adds the opening <think> token to the assistant generation prompt in the
    chat template. This enables models to generate explicit reasoning traces
    before providing final answers, supporting chain-of-thought style generation.

    Currently supports Qwen2.5 models with plans to extend to other model families.

    Args:
        tokenizer: The tokenizer whose chat template should be modified.
        base_model: The base model name to determine the appropriate modification.

    Returns:
        The modified tokenizer with updated chat template.

    Raises:
        NotImplementedError: If the base model is not supported for think token addition.

    Note:
        The function modifies the tokenizer in-place and also returns it.
    """
    if "qwen2.5" in base_model.lower():
        chat_template = tokenizer.chat_template
        assert (
            "<|im_start|>assistant" in chat_template
        ), f"<|im_start|>assistant not found in chat template"
        chat_template = chat_template.replace(
            "<|im_start|>assistant", r"<|im_start|>assistant\n<think>"
        )
        assert (
            "<think>" in chat_template
        ), f"<think> not found in the updated chat template"
        tokenizer.chat_template = chat_template
    else:
        raise NotImplementedError(
            f"Adding think tokens not implemented for {base_model}"
        )
    return tokenizer


def contains_chinese_chars(s: str) -> bool:
    """
    Checks if any Chinese characters are present in a given string.

    Detects the presence of Chinese characters in the Unicode range 0x4E00-0x9FFF,
    which covers most commonly used Chinese characters (CJK Unified Ideographs).
    Useful for filtering or special handling of multilingual datasets.

    Args:
        s: The string to check for Chinese characters.

    Returns:
        True if Chinese characters are found, False otherwise.

    Note:
        Excludes the first character in the range (which resembles an em-dash)
        to avoid false positives.
    """
    chinese_chars = [chr(i) for i in range(0x4E00, 0x9FFF + 1)]
    # remove the number 1 which looks like the em-dash
    chinese_chars = chinese_chars[1:]
    return any([c in s for c in chinese_chars])


def suppress_output_and_warnings(f: Callable, *args, **kwargs) -> Any:
    """
    Executes a function while suppressing stdout and warnings.

    Temporarily redirects stdout to devnull and suppresses all warning messages
    during function execution. Useful for running noisy functions that produce
    unnecessary output during batch processing.

    Args:
        f: The function to execute.
        *args: Positional arguments to pass to the function.
        **kwargs: Keyword arguments to pass to the function.

    Returns:
        The return value of the executed function.

    Note:
        stdout is restored after execution regardless of whether the function
        succeeds or raises an exception.
    """
    # Suppress stdout
    with open(os.devnull, "w") as devnull, warnings.catch_warnings():
        sys_stdout = sys.stdout  # Save real stdout
        sys.stdout = devnull
        warnings.simplefilter("ignore")
        try:
            result = f(*args, **kwargs)
        finally:
            sys.stdout = sys_stdout  # Restore
        return result


def run_func_with_timeout(
    f: Callable, timeout: int, suppress_out: bool = False, *args, **kwargs
) -> Any:
    """
    Executes a callable function with a specified timeout.

    Uses multiprocessing to run a function with a hard timeout limit. Optionally
    suppresses output and warnings during execution. Useful for preventing
    hanging operations in batch processing pipelines.

    Args:
        f: The function to execute.
        timeout: Maximum execution time in seconds.
        suppress_out: Whether to suppress stdout and warnings during execution.
        *args: Positional arguments to pass to the function.
        **kwargs: Keyword arguments to pass to the function.

    Returns:
        The return value of the executed function.

    Raises:
        TimeoutError: If the function execution exceeds the timeout limit.

    Note:
        The process pool is terminated if a timeout occurs, which forcefully
        stops the running function.
    """
    with multiprocessing.Pool(processes=1) as pool:
        if suppress_out:
            async_result = pool.apply_async(
                suppress_output_and_warnings, (f, args, kwargs)
            )
        else:
            async_result = pool.apply_async(f, args, kwargs)
        try:
            return async_result.get(timeout=timeout)
        except multiprocessing.TimeoutError:
            pool.terminate()
            print("Function timed out")
            raise TimeoutError


def load_dataset_with_fallback(
    primary: str, fallback: str, split="train"
) -> tuple[Dataset, bool]:
    """
    Attempts to load a primary dataset with automatic fallback to a secondary dataset.

    Tries to load the primary dataset first, and if it fails (doesn't exist or
    encounters an error), automatically falls back to loading the secondary dataset.
    This is useful for handling dataset availability issues or experimental setups
    where different versions of datasets might be used.

    Args:
        primary: Name/path of the primary dataset to attempt loading.
        fallback: Name/path of the fallback dataset to use if primary fails.
        split: Dataset split to load (default: "train").

    Returns:
        A tuple containing:
        - The loaded dataset (either primary or fallback)
        - Boolean indicating whether fallback was used (True) or primary was loaded (False)

    Raises:
        Exception: If both primary and fallback dataset loading fail.
    """
    try:
        huggingface_hub.dataset_info(primary)
        print(f"Loading primary dataset: {primary}")
        ds = load_dataset(primary, split=split)
        return ds, False

    except Exception as e:
        print(f"Failed with dataset '{primary}'. Error: {e}")
        print(f"Falling back to: {fallback}")
        ds = load_dataset(fallback, split=split)
        return ds, True


def get_split_size(dataset_name: str, split: str = "train", config: str = None) -> int:
    """
    Gets the number of examples in a dataset split without downloading the dataset.

    Efficiently retrieves dataset size information using HuggingFace's metadata
    without downloading the actual dataset files. Useful for planning experiments
    and understanding dataset characteristics before processing.

    Args:
        dataset_name: Name of the HuggingFace dataset.
        split: Dataset split to query (default: "train").
        config: Dataset configuration name. If None and dataset has only one
               configuration, it will be automatically selected.

    Returns:
        Number of examples in the specified dataset split.

    Raises:
        ValueError: If the dataset has multiple configs but none is specified,
                   or if the specified split doesn't exist.
    """
    infos = get_dataset_infos(dataset_name)

    # If no config is provided and the dataset only has one, auto-pick it
    if config is None:
        if len(infos) == 1:
            config = list(infos.keys())[0]
        else:
            raise ValueError(
                f"Dataset '{dataset_name}' has multiple configs: "
                f"{list(infos.keys())}. Please specify one."
            )

    try:
        return int(infos[config].splits[split].num_examples)
    except KeyError as e:
        raise ValueError(
            f"Split '{split}' not found in dataset '{dataset_name}' (config '{config}')"
        ) from e
