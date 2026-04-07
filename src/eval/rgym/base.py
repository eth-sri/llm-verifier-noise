import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import reasoning_gym
import tabulate
from reasoning_gym.utils import extract_answer as rgym_extract_answer
from tqdm import tqdm

from src.eval.base import Benchmark
from src.inference_models import Conversation, get_inference_model
from src.reward_almost_correct import levenshtein_is_correct, relerr_is_correct
from src.utils import esc, extract_boxed, pass_at_k


class RGym(Benchmark):
    """
    Unified Reasoning Gym evaluation.

    Responsibilities:
    - Build datasets from categories/datasets specs
    - Generate multi-sample model completions
    - Score using the dataset's score_answer implementation
    - Persist and reload results
    - Optionally compute stratified metrics on dataset entry metadata keys
    - Internally select variant-specific behavior (paths, filenames, prompt, stratified metrics)
      based on the datasets specified in `categories`.

    Variant selection rules (based on dataset names in categories):
      - If all datasets are 'chain_sum' (or unspecified -> defaults to 'chain_sum'):
          PATH_DIRNAME = 'RGYM_chain_sum'
          FILE_PREFIX  = 'rgym_chain_sum'
          STRATIFY_KEYS = ['num_terms', 'num_digits']
      - If all datasets are 'figlet_font':
          PATH_DIRNAME = 'RGYM_figlet_font'
          FILE_PREFIX  = 'rgym_figlet_font'
          STRATIFY_KEYS = ['font', 'space_letters']
      - If mixed or unknown datasets:
          PATH_DIRNAME = 'RGYM_mixed'
          FILE_PREFIX  = 'rgym_mixed'
          STRATIFY_KEYS = []
    """

    # Generic defaults (used for mixed/unknown and as fallbacks)
    PATH_DIRNAME = "RGYM_generic"
    FILE_PREFIX = "rgym_generic"
    PROMPT = "Please reason step by step, and put your final answer within \\boxed{}."
    STRATIFY_KEYS: List[str] = []

    # Variant definitions
    DEFAULT_PROMPT = (
        "Please reason step by step, and put your final answer within \\boxed{}."
    )
    VARIANT_CONFIGS = {
        "chain_sum": {
            # {'source_dataset': 'chain_sum', 'source_index': 0, 'num_terms': 2, 'num_digits': 1, 'expression': '4 + 3', 'difficulty': {'num_terms': (2, 6), 'num_digits': (1, 4)}}
            "PATH_DIRNAME": "RGYM_chain_sum",
            "FILE_PREFIX": "rgym_chain_sum",
            "PROMPT": DEFAULT_PROMPT,
            "STRATIFY_KEYS": ["num_terms", "num_digits"],
        },
        "figlet_font": {
            "PATH_DIRNAME": "RGYM_figlet_font",
            "FILE_PREFIX": "rgym_figlet_font",
            "PROMPT": DEFAULT_PROMPT,
            "STRATIFY_KEYS": ["font", "space_letters"],
        },
        "spell_backward": {
            "PATH_DIRNAME": "RGYM_spell_backward",
            "FILE_PREFIX": "rgym_spell_backward",
            "PROMPT": DEFAULT_PROMPT,
            "STRATIFY_KEYS": ["word_len"],
        },
        "number_sequence": {
            # {'source_dataset': 'number_sequence', 'source_index': 0, 'rule': 'double', 'complexity': 3, 'sequence': [3, 6, 12, 24, 48, 96, 192, 384, 768], 'difficulty': {'max_complexity': 3, 'terms': (4, 8)}}
            "PATH_DIRNAME": "RGYM_number_sequence",
            "FILE_PREFIX": "rgym_number_sequence",
            "PROMPT": DEFAULT_PROMPT,
            "STRATIFY_KEYS": ["len(sequence)", "complexity"],
        },
        # {'source_dataset': 'puzzle24', 'source_index': 0, 'numbers': [4, 3, 9, 8], 'difficulty': {'value': (1, 10)}}
        "puzzle24": {
            "PATH_DIRNAME": "RGYM_puzzle24",
            "FILE_PREFIX": "rgym_puzzle24",
            "PROMPT": DEFAULT_PROMPT,
            "STRATIFY_KEYS": [],
        },
        # {'source_dataset': 'countdown', 'source_index': 0, 'numbers': [36, 29, 95, 32, 4, 15], 'target': 139, 'expression': '15 - 4 + 95 + 36 - 32 + 29', 'difficulty': {'numbers': (4, 6), 'target': (100, 999), 'value': (1, 100)}}
        "countdown": {
            "PATH_DIRNAME": "RGYM_countdown",
            "FILE_PREFIX": "rgym_countdown",
            "PROMPT": DEFAULT_PROMPT,
            "STRATIFY_KEYS": ["len(numbers)"],
        },
        # {'source_dataset': 'simple_integration', 'source_index': 0, 'integrand': '70*x**6 + 12*x**2/5', 'variable': 'x', 'num_terms': 2, 'difficulty': {'terms': (2, 5)}}
        "simple_integration": {
            "PATH_DIRNAME": "RGYM_simple_integration",
            "FILE_PREFIX": "rgym_simple_integration",
            "PROMPT": DEFAULT_PROMPT,
            "STRATIFY_KEYS": ["num_terms"],
        },
        # {'source_dataset': 'simple_geometry', 'source_index': 0, 'n_sides': 3, 'known_angles': [16.0, 80.0], 'sum_of_known_angles': 96.0, 'missing_angle_raw': 84.0, 'missing_angle_rounded': 84, 'total_interior_sum': 180, 'difficulty': {'sides': (3, 6)}}
        "simple_geometry": {
            "PATH_DIRNAME": "RGYM_simple_geometry",
            "FILE_PREFIX": "rgym_simple_geometry",
            "PROMPT": DEFAULT_PROMPT
            + "If the answer is an integer value, do not include any decimal points (e.g. write '\\boxed{2}' instead of '\\boxed{2.0}').",
            "STRATIFY_KEYS": ["n_sides"],
        },
        # {'source_dataset': 'binary_alternation', 'source_index': 0, 'string': '011001010101010011101001010110', 'solution': 5, 'solvable': True, 'n': 30, 'difficulty': {'n': (10, 30)}}
        "binary_alternation": {
            "PATH_DIRNAME": "RGYM_binary_alternation",
            "FILE_PREFIX": "rgym_binary_alternation",
            "PROMPT": DEFAULT_PROMPT,
            "STRATIFY_KEYS": ["n", "solvable"],
        },
        # {'source_dataset': 'gcd', 'source_index': 0, 'numbers': [26, 760], 'result': 2, 'num_terms': 2, 'difficulty': {'num_terms': (2, 2), 'value': (1, 1000)}}
        "gcd": {
            "PATH_DIRNAME": "RGYM_gcd",
            "FILE_PREFIX": "rgym_gcd",
            "PROMPT": DEFAULT_PROMPT,
            "STRATIFY_KEYS": ["num_terms"],
        },
        # {'source_dataset': 'decimal_chain_sum', 'source_index': 0, 'num_terms': 2, 'num_digits': 1, 'expression': '4.23 + 3.96', 'difficulty': {'num_terms': (2, 6), 'num_digits': (1, 4), 'decimal_places': (1, 4)}}
        "decimal_chain_sum": {
            "PATH_DIRNAME": "RGYM_decimal_chain_sum",
            "FILE_PREFIX": "rgym_decimal_chain_sum",
            "PROMPT": DEFAULT_PROMPT,
            "STRATIFY_KEYS": ["num_terms", "num_digits", "decimal_places"],
        },
        # {'source_dataset': 'letter_counting', 'source_index': 0, 'span_length': 15, 'target_letter': 'a', 'span': ['bed', 'and', 'enters', 'his', 'mechanical', 'dresser', 'Two', 'minutes', 'later', 'the', 'machine', 'deposited', 'him', 'all', 'dressed'], 'difficulty': {'words': (5, 15)}}
        "letter_counting": {
            "PATH_DIRNAME": "RGYM_letter_counting",
            "FILE_PREFIX": "rgym_letter_counting",
            "PROMPT": DEFAULT_PROMPT,
            "STRATIFY_KEYS": ["span_length"],
        },
    }
    MIXED_FALLBACK = {
        "PATH_DIRNAME": "RGYM_mixed",
        "FILE_PREFIX": "rgym_mixed",
        "PROMPT": DEFAULT_PROMPT,
        "STRATIFY_KEYS": [],
    }

    @staticmethod
    def _collect_dataset_names(categories: List[Dict[str, Any]]) -> Set[str]:
        names: Set[str] = set()
        for cat in categories or []:
            for dcfg in cat.get("datasets", []) or []:
                ds = dcfg.get("dataset") or "chain_sum"
                names.add(ds)
        if not names:
            # If no datasets specified anywhere, default to chain_sum behavior
            names.add("chain_sum")
        return names

    @classmethod
    def _choose_variant(cls, categories: List[Dict[str, Any]]) -> Dict[str, Any]:
        names = cls._collect_dataset_names(categories)
        if len(names) == 1:
            key = next(iter(names))
            return cls.VARIANT_CONFIGS.get(key, cls.MIXED_FALLBACK)
        # Mixed or unknown datasets
        return cls.MIXED_FALLBACK

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
        n_samples: int = 10,
        max_workers: int = 128,
        vllm_port: int = 8000,
        timeout: int = 600,
        categories: list[dict[str, Any]] = [],
        prompt: Optional[str] = None,
        path_dirname: Optional[str] = None,
        file_prefix: Optional[str] = None,
        relerr: Optional[float] = None,
        leven: Optional[int] = None,
    ) -> None:
        # Decide variant first (before computing paths/prompts)
        variant = self._choose_variant(categories)
        # Configure stratified keys at instance-level (used during evaluation)
        self.STRATIFY_KEYS = variant.get("STRATIFY_KEYS", [])

        # Resolve effective prompt/path/filename using variant (allow explicit overrides)
        eff_prompt = prompt or variant.get("PROMPT", self.PROMPT)
        eff_path_dirname = path_dirname or variant.get(
            "PATH_DIRNAME", self.PATH_DIRNAME
        )
        eff_file_prefix = file_prefix or variant.get("FILE_PREFIX", self.FILE_PREFIX)

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
        self.timeout = timeout
        self.max_workers = max_workers

        # Output path
        self.path = (
            Path(__file__).parent.parent.parent.parent / f"results/{eff_path_dirname}"
        )
        self.path.mkdir(parents=True, exist_ok=True)

        self.save_path = self.path / esc(self.model_name)
        self.save_path.mkdir(parents=True, exist_ok=True)

        # Inference model abstraction used across the repo
        self.inference_model = get_inference_model(
            model_name=model_name,
            model_provider=model_provider,
            reasoning=reasoning,
            reasoning_effort=reasoning_effort,
            no_system_prompt=no_system_prompt,
            port=vllm_port,
            timeout=timeout,
        )

        # Config
        self.categories: list[dict[str, Any]] = categories
        self.prompt: str = eff_prompt
        self.file_prefix: str = eff_file_prefix
        self.relerr: Optional[float] = relerr
        self.leven: Optional[int] = leven

        # text file output
        if self.relerr is not None:
            self.display_type = f"relerr_{self.relerr}"
        elif self.leven is not None:
            self.display_type = f"leven_{self.leven}"
        else:
            self.display_type = "clean"

        # Results scaffold
        self.results: dict[str, Any] = {
            "metadata": {
                "model": self.model_name,
                "provider": self.model_provider,
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                "top_p": self.top_p,
                "n_samples": self.n_samples,
                "reasoning": self.reasoning,
                "reasoning_effort": self.reasoning_effort,
            },
            "categories": [],
        }

    def _dataset_iterables(
        self,
    ) -> List[Tuple[str, Dict[str, Any], List[Dict[str, Any]]]]:
        """
        Build RGym datasets and return iterables of entries.

        Returns:
            List of tuples: (category_name, dataset_config_dict, all_entries_list)
        """
        iters: List[Tuple[str, Dict[str, Any], List[Dict[str, Any]]]] = []
        for cat in self.categories:
            category_name = cat.get("category", "default")
            for dcfg in cat.get("datasets", []):
                dataset_name = dcfg.get("dataset")
                if not dataset_name:
                    # Default to chain_sum for backward compatibility if not specified
                    dataset_name = "chain_sum"
                # Flatten params (size, seed, params{})
                create_kwargs: Dict[str, Any] = {}
                # Top-level size/seed
                if dcfg.get("size") is not None:
                    create_kwargs["size"] = dcfg["size"]
                if dcfg.get("seed") is not None:
                    create_kwargs["seed"] = dcfg["seed"]
                # Nested params
                params = dcfg.get("params", {})
                if isinstance(params, dict):
                    create_kwargs.update(params)

                # Instantiate dataset and collect entries
                ds = reasoning_gym.create_dataset(dataset_name, **create_kwargs)
                entries = list(ds)

                # Attach the instantiated dataset object to config for later scoring
                dcfg["_dataset_instance"] = ds
                dcfg["_dataset_name"] = dataset_name

                iters.append((category_name, dcfg, entries))
        return iters

    def _save_json(self, data: Dict[str, Any]) -> None:
        with open(
            self.save_path
            / f"{self.file_prefix}_{self.n_samples}_{self.temperature}.json",
            "w",
        ) as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def _load_json(self) -> Dict[str, Any]:
        save_path = (
            self.save_path
            / f"{self.file_prefix}_{self.n_samples}_{self.temperature}.json"
        )
        with open(save_path, "r") as f:
            return json.load(f)

    def _build_conversations(self, entries: List[Dict[str, Any]]) -> List[Conversation]:
        return [
            Conversation().add_user_message(entry["question"] + f"\n{self.prompt}")
            for entry in entries
        ]

    def _extract_final_answer(self, response: str) -> Optional[str]:
        """
        Extract final answer from a completion.
        Priority:
          1) MATH-style \\boxed{...}
          2) Fall back to reasoning_gym.utils.extract_answer
        """
        try:
            extracted = extract_boxed(response)
            if extracted is not None:
                return str(extracted).strip()
        except Exception:
            pass

        # Fallback to RGym extractor (best-effort)
        try:
            extracted = rgym_extract_answer(response)
            if extracted is not None:
                return str(extracted).strip()
        except Exception:
            pass

        return None

    def generate_solutions(self, **kwargs) -> None:
        """
        Generate model responses for every entry in every dataset, with n_samples per prompt.
        """
        categories_results: List[Dict[str, Any]] = []

        for category_name, dcfg, entries in self._dataset_iterables():
            # Prepare dataset result scaffold
            dataset_result: Dict[str, Any] = {
                "name": dcfg["_dataset_name"],
                "category": category_name,
                "total_examples": len(entries),
                "config": {
                    "size": dcfg.get("size"),
                    "seed": dcfg.get("seed"),
                    **(dcfg.get("params", {}) or {}),
                },
                "completions_per_prompt": self.n_samples,
                "results": [
                    {
                        "question": entry.get("question"),
                        "expected_answer": str(entry.get("answer")),
                        # Preserve metadata for downstream stratified analysis if available
                        "metadata": entry.get("metadata", {}),
                        "completions": [],  # list of dicts: { "full_model_response": str }
                    }
                    for entry in entries
                ],
            }

            # For n_samples, generate one completion per entry per pass
            for sample_idx in range(self.n_samples):
                print(
                    f"[RGym] Generating sample {sample_idx + 1}/{self.n_samples} "
                    f"for dataset '{dcfg['_dataset_name']}' (category '{category_name}')..."
                )
                convs = self._build_conversations(entries)
                responses = self.inference_model.generate_multi_erb(
                    convs,
                    temperature=self.temperature,
                    max_workers=self.max_workers,
                    progress_bar=True,
                    base_delay=1.0,
                    max_delay=120.0,
                    max_retries=5,
                    max_tokens=self.max_tokens,
                    top_p=self.top_p,
                )

                for i, resp in enumerate(responses):
                    text = (
                        ("<think>\n" + resp.reasoning + "\n</think>\n" + resp.text)
                        if getattr(resp, "reasoning", None)
                        else resp.text
                    )
                    dataset_result["results"][i]["completions"].append(
                        {"full_model_response": text}
                    )

                # One-time estimate after the first sample only.
                # With a single sample, pass@1 is just exact-match accuracy.
                if sample_idx == 0:
                    dataset_instance = dcfg["_dataset_instance"]
                    correct = 0
                    total = len(entries)
                    for entry, resp in zip(entries, responses):
                        ans = self._extract_final_answer(resp.text)
                        try:
                            s = dataset_instance.score_answer(
                                answer=ans,
                                entry={
                                    "question": entry["question"],
                                    "answer": str(entry["answer"]),
                                },
                            )
                        except Exception:
                            s = 0.0
                        if s == 1.0:
                            correct += 1

                    tentative_pass1 = (correct / total) if total > 0 else 0.0
                    print(
                        f"[RGym] Tentative Pass@1 after sample 1/{self.n_samples}: "
                        f"{tentative_pass1:.2%}"
                    )

            categories_results.append(
                {"name": category_name, "datasets": [dataset_result]}
            )

        self.results["categories"] = categories_results
        self._save_json(self.results)

    def evaluate_solutions(self, **kwargs) -> None:
        """
        Score all generated responses using Reasoning Gym dataset scoring.
        Adds best_score and mean_score per entry, and aggregates per-dataset averages.
        Optionally computes stratified metrics for keys in STRATIFY_KEYS, using entry['metadata'].
        """
        # Reload to ensure we mutate the same file structure
        self.results = self._load_json()

        # Rebuild dataset instances to use their score implementations
        ds_iterables = self._dataset_iterables()
        # Build a quick lookup: (category, dataset_name) -> instance
        inst_map: Dict[Tuple[str, str], Any] = {}
        for category_name, dcfg, _entries in ds_iterables:
            inst_map[(category_name, dcfg["_dataset_name"])] = dcfg["_dataset_instance"]

        # Iterate through categories/datasets in results
        for cat in tqdm(self.results["categories"], desc="Evaluating categories"):
            category_name = cat["name"]
            for ds in tqdm(
                cat["datasets"],
                desc=f"Evaluating datasets in {category_name}",
                leave=False,
            ):
                ds_name = ds["name"]
                dataset_instance = inst_map[(category_name, ds_name)]

                total_best = 0.0
                total_mean = 0.0
                total_pass1 = 0.0
                total_pass5 = 0.0
                total_pass16 = 0.0

                for entry in tqdm(
                    ds["results"], desc=f"Scoring entries [{ds_name}]", leave=False
                ):
                    completions = entry.get("completions", [])
                    scores: List[float] = []
                    answers: List[Optional[str]] = []
                    best_idx: Optional[int] = None
                    best_score = -1.0

                    for c_idx, comp in enumerate(completions):
                        raw = comp.get("full_model_response", "")
                        model_ans = self._extract_final_answer(raw)
                        answers.append(model_ans)

                        try:
                            score = dataset_instance.score_answer(
                                answer=model_ans,
                                entry={
                                    "question": entry["question"],
                                    "answer": entry["expected_answer"],
                                },
                            )
                        except Exception:
                            score = 0.0

                        scores.append(score)
                        comp["score"] = score
                        comp["model_answer"] = model_ans

                        if score > best_score:
                            best_score = score
                            best_idx = c_idx

                    # Aggregate per-entry fields
                    if best_idx is not None and 0 <= best_idx < len(completions):
                        entry["best_score"] = float(best_score)
                        entry["best_model_answer"] = answers[best_idx]
                        entry["best_full_model_response"] = completions[best_idx][
                            "full_model_response"
                        ]
                    else:
                        entry["best_score"] = 0.0
                        entry["best_model_answer"] = None
                        entry["best_full_model_response"] = (
                            completions[0]["full_model_response"]
                            if completions
                            else None
                        )

                    mean_score = float(sum(scores) / len(scores)) if scores else 0.0
                    entry["mean_score"] = mean_score

                    # Compute pass@1, pass@5, and pass@16 using binary correctness from scores
                    n = len(scores)
                    if n > 0:
                        # Determine correctness flags with optional relative-error and Levenshtein tolerances
                        expected = entry.get("expected_answer")
                        correct_flags = []
                        for ans, s in zip(answers, scores):
                            ok_rel = (
                                relerr_is_correct(ans, expected, self.relerr)
                                if getattr(self, "relerr", None) is not None
                                else False
                            )

                            # NOTE: score_answer for spell_backwards does not differentiate lower/upper cases.
                            # it needs to be checked if it is the case across other datasets.
                            ok_lev = (
                                levenshtein_is_correct(
                                    ans, expected, self.leven, do_lower=True
                                )
                                if getattr(self, "leven", None) is not None
                                else False
                            )
                            correct_flags.append(
                                1 if (s == 1 or ok_rel or ok_lev) else 0
                            )

                        c = int(sum(correct_flags))
                        entry["correct_count"] = c
                        entry["pass_at_1"] = float(pass_at_k(k=min(1, n), c=c, n=n))
                        entry["pass_at_5"] = float(pass_at_k(k=min(5, n), c=c, n=n))
                        entry["pass_at_16"] = float(pass_at_k(k=min(16, n), c=c, n=n))
                    else:
                        entry["correct_count"] = 0
                        entry["pass_at_1"] = 0.0
                        entry["pass_at_5"] = 0.0
                        entry["pass_at_16"] = 0.0

                    total_best += entry["best_score"]
                    total_mean += mean_score
                    total_pass1 += entry["pass_at_1"]
                    total_pass5 += entry["pass_at_5"]
                    total_pass16 += entry["pass_at_16"]

                # Per-dataset aggregates
                total_examples = len(ds["results"])
                ds["average_best_score"] = (
                    (total_best / total_examples) if total_examples > 0 else 0.0
                )
                ds["average_mean_score"] = (
                    (total_mean / total_examples) if total_examples > 0 else 0.0
                )
                ds["pass_at_1"] = (
                    (total_pass1 / total_examples) if total_examples > 0 else 0.0
                )
                ds["pass_at_5"] = (
                    (total_pass5 / total_examples) if total_examples > 0 else 0.0
                )
                ds["pass_at_16"] = (
                    (total_pass16 / total_examples) if total_examples > 0 else 0.0
                )
                ds["total_examples"] = total_examples

                # Optional stratified metrics (e.g., by num_terms, num_digits)
                if self.STRATIFY_KEYS:
                    ds["stratified"] = self._compute_stratified(
                        ds["results"], self.STRATIFY_KEYS
                    )

        # Save back
        self._save_json(self.results)

    def _compute_stratified(
        self, entries: List[Dict[str, Any]], keys: List[str]
    ) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """
        Compute stratified aggregates per metadata key.

        Supports computed keys in the form ``len(path.to.value)``. In that case,
        stratification uses ``len(metadata[path][to][value])``.

        Returns:
          {
            "num_terms": {
              "2": {"count": 10, "average_best_score": 0.8, "average_mean_score": 0.7},
              "3": {...},
              ...
            },
            "num_digits": { ... }
          }
        """
        out: Dict[str, Dict[str, Dict[str, Any]]] = {}
        sums: Dict[str, Dict[str, Dict[str, float]]] = {}

        def _nested_get(obj: Any, dotted: str) -> Any:
            cur = obj
            for part in dotted.split("."):
                if not isinstance(cur, dict) or part not in cur:
                    return None
                cur = cur[part]
            return cur

        def _resolve_strat_value(meta: Dict[str, Any], key: str) -> Any:
            # Allow computed stratification keys like: len(sequence), len(difficulty.terms)
            if key.startswith("len(") and key.endswith(")"):
                inner = key[4:-1].strip()
                if not inner:
                    return None
                target = _nested_get(meta, inner)
                if target is None:
                    return None
                try:
                    return len(target)
                except Exception:
                    return None

            return _nested_get(meta, key) if "." in key else meta.get(key)

        for entry in entries:
            meta = entry.get("metadata") or {}
            for key in keys:
                val = _resolve_strat_value(meta, key)
                if val is None:
                    continue
                sval = str(val)
                out.setdefault(key, {}).setdefault(
                    sval,
                    {
                        "count": 0,
                        "average_best_score": 0.0,
                        "average_mean_score": 0.0,
                        "pass_at_1": 0.0,
                        "pass_at_5": 0.0,
                        "pass_at_16": 0.0,
                    },
                )
                sums.setdefault(key, {}).setdefault(
                    sval,
                    {"best": 0.0, "mean": 0.0, "p1": 0.0, "p5": 0.0, "p16": 0.0},
                )

                out[key][sval]["count"] += 1
                sums[key][sval]["best"] += float(entry.get("best_score", 0.0))
                sums[key][sval]["mean"] += float(entry.get("mean_score", 0.0))
                sums[key][sval]["p1"] += float(entry.get("pass_at_1", 0.0))
                sums[key][sval]["p5"] += float(entry.get("pass_at_5", 0.0))
                sums[key][sval]["p16"] += float(entry.get("pass_at_16", 0.0))

        for key, by_val in out.items():
            for sval, agg in by_val.items():
                cnt = max(agg["count"], 1)
                agg["average_best_score"] = sums[key][sval]["best"] / cnt
                agg["average_mean_score"] = sums[key][sval]["mean"] / cnt
                agg["pass_at_1"] = sums[key][sval]["p1"] / cnt
                agg["pass_at_5"] = sums[key][sval]["p5"] / cnt
                agg["pass_at_16"] = sums[key][sval]["p16"] / cnt

        return out

    def display_results(self, **kwargs) -> None:
        """
        Display a compact summary table and optional stratified breakdowns.
        """
        results_path = (
            self.save_path
            / f"{self.file_prefix}_{self.n_samples}_{self.temperature}.json"
        )
        if not results_path.exists():
            raise FileNotFoundError(f"Results file not found: {results_path}")

        with open(results_path, "r") as f:
            results = json.load(f)

        text_output_path = (
            self.save_path
            / f"{self.file_prefix}_{self.n_samples}_{self.temperature}_{self.display_type}.txt"
        )

        def print_and_save(s: str | None) -> None:
            print(s)

            with open(text_output_path, "a") as tf_handle:
                if s is None:
                    tf_handle.write("\n")
                else:
                    tf_handle.write(s + "\n")

        # Summarize
        total_datasets = 0
        total_examples = 0
        rows: List[List[Any]] = []

        for category in results.get("categories", []):
            category_name = category.get("name", "unknown")
            for ds in category.get("datasets", []):
                total_datasets += 1
                total_examples += ds.get("total_examples", 0)
                rows.append(
                    [
                        category_name,
                        ds.get("name", "unknown"),
                        f"{ds.get('pass_at_1', 0.0):.2%}",
                        f"{ds.get('pass_at_5', 0.0):.2%}",
                        f"{ds.get('pass_at_16', 0.0):.2%}",
                        ds.get("total_examples", 0),
                    ]
                )

        header = [
            "Category",
            "Dataset",
            "Pass@1",
            "Pass@5",
            "Pass@16",
            "#Examples",
        ]
        meta_table = [
            ["Display Type", self.display_type],
            ["Model", self.model_name],
            ["Provider", self.model_provider],
            ["Reasoning", self.reasoning],
            ["Reasoning Effort", self.reasoning_effort],
            ["Temperature", self.temperature],
            ["# Samples", self.n_samples],
        ]

        print_target_str = tabulate.tabulate(meta_table, tablefmt="plain") + "\n\n"
        print_target_str += (
            tabulate.tabulate(rows, headers=header, tablefmt="github") + "\n\n"
        )
        print_target_str += f"Total datasets: {total_datasets}\n"
        print_target_str += f"Total examples: {total_examples}\n"
        print_and_save(print_target_str)

        # Optional stratified breakdowns
        def try_sort_numeric(keys: list[str]) -> list[str]:
            try:
                return [k for _, k in sorted((int(k), k) for k in keys)]
            except Exception:
                return sorted(keys)

        for category in results.get("categories", []):
            category_name = category.get("name", "unknown")
            for ds in category.get("datasets", []):
                strat = ds.get("stratified")
                if not strat:
                    continue
                dataset_name = ds.get("name", "unknown")
                print_and_save(
                    f"\nStratified results for category='{category_name}', dataset='{dataset_name}':"
                )
                # 1D stratified breakdowns (per key), as before
                for key, by_val in strat.items():
                    sorted_vals = try_sort_numeric(list(by_val.keys()))
                    table_rows = []
                    for sval in sorted_vals:
                        agg = by_val[sval]
                        table_rows.append(
                            [
                                sval,
                                agg.get("count", 0),
                                f"{agg.get('pass_at_1', 0.0):.2%}",
                                f"{agg.get('pass_at_5', 0.0):.2%}",
                                f"{agg.get('pass_at_16', 0.0):.2%}",
                            ]
                        )
                    print_and_save(
                        tabulate.tabulate(
                            table_rows,
                            headers=[
                                key,
                                "Count",
                                "Pass@1",
                                "Pass@5",
                                "Pass@16",
                            ],
                            tablefmt="github",
                        )
                    )

                # Additionally, for chain_sum-style datasets with num_terms/num_digits metadata,
                # build joint N x M matrices where rows=num_terms, cols=num_digits.
                # This uses the per-entry pass@k values that were already computed.
                entries = ds.get("results", []) or []
                # Collect joint aggregates keyed by (num_terms, num_digits)
                joint_sums: Dict[tuple[int, int], Dict[str, float]] = {}
                terms_vals: Set[int] = set()
                digits_vals: Set[int] = set()

                for entry in entries:
                    meta = entry.get("metadata") or {}
                    t = meta.get("num_terms")
                    d = meta.get("num_digits")
                    if t is None or d is None:
                        continue
                    # Coerce to int when possible to ensure numeric sorting
                    try:
                        t_int = int(t)
                        d_int = int(d)
                    except Exception:
                        # Fallback: skip if not numeric; 1D tables above still show these
                        continue

                    terms_vals.add(t_int)
                    digits_vals.add(d_int)
                    key_td = (t_int, d_int)
                    if key_td not in joint_sums:
                        joint_sums[key_td] = {
                            "count": 0.0,
                            "p1": 0.0,
                            "p5": 0.0,
                            "p16": 0.0,
                        }
                    joint_sums[key_td]["count"] += 1.0
                    joint_sums[key_td]["p1"] += float(entry.get("pass_at_1", 0.0))
                    joint_sums[key_td]["p5"] += float(entry.get("pass_at_5", 0.0))
                    joint_sums[key_td]["p16"] += float(entry.get("pass_at_16", 0.0))

                if not terms_vals or not digits_vals:
                    # Nothing to build a matrix from (e.g., non-chain_sum datasets)
                    continue

                sorted_terms = sorted(terms_vals)
                sorted_digits = sorted(digits_vals)

                def build_matrix(metric_key: str) -> List[List[str]]:
                    """Build an N x M matrix with values formatted as

                    "<Pass@k as percent> (<#entries>)"

                    where the count is the number of dataset entries whose
                    (num_terms, num_digits) fall into that cell.
                    """
                    rows: List[List[str]] = []
                    for t_int in sorted_terms:
                        row: List[str] = [str(t_int)]
                        for d_int in sorted_digits:
                            cell = joint_sums.get((t_int, d_int))
                            if not cell or cell["count"] <= 0:
                                row.append("N/A")
                            else:
                                count = int(cell["count"])
                                if metric_key == "p1":
                                    avg = cell["p1"] / cell["count"]
                                elif metric_key == "p5":
                                    avg = cell["p5"] / cell["count"]
                                else:
                                    avg = cell["p16"] / cell["count"]
                                row.append(f"{avg:.2%} ({count})")
                        rows.append(row)
                    return rows

                header = ["num_terms \\ num_digits"] + [str(d) for d in sorted_digits]

                # Pretty-printed matrices in the log file (percent + count)
                print_and_save("Pass@1 matrix by (num_terms, num_digits):")
                print_and_save(
                    tabulate.tabulate(
                        build_matrix("p1"),
                        headers=header,
                        tablefmt="github",
                    )
                )

                print_and_save("Pass@5 matrix by (num_terms, num_digits):")
                print_and_save(
                    tabulate.tabulate(
                        build_matrix("p5"),
                        headers=header,
                        tablefmt="github",
                    )
                )

                print_and_save("Pass@16 matrix by (num_terms, num_digits):")
                print_and_save(
                    tabulate.tabulate(
                        build_matrix("p16"),
                        headers=header,
                        tablefmt="github",
                    )
                )

                # Additionally, write a single TSV for downstream analysis in a long format
                # with one row per (num_terms, num_digits) cell:
                #   num_terms\tnum_digits\tcount\tpass_at_1\tpass_at_5\tpass_at_16
                #
                # NOTE: This replaces the previous two-matrix TSV outputs
                # "*_accuracy.tsv" and "*_count.tsv" with a single combined file.

                combined_tsv_path = (
                    self.save_path
                    / f"{self.file_prefix}_{self.n_samples}_{self.temperature}_{self.display_type}.tsv"
                )

                # Build TSV content: one row per populated joint_sums cell.
                combined_lines: List[str] = []
                combined_header = [
                    "num_terms",
                    "num_digits",
                    "count",
                    "pass_at_1",
                    "pass_at_5",
                    "pass_at_16",
                ]
                combined_lines.append("\t".join(combined_header))

                # Sort rows first by num_terms, then by num_digits for reproducibility
                for t_int in sorted(terms_vals):
                    for d_int in sorted(digits_vals):
                        cell = joint_sums.get((t_int, d_int))
                        if not cell or cell["count"] <= 0:
                            # Skip empty cells: there were no examples for this combination
                            continue

                        count = int(cell["count"])
                        avg_p1 = cell["p1"] / cell["count"]
                        avg_p5 = cell["p5"] / cell["count"]
                        avg_p16 = cell["p16"] / cell["count"]

                        combined_lines.append(
                            "\t".join(
                                [
                                    str(t_int),
                                    str(d_int),
                                    str(count),
                                    f"{avg_p1:.6f}",
                                    f"{avg_p5:.6f}",
                                    f"{avg_p16:.6f}",
                                ]
                            )
                        )

                with open(combined_tsv_path, "w") as f_combined:
                    f_combined.write("\n".join(combined_lines) + "\n")
                print(f"Wrote combined stratified TSV to: {combined_tsv_path}")
