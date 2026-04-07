import argparse
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from transformers import AutoTokenizer

from src.statistics.token_stats import (
    Config,
    iter_step_files,
    main,
    plot,
    preprocess_texts,
)


def parse_args() -> Config:
    p = argparse.ArgumentParser(
        description="Compute per-token frequency and average reward/advantage from step-*.tsv logs."
    )
    p.add_argument(
        "--logdir", type=Path, required=True, help="Directory containing step-*.tsv"
    )
    p.add_argument(
        "--model-name", default="allenai/Olmo-3-1025-7B", help="Tokenizer model name"
    )
    p.add_argument("--chunksize", type=int, default=16_384)
    p.add_argument(
        "--min-f", type=float, default=1e-3, help="Min token frequency (0..1)"
    )
    p.add_argument(
        "--max-f", type=float, default=5e-2, help="Max token frequency (0..1)"
    )
    p.add_argument(
        "--min-id",
        type=int,
        default=0,
        help="min    number of step-*.tsv files to process.",
    )
    p.add_argument(
        "--max-id",
        type=int,
        default=10,
        help="Maximum number of step-*.tsv files to process.",
    )
    p.add_argument(
        "--replace-spaces-n",
        type=int,
        default=5,
        help='Replace runs of N spaces with "\\n" in completions (0 to disable).',
    )
    p.add_argument(
        "--sample-every",
        type=int,
        default=1,
        help="Keep one row every N rows (1 = no sampling).",
    )
    p.add_argument(
        "--save-name",
        default="token_impact_analysis.tsv",
        help="Output TSV filename (saved inside logdir unless absolute).",
    )
    p.add_argument(
        "--binsize",
        type=int,
        default=100,
        help="Length bin size for length-based analysis.",
    )
    p.add_argument(
        "--stratify",
        nargs="+",
        choices=["token", "length"],
        default=["token", "length"],
        help="Stratification methods to analyze by.",
    )
    p.add_argument(
        "--ngram-n",
        type=int,
        default=1,
        help="N-gram size for token-based analysis (1 = unigram).",
    )
    p.add_argument(
        "--include",
        nargs="+",
        type=int,
        default=[],
        help="List of token IDs to include regardless of frequency filters.",
    )
    p.add_argument(
        "--include_str",
        nargs="+",
        type=str,
        help="List of token strings to include regardless of frequency filters.",
    )
    # --calc-advantage-from-oracle-reward
    p.add_argument(
        "--calc-advantage-from-oracle",
        action="store_true",
        help="Calculate advantage by reading `oracle_reward` column.",
    )


    a = p.parse_args()

    # if include is str, tokenize to get token ids
    if a.include_str:
        tokenizer = AutoTokenizer.from_pretrained(a.model_name)
        include_tokens = []
        for text in a.include_str:
            ids = tokenizer.encode(text, add_special_tokens=False)
            assert len(ids) == 1, f"Included text '{text}' is tokenized into multiple tokens: {ids}"
            include_tokens.extend(ids)
        a.include.extend(include_tokens)

    logdir = a.logdir
    if not logdir.exists():
        raise FileNotFoundError(f"Logdir {logdir} does not exist.")
    if a.sample_every < 1:
        raise ValueError("--sample-every must be >= 1")
    if a.ngram_n < 1:
        raise ValueError("--ngram-n must be >= 1")
    if not (0.0 <= a.min_f <= 1.0 and 0.0 <= a.max_f <= 1.0 and a.min_f <= a.max_f):
        raise ValueError("--min-f/--max-f must satisfy 0<=min_f<=max_f<=1")

    return Config(**vars(a))


if __name__ == "__main__":
    args = parse_args()
    savepath = Path(args.save_name)
    if not savepath.is_absolute():
        savepath = args.logdir / savepath

    main(args, savepath)

    plot(args, savepath)
