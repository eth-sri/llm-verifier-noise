#!/usr/bin/env python3
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


@dataclass(frozen=True)
class Config:
    logdir: Path
    model_name: str
    chunksize: int
    min_f: float
    max_f: float
    min_id: int
    max_id: int
    replace_spaces_n: int
    sample_every: int
    save_name: str
    usecols: tuple[str, ...] = (
        "prompt",
        "completion",
        "reward",
        "advantage",
        "oracle_reward",
    )  # maybe not so useful
    binsize: int = 100
    stratify: list[str] = ("token", "length")  # analyze by 'token' and/or 'length'
    include: list[int] | None = None  # token ids to include regardless of frequency
    include_str : list[str] | None = None # token strings to include regardless of frequency
    calc_advantage_from_oracle: bool = False
    ngram_n: int = 1


def iter_step_files(
    logdir: Path, min_id: int = 0, max_id: int = 10**6
) -> Iterable[Path]:
    files = list(logdir.glob("step-*.tsv"))
    if not files:
        raise FileNotFoundError(f"No step-*.tsv found in {logdir}")

    def step_num(p: Path) -> int:
        m = re.compile(r"step-(\d+)\.tsv$").search(p.name)
        return int(m.group(1)) if m else 10**18  # put weird names at the end

    files.sort(key=step_num)
    return [f for f in files if min_id <= step_num(f) <= max_id]


def preprocess_texts(texts: pd.Series, replace_spaces_n: int) -> list[str]:
    texts = texts.fillna("")
    # NOTE linebreaks are changed to special tokens when saving log
    texts = texts.str.replace("[CRLF]", "\r\n")
    texts = texts.str.replace("[CR]", "\r")
    texts = texts.str.replace("[LF]", "\n")
    if replace_spaces_n and replace_spaces_n > 0:
        pat = re.compile(r" " * replace_spaces_n)
        texts = texts.map(lambda x: pat.sub("\n", x))
    return texts.tolist()


def main(cfg: Config, savepath_base: Path) -> None:
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, use_fast=True)

    if cfg.binsize <= 0:
        raise ValueError("--binsize must be > 0")
    if cfg.ngram_n <= 0:
        raise ValueError("--ngram-n must be > 0")

    # token/ngram stats
    tok_count = Counter()
    tok_reward_sum = defaultdict(float)
    tok_adv_sum = defaultdict(float)

    # length-bin stats (key = bin_left)
    len_count = defaultdict(int)
    len_reward_sum = defaultdict(float)
    len_adv_sum = defaultdict(float)

    total_rows = 0

    for step_file in iter_step_files(cfg.logdir, cfg.min_id, cfg.max_id):
        print(step_file)
        for chunk in pd.read_csv(
            step_file, sep="\t", usecols=list(cfg.usecols), chunksize=cfg.chunksize
        ):
            if cfg.sample_every > 1:
                chunk = chunk.iloc[:: cfg.sample_every, :]

            texts = preprocess_texts(chunk["completion"], cfg.replace_spaces_n)
            rewards = chunk["reward"].to_numpy(dtype=float, na_value=np.nan)
            if cfg.calc_advantage_from_oracle:
                oracle_rewards = chunk["oracle_reward"].to_numpy(
                    dtype=float, na_value=np.nan
                )
                g = chunk.groupby("prompt")["oracle_reward"]
                mu = g.transform("mean")
                sigma = g.transform(lambda s: s.std(ddof=1))
                chunk["advantage"] = (chunk["oracle_reward"] - mu) / (sigma + 1e-4)
                advs = chunk["advantage"].to_numpy(dtype=float, na_value=np.nan)
            else:
                advs = chunk["advantage"].to_numpy(dtype=float, na_value=np.nan)

            enc = tokenizer(
                texts,
                add_special_tokens=False,
                padding=False,
                truncation=False,
            )
            input_ids_batch = enc["input_ids"]

            for ids, r, a in zip(input_ids_batch, rewards, advs):
                if not np.isfinite(r) or not np.isfinite(a):
                    continue

                total_rows += 1

                # length bin
                L = len(ids)
                bin_left = (L // cfg.binsize) * cfg.binsize
                len_count[bin_left] += 1
                len_reward_sum[bin_left] += float(r)
                len_adv_sum[bin_left] += float(a)

                # token/ngram presence
                if cfg.ngram_n == 1:
                    present_features = set(ids)
                else:
                    if len(ids) < cfg.ngram_n:
                        present_features = set()
                    else:
                        present_features = {
                            tuple(ids[i : i + cfg.ngram_n])
                            for i in range(len(ids) - cfg.ngram_n + 1)
                        }

                for feature in present_features:
                    tok_count[feature] += 1
                    tok_reward_sum[feature] += float(r)
                    tok_adv_sum[feature] += float(a)

    if total_rows == 0:
        raise RuntimeError("No valid rows processed (total_rows=0).")

    # ---- build token/ngram df (apply min_f/max_f/include as before)
    token_rows = []
    for feature, c in tok_count.items():
        freq = c / total_rows
        include_match = cfg.include is not None and cfg.ngram_n == 1 and feature in cfg.include
        if cfg.min_f <= freq <= cfg.max_f or include_match:
            if cfg.ngram_n == 1:
                token_rows.append(
                    {
                        "token_id": feature,
                        "token_str": tokenizer.decode([feature]).replace("\n", "[LF]"),
                        "count": c,
                        "frequency": freq,
                        "avg_reward": tok_reward_sum[feature] / c,
                        "avg_advantage": tok_adv_sum[feature] / c,
                    }
                )
            else:
                ngram_ids = list(feature)
                token_rows.append(
                    {
                        "ngram_n": cfg.ngram_n,
                        "ngram_id": ",".join(map(str, ngram_ids)),
                        "ngram_str": tokenizer.decode(ngram_ids).replace("\n", "[LF]"),
                        "count": c,
                        "frequency": freq,
                        "avg_reward": tok_reward_sum[feature] / c,
                        "avg_advantage": tok_adv_sum[feature] / c,
                    }
                )
    df_token = pd.DataFrame(token_rows).sort_values("avg_advantage", ascending=False)

    # ---- build length df (keep all bins)
    len_rows = []
    for bin_left, c in len_count.items():
        len_rows.append(
            {
                "bin_left": bin_left,
                "bin_right": bin_left + cfg.binsize,
                "count": c,
                "frequency": c / total_rows,
                "avg_reward": len_reward_sum[bin_left] / c,
                "avg_advantage": len_adv_sum[bin_left] / c,
            }
        )
    df_len = pd.DataFrame(len_rows).sort_values("bin_left", ascending=True)

    print(f"Processed rows: {total_rows}")

    # save depending on mode
    if "token" in cfg.stratify:
        save_tok = rename_savepath(
            savepath_base, "token", cfg.binsize, cfg.min_id, cfg.max_id, cfg.ngram_n
        )
        print(f"Saving token analysis to {save_tok}")
        df_token.to_csv(save_tok, sep="\t", index=False)
        # only save tokens that consists of alphabets and spaces
        text_col = "token_str" if cfg.ngram_n == 1 else "ngram_str"
        df_token_eng = df_token[df_token[text_col].str.match(r"^[a-zA-Z\s]+$")]
        save_tok_eng = save_tok.with_name(save_tok.stem + ".eng_only.tsv")
        print(f"Saving token analysis (eng only) to {save_tok_eng}")
        df_token_eng.to_csv(save_tok_eng, sep="\t", index=False)

    if "length" in cfg.stratify:
        save_len = rename_savepath(
            savepath_base, "length", cfg.binsize, cfg.min_id, cfg.max_id, cfg.ngram_n
        )
        print(f"Saving length analysis to {save_len}")
        df_len.to_csv(save_len, sep="\t", index=False)


def rename_savepath(
    savepath: Path,
    stratify: str,
    binsize: int,
    min_id: int,
    max_id: int,
    ngram_n: int = 1,
) -> Path:
    if stratify == "token":
        new_savepath = savepath.with_suffix("")  # strip .tsv if present
        if ngram_n == 1:
            suffix = "token"
        else:
            suffix = f"{ngram_n}gram"
        new_savepath = (
            new_savepath.parent / (new_savepath.name + f".by_{suffix}.{min_id}_{max_id}.tsv")
        )
    elif stratify == "length":
        new_savepath = savepath.with_suffix("")  # strip .tsv if present
        new_savepath = new_savepath.parent / (
            new_savepath.name + f".by_length_bin{binsize}_{min_id}_{max_id}.tsv"
        )
    else:
        raise ValueError(f"Unknown stratify method: {stratify}")
    return new_savepath


def plot(cfg: Config, savepath: Path) -> None:
    plt.figure(figsize=(8, 6))
    if "token" in cfg.stratify:
        df = pd.read_csv(
            rename_savepath(
                savepath, "token", cfg.binsize, cfg.min_id, cfg.max_id, cfg.ngram_n
            ),
            sep="\t",
        )
        title_extra = " (by token)" if cfg.ngram_n == 1 else f" (by {cfg.ngram_n}-gram)"
        plt.scatter(df["frequency"] * 100, df["avg_advantage"], alpha=0.7, s=10)
        plt.xlim(df["frequency"].min() * 100 * 0.9, df["frequency"].max() * 100 * 1.1)
        plt.xlabel("Frequency (%)")
        plt.ylabel("Average Advantage")
        plt.title("Token Impact Analysis" + title_extra)
        plt.grid(True, which="both", ls="--", alpha=0.5)
        plt.tight_layout()
        save_token = rename_savepath(
            savepath, "token", cfg.binsize, cfg.min_id, cfg.max_id, cfg.ngram_n
        )
        plt.savefig(save_token.with_suffix(".png"), dpi=300)

    if "length" in cfg.stratify:
        df = pd.read_csv(
            rename_savepath(
                savepath, "length", cfg.binsize, cfg.min_id, cfg.max_id, cfg.ngram_n
            ),
            sep="\t",
        )
        title_extra = f" (bin size {cfg.binsize})"
        # filter out bins with min_f and max_f
        # df = df[(df["frequency"] >= cfg.min_f) & (df["frequency"] <= cfg.max_f)]
        plt.scatter(df["frequency"] * 100, df["avg_advantage"], alpha=0.7, s=10)
        plt.xlim(0, df["frequency"].max() * 100 * 1.1)
        # annotate {bin_left}-{bin_right}
        for _, row in df.iterrows():
            mean = (int(row["bin_left"]) + int(row["bin_right"])) // 2
            plt.annotate(
                f"{mean:d}",
                (row["frequency"] * 100, row["avg_advantage"]),
                textcoords="offset points",
                xytext=(0, 5),
                ha="center",
                fontsize=8,
            )
        plt.xlabel("Length Bin Frequency (%)")
        plt.ylabel("Average Advantage")
        plt.title("Token Impact Analysis" + title_extra)
        plt.grid(True, which="both", ls="--", alpha=0.5)
        plt.tight_layout()
        save_length = rename_savepath(
            savepath, "length", cfg.binsize, cfg.min_id, cfg.max_id, cfg.ngram_n
        )
        plt.savefig(save_length.with_suffix(".png"), dpi=300)
