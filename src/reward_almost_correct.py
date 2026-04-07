"""
Utilities for defining and reusing "almost correct" predicates in reward calculation during training and in evaluation.
"""

from __future__ import annotations

import time
import warnings

import nltk


def parse_float(val: object) -> float | None:
    """
    Best-effort parse of a numeric value from strings/numbers.
    """
    if val is None:
        return None
    if isinstance(val, (int, float)):
        try:
            return float(val)
        except Exception:
            return None
    try:
        s = str(val).strip().replace(",", "")
        return float(s)
    except Exception:
        return None


def relerr_is_correct(
    pred: object,
    truth: object,
    tolerance: float,
) -> bool:
    """
    Determines if the prediction is "almost correct" compared to the truth
    within a specified relative error tolerance.
    """
    if tolerance is None:
        return False

    p = parse_float(pred)
    t = parse_float(truth)
    relerr = calc_relerr(p, t)
    if relerr is None:
        return False
    return relerr <= tolerance


def calc_relerr(pred: float | None, truth: float | None) -> float | None:
    """
    Computes the relative error between prediction and truth.
    Returns None if parsing fails or truth is zero.
    """
    if pred is None or truth is None:
        return None
    if truth == 0.0:
        return None
    return abs(pred - truth) / abs(truth)


def parse_str(val: object) -> str | None:
    """
    Best-effort parse of a string value.
    """
    if val is None:
        return None
    try:
        return str(val).strip()
    except Exception:
        return None


def levenshtein_distance(
    pred: str | None, truth: str | None, do_lower: bool = False
) -> int | None:
    """
    Compute Levenshtein edit distance between two values coerced to strings.
    Returns None if either cannot be parsed to string.
    """
    if pred is None or truth is None:
        return None

    if do_lower:
        pred = pred.lower()
        truth = truth.lower()

    return nltk.edit_distance(pred, truth)


def levenshtein_is_correct(
    pred: object,
    truth: object,
    max_distance: int,
    do_lower: bool = False,
) -> bool:
    """
    Returns True if the edit distance between pred and truth is <= max_distance.
    If max_distance is None or parsing fails, returns False.
    """

    pred = parse_str(pred)
    truth = parse_str(truth)
    d = levenshtein_distance(pred, truth, do_lower=do_lower)

    if d is None:
        return False
    return d <= max_distance
