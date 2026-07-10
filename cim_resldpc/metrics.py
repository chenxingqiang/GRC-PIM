"""IR and hardware proxy metrics for CiM-ResLDPC."""
from __future__ import annotations

import math
from typing import Iterable, Sequence


def _dcg(relevances: Sequence[float], k: int) -> float:
    s = 0.0
    for i, rel in enumerate(relevances[:k]):
        s += (2.0 ** rel - 1.0) / math.log2(i + 2.0)
    return s


def ndcg_at_k(ranked_rels: Sequence[float], k: int = 10) -> float:
    """nDCG@k given relevance grades in rank order."""
    ideal = sorted(ranked_rels, reverse=True)
    idcg = _dcg(ideal, k)
    if idcg <= 0:
        return 0.0
    return _dcg(ranked_rels, k) / idcg


def average_precision(ranked_binary: Sequence[int]) -> float:
    """AP for a ranked list of 0/1 relevance labels."""
    hits = 0
    sum_prec = 0.0
    for i, rel in enumerate(ranked_binary, start=1):
        if rel:
            hits += 1
            sum_prec += hits / i
    if hits == 0:
        return 0.0
    return sum_prec / hits


def recall_at_k(ranked_binary: Sequence[int], n_relevant: int, k: int = 10) -> float:
    if n_relevant <= 0:
        return 0.0
    return sum(ranked_binary[:k]) / float(n_relevant)


def mean_metric(values: Iterable[float]) -> float:
    vals = list(values)
    if not vals:
        return 0.0
    return float(sum(vals) / len(vals))


def noise_degradation(clean_score: float, noisy_score: float) -> float:
    """Relative drop (positive means degradation)."""
    if abs(clean_score) < 1e-12:
        return 0.0
    return float((clean_score - noisy_score) / abs(clean_score))
