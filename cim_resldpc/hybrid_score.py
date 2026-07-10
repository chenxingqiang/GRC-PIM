"""Hybrid BM25 + dense + residual confidence scoring."""
from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass

import numpy as np


_TOKEN = re.compile(r"[a-z0-9_]+", re.I)


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


@dataclass
class BM25Index:
    doc_ids: list[str]
    doc_len: list[int]
    avgdl: float
    df: dict[str, int]
    tf: list[Counter]
    n_docs: int
    k1: float = 1.5
    b: float = 0.75

    def score(self, query: str) -> np.ndarray:
        q_tokens = tokenize(query)
        scores = np.zeros(self.n_docs, dtype=np.float32)
        for i, tf in enumerate(self.tf):
            dl = self.doc_len[i]
            s = 0.0
            for t in q_tokens:
                if t not in tf:
                    continue
                df = self.df.get(t, 0)
                idf = math.log(1.0 + (self.n_docs - df + 0.5) / (df + 0.5))
                f = tf[t]
                s += idf * (f * (self.k1 + 1.0)) / (
                    f + self.k1 * (1.0 - self.b + self.b * dl / self.avgdl)
                )
            scores[i] = s
        return scores


def build_bm25(doc_ids: list[str], texts: list[str]) -> BM25Index:
    tfs: list[Counter] = []
    df: dict[str, int] = defaultdict(int)
    lengths: list[int] = []
    for text in texts:
        toks = tokenize(text)
        c = Counter(toks)
        tfs.append(c)
        lengths.append(len(toks))
        for t in c:
            df[t] += 1
    avgdl = float(sum(lengths) / max(len(lengths), 1))
    return BM25Index(
        doc_ids=list(doc_ids),
        doc_len=lengths,
        avgdl=avgdl,
        df=dict(df),
        tf=tfs,
        n_docs=len(doc_ids),
    )


def fuse_scores(
    bm25: np.ndarray,
    dense: np.ndarray,
    residual_conf: np.ndarray,
    alpha: float = 0.3,
    beta: float = 0.5,
    gamma: float = 0.2,
) -> np.ndarray:
    """Min-max normalize each channel then weighted sum."""
    def _norm(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        lo, hi = float(x.min()), float(x.max())
        if hi - lo < 1e-8:
            return np.zeros_like(x)
        return (x - lo) / (hi - lo)

    return (
        alpha * _norm(bm25)
        + beta * _norm(dense)
        + gamma * _norm(residual_conf)
    ).astype(np.float32)
