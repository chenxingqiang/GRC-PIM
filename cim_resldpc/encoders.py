"""Text encoder backends for CiM-ResLDPC.

- hash: lightweight bag-of-hash (default, no deps beyond torch)
- tfidf_svd: stronger offline dense encoder (sklearn) — no model download
- st: optional sentence-transformers if installed
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence

import numpy as np


class TextEncoder(ABC):
    name: str = "base"

    @abstractmethod
    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """Return L2-normalized float32 embeddings (N, D)."""

    @property
    @abstractmethod
    def dim(self) -> int:
        ...


class HashEncoderAdapter(TextEncoder):
    """Wrap existing TONEL HashBagEncoder."""

    name = "hash"

    def __init__(self, tonel_encoder):
        self._enc = tonel_encoder

    @property
    def dim(self) -> int:
        return int(self._enc.config.embed_dim)

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        return self._enc.encode_numpy(list(texts))


class TfidfSVDEncoder(TextEncoder):
    """TF-IDF + fixed random projection to `dim` (stable on tiny corpora).

    TruncatedSVD alone collapses when N_docs < dim; we therefore project the
    sparse TF-IDF matrix with a frozen Gaussian matrix so embedding width is
    always the requested `dim` (IR-fair across corpus sizes).
    """

    name = "tfidf_svd"

    def __init__(self, dim: int = 128, ngram_range: tuple[int, int] = (1, 2), seed: int = 42):
        from sklearn.feature_extraction.text import TfidfVectorizer

        self._dim = dim
        self._seed = seed
        self._vectorizer = TfidfVectorizer(
            lowercase=True,
            ngram_range=ngram_range,
            min_df=1,
            max_features=8192,
            sublinear_tf=True,
        )
        self._proj: np.ndarray | None = None
        self._fitted = False

    @property
    def dim(self) -> int:
        return self._dim

    def fit(self, texts: Sequence[str]) -> "TfidfSVDEncoder":
        x = self._vectorizer.fit_transform(list(texts))
        vocab = x.shape[1]
        rng = np.random.default_rng(self._seed)
        # Gaussian random projection (Johnson–Lindenstrauss style)
        self._proj = (rng.normal(size=(vocab, self._dim)).astype(np.float32)
                      / np.sqrt(self._dim))
        self._fitted = True
        return self

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if not self._fitted or self._proj is None:
            raise RuntimeError("Call fit() before encode() for TfidfSVDEncoder")
        x = self._vectorizer.transform(list(texts))
        emb = x @ self._proj  # (N, dim)
        emb = np.asarray(emb, dtype=np.float32)
        norms = np.linalg.norm(emb, axis=-1, keepdims=True)
        safe = np.maximum(norms, 1e-8)
        out = emb / safe
        out = np.where(norms < 1e-8, 0.0, out)
        return out.astype(np.float32)


class SentenceTransformerEncoder(TextEncoder):
    """Optional strong neural encoder (requires sentence-transformers)."""

    name = "sentence_transformer"

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise ImportError(
                "sentence-transformers not installed; use encoder=tfidf_svd or hash"
            ) from e
        self._model = SentenceTransformer(model_name)
        # Compatible with sentence-transformers v5+ rename
        if hasattr(self._model, "get_embedding_dimension"):
            self._dim = int(self._model.get_embedding_dimension())
        else:
            self._dim = int(self._model.get_sentence_embedding_dimension())
        self._model_name = model_name

    @property
    def dim(self) -> int:
        return self._dim

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        emb = self._model.encode(
            list(texts), convert_to_numpy=True, normalize_embeddings=True,
        )
        return np.asarray(emb, dtype=np.float32)


def build_encoder(
    kind: str = "tfidf_svd",
    dim: int = 128,
    fit_texts: Sequence[str] | None = None,
    tonel_encoder=None,
    st_model: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> TextEncoder:
    kind = kind.lower()
    if kind == "hash":
        if tonel_encoder is None:
            from .tonel import TONELResEncoder
            from .config import CiMResLDPCConfig

            tonel_encoder = TONELResEncoder(CiMResLDPCConfig(embed_dim=dim))
        return HashEncoderAdapter(tonel_encoder)
    if kind in {"tfidf_svd", "tfidf"}:
        enc = TfidfSVDEncoder(dim=dim)
        if fit_texts is None:
            raise ValueError("tfidf_svd requires fit_texts")
        return enc.fit(fit_texts)
    if kind in {"st", "sentence_transformer", "minilm"}:
        return SentenceTransformerEncoder(st_model)
    raise ValueError(f"unknown encoder kind={kind!r}")
