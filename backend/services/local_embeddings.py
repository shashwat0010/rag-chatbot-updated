"""Lightweight TF-IDF embeddings (no API) for fallback when OpenAI is unavailable."""

import math
import re
from collections import Counter
from typing import List, Optional

import numpy as np

_TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
_STOPWORDS = {
    "a", "an", "the", "and", "or", "of", "in", "on", "for", "to", "is", "are",
    "was", "were", "be", "been", "with", "from", "at", "by", "as", "that",
    "this", "what", "how", "does", "do", "did", "can", "may", "should", "would",
    "about", "into", "their", "there", "which", "who", "when", "where", "why",
}


def _tokenize(text: str) -> List[str]:
    tokens = [t.lower() for t in _TOKEN_RE.findall(text)]
    return [t for t in tokens if len(t) > 2 and t not in _STOPWORDS]


class TfidfVectorizer:
    """Fit on document corpus; transform queries with the same vocabulary."""

    def __init__(self) -> None:
        self._vocab: List[str] = []
        self._idf: dict[str, float] = {}

    @property
    def is_fitted(self) -> bool:
        return bool(self._vocab)

    def fit(self, texts: List[str]) -> None:
        doc_tokens = [_tokenize(t) for t in texts]
        df: Counter[str] = Counter()
        for tokens in doc_tokens:
            for term in set(tokens):
                df[term] += 1
        n_docs = max(len(texts), 1)
        self._vocab = sorted(df.keys())
        self._idf = {
            term: math.log((1 + n_docs) / (1 + count)) + 1 for term, count in df.items()
        }

    def transform(self, texts: List[str]) -> List[List[float]]:
        if not self._vocab:
            return [[] for _ in texts]

        matrix = np.zeros((len(texts), len(self._vocab)), dtype=np.float32)
        term_to_idx = {t: i for i, t in enumerate(self._vocab)}

        for row, text in enumerate(texts):
            tokens = _tokenize(text)
            if not tokens:
                continue
            tf = Counter(tokens)
            for term, count in tf.items():
                if term not in term_to_idx:
                    continue
                j = term_to_idx[term]
                matrix[row, j] = (count / len(tokens)) * self._idf.get(term, 0.0)

        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1
        matrix = matrix / norms
        return matrix.tolist()


# Shared vectorizer for the current request's FAISS index
_active_vectorizer: Optional[TfidfVectorizer] = None


def fit_tfidf(texts: List[str]) -> List[List[float]]:
    global _active_vectorizer
    _active_vectorizer = TfidfVectorizer()
    _active_vectorizer.fit(texts)
    return _active_vectorizer.transform(texts)


def transform_tfidf(texts: List[str]) -> List[List[float]]:
    if _active_vectorizer is None or not _active_vectorizer.is_fitted:
        return fit_tfidf(texts)
    return _active_vectorizer.transform(texts)


def tfidf_embed(texts: List[str]) -> List[List[float]]:
    """Embed texts; fits vocabulary on all texts in this batch."""
    vectorizer = TfidfVectorizer()
    vectorizer.fit(texts)
    return vectorizer.transform(texts)


def get_local_embeddings(texts: List[str]) -> List[List[float]]:
    """Alias for tfidf_embed."""
    return tfidf_embed(texts)
