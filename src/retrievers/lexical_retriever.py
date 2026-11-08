from __future__ import annotations

import math
import re
from collections import Counter
from typing import Dict, List

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


class LexicalRetriever:
    """BM25 / TF-IDF keyword retriever used as a lightweight baseline."""

    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(2, 5),
            lowercase=True,
        )
        self.chunks = None
        self.tfidf_matrix = None
        self.bm25_documents = None
        self.bm25_doc_lengths = None
        self.bm25_idf = None
        self.bm25_avg_doc_length = 0.0

    @staticmethod
    def tokenize(text: str) -> List[str]:
        ascii_tokens = re.findall(r"[a-zA-Z0-9$]+", text.lower())
        chinese_tokens = []
        for sequence in re.findall(r"[\u4e00-\u9fff]{2,}", text):
            chinese_tokens.append(sequence)
            for size in (2, 3):
                chinese_tokens.extend(
                    sequence[i:i + size]
                    for i in range(0, max(len(sequence) - size + 1, 0))
                )
        return ascii_tokens + chinese_tokens

    @staticmethod
    def normalize_scores(scores: np.ndarray) -> np.ndarray:
        if scores.size == 0:
            return scores
        min_score = float(scores.min())
        max_score = float(scores.max())
        if max_score == min_score:
            if max_score > 0:
                return np.ones_like(scores, dtype=float)
            return np.zeros_like(scores, dtype=float)
        return (scores - min_score) / (max_score - min_score)

    def build_index(self, chunks: List[Dict[str, str]]) -> None:
        if not chunks:
            raise ValueError("No chunks provided. Cannot build lexical index.")

        self.chunks = chunks
        texts = [chunk["text"] for chunk in chunks]
        self.tfidf_matrix = self.vectorizer.fit_transform(texts)
        self._build_bm25_index(texts)

    def _build_bm25_index(self, texts: List[str]) -> None:
        tokenized_documents = [self.tokenize(text) for text in texts]
        self.bm25_documents = [Counter(tokens) for tokens in tokenized_documents]
        self.bm25_doc_lengths = [len(tokens) for tokens in tokenized_documents]
        self.bm25_avg_doc_length = (
            sum(self.bm25_doc_lengths) / len(self.bm25_doc_lengths)
            if self.bm25_doc_lengths
            else 0.0
        )

        document_frequencies = Counter()
        for tokens in tokenized_documents:
            document_frequencies.update(set(tokens))

        total_docs = len(tokenized_documents)
        self.bm25_idf = {
            term: math.log(1 + (total_docs - freq + 0.5) / (freq + 0.5))
            for term, freq in document_frequencies.items()
        }

    def scores(self, query: str) -> np.ndarray:
        if self.chunks is None:
            raise ValueError("Index has not been built yet.")
        query_terms = self.tokenize(query)
        scores = np.zeros(len(self.chunks), dtype=float)
        if not query_terms or not self.bm25_documents:
            return scores

        k1 = 1.5
        b = 0.75
        avgdl = self.bm25_avg_doc_length or 1.0
        for idx, doc_terms in enumerate(self.bm25_documents):
            doc_len = self.bm25_doc_lengths[idx] or 1
            score = 0.0
            for term in query_terms:
                term_freq = doc_terms.get(term, 0)
                if term_freq == 0:
                    continue
                idf = self.bm25_idf.get(term, 0.0)
                denominator = term_freq + k1 * (1 - b + b * doc_len / avgdl)
                score += idf * (term_freq * (k1 + 1)) / denominator
            scores[idx] = score
        return scores

    def search(self, query: str, top_k: int = 3) -> List[Dict[str, object]]:
        if self.chunks is None:
            raise ValueError("Index has not been built yet.")
        if not query.strip():
            raise ValueError("Query cannot be empty.")

        top_k = min(top_k, len(self.chunks))
        raw_scores = self.scores(query)
        normalized = self.normalize_scores(raw_scores)
        ranked_indices = normalized.argsort()[::-1][:top_k]

        results = []
        for idx in ranked_indices:
            chunk = self.chunks[int(idx)].copy()
            chunk["bm25_score"] = float(raw_scores[int(idx)])
            chunk["keyword_score"] = float(normalized[int(idx)])
            chunk["similarity_score"] = float(normalized[int(idx)])
            chunk["retrieval_mode"] = "lexical"
            chunk["retrieval_method"] = "lexical_bm25_tfidf"
            results.append(chunk)
        return results
