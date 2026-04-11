"""
S4 — Boundary Quality Filter
Scores each boundary with a CodeBLEU-inspired composite metric:

    BoundaryScore = α·BLEU_ngram + β·syntactic_overlap + γ·token_type_match

For code documents AST-aware token comparison is used; for prose, cosine
similarity between chunk embeddings replaces the BoundaryScore when
embeddings are available.

Boundaries with BoundaryScore > τ_sem are merged (too similar).
"""

import re
from collections import Counter
from typing import Any, Dict, List, Optional

import numpy as np


# Composite weights
_ALPHA = 0.40   # n-gram (BLEU) weight
_BETA  = 0.30   # syntactic overlap weight
_GAMMA = 0.30   # token-type match weight


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def filter_boundaries(
    chunks: List[Dict],
    doc_type: str,
    embeddings: Optional[List[List[float]]],
    config: Dict[str, Any],
) -> List[Dict]:
    """
    Score every boundary and merge those that exceed τ_sem.

    Adds to each chunk: boundary_score (float in [0,1]).
    Returns a new (possibly shorter) list of chunk dicts.
    """
    if not chunks:
        return chunks

    tau_sem = float(config.get("tau_sem", 0.75))
    n_max = int(config.get("n_max", 500))

    # First chunk always kept; its boundary_score reflects similarity to chunk 0
    result: List[Dict] = [dict(chunks[0])]
    result[0]["boundary_score"] = 0.0  # no previous boundary

    for idx in range(1, len(chunks)):
        prev = result[-1]
        curr = dict(chunks[idx])

        # --- Composite BoundaryScore ---
        if (
            embeddings
            and (idx - 1) < len(embeddings)
            and idx < len(embeddings)
        ):
            # Semantic similarity from embeddings
            cos_sim = _cosine_similarity(embeddings[idx - 1], embeddings[idx])
            # Blend with lexical signal
            lex_score = _lexical_boundary_score(
                prev["text"], curr["text"], doc_type
            )
            boundary_score = 0.5 * cos_sim + 0.5 * lex_score
        else:
            boundary_score = _lexical_boundary_score(
                prev["text"], curr["text"], doc_type
            )

        curr["boundary_score"] = round(float(boundary_score), 4)

        # Merge if chunks are too similar and combined size is acceptable
        prev_wc = len(prev["text"].split())
        curr_wc = len(curr["text"].split())
        if boundary_score > tau_sem and (prev_wc + curr_wc) <= n_max * 1.5:
            result[-1]["text"] = prev["text"] + "\n\n" + curr["text"]
            result[-1]["end"] = curr.get("end", prev.get("end", 0))
            result[-1]["boundary_score"] = round(float(boundary_score), 4)
        else:
            result.append(curr)

    return result


# ---------------------------------------------------------------------------
# Composite scoring helpers
# ---------------------------------------------------------------------------

def _lexical_boundary_score(text1: str, text2: str, doc_type: str) -> float:
    """Compute CodeBLEU-inspired boundary score between two text blocks."""
    bleu = _ngram_precision(text1, text2, n=2)
    syn  = _syntactic_overlap(text1, text2, doc_type)
    ttm  = _token_type_match(text1, text2)
    return _ALPHA * bleu + _BETA * syn + _GAMMA * ttm


def _ngram_precision(text1: str, text2: str, n: int = 2) -> float:
    """Modified n-gram precision (BLEU-style) between two texts."""
    tokens1 = _tokenize(text1)
    tokens2 = _tokenize(text2)

    if len(tokens1) < n or len(tokens2) < n:
        # Fall back to unigram overlap
        s1, s2 = set(tokens1), set(tokens2)
        union = s1 | s2
        return len(s1 & s2) / len(union) if union else 0.0

    ngrams1 = Counter(_ngrams(tokens1, n))
    ngrams2 = Counter(_ngrams(tokens2, n))

    clipped = sum(min(c, ngrams2[ng]) for ng, c in ngrams1.items())
    total = sum(ngrams1.values())
    return clipped / total if total else 0.0


def _ngrams(tokens: List[str], n: int):
    return [tuple(tokens[i: i + n]) for i in range(len(tokens) - n + 1)]


def _syntactic_overlap(text1: str, text2: str, doc_type: str) -> float:
    """
    For code: fraction of shared operator/keyword tokens.
    For prose: fraction of shared POS-proxy tokens (stopwords = function words).
    """
    if doc_type == "code":
        code_tokens1 = set(re.findall(r"[+\-*/%&|^~<>=!;:,.()\[\]{}]|\b\w+\b", text1))
        code_tokens2 = set(re.findall(r"[+\-*/%&|^~<>=!;:,.()\[\]{}]|\b\w+\b", text2))
        union = code_tokens1 | code_tokens2
        return len(code_tokens1 & code_tokens2) / len(union) if union else 0.0
    else:
        # Proxy: shared function words as syntactic signal
        func_words = {
            "the", "a", "an", "is", "was", "are", "were", "be", "been",
            "have", "has", "had", "do", "does", "did", "will", "would",
            "could", "should", "may", "might", "shall", "can", "of",
            "in", "on", "at", "by", "for", "with", "about", "as", "to",
        }
        t1 = [w for w in _tokenize(text1) if w in func_words]
        t2 = [w for w in _tokenize(text2) if w in func_words]
        c1, c2 = Counter(t1), Counter(t2)
        shared = sum(min(c1[w], c2[w]) for w in c1)
        total = max(len(t1), len(t2))
        return shared / total if total else 0.0


def _token_type_match(text1: str, text2: str) -> float:
    """Fraction of shared token *types* (vocabulary overlap)."""
    types1 = set(_tokenize(text1))
    types2 = set(_tokenize(text2))
    union = types1 | types2
    return len(types1 & types2) / len(union) if union else 0.0


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> List[str]:
    return re.findall(r"\b\w+\b", text.lower())


def _cosine_similarity(
    v1: List[float], v2: List[float]
) -> float:
    a = np.array(v1, dtype=np.float32)
    b = np.array(v2, dtype=np.float32)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))
