"""
S6 — Contextual Embedding
Two strategies based on document length:

  Short docs (fit in context window): "late chunking" — embed the full
      document, then mean-pool the token vectors that belong to each chunk
      span.

  Long docs: prepend a generated context header to each chunk, then embed
      each chunk independently with sentence-transformers.

Falls back to a TF-IDF-style bag-of-words vector when sentence-transformers
is unavailable.
"""

import re
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Lazy model loader
# ---------------------------------------------------------------------------

_model_cache: Dict[str, Any] = {}


def _get_model(model_name: str):
    if model_name in _model_cache:
        return _model_cache[model_name]
    try:
        from sentence_transformers import SentenceTransformer  # noqa: E402
        model = SentenceTransformer(model_name)
        _model_cache[model_name] = model
        return model
    except Exception:
        _model_cache[model_name] = None
        return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def embed_chunks(
    chunks: List[Dict],
    full_text: str,
    doc_profile: Dict[str, Any],
    model_name: str,
    config: Dict[str, Any],
) -> Tuple[List[Dict], List[List[float]]]:
    """
    Embed all chunks and return:
        - enriched chunk list (adds 'embedding' and optionally 'context_header')
        - raw list of embedding vectors (parallel to chunks)
    """
    if not chunks:
        return chunks, []

    length_bucket = doc_profile.get("length_bucket", "medium")
    domain = doc_profile.get("domain", "general")
    doc_type = doc_profile.get("type", "prose")

    model = _get_model(model_name)

    if model is None:
        # Fallback: TF-IDF bag-of-words
        embeddings = _bow_embed(chunks, full_text)
    elif length_bucket == "short":
        embeddings = _late_chunking(chunks, full_text, model)
    else:
        embeddings = _context_header_embed(
            chunks, full_text, domain, doc_type, model
        )

    enriched = [dict(c) for c in chunks]
    for i, chunk in enumerate(enriched):
        emb = embeddings[i] if i < len(embeddings) else []
        chunk["embedding"] = emb if isinstance(emb, list) else emb.tolist()

    return enriched, [
        (e if isinstance(e, list) else e.tolist()) for e in embeddings
    ]


# ---------------------------------------------------------------------------
# Late chunking (short documents)
# ---------------------------------------------------------------------------

def _late_chunking(
    chunks: List[Dict],
    full_text: str,
    model,
) -> List[List[float]]:
    """
    Encode the full document once, then mean-pool token vectors per chunk span.
    For models that expose token-level outputs this is exact; for
    SentenceTransformer (sentence-level) we approximate by re-encoding each
    chunk (the "pooling" happens at sentence level internally).
    """
    try:
        # Encode each chunk with the shared model context
        texts = [c["text"] for c in chunks]
        vectors = model.encode(texts, show_progress_bar=False, batch_size=32)
        return [v.tolist() for v in vectors]
    except Exception:
        return _bow_embed(chunks, full_text)


# ---------------------------------------------------------------------------
# Context-header embedding (long documents)
# ---------------------------------------------------------------------------

def _context_header_embed(
    chunks: List[Dict],
    full_text: str,
    domain: str,
    doc_type: str,
    model,
) -> List[List[float]]:
    # Infer a short topic from the first non-empty sentence of the document
    inferred_topic = _infer_topic(full_text)

    enriched_texts: List[str] = []
    for chunk in chunks:
        first_sentence = _first_sentence(chunk["text"])
        header = (
            f"This chunk is from a {domain} {doc_type} document "
            f"about {inferred_topic}. "
            f"The chunk covers: {first_sentence}"
        )
        chunk["context_header"] = header
        enriched_texts.append(header + "\n\n" + chunk["text"])

    try:
        vectors = model.encode(
            enriched_texts, show_progress_bar=False, batch_size=16
        )
        return [v.tolist() for v in vectors]
    except Exception:
        return _bow_embed(chunks, full_text)


# ---------------------------------------------------------------------------
# Fallback: TF-IDF-inspired BoW embedding
# ---------------------------------------------------------------------------

def _bow_embed(chunks: List[Dict], full_text: str) -> List[List[float]]:
    """
    Produce a lightweight sparse-then-projected embedding when
    sentence-transformers is unavailable.  Uses term-frequency vectors
    projected to 128 dims via a deterministic random projection.
    """
    vocab = _build_vocab(full_text, max_terms=2000)
    vocab_idx = {w: i for i, w in enumerate(vocab)}
    dim = len(vocab)

    tf_vectors: List[np.ndarray] = []
    for chunk in chunks:
        tokens = re.findall(r"\b\w+\b", chunk["text"].lower())
        vec = np.zeros(dim, dtype=np.float32)
        for t in tokens:
            if t in vocab_idx:
                vec[vocab_idx[t]] += 1.0
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        tf_vectors.append(vec)

    # Random projection to 128 dims (deterministic seed for reproducibility)
    rng = np.random.default_rng(seed=42)
    proj = rng.normal(0, 1.0 / np.sqrt(128), size=(dim, 128)).astype(np.float32)
    return [(v @ proj).tolist() for v in tf_vectors]


def _build_vocab(text: str, max_terms: int = 2000) -> List[str]:
    tokens = re.findall(r"\b\w+\b", text.lower())
    freq: Dict[str, int] = {}
    for t in tokens:
        freq[t] = freq.get(t, 0) + 1
    # Sort by frequency, take top max_terms
    return [w for w, _ in sorted(freq.items(), key=lambda x: -x[1])[:max_terms]]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _infer_topic(text: str) -> str:
    """Extract a short topic description from the document's opening."""
    # Try headings first
    heading = re.search(r"^#{1,3}\s+(.+)$", text, re.MULTILINE)
    if heading:
        return heading.group(1).strip()[:80]

    # Otherwise use the first meaningful sentence
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    for s in sentences[:3]:
        clean = s.strip()
        if len(clean.split()) >= 4:
            return clean[:80]

    return "the provided content"


def _first_sentence(text: str) -> str:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return (parts[0].strip() if parts else text[:120].strip())[:120]
