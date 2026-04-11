"""
S5 — Graph Enrichment
Runs spaCy NER on every chunk, builds a shared-entity graph where nodes
are chunks and edges link chunks sharing ≥1 named entity, then computes:

    h_graph(Cᵢ) = mean( embeddings of Cᵢ's neighbours, weighted by edge count )

The enriched vector plus entity metadata are stored on each chunk.
Falls back gracefully when spaCy is unavailable.
"""

import re
from collections import defaultdict
from typing import Any, Dict, List, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Lazy spaCy loader (avoids ImportError if not installed)
# ---------------------------------------------------------------------------

_nlp = None
_spacy_available = False


def _get_nlp():
    global _nlp, _spacy_available
    if _nlp is not None:
        return _nlp
    try:
        import spacy  # noqa: PLC0415
        _nlp = spacy.load("en_core_web_sm", disable=["parser", "tagger", "lemmatizer"])
        _spacy_available = True
    except Exception:
        _spacy_available = False
        _nlp = None
    return _nlp


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def enrich_graph(
    chunks: List[Dict],
    embeddings: Optional[List[List[float]]],
    config: Dict[str, Any],
) -> List[Dict]:
    """
    Enrich each chunk with:
        entities        – list of {text, label} dicts found by NER
        graph_neighbors – list of chunk indices sharing an entity
        graph_vector    – mean-pooled neighbour embedding (list[float])

    Returns the enriched list (same length).
    """
    if not chunks:
        return chunks

    enriched = [dict(c) for c in chunks]

    # ---- NER ----
    nlp = _get_nlp()
    for i, chunk in enumerate(enriched):
        text = chunk.get("text", "")
        if nlp is not None:
            try:
                doc = nlp(text[:10_000])  # cap to avoid OOM
                ents = [
                    {"text": ent.text.strip(), "label": ent.label_}
                    for ent in doc.ents
                    if ent.text.strip()
                ]
            except Exception:
                ents = _regex_ner(text)
        else:
            ents = _regex_ner(text)
        enriched[i]["entities"] = ents

    # ---- Build entity → chunk index mapping ----
    entity_to_chunks: Dict[str, List[int]] = defaultdict(list)
    for i, chunk in enumerate(enriched):
        for ent in chunk.get("entities", []):
            key = ent["text"].lower()
            if key not in entity_to_chunks[key]:
                entity_to_chunks[key].append(i)

    # ---- Build adjacency (edge_count between chunk pairs) ----
    edge_counts: Dict[tuple, int] = defaultdict(int)
    for ent_key, idxs in entity_to_chunks.items():
        for a in range(len(idxs)):
            for b in range(a + 1, len(idxs)):
                pair = (min(idxs[a], idxs[b]), max(idxs[a], idxs[b]))
                edge_counts[pair] += 1

    # ---- Assign neighbours and compute graph vectors ----
    for i, chunk in enumerate(enriched):
        neighbors: List[int] = []
        weights: List[int] = []

        for (a, b), count in edge_counts.items():
            if a == i:
                neighbors.append(b)
                weights.append(count)
            elif b == i:
                neighbors.append(a)
                weights.append(count)

        chunk["graph_neighbors"] = neighbors

        if embeddings and neighbors:
            valid_embs = []
            valid_weights = []
            for nb_idx, w in zip(neighbors, weights):
                if nb_idx < len(embeddings) and embeddings[nb_idx] is not None:
                    valid_embs.append(np.array(embeddings[nb_idx], dtype=np.float32))
                    valid_weights.append(float(w))
            if valid_embs:
                total_w = sum(valid_weights)
                graph_vec = sum(
                    (v * (wt / total_w) for v, wt in zip(valid_embs, valid_weights)),
                    np.zeros_like(valid_embs[0]),
                )
                chunk["graph_vector"] = graph_vec.tolist()
            else:
                chunk["graph_vector"] = (
                    embeddings[i].copy()
                    if i < len(embeddings) and embeddings[i]
                    else []
                )
        else:
            chunk["graph_vector"] = (
                embeddings[i].copy()
                if embeddings and i < len(embeddings) and embeddings[i]
                else []
            )

    return enriched


def build_entity_graph_data(chunks: List[Dict]) -> Dict[str, Any]:
    """
    Build a serialisable graph dict for the frontend SVG renderer.
    Returns { nodes: [{id, label, entity_count}], edges: [{source, target, weight}] }
    """
    nodes = []
    for i, chunk in enumerate(chunks):
        ent_count = len(chunk.get("entities", []))
        label = f"C{i}"
        nodes.append({"id": i, "label": label, "entity_count": ent_count})

    edges: List[Dict] = []
    seen: set = set()

    for i, chunk in enumerate(chunks):
        for nb in chunk.get("graph_neighbors", []):
            pair = (min(i, nb), max(i, nb))
            if pair not in seen:
                seen.add(pair)
                edges.append({"source": pair[0], "target": pair[1], "weight": 1})

    return {"nodes": nodes, "edges": edges}


# ---------------------------------------------------------------------------
# Regex-based NER fallback
# ---------------------------------------------------------------------------

def _regex_ner(text: str) -> List[Dict]:
    """Very lightweight NER using capitalisation heuristics."""
    entities: List[Dict] = []

    # Proper nouns: consecutive title-cased words
    for match in re.finditer(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,4})\b", text):
        word = match.group(1)
        if len(word) > 3 and word not in {
            "The", "This", "That", "These", "Those", "They", "Their",
            "When", "Where", "What", "Which", "While", "With", "Also",
        }:
            entities.append({"text": word, "label": "ENTITY"})

    # Deduplicate by text
    seen: set = set()
    unique: List[Dict] = []
    for e in entities:
        key = e["text"].lower()
        if key not in seen:
            seen.add(key)
            unique.append(e)

    return unique[:30]  # cap to avoid noise
