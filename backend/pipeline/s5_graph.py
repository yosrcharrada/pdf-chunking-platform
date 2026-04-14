"""
S5 — Graph Enrichment
Runs spaCy NER on every chunk, builds a shared-entity graph where nodes
are chunks and edges link chunks sharing >= 1 named entity, then computes:

    h_graph(Ci) = mean( embeddings of Ci's neighbours, weighted by edge count )

The enriched vector plus entity metadata are stored on each chunk.
Falls back gracefully when spaCy is unavailable.

KG Store (Fix 3)
-----------------
A module-level KGStore singleton persists entity co-occurrence relationships
across pipeline runs as a JSON file.  enrich_graph() reads prior edge weights
BEFORE building the current graph (so the KG acts as an input to S5, not only
a post-hoc write), then writes the merged result back after processing.
"""

import json
import os
import re
from collections import defaultdict
from typing import Any, Dict, List, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Persistent Knowledge Graph Store  (Fix 3)
# ---------------------------------------------------------------------------

KG_STORE_PATH = os.path.join(os.path.dirname(__file__), "..", "kg_store.json")


class KGStore:
    """
    Persistent knowledge graph store.

    Persists entity co-occurrence relationships across pipeline runs.
    S5 reads prior entity edges before building the current graph,
    then writes the merged result back — making the KG cumulative.
    """

    def __init__(self, path: str = KG_STORE_PATH):
        self.path = os.path.abspath(path)
        self.entity_cooccurrence: Dict[str, Dict[str, int]] = defaultdict(
            lambda: defaultdict(int)
        )
        self.entity_chunk_index: Dict[str, List[str]] = defaultdict(list)
        self._load()

    def _load(self) -> None:
        if os.path.exists(self.path):
            try:
                with open(self.path, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
                for ent_a, neighbours in data.get("cooccurrence", {}).items():
                    for ent_b, weight in neighbours.items():
                        self.entity_cooccurrence[ent_a][ent_b] = int(weight)
                self.entity_chunk_index = defaultdict(
                    list, data.get("chunk_index", {})
                )
            except Exception:
                pass  # start fresh if file is corrupt

    def save(self) -> None:
        try:
            data = {
                "cooccurrence": {
                    k: dict(v) for k, v in self.entity_cooccurrence.items()
                },
                "chunk_index": dict(self.entity_chunk_index),
            }
            with open(self.path, "w", encoding="utf-8") as fh:
                json.dump(data, fh, ensure_ascii=False, indent=2)
        except Exception:
            pass  # non-fatal — KG store write failure should not break pipeline

    def add_chunk_entities(self, chunk_id: str, entities: List[str]) -> None:
        """Register entities from a new chunk and update co-occurrence counts."""
        for ent in entities:
            if chunk_id not in self.entity_chunk_index[ent]:
                self.entity_chunk_index[ent].append(chunk_id)
        for idx_a, ent_a in enumerate(entities):
            for ent_b in entities[idx_a + 1:]:
                self.entity_cooccurrence[ent_a][ent_b] += 1
                self.entity_cooccurrence[ent_b][ent_a] += 1

    def get_prior_weight(self, ent_a: str, ent_b: str) -> int:
        """Return historical co-occurrence count between two entities."""
        return int(self.entity_cooccurrence.get(ent_a, {}).get(ent_b, 0))

    def get_prior_chunk_ids(self, entity: str) -> List[str]:
        """Return all chunk IDs that previously contained this entity."""
        return list(self.entity_chunk_index.get(entity, []))


# Module-level KGStore singleton — shared across pipeline runs
_kg_store = KGStore()


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
        import spacy  # noqa: E402
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
        entities        - list of {text, label} dicts found by NER
        graph_neighbors - list of chunk indices sharing an entity
        graph_vector    - mean-pooled neighbour embedding (list[float])

    Prior entity co-occurrence knowledge from the KGStore is read FIRST to
    boost current edge weights, then new entity data is written back so the
    KG grows cumulatively across runs.

    Returns the enriched list (same length).
    """
    if not chunks:
        return chunks

    enriched = [dict(c) for c in chunks]
    job_id = config.get("job_id", "unknown")

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

    # ---- Build entity -> chunk index mapping ----
    entity_to_chunks: Dict[str, List[int]] = defaultdict(list)
    for i, chunk in enumerate(enriched):
        for ent in chunk.get("entities", []):
            key = ent["text"].lower()
            if i not in entity_to_chunks[key]:
                entity_to_chunks[key].append(i)

    # ---- Build adjacency from current run ----
    edge_counts: Dict[tuple, int] = defaultdict(int)
    for ent_key, idxs in entity_to_chunks.items():
        for a in range(len(idxs)):
            for b in range(a + 1, len(idxs)):
                pair = (min(idxs[a], idxs[b]), max(idxs[a], idxs[b]))
                edge_counts[pair] += 1

    # ---- Boost edge weights using KG prior knowledge (Fix 3: KG as input) ----
    for i in range(len(enriched)):
        for j in range(i + 1, len(enriched)):
            ents_i = {e["text"].lower() for e in enriched[i].get("entities", [])}
            ents_j = {e["text"].lower() for e in enriched[j].get("entities", [])}
            for ei in ents_i:
                for ej in ents_j:
                    prior = _kg_store.get_prior_weight(ei, ej)
                    if prior > 0:
                        pair = (i, j)
                        edge_counts[pair] += prior

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
                    list(embeddings[i])
                    if i < len(embeddings) and embeddings[i]
                    else []
                )
        else:
            chunk["graph_vector"] = (
                list(embeddings[i])
                if embeddings and i < len(embeddings) and embeddings[i]
                else []
            )

    # ---- Write new entity co-occurrences back to KG store (Fix 3: KG write) ----
    for i, chunk in enumerate(enriched):
        chunk_id = f"{job_id}::C{i}"
        entity_texts = [e["text"].lower() for e in chunk.get("entities", [])]
        _kg_store.add_chunk_entities(chunk_id, entity_texts)

    _kg_store.save()

    return enriched


def build_entity_graph_data(chunks: List[Dict]) -> Dict[str, Any]:
    """
    Build a serialisable graph dict for the frontend SVG renderer.
    Returns { nodes: [{id, label, entity_count}], edges: [{source, target, weight}] }
    """
    nodes = []
    for i, chunk in enumerate(chunks):
        ent_count = len(chunk.get("entities", []))
        nodes.append({"id": i, "label": f"C{i}", "entity_count": ent_count})

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
