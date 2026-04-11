"""
S3 — Entropy Boundary Refinement
Computes Jensen-Shannon Divergence (JSD) over unigram token distributions
between adjacent chunks, then applies threshold logic to merge/confirm
boundaries.  An EMA-based rolling hidden state detects macro topic shifts.
"""

import re
from typing import Any, Dict, List

import numpy as np


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def refine_boundaries(
    chunks: List[Dict], config: Dict[str, Any]
) -> List[Dict]:
    """
    Refine chunk boundaries using JSD entropy analysis.

    Adds to each chunk dict:
        jsd_score      – JSD against the *next* chunk (0 for last chunk)
        hidden_state   – EMA-accumulated entropy signal at this boundary
        boundary_type  – 'merged' | 'hard' | 'soft' | 'end' | 'single'

    Returns a new list of chunk dicts (may be shorter due to merges).
    """
    if not chunks:
        return chunks

    if len(chunks) == 1:
        c = dict(chunks[0])
        c.update({"jsd_score": 0.0, "hidden_state": 0.0, "boundary_type": "single"})
        return [c]

    tau_low = float(config.get("tau_jsd_low", 0.15))
    tau_high = float(config.get("tau_jsd_high", 0.45))

    # ---- Compute JSD between every adjacent pair ----
    jsd_scores: List[float] = [
        _compute_jsd(chunks[i]["text"], chunks[i + 1]["text"])
        for i in range(len(chunks) - 1)
    ]

    # ---- Rolling hidden state (LSTM-inspired EMA) ----
    alpha_ema = 0.3
    h = jsd_scores[0]
    hidden_states: List[float] = [h]
    for jsd in jsd_scores[1:]:
        h = alpha_ema * h + (1.0 - alpha_ema) * jsd
        hidden_states.append(h)

    # ---- Merge / confirm pass ----
    result: List[Dict] = []
    i = 0
    while i < len(chunks):
        chunk = dict(chunks[i])

        if i < len(jsd_scores):
            local_jsd = jsd_scores[i]
            hidden = hidden_states[i]
            # Use the max of local JSD and a scaled hidden signal to catch
            # macro topic shifts even when the local window is small.
            effective_jsd = max(local_jsd, hidden * 0.55)

            chunk["jsd_score"] = round(local_jsd, 4)
            chunk["hidden_state"] = round(hidden, 4)

            if effective_jsd < tau_low and i + 1 < len(chunks):
                # ---- Merge ----
                nxt = chunks[i + 1]
                chunk["text"] = chunk["text"] + "\n\n" + nxt["text"]
                chunk["end"] = nxt.get("end", chunk.get("end", 0))
                chunk["boundary_type"] = "merged"
                result.append(chunk)
                i += 2
                continue
            elif effective_jsd > tau_high:
                chunk["boundary_type"] = "hard"
            else:
                chunk["boundary_type"] = "soft"
        else:
            chunk["jsd_score"] = 0.0
            chunk["hidden_state"] = 0.0
            chunk["boundary_type"] = "end"

        result.append(chunk)
        i += 1

    return result


def get_jsd_series(chunks: List[Dict]) -> List[float]:
    """Extract the JSD score series for visualisation (Tab 3 line chart)."""
    return [c.get("jsd_score", 0.0) for c in chunks]


# ---------------------------------------------------------------------------
# JSD computation
# ---------------------------------------------------------------------------

def _compute_jsd(text1: str, text2: str) -> float:
    """Jensen-Shannon Divergence between unigram distributions of two texts."""
    tokens1 = _tokenize(text1)
    tokens2 = _tokenize(text2)

    if not tokens1 or not tokens2:
        return 0.5

    vocab = list(set(tokens1) | set(tokens2))
    count1: Dict[str, int] = {}
    count2: Dict[str, int] = {}
    for t in tokens1:
        count1[t] = count1.get(t, 0) + 1
    for t in tokens2:
        count2[t] = count2.get(t, 0) + 1

    total1 = len(tokens1)
    total2 = len(tokens2)

    p = np.array([count1.get(w, 0) / total1 for w in vocab], dtype=np.float64)
    q = np.array([count2.get(w, 0) / total2 for w in vocab], dtype=np.float64)

    m = (p + q) / 2.0
    jsd = 0.5 * _kl(p, m) + 0.5 * _kl(q, m)
    return float(np.clip(jsd, 0.0, 1.0))


def _kl(p: np.ndarray, q: np.ndarray) -> float:
    """KL divergence D(P ‖ Q), safe against zeros."""
    eps = 1e-12
    mask = p > 0
    return float(np.sum(p[mask] * np.log(p[mask] / np.maximum(q[mask], eps))))


def _tokenize(text: str) -> List[str]:
    return re.findall(r"\b\w+\b", text.lower())
