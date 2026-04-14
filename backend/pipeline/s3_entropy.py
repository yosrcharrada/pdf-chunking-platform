"""
S3 — Entropy Boundary Refinement
Computes Jensen-Shannon Divergence (JSD) over unigram token distributions
between adjacent chunks, then applies threshold logic to merge/confirm
boundaries.

A true single-layer LSTM cell (LSTMEntropyMemory) maintains both a hidden
state hₜ and a cell state cₜ across chunk boundaries, implementing the
architectural claim hₜ = f(hₜ₋₁, xₜ) with proper input/forget/cell/output
gates.  The LSTM output scalar is blended with the raw JSD to decide
boundaries, so macro topic shifts are detected even when local JSD is weak.
"""

import re
from typing import Any, Dict, List

import numpy as np


# ---------------------------------------------------------------------------
# LSTM entropy memory  (Fix 1 + Fix 2)
# ---------------------------------------------------------------------------

class LSTMEntropyMemory:
    """
    Minimal single-layer LSTM cell for tracking entropy drift across chunk
    boundaries.

    Input dimension : 3  (jsd, shannon_entropy, token_overlap)
    Hidden dimension: 8
    Output          : scalar boundary confidence in [0, 1]

    Weights are fixed (not trained) using Xavier initialisation with a fixed
    seed, making the computation deterministic and reproducible.
    """

    def __init__(self, input_dim: int = 3, hidden_dim: int = 8, seed: int = 42):
        rng = np.random.RandomState(seed)
        scale = np.sqrt(2.0 / (input_dim + hidden_dim))

        # Input gate
        self.Wi = rng.randn(hidden_dim, input_dim)  * scale
        self.Ui = rng.randn(hidden_dim, hidden_dim) * scale
        self.bi = np.zeros(hidden_dim)

        # Forget gate  (bias initialised to 1 — standard LSTM practice)
        self.Wf = rng.randn(hidden_dim, input_dim)  * scale
        self.Uf = rng.randn(hidden_dim, hidden_dim) * scale
        self.bf = np.ones(hidden_dim)

        # Cell gate
        self.Wg = rng.randn(hidden_dim, input_dim)  * scale
        self.Ug = rng.randn(hidden_dim, hidden_dim) * scale
        self.bg = np.zeros(hidden_dim)

        # Output gate
        self.Wo = rng.randn(hidden_dim, input_dim)  * scale
        self.Uo = rng.randn(hidden_dim, hidden_dim) * scale
        self.bo = np.zeros(hidden_dim)

        # Projection: hidden_dim -> scalar
        self.Wp = rng.randn(1, hidden_dim) * scale

        self.hidden_dim = hidden_dim
        self.h = np.zeros(hidden_dim)   # hidden state
        self.c = np.zeros(hidden_dim)   # cell state

    def reset(self) -> None:
        self.h = np.zeros(self.hidden_dim)
        self.c = np.zeros(self.hidden_dim)

    def step(self, jsd: float, shannon_h: float, token_overlap: float) -> float:
        """
        Run one LSTM step given the current boundary's entropy features.

        Args:
            jsd          : Jensen-Shannon divergence between adjacent chunks
            shannon_h    : normalised Shannon entropy of the left chunk
            token_overlap: Jaccard overlap between the two chunk vocabularies

        Returns:
            scalar boundary confidence score in [0, 1]
        """
        x = np.array([jsd, shannon_h, token_overlap], dtype=np.float32)

        i_gate = self._sigmoid(self.Wi @ x + self.Ui @ self.h + self.bi)
        f_gate = self._sigmoid(self.Wf @ x + self.Uf @ self.h + self.bf)
        g_gate = np.tanh(      self.Wg @ x + self.Ug @ self.h + self.bg)
        o_gate = self._sigmoid(self.Wo @ x + self.Uo @ self.h + self.bo)

        self.c = f_gate * self.c + i_gate * g_gate
        self.h = o_gate * np.tanh(self.c)

        score = float(self._sigmoid(self.Wp @ self.h)[0])
        return score

    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(x, -20.0, 20.0)))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def refine_boundaries(
    chunks: List[Dict], config: Dict[str, Any]
) -> List[Dict]:
    """
    Refine chunk boundaries using JSD entropy analysis and the LSTM memory.

    Adds to each chunk dict:
        jsd_score      - JSD against the *next* chunk (0.0 for last chunk)
        hidden_state   - scalar output of the LSTM at this boundary
        lstm_cell      - LSTM cell-state vector at this boundary (list[float])
        boundary_type  - 'merged' | 'hard' | 'soft' | 'end' | 'single'

    Returns a new list of chunk dicts (may be shorter due to merges).
    """
    if not chunks:
        return chunks

    if len(chunks) == 1:
        c = dict(chunks[0])
        c.update({
            "jsd_score":     0.0,
            "hidden_state":  0.0,
            "lstm_cell":     [],
            "boundary_type": "single",
        })
        return [c]

    tau_low  = float(config.get("tau_jsd_low",  0.15))
    tau_high = float(config.get("tau_jsd_high", 0.45))

    lstm = LSTMEntropyMemory(input_dim=3, hidden_dim=8, seed=42)
    lstm.reset()

    # Work on a mutable copy; we may shorten the list during merges
    work = list(chunks)
    result: List[Dict] = []

    i = 0
    while i < len(work):
        chunk = dict(work[i])

        if i < len(work) - 1:
            text_a = chunk["text"]
            text_b = work[i + 1]["text"]

            jsd           = _compute_jsd(text_a, text_b)
            shannon_h     = _compute_shannon_entropy(text_a)
            token_overlap = _compute_token_overlap(text_a, text_b)

            lstm_score = lstm.step(jsd, shannon_h, token_overlap)

            chunk["jsd_score"]    = round(jsd, 4)
            chunk["hidden_state"] = round(lstm_score, 4)
            chunk["lstm_cell"]    = [round(float(v), 4) for v in lstm.c.tolist()]

            # Decision: blend raw JSD with LSTM confidence
            effective_signal = 0.6 * jsd + 0.4 * lstm_score

            if effective_signal < tau_low:
                # Merge with next chunk
                next_chunk = work[i + 1]
                chunk["text"] = chunk["text"] + "\n\n" + next_chunk["text"]
                chunk["end"]  = next_chunk.get("end", chunk.get("end", 0))
                chunk["boundary_type"] = "merged"
                # Replace the current slot with the merged chunk, drop the next
                work = work[: i + 1] + work[i + 2:]
                work[i] = chunk
                # Do not advance i — re-evaluate the merged chunk against the new next
                continue
            elif effective_signal > tau_high:
                chunk["boundary_type"] = "hard"
            else:
                chunk["boundary_type"] = "soft"
        else:
            # Last chunk
            chunk["jsd_score"]    = 0.0
            chunk["hidden_state"] = 0.0
            chunk["lstm_cell"]    = []
            chunk["boundary_type"] = "end"

        result.append(chunk)
        i += 1

    return result


def get_jsd_series(chunks: List[Dict]) -> List[float]:
    """Extract the JSD score series for visualisation (Tab 3 line chart)."""
    return [c.get("jsd_score", 0.0) for c in chunks]


# ---------------------------------------------------------------------------
# Entropy / overlap helpers
# ---------------------------------------------------------------------------

def _compute_shannon_entropy(text: str) -> float:
    """Compute normalised Shannon entropy of the token distribution in a chunk."""
    tokens = re.findall(r"\b\w+\b", text.lower())
    if not tokens:
        return 0.0
    counts: Dict[str, int] = {}
    for t in tokens:
        counts[t] = counts.get(t, 0) + 1
    total = len(tokens)
    probs = [c / total for c in counts.values()]
    H = -sum(p * np.log2(p) for p in probs if p > 0)
    max_H = float(np.log2(total)) if total > 1 else 1.0
    return float(H / max_H)   # normalised to [0, 1]


def _compute_token_overlap(text_a: str, text_b: str) -> float:
    """Jaccard overlap between the unigram sets of two adjacent chunks."""
    va = set(re.findall(r"\b\w+\b", text_a.lower()))
    vb = set(re.findall(r"\b\w+\b", text_b.lower()))
    union = va | vb
    if not union:
        return 0.0
    return float(len(va & vb) / len(union))


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
    """KL divergence D(P || Q), safe against zeros."""
    eps = 1e-12
    mask = p > 0
    return float(np.sum(p[mask] * np.log(p[mask] / np.maximum(q[mask], eps))))


def _tokenize(text: str) -> List[str]:
    return re.findall(r"\b\w+\b", text.lower())
