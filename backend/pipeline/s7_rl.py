"""
S7 — RL Reward Calibration Loop
Iteratively adjusts chunking thresholds to maximise a composite reward:

    r = α·mean_CodeBLEU + β·recall_proxy − λ·chunk_count_penalty

State:   [mean_JSD, mean_CodeBLEU, mean_ICC, chunk_count, size_variance]
Action:  small delta perturbations to τ_jsd_low, τ_jsd_high, τ_sem, n_min, n_max
"""

import copy
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Target average words-per-chunk used in the reward count-penalty calculation
_TARGET_WORDS_PER_CHUNK = 300

# Pipeline stages that are re-run each iteration
from .s2_chunkers import run_all_chunkers, select_best_strategy
from .s3_entropy import refine_boundaries
from .s4_boundary import filter_boundaries, _lexical_boundary_score
from .s5_graph import enrich_graph
from .s6_embedding import embed_chunks


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_rl_loop(
    text: str,
    doc_profile: Dict[str, Any],
    initial_chunks: List[Dict],
    config: Dict[str, Any],
) -> Tuple[List[Dict], List[float], Dict[str, Any]]:
    """
    Run the RL calibration loop.

    Returns:
        best_chunks     – optimised chunk list
        reward_history  – reward value at each iteration
        final_config    – the config dict used for the best reward
    """
    max_iters = int(config.get("max_iterations", 10))
    alpha = float(config.get("alpha", 0.4))
    beta  = float(config.get("beta", 0.4))
    lam   = float(config.get("lambda", 0.2))
    model_name = config.get("embedding_model", "all-MiniLM-L6-v2")
    doc_type = doc_profile.get("type", "prose")

    # Generate probe queries once from document headings / first sentences
    probe_queries = _generate_probes(text, n=5)

    # ---- Evaluate initial chunks ----
    current_config = copy.deepcopy(config)
    best_chunks = initial_chunks
    best_config = copy.deepcopy(current_config)

    embs: List[List[float]] = [
        c.get("embedding", []) for c in initial_chunks
    ]
    best_reward = _compute_reward(
        initial_chunks, embs, probe_queries, alpha, beta, lam
    )
    reward_history: List[float] = [round(best_reward, 4)]

    for iteration in range(max_iters):
        # ---- Perturb config ----
        trial_config = _perturb_config(current_config, iteration)

        # ---- Re-run S2 → S6 with trial config ----
        try:
            all_chunks = run_all_chunkers(text, doc_type, trial_config)
            trial_chunks = select_best_strategy(all_chunks, doc_type, trial_config)

            if not trial_chunks:
                reward_history.append(round(best_reward, 4))
                continue

            trial_chunks = refine_boundaries(trial_chunks, trial_config)
            # Preliminary embeddings for S4 (may be empty initially)
            prelim_embs: List[List[float]] = []
            trial_chunks = filter_boundaries(
                trial_chunks, doc_type, prelim_embs, trial_config
            )
            trial_chunks = enrich_graph(trial_chunks, prelim_embs, trial_config)
            trial_chunks, trial_embs = embed_chunks(
                trial_chunks, text, doc_profile, model_name, trial_config
            )
        except Exception:
            reward_history.append(round(best_reward, 4))
            continue

        trial_reward = _compute_reward(
            trial_chunks, trial_embs, probe_queries, alpha, beta, lam
        )

        if trial_reward > best_reward:
            best_reward = trial_reward
            best_chunks = trial_chunks
            best_config = copy.deepcopy(trial_config)
            current_config = trial_config  # carry forward
        # else: revert (keep current_config unchanged)

        reward_history.append(round(trial_reward, 4))

    return best_chunks, reward_history, best_config


# ---------------------------------------------------------------------------
# Reward computation
# ---------------------------------------------------------------------------

def _compute_reward(
    chunks: List[Dict],
    embeddings: List[List[float]],
    probe_queries: List[str],
    alpha: float,
    beta: float,
    lam: float,
) -> float:
    if not chunks:
        return -1.0

    mean_boundary_score = float(
        np.mean([c.get("boundary_score", 0.5) for c in chunks])
    )
    mean_icc = float(np.mean([c.get("icc", 0.5) for c in chunks]))

    recall = _recall_proxy(chunks, probe_queries)

    # Chunk count penalty: penalise both too many and too few
    chunk_count = len(chunks)
    target_count = max(3.0, float(
        sum(len(c["text"].split()) for c in chunks) / _TARGET_WORDS_PER_CHUNK
    ))
    count_penalty = abs(chunk_count - target_count) / max(target_count, 1.0)

    reward = (
        alpha * mean_boundary_score
        + beta * recall
        - lam * count_penalty
    )
    return float(reward)


def _recall_proxy(chunks: List[Dict], probes: List[str]) -> float:
    """Fraction of probe queries where at least one chunk contains key terms."""
    if not probes:
        return 0.5
    hit_count = 0
    for query in probes:
        key_terms = set(re.findall(r"\b\w{4,}\b", query.lower()))
        if not key_terms:
            hit_count += 1
            continue
        for chunk in chunks:
            chunk_terms = set(re.findall(r"\b\w+\b", chunk["text"].lower()))
            if key_terms & chunk_terms:
                hit_count += 1
                break
    return hit_count / len(probes)


# ---------------------------------------------------------------------------
# State vector
# ---------------------------------------------------------------------------

def _compute_state(
    chunks: List[Dict], embeddings: List[List[float]]
) -> np.ndarray:
    mean_jsd = float(np.mean([c.get("jsd_score", 0.0) for c in chunks]))
    mean_boundary = float(np.mean([c.get("boundary_score", 0.5) for c in chunks]))
    mean_icc = float(np.mean([c.get("icc", 0.5) for c in chunks]))
    chunk_count = float(len(chunks))
    sizes = [len(c["text"].split()) for c in chunks]
    size_variance = float(np.var(sizes)) if sizes else 0.0

    return np.array(
        [mean_jsd, mean_boundary, mean_icc, chunk_count, size_variance],
        dtype=np.float32,
    )


# ---------------------------------------------------------------------------
# Config perturbation (action)
# ---------------------------------------------------------------------------

# Delta magnitudes for each parameter
_DELTAS = {
    "tau_jsd_low":  0.03,
    "tau_jsd_high": 0.04,
    "tau_sem":      0.04,
    "n_min":        15,
    "n_max":        40,
}


def _perturb_config(config: Dict[str, Any], seed: int) -> Dict[str, Any]:
    """Perturb one or two config parameters by a small random delta."""
    rng = np.random.default_rng(seed=seed + 7)
    trial = copy.deepcopy(config)

    # Pick 1-2 parameters to adjust
    keys = list(_DELTAS.keys())
    rng.shuffle(keys)
    chosen = keys[: rng.integers(1, 3)]

    for key in chosen:
        delta = _DELTAS[key]
        direction = rng.choice([-1, 1])
        current = float(trial.get(key, _defaults(key)))
        trial[key] = current + direction * delta

    # Enforce constraints
    trial["tau_jsd_low"] = float(
        np.clip(trial.get("tau_jsd_low", 0.15), 0.05, 0.40)
    )
    trial["tau_jsd_high"] = float(
        np.clip(trial.get("tau_jsd_high", 0.45), 0.25, 0.80)
    )
    # Ensure low < high
    if trial["tau_jsd_low"] >= trial["tau_jsd_high"]:
        trial["tau_jsd_high"] = trial["tau_jsd_low"] + 0.10

    trial["tau_sem"] = float(np.clip(trial.get("tau_sem", 0.75), 0.40, 0.95))
    trial["n_min"] = int(np.clip(trial.get("n_min", 100), 20, 300))
    trial["n_max"] = int(np.clip(trial.get("n_max", 500), 100, 1000))
    if trial["n_min"] >= trial["n_max"]:
        trial["n_max"] = trial["n_min"] + 100

    return trial


def _defaults(key: str) -> float:
    return {
        "tau_jsd_low": 0.15,
        "tau_jsd_high": 0.45,
        "tau_sem": 0.75,
        "n_min": 100.0,
        "n_max": 500.0,
    }.get(key, 0.0)


# ---------------------------------------------------------------------------
# Probe query generation
# ---------------------------------------------------------------------------

def _generate_probes(text: str, n: int = 5) -> List[str]:
    """Extract probe queries from headings and first sentences."""
    probes: List[str] = []

    # Markdown / section headings
    for m in re.finditer(r"^#{1,3}\s+(.+)$", text, re.MULTILINE):
        probes.append(m.group(1).strip())
        if len(probes) >= n:
            return probes

    # First sentence of each paragraph
    for para in re.split(r"\n{2,}", text):
        para = para.strip()
        if not para:
            continue
        sentences = re.split(r"(?<=[.!?])\s+", para)
        if sentences and len(sentences[0].split()) >= 5:
            probes.append(sentences[0].strip())
        if len(probes) >= n:
            return probes

    return probes[:n]
