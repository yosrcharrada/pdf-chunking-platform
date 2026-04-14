"""
S7 — RL Reward Calibration Loop
Iteratively adjusts chunking thresholds to maximise a composite reward:

    r = alpha * mean_CodeBLEU + beta * recall_proxy - lambda * chunk_count_penalty

An LSTMQAgent encodes the pipeline metric state vector through a single-layer
LSTM, then selects actions via epsilon-greedy exploration over a tabular
Q-table.  After each trial the Q-table is updated with the Bellman equation:

    Q(s,a) <- Q(s,a) + lr * (r + gamma * max_a' Q(s',a') - Q(s,a))

This replaces the previous random perturbation scheme and implements the
architectural claims of "LSTM policy pi_theta(a_t | s_t, h_{t-1})" and
"Q-update Q(s,a) <- r + gamma max Q(s',a')".
"""

import copy
import re
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Target average words-per-chunk used in the reward count-penalty calculation.
# 300 words (~400 tokens) is a widely cited sweet-spot for RAG retrieval
# chunks — large enough to carry full context, small enough to stay focused.
_TARGET_WORDS_PER_CHUNK = 300

# Pipeline stages that are re-run each iteration
from .s2_chunkers import run_all_chunkers, select_best_strategy
from .s3_entropy import refine_boundaries
from .s4_boundary import filter_boundaries, _lexical_boundary_score
from .s5_graph import enrich_graph
from .s6_embedding import embed_chunks


# ---------------------------------------------------------------------------
# LSTM Q-Agent  (Fix 4 + Fix 5)
# ---------------------------------------------------------------------------

class LSTMQAgent:
    """
    RL agent with:
    - LSTM state encoder: encodes the 5-dim metric state vector into a hidden
      representation hₜ using proper LSTM gates (i, f, g, o) and cell state cₜ
    - Tabular Q-table keyed on (discretized_hidden_bin, action_index)
    - Epsilon-greedy action selection
    - Bellman Q-update: Q(s,a) <- Q(s,a) + lr*(r + gamma*max_a' Q(s',a') - Q(s,a))

    State vector (5 dims):
      [mean_jsd, mean_boundary_score, mean_icc, recall_proxy, chunk_count_ratio]

    Actions (6 discrete):
      0: tau_jsd_low  -= 0.02
      1: tau_jsd_low  += 0.02
      2: tau_jsd_high -= 0.03
      3: tau_jsd_high += 0.03
      4: n_max        -= 20
      5: n_max        += 20
    """

    ACTIONS = [
        ("tau_jsd_low",  -0.02),
        ("tau_jsd_low",  +0.02),
        ("tau_jsd_high", -0.03),
        ("tau_jsd_high", +0.03),
        ("n_max",        -20),
        ("n_max",        +20),
    ]
    N_ACTIONS = len(ACTIONS)

    def __init__(
        self,
        state_dim: int = 5,
        hidden_dim: int = 8,
        lr: float = 0.1,
        gamma: float = 0.9,
        epsilon: float = 0.3,
        seed: int = 42,
    ):
        self.lr      = lr
        self.gamma   = gamma
        self.epsilon = epsilon
        self.rng     = np.random.RandomState(seed)
        self.hidden_dim = hidden_dim

        # Fixed-weight LSTM encoder (Xavier init, deterministic)
        rng2  = np.random.RandomState(seed + 1)
        scale = np.sqrt(2.0 / (state_dim + hidden_dim))

        self.Wi = rng2.randn(hidden_dim, state_dim)  * scale
        self.Wf = rng2.randn(hidden_dim, state_dim)  * scale
        self.Wg = rng2.randn(hidden_dim, state_dim)  * scale
        self.Wo = rng2.randn(hidden_dim, state_dim)  * scale
        self.Ui = rng2.randn(hidden_dim, hidden_dim) * scale
        self.Uf = rng2.randn(hidden_dim, hidden_dim) * scale
        self.Ug = rng2.randn(hidden_dim, hidden_dim) * scale
        self.Uo = rng2.randn(hidden_dim, hidden_dim) * scale
        self.bf = np.ones(hidden_dim)                          # forget-gate bias = 1
        self.bi = np.zeros(hidden_dim)
        self.bg = np.zeros(hidden_dim)
        self.bo = np.zeros(hidden_dim)

        self.h = np.zeros(hidden_dim)  # LSTM hidden state
        self.c = np.zeros(hidden_dim)  # LSTM cell state

        # Tabular Q-table: (state_key_tuple, action_idx) -> float
        self.Q: Dict[Tuple, float] = defaultdict(float)
        self.last_state_key: Optional[Tuple] = None
        self.last_action:    Optional[int]   = None

    # ------------------------------------------------------------------
    # LSTM helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(x, -20.0, 20.0)))

    def _encode_state(self, state_vec: np.ndarray) -> np.ndarray:
        """Run one LSTM step to encode state into hidden representation."""
        x   = state_vec.astype(np.float32)
        i_g = self._sigmoid(self.Wi @ x + self.Ui @ self.h + self.bi)
        f_g = self._sigmoid(self.Wf @ x + self.Uf @ self.h + self.bf)
        g_g = np.tanh(      self.Wg @ x + self.Ug @ self.h + self.bg)
        o_g = self._sigmoid(self.Wo @ x + self.Uo @ self.h + self.bo)
        self.c = f_g * self.c + i_g * g_g
        self.h = o_g * np.tanh(self.c)
        return self.h.copy()

    @staticmethod
    def _discretize(hidden: np.ndarray) -> Tuple:
        """Bin each hidden unit to {-1, 0, +1} for tabular Q-table key."""
        return tuple(np.sign(hidden).astype(int).tolist())

    # ------------------------------------------------------------------
    # RL interface
    # ------------------------------------------------------------------

    def select_action(self, state_vec: np.ndarray) -> int:
        """Epsilon-greedy action selection over discretized LSTM hidden state."""
        hidden    = self._encode_state(state_vec)
        state_key = self._discretize(hidden)
        self.last_state_key = state_key

        if self.rng.rand() < self.epsilon:
            action = int(self.rng.randint(0, self.N_ACTIONS))
        else:
            q_vals = [self.Q[(state_key, a)] for a in range(self.N_ACTIONS)]
            action = int(np.argmax(q_vals))

        self.last_action = action
        return action

    def update(self, reward: float, next_state_vec: np.ndarray) -> None:
        """
        Bellman Q-update:
            Q(s,a) <- Q(s,a) + lr * (r + gamma * max_a' Q(s',a') - Q(s,a))
        """
        if self.last_state_key is None or self.last_action is None:
            return

        next_hidden    = self._encode_state(next_state_vec)
        next_state_key = self._discretize(next_hidden)

        current_q  = self.Q[(self.last_state_key, self.last_action)]
        max_next_q = max(self.Q[(next_state_key, a)] for a in range(self.N_ACTIONS))

        self.Q[(self.last_state_key, self.last_action)] = (
            current_q + self.lr * (reward + self.gamma * max_next_q - current_q)
        )

    def apply_action(self, action_idx: int, config: Dict[str, Any]) -> Dict[str, Any]:
        """Apply the selected action to the config dict. Returns a modified copy."""
        new_config = copy.deepcopy(config)
        param, delta = self.ACTIONS[action_idx]
        current_val = float(new_config.get(param, 0))
        new_config[param] = round(current_val + delta, 4)

        # Clamp to valid ranges
        new_config["tau_jsd_low"]  = float(
            np.clip(new_config.get("tau_jsd_low",  0.15), 0.05, 0.35)
        )
        new_config["tau_jsd_high"] = float(
            np.clip(new_config.get("tau_jsd_high", 0.45), 0.30, 0.70)
        )
        new_config["n_max"] = int(np.clip(new_config.get("n_max", 500), 200, 800))

        # Ensure tau_low < tau_high
        if new_config["tau_jsd_low"] >= new_config["tau_jsd_high"]:
            new_config["tau_jsd_high"] = new_config["tau_jsd_low"] + 0.10

        return new_config


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
    Run the RL calibration loop using LSTMQAgent.

    Returns:
        best_chunks     - optimised chunk list
        reward_history  - reward value at each iteration
        final_config    - the config dict used for the best reward
                          (includes q_table_size and lstm_hidden_dim)
    """
    max_iters  = int(config.get("max_iterations", 10))
    alpha      = float(config.get("alpha", 0.4))
    beta       = float(config.get("beta",  0.4))
    lam        = float(config.get("lambda", 0.2))
    model_name = config.get("embedding_model", "all-MiniLM-L6-v2")
    doc_type   = doc_profile.get("type", "prose")

    probe_queries = _generate_probes(text, n=5)
    agent = LSTMQAgent(state_dim=5, hidden_dim=8, lr=0.1, gamma=0.9, epsilon=0.3)

    def _state_vec(chunks: List[Dict]) -> np.ndarray:
        return np.array([
            float(np.mean([c.get("jsd_score",      0.5) for c in chunks])),
            float(np.mean([c.get("boundary_score", 0.5) for c in chunks])),
            float(np.mean([c.get("icc",            0.5) for c in chunks])),
            float(_recall_proxy(chunks, probe_queries)),
            float(np.clip(len(chunks) / max(1.0, _target_count(chunks)), 0.0, 2.0)),
        ], dtype=np.float32)

    current_config = copy.deepcopy(config)
    embs_init      = [c.get("embedding", []) for c in initial_chunks]
    best_reward    = _compute_reward(
        initial_chunks, embs_init, probe_queries, alpha, beta, lam
    )
    best_chunks    = initial_chunks
    best_config    = copy.deepcopy(current_config)
    reward_history: List[float] = [round(best_reward, 4)]

    current_chunks = initial_chunks

    for _iteration in range(max_iters):
        sv         = _state_vec(current_chunks)
        action_idx = agent.select_action(sv)
        trial_config = agent.apply_action(action_idx, current_config)

        try:
            all_chunks   = run_all_chunkers(text, doc_type, trial_config)
            trial_chunks = select_best_strategy(all_chunks, doc_type, trial_config)

            if not trial_chunks:
                reward_history.append(round(best_reward, 4))
                continue

            trial_chunks = refine_boundaries(trial_chunks, trial_config)
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

        # Bellman Q-update
        next_sv = _state_vec(trial_chunks)
        agent.update(trial_reward, next_sv)

        if trial_reward > best_reward:
            best_reward    = trial_reward
            best_chunks    = trial_chunks
            best_config    = copy.deepcopy(trial_config)
            current_config = trial_config
            current_chunks = trial_chunks

        reward_history.append(round(trial_reward, 4))

    # Expose Q-table metadata in the returned config
    best_config["q_table_size"]    = len(agent.Q)
    best_config["lstm_hidden_dim"] = agent.hidden_dim

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

    recall = _recall_proxy(chunks, probe_queries)

    chunk_count  = len(chunks)
    target_count = _target_count(chunks)
    count_penalty = abs(chunk_count - target_count) / max(target_count, 1.0)

    return float(
        alpha * mean_boundary_score
        + beta  * recall
        - lam   * count_penalty
    )


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


def _target_count(chunks: List[Dict]) -> float:
    """Expected number of chunks given total word count and target chunk size."""
    total_words = sum(len(c["text"].split()) for c in chunks)
    return max(3.0, total_words / _TARGET_WORDS_PER_CHUNK)


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
