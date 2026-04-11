"""
S1 — Document Profiler
Classifies document type/domain, computes intrinsic quality metrics,
and suggests initial hyperparameters for downstream stages.
"""

import re
from typing import Dict, Any, List

import numpy as np


def profile_document(text: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Profile the document and return classification, metrics, and
    suggested hyperparameters.

    Returns:
        dict with keys: type, domain, length_bucket, token_count,
                        metrics (RC, ICC, DCC, BI, SC, overall),
                        suggested_config
    """
    tokens = text.split()
    token_count = len(tokens)

    doc_type = _classify_type(text)
    domain = _classify_domain(text)

    if token_count < 1000:
        length_bucket = "short"
    elif token_count <= 10000:
        length_bucket = "medium"
    else:
        length_bucket = "long"

    metrics = _compute_metrics(text, tokens, doc_type)
    suggested = _suggest_hyperparams(token_count, doc_type, metrics, config)

    return {
        "type": doc_type,
        "domain": domain,
        "length_bucket": length_bucket,
        "token_count": token_count,
        "metrics": metrics,
        "suggested_config": suggested,
    }


# ---------------------------------------------------------------------------
# Classification helpers
# ---------------------------------------------------------------------------

def _classify_type(text: str) -> str:
    lines = text.split("\n")
    total = max(len(lines), 1)

    code_score = 0
    code_patterns = [
        r"^\s*(def |class |async def )",
        r"^\s*(import |from \w+ import )",
        r"^\s*(public |private |protected |static )",
        r"^\s*(function |const |let |var |=>)",
        r"^\s*(#include|#define|#pragma)",
        r"[{};]\s*$",
    ]
    for line in lines[:200]:
        for pat in code_patterns:
            if re.search(pat, line):
                code_score += 1
                break

    md_header_count = sum(1 for l in lines if re.match(r"^#{1,6}\s+\S", l))
    table_count = sum(1 for l in lines if re.match(r"^\s*\|.+\|", l))

    code_ratio = code_score / total
    if code_ratio > 0.08:
        return "code"
    if table_count / total > 0.08:
        return "table"
    if md_header_count >= 3 or (md_header_count > 0 and table_count > 0):
        return "mixed"
    return "prose"


def _classify_domain(text: str) -> str:
    text_lower = text.lower()
    domains: Dict[str, List[str]] = {
        "technical": [
            "algorithm", "function", "api", "database", "server",
            "software", "hardware", "network", "protocol", "interface",
            "implementation", "framework", "repository", "deployment",
        ],
        "financial": [
            "revenue", "profit", "loss", "investment", "portfolio",
            "equity", "dividend", "fiscal", "balance sheet",
            "cash flow", "earnings", "gdp", "bond",
        ],
        "clinical": [
            "patient", "diagnosis", "treatment", "symptom", "clinical",
            "medical", "therapy", "drug", "dosage", "disease",
            "procedure", "physician", "hospital",
        ],
        "narrative": [
            "story", "character", "chapter", "novel", "author",
            "narrative", "protagonist", "plot", "setting", "scene",
        ],
    }

    scores = {
        domain: sum(text_lower.count(kw) for kw in kws)
        for domain, kws in domains.items()
    }
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "general"


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

def _compute_metrics(
    text: str, tokens: List[str], doc_type: str
) -> Dict[str, float]:
    sentences = _split_sentences(text)
    paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]

    # RC — References Completeness
    ref_hits = len(re.findall(r"\[\d+\]|\([\w\s\-]+,\s*\d{4}\)", text))
    rc = float(np.clip(ref_hits / max(1, len(paragraphs)) * 1.5, 0.0, 1.0))

    # ICC — Intra-chunk Cohesion (vocabulary Jaccard between consecutive sentences)
    icc = _jaccard_consecutive(sentences) if len(sentences) >= 2 else 0.5

    # DCC — Document Contextual Coherence (Jaccard between consecutive paragraphs)
    dcc = _jaccard_consecutive(paragraphs) if len(paragraphs) >= 2 else 0.5

    # BI — Block Integrity (fraction of sentences that end with terminal punctuation)
    complete = sum(
        1 for s in sentences if s.rstrip().endswith((".", "!", "?", ":", ";"))
    )
    bi = float(complete / max(1, len(sentences)))

    # SC — Size Compliance (how close average sentence length is to ideal 10-50 words)
    lengths = [len(s.split()) for s in sentences]
    avg_len = float(np.mean(lengths)) if lengths else 20.0
    ideal_mid = 30.0
    sc = float(np.clip(1.0 - abs(avg_len - ideal_mid) / ideal_mid, 0.0, 1.0))

    overall = float(np.mean([rc, icc, dcc, bi, sc]))
    return {
        "RC": round(rc, 4),
        "ICC": round(icc, 4),
        "DCC": round(dcc, 4),
        "BI": round(bi, 4),
        "SC": round(sc, 4),
        "overall": round(overall, 4),
    }


def _jaccard_consecutive(items: List[str]) -> float:
    overlaps: List[float] = []
    for i in range(len(items) - 1):
        a = set(re.findall(r"\b\w+\b", items[i].lower()))
        b = set(re.findall(r"\b\w+\b", items[i + 1].lower()))
        union = a | b
        if union:
            overlaps.append(len(a & b) / len(union))
    return float(np.mean(overlaps)) if overlaps else 0.5


def _split_sentences(text: str) -> List[str]:
    raw = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in raw if s.strip()]


# ---------------------------------------------------------------------------
# Hyperparameter suggestion
# ---------------------------------------------------------------------------

def _suggest_hyperparams(
    token_count: int,
    doc_type: str,
    metrics: Dict[str, float],
    config: Dict[str, Any],
) -> Dict[str, Any]:
    n_min = int(config.get("n_min", 100))
    n_max = int(config.get("n_max", 500))

    # Adjust for document length
    if token_count < 1000:
        n_min = min(n_min, 50)
        n_max = min(n_max, 200)
    elif token_count > 10000:
        n_min = max(n_min, 150)
        n_max = max(n_max, 600)

    # Code docs benefit from tighter chunks
    if doc_type == "code":
        n_min = max(30, n_min - 30)
        n_max = min(300, n_max)

    # Adjust JSD thresholds based on measured coherence
    tau_low = float(config.get("tau_jsd_low", 0.15))
    tau_high = float(config.get("tau_jsd_high", 0.45))
    icc = metrics.get("ICC", 0.5)
    if icc > 0.6:
        # High cohesion doc → be stricter about merging
        tau_low = max(0.08, tau_low - 0.05)
    elif icc < 0.3:
        # Low cohesion → merge less aggressively
        tau_low = min(0.25, tau_low + 0.05)

    return {
        "n_min": n_min,
        "n_max": n_max,
        "tau_jsd_low": round(tau_low, 3),
        "tau_jsd_high": round(tau_high, 3),
        "tau_sem": float(config.get("tau_sem", 0.75)),
    }
