"""
S2 — Parallel Heuristic Chunkers
Runs three chunking strategies simultaneously:
  (A) Recursive Character Split
  (B) Fixed / Sliding Window
  (C) Structure-based (Markdown / LaTeX / HTML / code AST)
"""

import re
from typing import Dict, Any, List


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_all_chunkers(
    text: str, doc_type: str, config: Dict[str, Any]
) -> Dict[str, List[Dict]]:
    """
    Run all three chunkers and return a dict with their results.

    Returns:
        {
            "recursive": [...],
            "sliding_window": [...],
            "structure": [...],
        }
    """
    n_min = int(config.get("n_min", 100))
    n_max = int(config.get("n_max", 500))
    overlap = max(10, int(n_max * 0.2))

    return {
        "recursive": recursive_character_split(text, n_min, n_max),
        "sliding_window": sliding_window_split(text, n_max, overlap),
        "structure": structure_based_split(text, doc_type),
    }


def select_best_strategy(
    all_chunks: Dict[str, List[Dict]], doc_type: str, config: Dict[str, Any]
) -> List[Dict]:
    """Choose the strategy whose chunk count is closest to a target, with
    type-aware preferences."""
    n_max = int(config.get("n_max", 500))
    target_tokens = n_max * 0.7  # ideal average size

    preference_order = {
        "code": ["structure", "recursive", "sliding_window"],
        "table": ["structure", "recursive", "sliding_window"],
        "mixed": ["structure", "recursive", "sliding_window"],
        "prose": ["recursive", "sliding_window", "structure"],
    }.get(doc_type, ["recursive", "sliding_window", "structure"])

    for strategy in preference_order:
        chunks = all_chunks.get(strategy, [])
        if len(chunks) >= 2:
            avg_tokens = sum(len(c["text"].split()) for c in chunks) / len(chunks)
            if abs(avg_tokens - target_tokens) / target_tokens < 1.0:
                return chunks

    # Fallback: pick strategy with most chunks (most granular)
    best = max(all_chunks.items(), key=lambda kv: len(kv[1]))
    return best[1] if best[1] else []


# ---------------------------------------------------------------------------
# (A) Recursive Character Split
# ---------------------------------------------------------------------------

def recursive_character_split(
    text: str, n_min: int, n_max: int
) -> List[Dict]:
    separators = ["\n\n", "\n", " ", ""]
    raw_chunks = _recursive_split(text, separators, n_min, n_max)

    result: List[Dict] = []
    search_start = 0
    for chunk in raw_chunks:
        if not chunk.strip():
            continue
        idx = text.find(chunk, search_start)
        if idx == -1:
            idx = search_start
        end = idx + len(chunk)
        result.append(
            {"text": chunk, "start": idx, "end": end, "method": "recursive"}
        )
        search_start = end

    return result


def _recursive_split(
    text: str, separators: List[str], n_min: int, n_max: int
) -> List[str]:
    if not text.strip():
        return []

    sep = separators[0] if separators else ""
    remaining_seps = separators[1:] if len(separators) > 1 else []

    if sep == "":
        # Character-level split
        words = text.split()
        chunks: List[str] = []
        buf: List[str] = []
        for word in words:
            buf.append(word)
            if len(buf) >= n_max:
                chunks.append(" ".join(buf))
                buf = []
        if buf:
            chunks.append(" ".join(buf))
        return chunks

    parts = text.split(sep)
    chunks: List[str] = []
    current_parts: List[str] = []

    for part in parts:
        candidate_words = " ".join(current_parts + [part]).split()
        if len(candidate_words) <= n_max:
            current_parts.append(part)
        else:
            if current_parts:
                current_text = sep.join(current_parts).strip()
                current_wc = len(current_text.split())
                if current_wc >= n_min or not remaining_seps:
                    if current_wc > n_max and remaining_seps:
                        chunks.extend(
                            _recursive_split(
                                current_text, remaining_seps, n_min, n_max
                            )
                        )
                    else:
                        if current_text:
                            chunks.append(current_text)
                elif remaining_seps:
                    chunks.extend(
                        _recursive_split(
                            current_text, remaining_seps, n_min, n_max
                        )
                    )
            # Start fresh with the current part
            if len(part.split()) > n_max and remaining_seps:
                chunks.extend(
                    _recursive_split(part, remaining_seps, n_min, n_max)
                )
                current_parts = []
            else:
                current_parts = [part]

    if current_parts:
        final_text = sep.join(current_parts).strip()
        final_wc = len(final_text.split())
        if final_wc > n_max and remaining_seps:
            chunks.extend(
                _recursive_split(final_text, remaining_seps, n_min, n_max)
            )
        elif final_text:
            if chunks and final_wc < n_min:
                # Merge small tail into previous chunk
                combined = chunks[-1] + (sep if sep else " ") + final_text
                if len(combined.split()) <= n_max * 1.3:
                    chunks[-1] = combined
                else:
                    chunks.append(final_text)
            else:
                chunks.append(final_text)

    return [c for c in chunks if c.strip()]


# ---------------------------------------------------------------------------
# (B) Fixed / Sliding Window
# ---------------------------------------------------------------------------

def sliding_window_split(
    text: str, window_size: int, overlap: int
) -> List[Dict]:
    tokens = text.split()
    if not tokens:
        return []

    step = max(1, window_size - overlap)
    chunks: List[Dict] = []
    token_offsets = _build_token_offsets(text, tokens)

    i = 0
    while i < len(tokens):
        end_idx = min(i + window_size, len(tokens))
        window_tokens = tokens[i:end_idx]
        chunk_text = " ".join(window_tokens)

        char_start = token_offsets[i]
        char_end = token_offsets[end_idx - 1] + len(tokens[end_idx - 1])

        chunks.append(
            {
                "text": chunk_text,
                "start": char_start,
                "end": char_end,
                "method": "sliding_window",
            }
        )

        if end_idx == len(tokens):
            break
        i += step

    return chunks


def _build_token_offsets(text: str, tokens: List[str]) -> List[int]:
    offsets: List[int] = []
    search_pos = 0
    for token in tokens:
        idx = text.find(token, search_pos)
        if idx == -1:
            idx = search_pos
        offsets.append(idx)
        search_pos = idx + len(token)
    return offsets


# ---------------------------------------------------------------------------
# (C) Structure-based Split
# ---------------------------------------------------------------------------

def structure_based_split(text: str, doc_type: str) -> List[Dict]:
    if doc_type == "code":
        chunks = _split_code(text)
    else:
        chunks = _split_markup(text)

    # Fall back to paragraph split if no structure was found
    if len(chunks) <= 1:
        chunks = _split_paragraphs(text)

    return chunks


def _split_markup(text: str) -> List[Dict]:
    """Split on Markdown headers, HTML headers, or LaTeX sections."""
    header_re = re.compile(
        r"^(#{1,6}\s+.+|"
        r"<h[1-6][^>]*>.+?</h[1-6]>|"
        r"\\(?:chapter|section|subsection|subsubsection)\{[^}]+\})",
        re.MULTILINE | re.IGNORECASE,
    )

    lines = text.split("\n")
    chunks: List[Dict] = []
    current_lines: List[str] = []
    current_start = 0
    char_pos = 0

    for line in lines:
        is_header = bool(header_re.match(line))
        if is_header and current_lines:
            chunk_text = "\n".join(current_lines).strip()
            if chunk_text:
                chunks.append(
                    {
                        "text": chunk_text,
                        "start": current_start,
                        "end": current_start + len(chunk_text),
                        "method": "structure",
                    }
                )
            current_lines = [line]
            current_start = char_pos
        else:
            current_lines.append(line)
        char_pos += len(line) + 1  # +1 for the \n we split on

    if current_lines:
        chunk_text = "\n".join(current_lines).strip()
        if chunk_text:
            chunks.append(
                {
                    "text": chunk_text,
                    "start": current_start,
                    "end": current_start + len(chunk_text),
                    "method": "structure",
                }
            )

    return chunks


def _split_code(text: str) -> List[Dict]:
    """Split on function / class definitions (Python, JS, Java, C/C++)."""
    boundary_re = re.compile(
        r"^(async\s+)?(?:def |class |public |private |protected |static |function |"
        r"const\s+\w+\s*=\s*(?:async\s*)?\(|let\s+\w+\s*=\s*(?:async\s*)?\()",
        re.MULTILINE,
    )

    lines = text.split("\n")
    chunks: List[Dict] = []
    current_lines: List[str] = []
    current_start = 0
    char_pos = 0

    for line in lines:
        is_boundary = bool(boundary_re.match(line))
        # Only split if we have enough content to justify a new chunk
        if is_boundary and len("\n".join(current_lines).split()) > 5:
            chunk_text = "\n".join(current_lines).strip()
            if chunk_text:
                chunks.append(
                    {
                        "text": chunk_text,
                        "start": current_start,
                        "end": current_start + len(chunk_text),
                        "method": "structure",
                    }
                )
            current_lines = [line]
            current_start = char_pos
        else:
            current_lines.append(line)
        char_pos += len(line) + 1

    if current_lines:
        chunk_text = "\n".join(current_lines).strip()
        if chunk_text:
            chunks.append(
                {
                    "text": chunk_text,
                    "start": current_start,
                    "end": current_start + len(chunk_text),
                    "method": "structure",
                }
            )

    return chunks


def _split_paragraphs(text: str) -> List[Dict]:
    """Split on double newlines (paragraphs)."""
    chunks: List[Dict] = []
    search_pos = 0
    for para in re.split(r"\n{2,}", text):
        para_stripped = para.strip()
        if not para_stripped:
            continue
        idx = text.find(para_stripped, search_pos)
        if idx == -1:
            idx = search_pos
        end = idx + len(para_stripped)
        chunks.append(
            {
                "text": para_stripped,
                "start": idx,
                "end": end,
                "method": "structure",
            }
        )
        search_pos = end
    return chunks
