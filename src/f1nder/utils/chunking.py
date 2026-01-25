from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple


@dataclass(frozen=True)
class Chunk:
    """A chunk of text belonging to a parent document."""
    text: str
    chunk_id: int
    n_tokens: int
    start_token: int
    end_token: int


def _window_spans(n_tokens: int, *, window: int, overlap: int) -> List[Tuple[int, int]]:
    """
    Returns a list of (start, end) spans with overlap, covering the sequence.
    """
    if window <= 0:
        raise ValueError("window must be > 0")
    if overlap < 0:
        raise ValueError("overlap must be >= 0")
    if overlap >= window:
        raise ValueError("overlap must be < window")

    spans: List[Tuple[int, int]] = []
    step = window - overlap
    start = 0
    while start < n_tokens:
        end = min(start + window, n_tokens)
        spans.append((start, end))
        if end >= n_tokens:
            break
        start += step
    return spans


def chunk_text_sliding_window(
    text: str,
    *,
    tokenizer,
    chunk_tokens: int = 384,
    overlap_tokens: int = 64,
    min_last_chunk_tokens: int = 80,
    date_tokens_budget: int = 0,
) -> List[Chunk]:
    """
    Token-based sliding window chunking.

    Assumption (your case):
      - each input 'text' is already a single paragraph / passage.

    Design:
      - tokenize once
      - generate spans with overlap
      - decode each span to text
      - if last chunk is too small, merge it into previous (reduces noisy tiny chunks)

    Returns:
      list[Chunk] with (text, chunk_id, n_tokens, start_token, end_token).
    """
    # --- basic validation ---
    if chunk_tokens <= 0:
        raise ValueError("chunk_tokens must be > 0")
    if overlap_tokens < 0 or overlap_tokens >= chunk_tokens:
        raise ValueError("overlap_tokens must satisfy 0 <= overlap < chunk_tokens")
    if min_last_chunk_tokens < 0:
        raise ValueError("min_last_chunk_tokens must be >= 0")
    
    # --- compute the *effective* content window ---
    effective_chunk_tokens = int(chunk_tokens) - int(date_tokens_budget)
    if effective_chunk_tokens <= 0:
        raise ValueError(
            f"chunk_tokens ({chunk_tokens}) must be > date_tokens_budget ({date_tokens_budget}). "
            "Otherwise there is no room left for content."
        )

    # overlap must be defined relative to the effective window
    if overlap_tokens < 0 or overlap_tokens >= effective_chunk_tokens:
        raise ValueError(
            "overlap_tokens must satisfy 0 <= overlap < (chunk_tokens - date_tokens_budget). "
            f"Got overlap={overlap_tokens}, effective_chunk_tokens={effective_chunk_tokens}."
        )

    t = text.strip()
    if not t:
        return [Chunk(text="", chunk_id=0, n_tokens=0, start_token=0, end_token=0)]

    # tokenize once (why: efficiency; windowing should be on token ids)
    token_ids = tokenizer.encode(t, add_special_tokens=False)
    n = len(token_ids)

    # If it's already short enough, single chunk (of content)
    if n <= effective_chunk_tokens:
        return [Chunk(text=t, chunk_id=0, n_tokens=n, start_token=0, end_token=n)]

    spans = _window_spans(n, window=effective_chunk_tokens, overlap=overlap_tokens)

    chunks: List[Chunk] = []
    for (s, e) in spans:
        ids = token_ids[s:e]
        ch_text = tokenizer.decode(
            ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        ).strip()
        if not ch_text:
            continue
        chunks.append(
            Chunk(
                text=ch_text,
                chunk_id=len(chunks),
                n_tokens=len(ids),
                start_token=s,
                end_token=e,
            )
        )

    # Optional tail handling: replace tiny tail with a final full-size window
    # (why: avoid noisy last chunk with very few tokens)
    if len(chunks) >= 2 and chunks[-1].n_tokens < min_last_chunk_tokens:
        end = n
        start = max(0, end - effective_chunk_tokens)
        ids = token_ids[start:end]
        ch_text = tokenizer.decode(
            ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        ).strip()

        # drop old tiny tail and append the replacement
        chunks = chunks[:-1]
        chunks.append(
            Chunk(
                text=ch_text,
                chunk_id=len(chunks),
                n_tokens=len(ids),
                start_token=start,
                end_token=end,
            )
        )

    return chunks

    # # Merge tiny tail chunk into the previous one (optional but usually helps)
    # if (
    #     len(chunks) >= 2
    #     and chunks[-1].n_tokens < min_last_chunk_tokens
    # ):
    #     prev = chunks[-2]
    #     last = chunks[-1]

    #     merged_text = (prev.text + " " + last.text).strip()
    #     merged_ids = tokenizer.encode(merged_text, add_special_tokens=False)

    #     # keep prev span start, last span end (approx)
    #     merged = Chunk(
    #         text=merged_text,
    #         chunk_id=prev.chunk_id,
    #         n_tokens=len(merged_ids),
    #         start_token=prev.start_token,
    #         end_token=last.end_token,
    #     )
    #     chunks = chunks[:-2] + [merged]

    #     # re-number chunk_id sequentially
    #     chunks = [
    #         Chunk(
    #             text=c.text,
    #             chunk_id=i,
    #             n_tokens=c.n_tokens,
    #             start_token=c.start_token,
    #             end_token=c.end_token,
    #         )
    #         for i, c in enumerate(chunks)
    #     ]

    # Replace tiny tail chunk with a final window ending at the end
    if len(chunks) >= 2 and chunks[-1].n_tokens < min_last_chunk_tokens:
        end = n
        start = max(0, end - chunk_tokens)
        ids = token_ids[start:end]
        ch_text = tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=True).strip()

        # drop the tiny tail
        chunks = chunks[:-1]

        # append the new final chunk (within chunk_tokens)
        chunks.append(
            Chunk(
                text=ch_text,
                chunk_id=len(chunks),
                n_tokens=len(ids),
                start_token=start,
                end_token=end,
            )
        )

    return chunks