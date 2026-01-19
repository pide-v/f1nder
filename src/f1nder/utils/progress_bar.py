# f1nder/utils/progress_bar.py
from __future__ import annotations

from typing import Iterable, Iterator, Optional, TypeVar

T = TypeVar("T")


def progress(
    iterable: Iterable[T],
    *,
    total: Optional[int] = None,
    desc: str = "",
    enabled: bool = True,
    leave: bool = False,
) -> Iterator[T]:
    """
    Wraps an iterable with a progress bar (tqdm) when enabled.

    Why this exists:
    - Central place to control progress-bar behaviour across scripts.
    - Keeps your core code clean (no tqdm imports everywhere).
    - If tqdm isn't installed, it gracefully falls back to a plain iterator.

    tqdm usage: wrap any iterable as tqdm(iterable) to show a progress bar.  [oai_citation:1‡tqdm.github.io](https://tqdm.github.io/?utm_source=chatgpt.com)
    """
    if not enabled:
        yield from iterable
        return

    try:
        from tqdm import tqdm  # type: ignore
    except Exception:
        # Fallback: tqdm not installed -> just iterate
        yield from iterable
        return

    yield from tqdm(iterable, total=total, desc=desc, leave=leave, dynamic_ncols=True)


def log(message: str, *, verbose: bool = True) -> None:
    """
    Logs a message to stdout if verbose is True.

    Why this exists:
    - Central place to control logging verbosity across scripts.
    - Keeps your core code clean (no print statements everywhere).
    """
    if verbose:
        print(message)