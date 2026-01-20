# f1nder/utils/progress_bar.py
from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator
from pathlib import Path


@contextmanager
def progress_counter(
    *,
    total: int,
    desc: str = "",
    enabled: bool = True,
) -> Iterator[object]:
    """
    A progress bar that you can update manually with pbar.update(n).

    Why:
    - You want progress per document, but encoding happens in batches.
    - This bar tracks *documents processed*, not *batches processed*.

    If tqdm is missing or disabled, yields a dummy object with update().
    """
    if not enabled:
        class Dummy:
            def update(self, n: int) -> None: ...
            def close(self) -> None: ...
        yield Dummy()
        return

    try:
        from tqdm import tqdm  # type: ignore
        pbar = tqdm(total=total, desc=desc, dynamic_ncols=True)
        try:
            yield pbar
        finally:
            pbar.close()
    except Exception:
        class Dummy:
            def update(self, n: int) -> None: ...
            def close(self) -> None: ...
        yield Dummy()


def log(msg: str, verbose: bool = True) -> None:
    if verbose:
        print(msg)
