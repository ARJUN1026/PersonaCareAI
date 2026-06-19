from __future__ import annotations

import random
import time
from typing import Callable, TypeVar

T = TypeVar("T")

def call_with_backoff(func: Callable[..., T], *args, max_retries: int = 5, base_delay: float = 1.0, **kwargs) -> T:
    """Retry Gemini/Chroma calls with exponential backoff for transient rate-limit/network failures."""
    last_error: Exception | None = None
    for attempt in range(max_retries):
        try:
            return func(*args, **kwargs)
        except Exception as exc:  # production version should filter retryable exceptions
            last_error = exc
            if attempt == max_retries - 1:
                break
            sleep_for = base_delay * (2 ** attempt) + random.uniform(0, 0.75)
            time.sleep(sleep_for)
    raise last_error  # type: ignore[misc]
