"""
Common utilities used across the project.
"""

import uuid


def generate_request_id() -> str:
    """Unique request ID for logging and tracing (12 hex chars)."""
    return uuid.uuid4().hex[:12]


def chunked_iterable(iterable, size: int):
    """Yield successive `size`-sized chunks from `iterable`.

    Useful for batching large OCR jobs or sending chunks to external APIs.
    """
    it = iter(iterable)
    while True:
        chunk = []
        try:
            for _ in range(size):
                chunk.append(next(it))
        except StopIteration:
            if chunk:
                yield chunk
            break
        yield chunk
