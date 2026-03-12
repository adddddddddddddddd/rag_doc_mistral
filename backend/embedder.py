"""
embedder.py — generates embeddings via Mistral API (mistral-embed model).
Handles batching to respect API limits.
"""

import os
import re
import time
from typing import Generator
from mistralai import Mistral
from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file

MISTRAL_MODEL = "mistral-embed-2312"
BATCH_SIZE = 32          # Mistral allows up to 128, keep conservative
RETRY_ATTEMPTS = 3
RETRY_DELAY = 2.0        # seconds between retries
MISTRAL_MAX_TOKENS = 8192
# Use 2 chars/token — conservative enough for code-heavy markdown
_SAFE_CHAR_LIMIT = MISTRAL_MAX_TOKENS * 2


client = Mistral(api_key=os.getenv("MISTRAL_API_KEY"))

def _batches(items: list, size: int) -> Generator[list, None, None]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


def _truncate_oversized(texts: list[str]) -> list[str]:
    """Pre-flight: truncate any text exceeding the safe character limit."""
    result = list(texts)
    truncated_count = 0
    for i, t in enumerate(result):
        if len(t) > _SAFE_CHAR_LIMIT:
            result[i] = t[:_SAFE_CHAR_LIMIT]
            truncated_count += 1
    if truncated_count:
        print(f"  ⚠ Pre-truncated {truncated_count} text(s) to ~{MISTRAL_MAX_TOKENS} token limit.")
    return result


def _truncate_from_error(error_msg: str, batch: list[str]) -> list[str] | None:
    """
    Parse a Mistral 'Input id X has Y tokens' error and truncate the offending
    input(s) using the actual token count reported. Returns None if not parseable.
    """
    matches = re.findall(r"Input id (\d+) has (\d+) tokens", str(error_msg))
    if not matches:
        return None
    patched = list(batch)
    for idx_str, token_str in matches:
        idx = int(idx_str)
        actual_tokens = int(token_str)
        if idx >= len(patched):
            continue
        actual_chars = len(patched[idx])
        # Compute real chars/token ratio, cap safe chars at 8000 tokens worth
        chars_per_token = actual_chars / actual_tokens if actual_tokens else 4
        safe_chars = int(8000 * chars_per_token)
        patched[idx] = patched[idx][:safe_chars]
        print(f"  ⚠ Input {idx} in batch truncated: {actual_tokens} → ~8000 tokens")
    return patched


def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Embed a list of strings using mistral-embed.
    Returns a list of float vectors (one per input text).
    """
    all_embeddings: list[list[float]] = []

    texts = _truncate_oversized(texts)

    for batch_idx, batch in enumerate(_batches(texts, BATCH_SIZE)):
        print(f"  Embedding batch {batch_idx + 1} / {-(-len(texts) // BATCH_SIZE)} "
              f"({len(batch)} texts)...")

        for attempt in range(RETRY_ATTEMPTS):
            try:
                response = client.embeddings.create(
                    model=MISTRAL_MODEL,
                    inputs=batch,
                )
                # Sort by index to preserve order
                sorted_data = sorted(response.data, key=lambda e: e.index)
                all_embeddings.extend([e.embedding for e in sorted_data])
                break
            except Exception as e:
                patched = _truncate_from_error(str(e), batch)
                if patched is not None:
                    batch = patched
                    print(f"  ↳ Retrying after truncating oversized input(s)...")
                    continue
                if attempt < RETRY_ATTEMPTS - 1:
                    print(f"  ⚠ Attempt {attempt + 1} failed: {e}. Retrying in {RETRY_DELAY}s...")
                    time.sleep(RETRY_DELAY)
                else:
                    raise RuntimeError(f"Embedding failed after {RETRY_ATTEMPTS} attempts: {e}") from e

    return all_embeddings


def embed_query(query: str) -> list[float]:
    """Embed a single query string."""
    return embed_texts([query])[0]
