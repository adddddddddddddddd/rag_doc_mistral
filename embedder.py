"""
embedder.py — generates embeddings via Mistral API (mistral-embed model).
Handles batching to respect API limits.
"""

import os
import time
from typing import Generator
from mistralai import Mistral
from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file

MISTRAL_MODEL = "mistral-embed-2312"
BATCH_SIZE = 32          # Mistral allows up to 128, keep conservative
RETRY_ATTEMPTS = 3
RETRY_DELAY = 2.0        # seconds between retries


client = Mistral(api_key=os.getenv("MISTRAL_API_KEY"))

def _batches(items: list, size: int) -> Generator[list, None, None]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Embed a list of strings using mistral-embed.
    Returns a list of float vectors (one per input text).
    """
    all_embeddings: list[list[float]] = []

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
                if attempt < RETRY_ATTEMPTS - 1:
                    print(f"  ⚠ Attempt {attempt + 1} failed: {e}. Retrying in {RETRY_DELAY}s...")
                    time.sleep(RETRY_DELAY)
                else:
                    raise RuntimeError(f"Embedding failed after {RETRY_ATTEMPTS} attempts: {e}") from e

    return all_embeddings


def embed_query(query: str) -> list[float]:
    """Embed a single query string."""
    return embed_texts([query])[0]