"""
rag.py — Retrieval-Augmented Generation over your indexed documentation.

Usage:
    python rag.py "how do I configure authentication?"
    python rag.py "what are the deployment steps?" --top 5 --mode hybrid
"""

import os
import fire
from mistralai import Mistral

from embedder import embed_query
from vespa_utils import search
from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file
import logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

RAG_MODEL = "mistral-large-latest"
RERANK_MODEL = "mistral-rerank-latest"
SYSTEM_PROMPT = """You are a helpful assistant that answers questions based strictly on the provided documentation excerpts.
- If the answer is in the excerpts, answer clearly and cite which section it comes from.
- If the answer is not in the excerpts, say so — do not make up information.
- Keep answers concise and precise."""


def build_context(hits: list[dict]) -> str:
    parts = []
    for i, hit in enumerate(hits, 1):
        parts.append(
            f"[{i}] {hit['heading']} (from {hit['source_file']})\n{hit['body']}"
        )
    return "\n\n---\n\n".join(parts)


def rerank_hits(hits: list[dict], query: str, top_n: int) -> list[dict]:
    """
    Rerank retrieved hits using Mistral's reranker and return the top_n results.

    Args:
        hits:  Chunks returned by Vespa search.
        query: The user query.
        top_n: How many chunks to keep after reranking.

    Returns:
        Reranked and trimmed list of hit dicts, with a 'rerank_score' field added.
    """
    client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])
    response = client.rerank.reranking(
        model=RERANK_MODEL,
        query=query,
        documents=[h["body"] for h in hits],
        top_n=top_n,
    )
    return [
        {**hits[r.index], "rerank_score": r.relevance_score}
        for r in response.results
    ]


def rag_from_hits(hits: list[dict], query: str, rerank: bool = False, top_n: int = 3) -> str:
    """
    Generate an answer from pre-fetched Vespa hits.
    Use this when you already have the hits from search() and want to avoid
    re-embedding / re-searching (e.g. during evaluation).

    Args:
        hits:   Chunks returned by search().
        query:  The user query.
        rerank: Whether to rerank hits before generating (default: False).
        top_n:  How many chunks to keep after reranking (default: 3).
    """
    if not hits:
        return "No relevant documentation found for your query."
    if rerank:
        hits = rerank_hits(hits, query, top_n=min(top_n, len(hits)))
    context = build_context(hits)
    client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])
    response = client.chat.complete(
        model=RAG_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Documentation excerpts:\n\n{context}\n\n---\n\nQuestion: {query}"},
        ],
    )
    return response.choices[0].message.content.strip()


def rag(query: str, top_k: int = 10, mode: str = "hybrid", rerank: bool = True, top_n: int = 3) -> str:
    # 1. Embed query
    query_vec = embed_query(query)

    # 2. Retrieve a wider pool of chunks from Vespa
    hits = search(query_vec, top_k=top_k, rank_profile=mode)
    if not hits:
        return "No relevant documentation found for your query."
    logger.info(f"🔍 Retrieved {len(hits)} chunk(s) from Vespa:")
    for i, hit in enumerate(hits, 1):
        logger.info(f"  [{i}] {hit['heading']} (from {hit['source_file']})")

    # 3. Rerank and keep top_n
    if rerank:
        hits = rerank_hits(hits, query, top_n=min(top_n, len(hits)))
        logger.info(f"⚡ Reranked — keeping top {len(hits)}:")
        for i, hit in enumerate(hits, 1):
            logger.info(f"  [{i}] {hit['heading']} (score={hit['rerank_score']:.4f})")

    # 4. Build context and generate answer
    context = build_context(hits)
    client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])
    response = client.chat.complete(
        model=RAG_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Documentation excerpts:\n\n{context}\n\n---\n\nQuestion: {query}"},
        ],
    )

    return response.choices[0].message.content


def main(query: str, top: int = 10, mode: str = "hybrid", rerank: bool = True, top_n: int = 3) -> None:
    """RAG over your documentation

    Args:
        query:  Your question
        top:    Chunks to retrieve from Vespa (default: 10)
        mode:   Retrieval mode - 'semantic' or 'hybrid' (default: hybrid)
        rerank: Rerank retrieved chunks before generation (default: True)
        top_n:  Chunks to keep after reranking (default: 3)
    """
    logger.info(f"🔍 Query: {query!r}\n")
    answer = rag(query, top_k=top, mode=mode, rerank=rerank, top_n=top_n)
    logger.info(f"💬 Answer:\n{answer}\n")


if __name__ == "__main__":
    fire.Fire(main)