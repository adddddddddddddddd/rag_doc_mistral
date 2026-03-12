"""
rag.py — Retrieval-Augmented Generation over your indexed documentation.

Usage:
    uv run python -m backend.rag "how do I configure authentication?"
    uv run python -m backend.rag "what are the deployment steps?" --top 5 --mode hybrid
"""

import os
import fire
from mistralai import Mistral

from backend.embedder import embed_query
from backend.vespa_utils import search, search_with_section
from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file
import logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

RAG_MODEL = "mistral-large-latest"
SYSTEM_PROMPT = """You are a helpful assistant that answers questions based strictly on the provided documentation excerpts.
- If the answer is in the excerpts, answer clearly and cite which section it comes from.
- If the answer is not in the excerpts, say so — do not make up information.
- Keep answers concise and precise."""

HYDE_SYSTEM_PROMPT = """You are a technical documentation assistant. Given a user question, write a short, plausible documentation excerpt that would answer it.
- Write as if you are the documentation, not as an assistant.
- Be concise (2-5 sentences). Do not hedge or say you are unsure — just write the hypothetical answer directly.
- The goal is to produce text whose embedding will match real documentation, not to give a final answer."""


def generate_hypothetical_answer(query: str) -> str:
    """Generate a hypothetical documentation excerpt for the given query (HyDE step)."""
    client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])
    response = client.chat.complete(
        model=RAG_MODEL,
        messages=[
            {"role": "system", "content": HYDE_SYSTEM_PROMPT},
            {"role": "user", "content": query},
        ],
    )
    return response.choices[0].message.content.strip()


def build_context(hits: list[dict]) -> str:
    parts = []
    for i, hit in enumerate(hits, 1):
        parts.append(
            f"[{i}] {hit['heading']} (from {hit['source_file']})\n{hit['body']}"
        )
    return "\n\n---\n\n".join(parts)


def rag_from_hits(hits: list[dict], query: str) -> str:
    """
    Generate an answer from pre-fetched Vespa hits.
    Use this when you already have the hits from search() and want to avoid
    re-embedding / re-searching (e.g. during evaluation).
    """
    if not hits:
        return "No relevant documentation found for your query."
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


def rag(query: str, top_k: int = 5, mode: str = "hybrid") -> str:
    # 1. Embed query
    query_vec = embed_query(query)

    # 2. Retrieve relevant chunks from Vespa
    hits = search(query_vec, top_k=top_k, rank_profile=mode)
    if not hits:
        return "No relevant documentation found for your query."
    logger.info(f"🔍 Retrieved {len(hits)} relevant chunk(s) from Vespa:")
    for i, hit in enumerate(hits, 1):
        logger.info(f"  [{i}] {hit['heading']} (from {hit['source_file']})")
    # 3. Build context from retrieved chunks
    context = build_context(hits)

    # 4. Generate answer with Mistral
    client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])
    response = client.chat.complete(
        model=RAG_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Documentation excerpts:\n\n{context}\n\n---\n\nQuestion: {query}"},
        ],
    )

    return response.choices[0].message.content


def rag_with_section(
    query: str,
    section_paths: list[str],
    top_k: int = 5,
    mode: str = "hybrid",
) -> str:
    query_vec = embed_query(query)
    hits = search_with_section(query_vec, section_paths=section_paths, top_k=top_k, rank_profile=mode)
    if not hits:
        return "No relevant documentation found for your query."
    logger.info(f"🔍 Retrieved {len(hits)} relevant chunk(s) from Vespa:")
    for i, hit in enumerate(hits, 1):
        logger.info(f"  [{i}] {hit['heading']} (from {hit['source_file']})")
    return rag_from_hits(hits, query)


def rag_with_hyde(query: str, top_k: int = 5, mode: str = "hybrid") -> str:
    """RAG with Hypothetical Document Embedding (HyDE).

    Instead of embedding the raw query, generates a hypothetical answer first
    and embeds that — improving semantic alignment with real documentation chunks.
    """
    # 1. Generate hypothetical answer and embed it
    hypothetical = generate_hypothetical_answer(query)
    logger.info(f"📝 HyDE hypothetical answer:\n{hypothetical}\n")
    query_vec = embed_query(hypothetical)

    # 2. Retrieve using the hypothetical embedding
    hits = search(query_vec, top_k=top_k, rank_profile=mode)
    if not hits:
        return "No relevant documentation found for your query."
    logger.info(f"🔍 Retrieved {len(hits)} relevant chunk(s) from Vespa:")
    for i, hit in enumerate(hits, 1):
        logger.info(f"  [{i}] {hit['heading']} (from {hit['source_file']})")

    # 3. Answer using the *original* query (not the hypothetical)
    return rag_from_hits(hits, query)


def rag_with_section_hyde(
    query: str,
    section_paths: list[str],
    top_k: int = 5,
    mode: str = "hybrid",
) -> str:
    """Section-filtered RAG with HyDE."""
    hypothetical = generate_hypothetical_answer(query)
    logger.info(f"📝 HyDE hypothetical answer:\n{hypothetical}\n")
    query_vec = embed_query(hypothetical)
    hits = search_with_section(query_vec, section_paths=section_paths, top_k=top_k, rank_profile=mode)
    if not hits:
        return "No relevant documentation found for your query."
    logger.info(f"🔍 Retrieved {len(hits)} relevant chunk(s) from Vespa:")
    for i, hit in enumerate(hits, 1):
        logger.info(f"  [{i}] {hit['heading']} (from {hit['source_file']})")
    return rag_from_hits(hits, query)


def main(query: str, top: int = 5, mode: str = "hybrid") -> None:
    """RAG over your documentation
    
    Args:
        query: Your question
        top: Chunks to retrieve (default: 5)
        mode: Retrieval mode - 'semantic' or 'hybrid' (default: hybrid)
    """
    logger.info(f"🔍 Query: {query!r}\n")
    answer = rag(query, top_k=top, mode=mode)
    logger.info(f"💬 Answer:\n{answer}\n")


if __name__ == "__main__":
    fire.Fire(main)