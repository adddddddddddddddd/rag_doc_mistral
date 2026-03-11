"""
index.py — orchestrates chunking, embedding, and indexing of markdown files into Vespa.

Usage:
    python index.py path/to/file1.md path/to/file2.md ...

    # Or pipe from your existing path-retrieval script:
    python get_paths.py | xargs python index.py
"""

import hashlib
import sys
from pathlib import Path

import fire
from chunker import chunk_file, Chunk
from embedder import embed_texts
from vespa_utils import wait_for_vespa, deploy_app, feed_all, file_already_indexed
from extract_all_md_files import list_markdown_paths
import logging

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

DOCS_GITHUB_PATH = Path("platform-docs-public")

def main(include_folders=None, exclude_folders=None) -> None:
    md_paths = list_markdown_paths(include_folders=include_folders, exclude_folders=exclude_folders, base_path=DOCS_GITHUB_PATH)
    if not md_paths:
        logging.info("Usage: python index.py file1.md file2.md ...")
        logging.info("   or: python get_paths.py | xargs python index.py")
        sys.exit(1)

    # ── 1. Validate paths ────────────────────────────────────────────────────
    paths = []
    for p in md_paths:
        path = Path(p)
        if not path.exists():
            logging.info(f"⚠ Skipping missing file: {p}")
            continue
        if path.suffix.lower() != ".md":
            logging.info(f"⚠ Skipping non-markdown file: {p}")
            continue
        paths.append(path)

    if not paths:
        logging.info("❌ No valid markdown files found.")
        sys.exit(1)

    logging.info(f"📄 Found {len(paths)} markdown file(s)")

    # ── 2. Chunk all files ───────────────────────────────────────────────────
    all_chunks: list[Chunk] = []
    skipped = 0
    for path in paths:
        if file_already_indexed(str(path)):
            logging.info(f"  ⏭  Skipping {path.name} (already indexed)")
            skipped += 1
            continue
        file_chunks = chunk_file(path)
        logging.info(f"  {path.name}: {len(file_chunks)} chunk(s)")
        all_chunks.extend(file_chunks)

    if skipped:
        logging.info(f"\n⏭  Skipped {skipped} already-indexed file(s).")

    # ── 3. Deduplicate chunks by body hash; merge headings when body is shared ─
    body_to_chunk: dict[str, Chunk] = {}
    for chunk in all_chunks:
        body_hash = hashlib.sha256(chunk.body.encode()).hexdigest()
        if body_hash not in body_to_chunk:
            body_to_chunk[body_hash] = chunk
        else:
            existing = body_to_chunk[body_hash]
            if chunk.heading not in existing.heading:
                existing.heading = f"{existing.heading} / {chunk.heading}"
    unique_chunks = list(body_to_chunk.values())
    duplicates = len(all_chunks) - len(unique_chunks)
    if duplicates:
        logging.info(f"🔁 Merged {duplicates} duplicate chunk(s) into shared-body groups.")
    all_chunks = unique_chunks

    if not all_chunks:
        logging.info("✅ Nothing new to index.")
        return

    logging.info(f"\n✂️  Total chunks: {len(all_chunks)}")

    # ── 4. Embed all chunks ──────────────────────────────────────────────────
    logging.info("\n🔮 Generating embeddings via mistral-embed...")
    texts = [f"{c.heading}\n\n{c.body}" for c in all_chunks]
    embeddings = embed_texts(texts)
    logging.info(f"✅ Got {len(embeddings)} embeddings (dim={len(embeddings[0])})")

    # ── 5. Deploy Vespa schema (idempotent) ──────────────────────────────────
    logging.info("\n🏗️  Deploying Vespa application schema...")
    wait_for_vespa()
    deploy_app()

    # ── 6. Feed into Vespa ───────────────────────────────────────────────────
    feed_all(all_chunks, embeddings)

    logging.info(f"\n🎉 Done! {len(all_chunks)} chunks indexed from {len(paths)} file(s).")


if __name__ == "__main__":
    fire.Fire(main)