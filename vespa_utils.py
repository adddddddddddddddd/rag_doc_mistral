"""
vespa_client.py — handles deploying the Vespa app schema and feeding/querying documents.
"""

import json
import time
import urllib.request
import urllib.error
from chunker import Chunk
import fire

VESPA_URL = "http://localhost:8081"
VESPA_CONFIG_URL = "http://localhost:19072"
APP_PATH = "my-vespa-app"
NAMESPACE = "docs"
DOC_TYPE = "doc"
EMBEDDING_DIM = 1024


# ─── Deployment ──────────────────────────────────────────────────────────────

def deploy_app(app_path: str = APP_PATH) -> None:
    """
    Deploy the Vespa application package via the config server API.
    Equivalent to: vespa deploy my-vespa-app/
    """
    import zipfile, io, os

    print("📦 Packaging Vespa application...")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(app_path):
            for fname in files:
                full = os.path.join(root, fname)
                arcname = os.path.relpath(full, app_path)
                zf.write(full, arcname)
    buf.seek(0)

    print("🚀 Deploying to Vespa config server...")
    req = urllib.request.Request(
        f"{VESPA_CONFIG_URL}/application/v2/tenant/default/prepareandactivate",
        data=buf.read(),
        headers={"Content-Type": "application/zip"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req) as resp:
            result = json.loads(resp.read())
            print(f"✅ Deployed: {result.get('message', 'OK')}")
    except urllib.error.HTTPError as e:
        body = e.read().decode()
        raise RuntimeError(f"Deploy failed [{e.code}]: {body}") from e


def wait_for_vespa(timeout: int = 120) -> None:
    """Wait until Vespa is ready to accept requests."""
    print("⏳ Waiting for Vespa to be ready...", end="", flush=True)
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"{VESPA_URL}/state/v1/health") as r:
                data = json.loads(r.read())
                if data.get("status", {}).get("code") == "up":
                    print(" ✅ Ready!")
                    return
        except Exception:
            pass
        print(".", end="", flush=True)
        time.sleep(2)
    raise TimeoutError("Vespa did not become ready in time.")


# ─── Feeding ─────────────────────────────────────────────────────────────────

def _feed_url(doc_id: str) -> str:
    return f"{VESPA_URL}/document/v1/{NAMESPACE}/{DOC_TYPE}/docid/{doc_id}"


def feed_chunk(chunk: Chunk, embedding: list[float]) -> None:
    """Feed a single chunk + its embedding to Vespa."""
    vespa_id = f"id:{NAMESPACE}:{DOC_TYPE}::{chunk.id}"
    doc = {
        "put": vespa_id,
        "fields": {
            "id": chunk.id,
            "source_file": chunk.source_file,
            "heading": chunk.heading,
            "body": chunk.body,
            "embedding": {"values": embedding},
        }
    }
    payload = json.dumps(doc).encode("utf-8")
    req = urllib.request.Request(
        _feed_url(chunk.id),
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req) as resp:
            _ = resp.read()
    except urllib.error.HTTPError as e:
        body = e.read().decode()
        raise RuntimeError(f"Feed failed for {chunk.id} [{e.code}]: {body}") from e


def feed_all(chunks: list[Chunk], embeddings: list[list[float]]) -> None:
    """Feed all chunks with their embeddings."""
    print(f"📤 Feeding {len(chunks)} chunks to Vespa...")
    for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
        feed_chunk(chunk, emb)
        if (i + 1) % 10 == 0 or (i + 1) == len(chunks):
            print(f"  Fed {i + 1}/{len(chunks)} chunks")
    print("✅ All chunks indexed!")


# ─── Existence check ─────────────────────────────────────────────────────────

def file_already_indexed(source_file: str) -> bool:
    """
    Return True if at least one chunk from `source_file` already exists in Vespa.
    Uses a YQL query filtering on the source_file attribute.
    """
    body = {
        "yql": f'select id from {DOC_TYPE} where source_file contains "{source_file}"',
        "hits": 1,   # we only need to know if at least one exists
    }
    payload = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        f"{VESPA_URL}/search/",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read())
        total = data.get("root", {}).get("fields", {}).get("totalCount", 0)
        return total > 0
    except Exception as e:
        raise RuntimeError(f"Existence check failed for {source_file!r}: {e}") from e


# ─── Querying ─────────────────────────────────────────────────────────────────

def search(
    query_embedding: list[float],
    top_k: int = 5,
    rank_profile: str = "semantic",
) -> list[dict]:
    """
    Run an ANN search against Vespa.
    Returns a list of hit dicts with id, heading, body, source_file, relevance.
    """
    body = {
        "yql": f'select id, heading, body, source_file from {DOC_TYPE} where '
               f'{{targetHits: {top_k}}}nearestNeighbor(embedding, q)',
        "input.query(q)": query_embedding,
        "ranking": rank_profile,
        "hits": top_k,
    }
    payload = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        f"{VESPA_URL}/search/",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req) as resp:
        data = json.loads(resp.read())

    hits = data.get("root", {}).get("children", [])
    return [
        {
            "id": h["fields"].get("id"),
            "heading": h["fields"].get("heading"),
            "body": h["fields"].get("body"),
            "source_file": h["fields"].get("source_file"),
            "relevance": h.get("relevance"),
        }
        for h in hits
    ]
    
if __name__ == "__main__":
    fire.Fire({
        "deploy_app": deploy_app,
        "wait_for_vespa": wait_for_vespa,
        "feed_all": feed_all,
        "file_already_indexed": file_already_indexed,
        "search": search,
    })