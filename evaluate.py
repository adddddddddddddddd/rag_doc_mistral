"""
evaluate.py — RAG evaluation pipeline.

Phase 1a — generate:          sample chunks from Vespa, generate one question per chunk,
                               save to eval_dataset.json.

Phase 1b — add_ground_truth:  backfill a ground_truth_answer per entry (separate loop,
                               resume-safe).

Phase 2a — build_eval_json:        for each entry embed query → retrieve → generate answer,
                                    save rich intermediate JSON for downstream scorers.

Phase 2b — evaluate:               quick Recall@k (gold chunk in retrieved set).

Phase 2c — score_faithfulness:     LLM judge — break RAG answer into claims, check each
                                    claim is supported by the retrieved contexts.

Phase 2d — score_answer_relevancy: LLM judge — float score for how well the answer
                                    addresses the question.

Phase 2e — score_completeness:     LLM judge — extract key facts from ground_truth_body,
                                    check how many are covered in the RAG answer.

run_all:                            build_eval_json → faithfulness → answer_relevancy
                                    → completeness → print summary.

Phase 3  — ragas_eval:             full RAGAS scoring (Faithfulness, AnswerRelevancy,
                                    ContextPrecision, ContextRecall).

Usage:
    uv run python evaluate.py generate --samples 200 --out eval_dataset.json
    uv run python evaluate.py add_ground_truth --dataset eval_dataset.json --model mistral-large-latest --out eval_dataset_with_gt.json
    uv run python evaluate.py build_eval_json --dataset eval_dataset.json --top 5 --mode hybrid
    uv run python evaluate.py evaluate --dataset eval_dataset.json --top 5 --mode hybrid
    uv run python evaluate.py score_recall_at_k --intermediate eval_intermediate_top5_hybrid.json
    uv run python evaluate.py score_precision_at_k --intermediate eval_intermediate_top5_hybrid.json
    uv run python evaluate.py score_faithfulness --intermediate eval_intermediate_top5_hybrid.json
    uv run python evaluate.py score_answer_relevancy --intermediate eval_intermediate_top5_hybrid.json
    uv run python evaluate.py score_completeness --intermediate eval_intermediate_top5_hybrid.json
    uv run python evaluate.py run_all --dataset eval_dataset.json --top 5 --mode hybrid --samples 50
    uv run python evaluate.py ragas_eval --dataset eval_dataset.json --top 5 --mode hybrid --samples 50
"""

import json
import os
import re
import time
import urllib.request
import urllib.error
import fire
from mistralai import Mistral
from dotenv import load_dotenv

from embedder import embed_query
from vespa_utils import VESPA_URL, NAMESPACE, DOC_TYPE, search
from rag import SYSTEM_PROMPT, rag_from_hits

load_dotenv()

EVAL_MODEL = "mistral-large-latest"
FAITHFULNESS_EVALUATION_MODEL = "magistral-small-2509"
DEFAULT_DATASET = "eval_dataset.json"

QUESTION_PROMPT = """You are creating an evaluation dataset for a RAG system.
Given the documentation chunk below, write exactly ONE question that:
- Can be answered solely from this chunk (no outside knowledge needed)
- Is specific enough that only this chunk would be relevant
- Reads like a natural user question

Respond with only the question, no preamble, no quotes."""

GROUND_TRUTH_PROMPT = """You are creating ground truth answers for a RAG evaluation dataset.
Given the documentation chunk and the question below, write a concise and accurate answer
based solely on the chunk — do not use outside knowledge.

Respond with only the answer, no preamble."""


# ─── Vespa document visit ─────────────────────────────────────────────────────


def _visit_chunks(max_docs: int = 500) -> list[dict]:
    """
    Fetch up to `max_docs` documents from Vespa using the document visit API.
    Returns list of dicts with id, heading, body, source_file.
    """
    chunks = []
    continuation = None

    while len(chunks) < max_docs:
        want = min(100, max_docs - len(chunks))
        url = (
            f"{VESPA_URL}/document/v1/{NAMESPACE}/{DOC_TYPE}/docid"
            f"?wantedDocumentCount={want}"
        )
        if continuation:
            url += f"&continuation={continuation}"

        try:
            with urllib.request.urlopen(url) as resp:
                data = json.loads(resp.read())
        except urllib.error.HTTPError as e:
            raise RuntimeError(
                f"Vespa visit failed [{e.code}]: {e.read().decode()}"
            ) from e

        for doc in data.get("documents", []):
            f = doc.get("fields", {})
            chunks.append(
                {
                    "id": f.get("id"),
                    "heading": f.get("heading", ""),
                    "body": f.get("body", ""),
                    "source_file": f.get("source_file", ""),
                }
            )

        continuation = data.get("continuation")
        if not continuation:
            break

    return chunks


# ─── Phase 1: generate ───────────────────────────────────────────────────────


def generate(
    samples: int = 200, out: str = DEFAULT_DATASET, delay: float = 0.3
) -> None:
    """
    Sample chunks from Vespa, generate one question per chunk, save to JSON.

    Args:
        samples: Number of chunks to sample (default: 200)
        out:     Output JSON file path (default: eval_dataset.json)
        delay:   Seconds between Mistral API calls to avoid rate limiting (default: 0.3)
    """
    print(f"📥 Fetching up to {samples} chunks from Vespa...")
    chunks = _visit_chunks(max_docs=samples)
    print(f"  Got {len(chunks)} chunks")

    # Skip chunks with very short bodies
    chunks = [c for c in chunks if len(c["body"].split()) >= 20]
    print(f"  {len(chunks)} chunks after filtering short ones")

    client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])
    dataset: list[dict] = []

    # Load existing dataset to allow resuming
    if os.path.exists(out):
        with open(out) as f:
            dataset = json.load(f)
        existing_ids = {d["chunk_id"] for d in dataset}
        chunks = [c for c in chunks if c["id"] not in existing_ids]
        print(f"  Resuming — {len(existing_ids)} already done, {len(chunks)} remaining")

    for i, chunk in enumerate(chunks):
        prompt = f"{QUESTION_PROMPT}\n\n---\nHeading: {chunk['heading']}\n\n{chunk['body']}\n---"
        try:
            resp = client.chat.complete(
                model=EVAL_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
            )
            question = resp.choices[0].message.content.strip()
        except Exception as e:
            print(f"  ⚠ Skipping chunk {chunk['id']}: {e}")
            continue

        dataset.append(
            {
                "chunk_id": chunk["id"],
                "source_file": chunk["source_file"],
                "heading": chunk["heading"],
                "body": chunk["body"],
                "question": question,
            }
        )

        if (i + 1) % 10 == 0 or (i + 1) == len(chunks):
            with open(out, "w") as f:
                json.dump(dataset, f, indent=2, ensure_ascii=False)
            print(f"  [{i + 1}/{len(chunks)}] saved — last: {question[:80]!r}")

        time.sleep(delay)

    print(f"\n✅ Dataset saved to {out!r} ({len(dataset)} entries)")


# ─── Phase 1b: add_ground_truth ──────────────────────────────────────────────


def add_ground_truth(
    dataset: str = DEFAULT_DATASET,
    model: str = EVAL_MODEL,
    out: str = None,
    delay: float = 0.3,
) -> None:
    """
    Add a ground_truth_answer to every entry in the dataset that doesn't have one yet.
    Safe to re-run — skips entries that already have ground_truth_answer.

    Args:
        dataset: Path to the input JSON dataset (default: eval_dataset.json)
        model:   Mistral model to use for generating answers (default: mistral-large-latest)
        out:     Output file path (default: same as dataset, i.e. in-place)
        delay:   Seconds between Mistral API calls to avoid rate limiting (default: 0.3)
    """
    if not os.path.exists(dataset):
        raise FileNotFoundError(
            f"Dataset not found: {dataset!r}. Run 'generate' first."
        )

    out = out or dataset

    with open(dataset) as f:
        entries = json.load(f)

    to_process = [e for e in entries if not e.get("ground_truth_answer")]
    print(
        f"📝 Adding ground truth answers | {len(to_process)} entries to process "
        f"({len(entries) - len(to_process)} already done)\n"
    )

    if not to_process:
        print("✅ All entries already have ground_truth_answer.")
        return

    client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])
    index = {e["chunk_id"]: e for e in entries}
    print(f"  Model : {model}")
    print(f"  Out   : {out}\n")

    for i, entry in enumerate(to_process):
        prompt = (
            f"{GROUND_TRUTH_PROMPT}\n\n"
            f"---\nHeading: {entry['heading']}\n\n{entry['body']}\n---\n\n"
            f"Question: {entry['question']}"
        )
        try:
            resp = client.chat.complete(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
            )
            index[entry["chunk_id"]]["ground_truth_answer"] = resp.choices[
                0
            ].message.content.strip()
        except Exception as e:
            print(f"  ⚠ Skipping {entry['chunk_id']}: {e}")
            continue

        if (i + 1) % 10 == 0 or (i + 1) == len(to_process):
            with open(out, "w") as f:
                json.dump(list(index.values()), f, indent=2, ensure_ascii=False)
            print(f"  [{i + 1}/{len(to_process)}] saved")

        time.sleep(delay)

    print(f"\n✅ Ground truth answers added and saved to {out!r}")


# ─── Phase 2: evaluate ───────────────────────────────────────────────────────


def evaluate(
    dataset: str = DEFAULT_DATASET, top: int = 5, mode: str = "hybrid"
) -> None:
    """
    Compute Precision@k for each question in the dataset.

    Args:
        dataset: Path to the JSON dataset (default: eval_dataset.json)
        top:     k — number of chunks retrieved per query (default: 5)
        mode:    Retrieval mode: 'semantic' or 'hybrid' (default: hybrid)
    """
    if not os.path.exists(dataset):
        raise FileNotFoundError(
            f"Dataset not found: {dataset!r}. Run 'generate' first."
        )

    with open(dataset) as f:
        entries = json.load(f)

    print(f"📊 Evaluating {len(entries)} questions | top={top} | mode={mode}\n")

    hits_at_k = 0
    results = []

    for i, entry in enumerate(entries):
        query_vec = embed_query(entry["question"])
        retrieved = search(query_vec, top_k=top, rank_profile=mode)
        retrieved_ids = [h["id"] for h in retrieved]

        hit = entry["chunk_id"] in retrieved_ids
        hits_at_k += int(hit)

        results.append({**entry, "retrieved_ids": retrieved_ids, "hit": hit})

        if (i + 1) % 20 == 0 or (i + 1) == len(entries):
            current_p = hits_at_k / (i + 1)
            print(f"  [{i + 1}/{len(entries)}] Precision@{top} so far: {current_p:.3f}")

    precision = hits_at_k / len(entries)
    print(
        f"\n✅ Precision@{top} ({mode}): {precision:.3f}  ({hits_at_k}/{len(entries)} hits)"
    )

    # Save detailed results
    results_path = dataset.replace(".json", f"_results_top{top}_{mode}.json")
    with open(results_path, "w") as f:
        json.dump(
            {"precision_at_k": precision, "k": top, "mode": mode, "results": results},
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"📄 Detailed results saved to {results_path!r}")


# ─── Phase 2b: build_eval_json ───────────────────────────────────────────────


def build_eval_json(
    dataset: str = DEFAULT_DATASET,
    top: int = 5,
    mode: str = "hybrid",
    samples: int = None,
    out: str = None,
    delay: float = 0.3,
) -> None:
    """
    For each entry: embed query → retrieve top-k chunks from Vespa → generate RAG answer.
    Saves a rich intermediate JSON used by downstream metric scorers.

    Each record contains:
        question, ground_truth_answer, ground_truth_chunk_id, ground_truth_source_file,
        ground_truth_heading, ground_truth_body,
        retrieved_hits (id, heading, body, source_file, relevance),
        rag_answer, recall_at_k (bool), top_k, mode

    Args:
        dataset: Path to eval dataset JSON (default: eval_dataset.json)
        top:     Number of chunks to retrieve per query (default: 5)
        mode:    Retrieval mode: 'semantic' or 'hybrid' (default: hybrid)
        samples: Limit to first N entries (default: all)
        out:     Output file path (default: eval_intermediate_top{top}_{mode}.json)
        delay:   Seconds between Mistral API calls (default: 0.3)
    """
    if not os.path.exists(dataset):
        raise FileNotFoundError(
            f"Dataset not found: {dataset!r}. Run 'generate' first."
        )

    if out is None:
        out = f"eval_intermediate_top{top}_{mode}.json"

    with open(dataset) as f:
        entries = json.load(f)

    if samples:
        entries = entries[:samples]

    # Resume support: skip entries already present in the output file
    done_ids: set[str] = set()
    records: list[dict] = []
    if os.path.exists(out):
        with open(out) as f:
            records = json.load(f)
        done_ids = {r["ground_truth_chunk_id"] for r in records}
        entries = [e for e in entries if e["chunk_id"] not in done_ids]
        print(f"  Resuming — {len(done_ids)} already done, {len(entries)} remaining")

    print(f"🔍 Building eval JSON | {len(entries)} entries | top={top} | mode={mode}\n")

    for i, entry in enumerate(entries):
        question = entry["question"]

        # 1. Embed + retrieve
        query_vec = embed_query(question)
        hits = search(query_vec, top_k=top, rank_profile=mode)

        # 2. Generate RAG answer from the retrieved hits
        rag_answer = rag_from_hits(hits, question)

        # 3. Recall@k — did the gold chunk appear in the retrieved set?
        retrieved_ids = [h["id"] for h in hits]
        recall = entry["chunk_id"] in retrieved_ids

        records.append(
            {
                # Query
                "question": question,
                # Ground truth (from generate phase)
                "ground_truth_answer": entry.get("ground_truth_answer"),
                "ground_truth_chunk_id": entry["chunk_id"],
                "ground_truth_source_file": entry["source_file"],
                "ground_truth_heading": entry["heading"],
                "ground_truth_body": entry["body"],
                # Retrieval
                "retrieved_hits": [
                    {
                        "id": h["id"],
                        "heading": h["heading"],
                        "body": h["body"],
                        "source_file": h["source_file"],
                        "relevance": h.get("relevance"),
                    }
                    for h in hits
                ],
                "retrieved_ids": retrieved_ids,
                # Generation
                "rag_answer": rag_answer,
                # Retrieval metric
                "recall_at_k": recall,
                # Run metadata
                "top_k": top,
                "mode": mode,
            }
        )

        if (i + 1) % 10 == 0 or (i + 1) == len(entries):
            with open(out, "w") as f:
                json.dump(records, f, indent=2, ensure_ascii=False)
            print(f"  [{i + 1}/{len(entries)}] recall={recall} | {question[:70]!r}")

        time.sleep(delay)

    recall_at_k = (
        sum(r["recall_at_k"] for r in records) / len(records) if records else 0.0
    )
    print(f"\n✅ Saved {len(records)} records to {out!r}")
    print(f"   Recall@{top} ({mode}): {recall_at_k:.3f}")


# ─── Phase 2c: score_recall_at_k ─────────────────────────────────────────────


def score_recall_at_k(
    intermediate: str = "eval_intermediate_top5_hybrid.json",
) -> float:
    """
    Compute Recall@k from the intermediate eval JSON.

    Recall@k = fraction of queries where the gold chunk appears in the top-k results.
    The boolean per-record value is already stored by build_eval_json; this function
    aggregates it and prints the result.

    Args:
        intermediate: Path to the intermediate eval JSON (output of build_eval_json)

    Returns:
        Mean Recall@k across all records.
    """
    if not os.path.exists(intermediate):
        raise FileNotFoundError(
            f"Intermediate file not found: {intermediate!r}. Run 'build_eval_json' first."
        )

    with open(intermediate) as f:
        records = json.load(f)

    k = records[0]["top_k"] if records else "?"
    mode = records[0]["mode"] if records else "?"
    hits = sum(r["recall_at_k"] for r in records)
    recall = hits / len(records)

    print(f"📊 Recall@{k} ({mode}): {recall:.3f}  ({hits}/{len(records)} hits)")
    return recall


# ─── Phase 2d: score_precision_at_k ──────────────────────────────────────────


def score_precision_at_k(
    intermediate: str = "eval_intermediate_top5_hybrid.json",
) -> float:
    """
    Compute Precision@k from the intermediate eval JSON.

    With one relevant document per query:
        Precision@k = 1/k  if the gold chunk is in the top-k results, else 0.
    The mean across queries gives the average precision.

    Args:
        intermediate: Path to the intermediate eval JSON (output of build_eval_json)

    Returns:
        Mean Precision@k across all records.
    """
    if not os.path.exists(intermediate):
        raise FileNotFoundError(
            f"Intermediate file not found: {intermediate!r}. Run 'build_eval_json' first."
        )

    with open(intermediate) as f:
        records = json.load(f)

    k = records[0]["top_k"] if records else 1
    mode = records[0]["mode"] if records else "?"
    hits = sum(r["recall_at_k"] for r in records)
    precision = (hits / k) / len(records)

    print(
        f"📊 Precision@{k} ({mode}): {precision:.3f}  ({hits}/{len(records)} hits, 1/{k} per hit)"
    )
    return precision


# ─── LLM-judge helpers ───────────────────────────────────────────────────────

_CLAIMS_PROMPT = """\
Break the following answer into atomic factual claims.
Each claim must be a single, self-contained, verifiable statement.

Answer:
{answer}

Respond with a JSON array of strings only. No explanation, no markdown fences."""

_SUPPORT_PROMPT = """\
You are a faithfulness judge analysing the response of another LLM. Your task is to determine whether the following claim is fully supported by the retrieved documentation contexts below or if it is made up.

Contexts:
{contexts}

Claim: {claim}

Respond with only "yes" or "no"."""

_RELEVANCY_PROMPT = """\
You are an answer quality judge. Does the following answer directly and relevantly address the question?

Question: {question}
Answer: {answer}

Rate from 0.0 (completely irrelevant or off-topic) to 1.0 (fully relevant and on-topic).
Respond with only a decimal number."""

_FACTS_PROMPT = """\
Extract the key facts and pieces of information from the following documentation chunk.
Each fact must be a single, concrete, self-contained statement.

Chunk:
{body}

Respond with a JSON array of strings only. No explanation, no markdown fences."""

_COVERAGE_PROMPT = """\
You are a completeness judge. Is the following fact from the source documentation addressed or mentioned in the answer?

Answer:
{answer}

Fact: {fact}

Respond with only "yes" or "no"."""


def _llm_json_list(client, prompt: str) -> list[str]:
    """Call the LLM and parse a JSON array from the response."""
    resp = client.chat.complete(
        model=EVAL_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
    )
    text = resp.choices[0].message.content.strip()
    # Strip markdown code fences if present
    if text.startswith("```"):
        text = "\n".join(text.split("\n")[1:]).rstrip("`").strip()
    try:
        result = json.loads(text)
        return [str(x) for x in result] if isinstance(result, list) else []
    except json.JSONDecodeError:
        return []


def _llm_yn(client, prompt: str, model: str = EVAL_MODEL, reasoning_model : bool = False) -> bool:
    """Call the LLM with a yes/no question, return True for 'yes'."""
    if reasoning_model:
        resp = client.chat.complete(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": "# HOW YOU SHOULD THINK AND ANSWER\n\nFirst draft your thinking process (inner monologue) until you arrive at a response. Format your response using Markdown, and use LaTeX for any mathematical equations. Write both your thoughts and the response in the same language as the input.\n\nYour thinking process must follow the template below:",
                        },
                        {
                            "type": "thinking",
                            "thinking": [
                                {
                                    "type": "text",
                                    "text": "Your thoughts or/and draft, like working through an exercise on scratch paper. Be as casual and as long as you want until you are confident to generate the response to the user.",
                                }
                            ],
                        },
                        {
                            "type": "text",
                            "text": "Here, provide a self-contained response.",
                        },
                    ],
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
        )
        text_answer = ""
        for content in resp.choices[0].message.content:
            if content.type == "text":
                text_answer += content.text
        return text_answer.strip().lower().startswith("yes")
    else:
        resp = client.chat.complete(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
    return resp.choices[0].message.content.strip().lower().startswith("yes")


def _llm_float(client, prompt: str) -> float:
    """Call the LLM and parse a 0.0–1.0 float from the response."""
    resp = client.chat.complete(
        model=EVAL_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
    )
    text = resp.choices[0].message.content.strip()
    try:
        return max(0.0, min(1.0, float(text)))
    except ValueError:
        m = re.search(r"[0-9]+(?:\.[0-9]+)?", text)
        return max(0.0, min(1.0, float(m.group()))) if m else 0.5


# ─── Phase 2c: score_faithfulness ────────────────────────────────────────────


def score_faithfulness(
    intermediate: str = "eval_intermediate_top5_hybrid.json",
    delay: float = 0.3,
) -> None:
    """
    Score faithfulness for each record in the intermediate eval JSON.

    Breaks the RAG answer into atomic claims, then checks each claim against
    the retrieved contexts using an LLM judge.
    Score = supported_claims / total_claims.
    Updates the intermediate file in-place. Resume-safe.

    Args:
        intermediate: Path to the intermediate eval JSON (output of build_eval_json)
        delay:        Seconds between API calls (default: 0.3)
    """
    if not os.path.exists(intermediate):
        raise FileNotFoundError(
            f"Intermediate file not found: {intermediate!r}. Run 'build_eval_json' first."
        )

    with open(intermediate) as f:
        records = json.load(f)

    to_process = [r for r in records if "faithfulness" not in r]
    done = len(records) - len(to_process)
    print(
        f"🔍 Scoring faithfulness | {len(to_process)} to process ({done} already done)\n"
    )

    if not to_process:
        scored = [
            r["faithfulness"] for r in records if r.get("faithfulness") is not None
        ]
        print(f"✅ All done. Mean faithfulness: {sum(scored) / len(scored):.3f}")
        return

    client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])
    # Build index so updates propagate into `records`
    index = {r["ground_truth_chunk_id"]: r for r in records}

    for i, record in enumerate(to_process):
        cid = record["ground_truth_chunk_id"]
        contexts = "\n\n---\n\n".join(
            f"[{j + 1}] {h['heading']}\n{h['body']}"
            for j, h in enumerate(record["retrieved_hits"])
        )

        claims = _llm_json_list(
            client, _CLAIMS_PROMPT.format(answer=record["rag_answer"])
        )
        time.sleep(delay)

        if not claims:
            index[cid]["faithfulness"] = None
            index[cid]["faithfulness_claims"] = []
            continue

        supported = 0
        claim_details = []
        for claim in claims:
            yn = _llm_yn(
                client=client,
                prompt=_SUPPORT_PROMPT.format(contexts=contexts, claim=claim),
                model=FAITHFULNESS_EVALUATION_MODEL,
                reasoning_model=True,
            )
            claim_details.append({"claim": claim, "supported": yn})
            supported += int(yn)
            time.sleep(delay)

        score = supported / len(claims)
        index[cid]["faithfulness"] = score
        index[cid]["faithfulness_claims"] = claim_details

        if (i + 1) % 10 == 0 or (i + 1) == len(to_process):
            with open(intermediate, "w") as f:
                json.dump(records, f, indent=2, ensure_ascii=False)
            print(
                f"  [{i + 1}/{len(to_process)}] faithfulness={score:.2f} ({supported}/{len(claims)} claims supported)"
            )

    with open(intermediate, "w") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    scored = [r["faithfulness"] for r in records if r.get("faithfulness") is not None]
    print(
        f"\n✅ Faithfulness scored. Mean: {sum(scored) / len(scored):.3f} ({len(scored)} records)"
    )


# ─── Phase 2d: score_answer_relevancy ────────────────────────────────────────


def score_answer_relevancy(
    intermediate: str = "eval_intermediate_top5_hybrid.json",
    delay: float = 0.3,
) -> None:
    """
    Score answer relevancy for each record in the intermediate eval JSON.

    Asks an LLM judge whether the RAG answer directly and relevantly addresses
    the question. Score is a float in [0.0, 1.0].
    Updates the intermediate file in-place. Resume-safe.

    Args:
        intermediate: Path to the intermediate eval JSON (output of build_eval_json)
        delay:        Seconds between API calls (default: 0.3)
    """
    if not os.path.exists(intermediate):
        raise FileNotFoundError(
            f"Intermediate file not found: {intermediate!r}. Run 'build_eval_json' first."
        )

    with open(intermediate) as f:
        records = json.load(f)

    to_process = [r for r in records if "answer_relevancy" not in r]
    done = len(records) - len(to_process)
    print(
        f"🔍 Scoring answer relevancy | {len(to_process)} to process ({done} already done)\n"
    )

    if not to_process:
        scored = [
            r["answer_relevancy"]
            for r in records
            if r.get("answer_relevancy") is not None
        ]
        print(f"✅ All done. Mean answer relevancy: {sum(scored) / len(scored):.3f}")
        return

    client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])
    index = {r["ground_truth_chunk_id"]: r for r in records}

    for i, record in enumerate(to_process):
        cid = record["ground_truth_chunk_id"]
        score = _llm_float(
            client,
            _RELEVANCY_PROMPT.format(
                question=record["question"], answer=record["rag_answer"]
            ),
        )
        index[cid]["answer_relevancy"] = score
        time.sleep(delay)

        if (i + 1) % 10 == 0 or (i + 1) == len(to_process):
            with open(intermediate, "w") as f:
                json.dump(records, f, indent=2, ensure_ascii=False)
            print(f"  [{i + 1}/{len(to_process)}] answer_relevancy={score:.2f}")

    with open(intermediate, "w") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    scored = [
        r["answer_relevancy"] for r in records if r.get("answer_relevancy") is not None
    ]
    print(
        f"\n✅ Answer relevancy scored. Mean: {sum(scored) / len(scored):.3f} ({len(scored)} records)"
    )


# ─── Phase 2e: score_completeness ────────────────────────────────────────────


def score_completeness(
    intermediate: str = "eval_intermediate_top5_hybrid.json",
    delay: float = 0.3,
) -> None:
    """
    Score completeness for each record in the intermediate eval JSON.

    Extracts key facts from ground_truth_body (the raw source chunk), then
    checks how many of those facts are addressed in the RAG answer.
    Score = covered_facts / total_facts.
    Updates the intermediate file in-place. Resume-safe.

    Args:
        intermediate: Path to the intermediate eval JSON (output of build_eval_json)
        delay:        Seconds between API calls (default: 0.3)
    """
    if not os.path.exists(intermediate):
        raise FileNotFoundError(
            f"Intermediate file not found: {intermediate!r}. Run 'build_eval_json' first."
        )

    with open(intermediate) as f:
        records = json.load(f)

    to_process = [r for r in records if "completeness" not in r]
    done = len(records) - len(to_process)
    print(
        f"🔍 Scoring completeness | {len(to_process)} to process ({done} already done)\n"
    )

    if not to_process:
        scored = [
            r["completeness"] for r in records if r.get("completeness") is not None
        ]
        print(f"✅ All done. Mean completeness: {sum(scored) / len(scored):.3f}")
        return

    client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])
    index = {r["ground_truth_chunk_id"]: r for r in records}

    for i, record in enumerate(to_process):
        cid = record["ground_truth_chunk_id"]

        facts = _llm_json_list(
            client, _FACTS_PROMPT.format(body=record["ground_truth_body"])
        )
        time.sleep(delay)

        if not facts:
            index[cid]["completeness"] = None
            index[cid]["completeness_facts"] = []
            continue

        covered = 0
        fact_details = []
        for fact in facts:
            yn = _llm_yn(
                client, _COVERAGE_PROMPT.format(answer=record["rag_answer"], fact=fact)
            )
            fact_details.append({"fact": fact, "covered": yn})
            covered += int(yn)
            time.sleep(delay)

        score = covered / len(facts)
        index[cid]["completeness"] = score
        index[cid]["completeness_facts"] = fact_details

        if (i + 1) % 10 == 0 or (i + 1) == len(to_process):
            with open(intermediate, "w") as f:
                json.dump(records, f, indent=2, ensure_ascii=False)
            print(
                f"  [{i + 1}/{len(to_process)}] completeness={score:.2f} ({covered}/{len(facts)} facts covered)"
            )

    with open(intermediate, "w") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    scored = [r["completeness"] for r in records if r.get("completeness") is not None]
    print(
        f"\n✅ Completeness scored. Mean: {sum(scored) / len(scored):.3f} ({len(scored)} records)"
    )


# ─── run_all ─────────────────────────────────────────────────────────────────


def run_all(
    dataset: str = DEFAULT_DATASET,
    top: int = 5,
    mode: str = "hybrid",
    samples: int = None,
    delay: float = 0.3,
) -> None:
    """
    Run the full evaluation pipeline in sequence and print a summary.

    Steps:
        1. build_eval_json        — retrieve top-k chunks + generate RAG answers
        2. score_recall_at_k      — fraction of queries where gold chunk is retrieved
        3. score_precision_at_k   — mean precision (hit / k) across queries
        4. score_faithfulness     — claim extraction + LLM support check
        5. score_answer_relevancy — LLM float judge
        6. score_completeness     — fact extraction from ground_truth_body + coverage check

    Args:
        dataset: Path to eval dataset JSON (default: eval_dataset.json)
        top:     Number of chunks to retrieve (default: 5)
        mode:    Retrieval mode: 'semantic' or 'hybrid' (default: hybrid)
        samples: Limit to first N entries (default: all)
        delay:   Seconds between API calls (default: 0.3)
    """
    intermediate = f"eval_intermediate_top{top}_{mode}.json"

    sep = "=" * 60
    print(f"{sep}\nSTEP 1/6 — build_eval_json (retrieve + generate)\n{sep}")
    build_eval_json(dataset=dataset, top=top, mode=mode, samples=samples, delay=delay)

    print(f"\n{sep}\nSTEP 2/6 — score_recall_at_k\n{sep}")
    recall = score_recall_at_k(intermediate=intermediate)

    print(f"\n{sep}\nSTEP 3/6 — score_precision_at_k\n{sep}")
    precision = score_precision_at_k(intermediate=intermediate)

    print(f"\n{sep}\nSTEP 4/6 — score_faithfulness\n{sep}")
    score_faithfulness(intermediate=intermediate, delay=delay)

    print(f"\n{sep}\nSTEP 5/6 — score_answer_relevancy\n{sep}")
    score_answer_relevancy(intermediate=intermediate, delay=delay)

    print(f"\n{sep}\nSTEP 6/6 — score_completeness\n{sep}")
    score_completeness(intermediate=intermediate, delay=delay)

    # ── Summary ──────────────────────────────────────────────────────────────
    with open(intermediate) as f:
        records = json.load(f)

    n = len(records)
    faith = [r["faithfulness"] for r in records if r.get("faithfulness") is not None]
    relev = [
        r["answer_relevancy"] for r in records if r.get("answer_relevancy") is not None
    ]
    compl = [r["completeness"] for r in records if r.get("completeness") is not None]

    print(f"\n{sep}")
    print("EVALUATION SUMMARY")
    print(sep)
    print(f"  Records          : {n}")
    print(f"  Recall@{top:<3} ({mode})  : {recall:.3f}")
    print(f"  Precision@{top:<3} ({mode}): {precision:.3f}")
    if faith:
        print(
            f"  Faithfulness     : {sum(faith) / len(faith):.3f}  ({len(faith)} scored)"
        )
    if relev:
        print(
            f"  Ans. relevancy   : {sum(relev) / len(relev):.3f}  ({len(relev)} scored)"
        )
    if compl:
        print(
            f"  Completeness     : {sum(compl) / len(compl):.3f}  ({len(compl)} scored)"
        )
    with open("README.md", "w") as f:
        f.write(f"# RAG Evaluation Results\n\n")
        f.write(f"**Dataset:** {dataset}\n\n")
        f.write(f"**Records:** {n}\n\n")
        f.write(f"**Recall@{top} ({mode}):** {recall:.3f}\n\n")
        f.write(f"**Precision@{top} ({mode}):** {precision:.3f}\n\n")
        if faith:
            f.write(
                f"**Faithfulness:** {sum(faith) / len(faith):.3f}  ({len(faith)} scored)\n\n"
            )
        if relev:
            f.write(
                f"**Answer Relevancy:** {sum(relev) / len(relev):.3f}  ({len(relev)} scored)\n\n"
            )
        if compl:
            f.write(
                f"**Completeness:** {sum(compl) / len(compl):.3f}  ({len(compl)} scored)\n\n"
            )
    print(sep)


def retrieve_items_where_faithfulness_below(
    record_path: str, threshold: float = 0.5
) -> list[dict]:
    """
    Helper to filter records with faithfulness below a certain threshold.

    Args:
        record_path: Path to the JSON file containing evaluation records.
        threshold:   Faithfulness threshold below which records are returned.

    Returns:
        A list of records with faithfulness below the specified threshold.
    """
    with open(record_path) as f:
        records = json.load(f)
    items_where_faithfulness_below = [
        r
        for r in records
        if r.get("faithfulness") is not None and r["faithfulness"] < threshold
    ]
    with open(f"unfaithful_records_below_{threshold}.json", "w") as f:
        json.dump(items_where_faithfulness_below, f, indent=2, ensure_ascii=False)
    return items_where_faithfulness_below


# ─── Full pipeline ───────────────────────────────────────────────────────────


def run_full_pipeline(
    samples: int = 200,
    top: int = 5,
    mode: str = "hybrid",
    dataset: str = DEFAULT_DATASET,
    delay: float = 0.3,
) -> None:
    """
    Run the full eval pipeline end-to-end (assumes Vespa is already indexed):
        1. Generate eval dataset from indexed chunks
        2. Add ground truth answers
        3. Run full evaluation (recall, precision, faithfulness, relevancy, completeness)

    Args:
        samples: Number of chunks to sample for eval dataset (default: 200)
        top:     Number of chunks to retrieve per query (default: 5)
        mode:    Retrieval mode: 'semantic' or 'hybrid' (default: hybrid)
        dataset: Path to eval dataset JSON (default: eval_dataset.json)
        delay:   Seconds between Mistral API calls (default: 0.3)
    """
    sep = "=" * 60

    print(f"{sep}\nSTEP 1/3 — generate eval dataset ({samples} samples)\n{sep}")
    generate(samples=samples, out=dataset, delay=delay)

    print(f"\n{sep}\nSTEP 2/3 — add ground truth answers\n{sep}")
    add_ground_truth(dataset=dataset, delay=delay)

    print(f"\n{sep}\nSTEP 3/3 — evaluate (top={top}, mode={mode})\n{sep}")
    run_all(dataset=dataset, top=top, mode=mode, delay=delay)


# ─── Phase 3: RAGAS evaluation ───────────────────────────────────────────────


def ragas_eval(
    dataset: str = DEFAULT_DATASET,
    top: int = 5,
    mode: str = "hybrid",
    samples: int = None,
    out: str = "ragas_results.json",
) -> None:
    """
    Run RAGAS evaluation (Faithfulness, AnswerRelevancy, ContextPrecision, ContextRecall).

    For each question the full RAG pipeline is executed (retrieve + generate), then
    RAGAS scores the results using Mistral as the judge LLM.

    Args:
        dataset: Path to eval dataset JSON (default: eval_dataset.json)
        top:     Number of chunks to retrieve (default: 5)
        mode:    Retrieval mode: 'semantic' or 'hybrid' (default: hybrid)
        samples: Limit to first N entries — useful for quick smoke-tests (default: all)
        out:     Output file for RAGAS scores (default: ragas_results.json)
    """
    from ragas import evaluate as ragas_evaluate, EvaluationDataset, SingleTurnSample
    from ragas.metrics import (
        Faithfulness,
        AnswerRelevancy,
        ContextPrecision,
        ContextRecall,
    )
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from langchain_mistralai import ChatMistralAI
    from langchain_mistralai.embeddings import MistralAIEmbeddings

    if not os.path.exists(dataset):
        raise FileNotFoundError(
            f"Dataset not found: {dataset!r}. Run 'generate' first."
        )

    with open(dataset) as f:
        entries = json.load(f)

    if samples:
        entries = entries[:samples]

    print(
        f"🔍 Running RAGAS evaluation on {len(entries)} entries | top={top} | mode={mode}\n"
    )

    api_key = os.environ["MISTRAL_API_KEY"]
    mistral_client = Mistral(api_key=api_key)

    ragas_llm = LangchainLLMWrapper(
        ChatMistralAI(model="mistral-large-latest", api_key=api_key)
    )
    ragas_embeddings = LangchainEmbeddingsWrapper(
        MistralAIEmbeddings(api_key=api_key, model="mistral-embed")
    )

    ragas_samples = []

    for i, entry in enumerate(entries):
        question = entry["question"]
        ground_truth = entry["body"]  # chunk the question was generated from

        # Retrieve
        query_vec = embed_query(question)
        hits = search(query_vec, top_k=top, rank_profile=mode)
        contexts = [h["body"] for h in hits]

        # Generate answer
        context_text = "\n\n---\n\n".join(
            f"[{j + 1}] {h['heading']} (from {h['source_file']})\n{h['body']}"
            for j, h in enumerate(hits)
        )
        resp = mistral_client.chat.complete(
            model=EVAL_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": f"Documentation excerpts:\n\n{context_text}\n\n---\n\nQuestion: {question}",
                },
            ],
        )
        answer = resp.choices[0].message.content.strip()

        ragas_samples.append(
            SingleTurnSample(
                user_input=question,
                response=answer,
                retrieved_contexts=contexts,
                reference=ground_truth,
            )
        )

        print(f"  [{i + 1}/{len(entries)}] {question[:70]!r}")
        time.sleep(0.3)

    eval_dataset = EvaluationDataset(samples=ragas_samples)

    metrics = [
        Faithfulness(llm=ragas_llm),
        AnswerRelevancy(llm=ragas_llm, embeddings=ragas_embeddings),
        ContextPrecision(llm=ragas_llm),
        ContextRecall(llm=ragas_llm),
    ]

    print("\n⚙️  Running RAGAS metrics...")
    results = ragas_evaluate(eval_dataset, metrics=metrics)

    df = results.to_pandas()
    print("\n📊 RAGAS Results:")
    print(
        df[["faithfulness", "answer_relevancy", "context_precision", "context_recall"]]
        .describe()
        .to_string()
    )

    with open(out, "w") as f:
        json.dump(df.to_dict(orient="records"), f, indent=2, ensure_ascii=False)
    print(f"\n✅ RAGAS results saved to {out!r}")


if __name__ == "__main__":
    fire.Fire(
        {
            "generate": generate,
            "add_ground_truth": add_ground_truth,
            "build_eval_json": build_eval_json,
            "evaluate": evaluate,
            "score_recall_at_k": score_recall_at_k,
            "score_precision_at_k": score_precision_at_k,
            "score_faithfulness": score_faithfulness,
            "score_answer_relevancy": score_answer_relevancy,
            "score_completeness": score_completeness,
            "run_all": run_all,
            "run_full_pipeline": run_full_pipeline,
            "ragas_eval": ragas_eval,
            "retrieve_items_where_faithfulness_below": retrieve_items_where_faithfulness_below,
        }
    )
