import fastapi
import uvicorn
from fastapi.middleware.cors import CORSMiddleware

from backend.rag import rag, rag_with_section, rag_with_hyde, rag_with_section_hyde

app = fastapi.FastAPI()

origins = [
    "http://localhost.tiangolo.com",
    "https://localhost.tiangolo.com",
    "http://localhost",
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/rag")
def rag_endpoint(query: str, top: int = 5, mode: str = "hybrid"):
    response = rag(query=query, top_k=top, mode=mode)
    return {"message": response}


sections = {
    "getting_started": [
        "platform-docs-public/public/getting_started/clients.md",
        "platform-docs-public/public/getting_started/glossary.md",
        "platform-docs-public/public/getting_started/quickstart.md",
    ],
    "capabilities": [
        "platform-docs-public/public/capabilities/audio_transcription.md",
        "platform-docs-public/public/capabilities/embeddings.md",
        "platform-docs-public/public/capabilities/function_calling.md",
        "platform-docs-public/public/capabilities/guardrailing.md",
        "platform-docs-public/public/capabilities/document_ai.md",
        "platform-docs-public/public/capabilities/batch.md",
        "platform-docs-public/public/capabilities/citations.md",
        "platform-docs-public/public/capabilities/code_generation.md",
        "platform-docs-public/public/capabilities/completion.md",
        "platform-docs-public/public/capabilities/predicted_outputs.md",
        "platform-docs-public/public/capabilities/reasoning.md",
        "platform-docs-public/public/capabilities/structured_output.md",

    ],
    "agents": [
        "platform-docs-public/public/agents/agents.md",
        "platform-docs-public/public/agents/handoffs.md",
        "platform-docs-public/public/agents/tools.md",
        "platform-docs-public/public/agents/introduction.md",
        "platform-docs-public/public/agents/tools/built-in.md",
        "platform-docs-public/public/agents/tools/function_calling.md",
        "platform-docs-public/public/agents/tools/built-in/code_interpreter.md",
        "platform-docs-public/public/agents/tools/built-in/document_library.md",
        "platform-docs-public/public/agents/tools/built-in/image_generation.md",
        "platform-docs-public/public/agents/tools/built-in/websearch.md",
    ],
}


@app.get("/sections")
def get_sections():
    return {"sections": list(sections.keys())}


@app.post("/rag/section/{section_name}")
def rag_section_endpoint(
    section_name: str, query: str, top: int = 5, mode: str = "hybrid"
):
    section_paths = sections.get(section_name)
    if section_paths is None:
        raise fastapi.HTTPException(
            status_code=404, detail=f"Unknown section: {section_name!r}"
        )
    response = rag_with_section(
        query=query, section_paths=section_paths, top_k=top, mode=mode
    )
    return {"message": response}


@app.post("/rag/hyde")
def rag_hyde_endpoint(query: str, top: int = 5, mode: str = "hybrid"):
    response = rag_with_hyde(query=query, top_k=top, mode=mode)
    return {"message": response}


@app.post("/rag/section/{section_name}/hyde")
def rag_section_hyde_endpoint(
    section_name: str, query: str, top: int = 3, mode: str = "hybrid"
):
    section_paths = sections.get(section_name)
    if section_paths is None:
        raise fastapi.HTTPException(
            status_code=404, detail=f"Unknown section: {section_name!r}"
        )
    response = rag_with_section_hyde(
        query=query, section_paths=section_paths, top_k=top, mode=mode
    )
    return {"message": response}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
