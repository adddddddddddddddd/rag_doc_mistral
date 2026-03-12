import fastapi
import uvicorn
from fastapi.middleware.cors import CORSMiddleware

from rag import rag, rag_with_section

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
    "agents" : [
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
    ]
}

@app.post("/rag/section/{section_name}")
def rag_section_endpoint(section_name: str, query: str, top: int = 5, mode: str = "hybrid"):
    section_paths = sections.get(section_name)
    if section_paths is None:
        raise fastapi.HTTPException(status_code=404, detail=f"Unknown section: {section_name!r}")
    response = rag_with_section(query=query, section_paths=section_paths, top_k=top, mode=mode)
    return {"message": response}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)