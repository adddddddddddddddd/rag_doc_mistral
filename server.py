import fastapi
import uvicorn
from fastapi.middleware.cors import CORSMiddleware

from rag import rag

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

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)