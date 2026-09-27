import json

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from rag_pipeline import ask_rag


app = FastAPI(title="نظامي API")


# =====================================================
# CORS
# =====================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)



# =====================================================
# Load Dataset
# =====================================================

with open(
    "saudi_labor_law_dataset_full.json",
    encoding="utf-8"
) as f:
    _dataset = json.load(f)



# =====================================================
# Request Model
# =====================================================

class AskRequest(BaseModel):
    question: str



# =====================================================
# Frontend
# =====================================================

@app.get("/")
def serve_index():
    return FileResponse("index.html")



# =====================================================
# API Info
# =====================================================

@app.get("/api/info")
def info():
    return {
        "name": "نظامي",
        "articles_count": len(_dataset),
    }



# =====================================================
# Ask RAG
# =====================================================

@app.post("/api/ask")
def ask(payload: AskRequest):

    answer, sources, contexts = ask_rag(
        payload.question
    )

    return {
        "answer": answer,
        "sources": sources,
    }



# =====================================================
# Run Local
# =====================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8501
    )
