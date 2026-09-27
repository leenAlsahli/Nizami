import json

import traceback

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from rag_pipeline import ask_rag


app = FastAPI(title="نظامي API")

# CORS مفتوح -- مو ضروري أصلاً بما إن الواجهة والـ API على نفس الأصل،
# بس نخليها موجودة احتياط لو حبيتي تفصلين الواجهة لسيرفر ثاني مستقبلاً
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# =====================================================
# تحميل عدد المواد -- يستخدم بصفحة "حول النظام"
# =====================================================

with open("saudi_labor_law_dataset_full.json", encoding="utf-8") as f:
    _dataset = json.load(f)


class AskRequest(BaseModel):
    question: str



@app.get("/")
def serve_index():
    return FileResponse("index.html")


# =====================================================
# API
# =====================================================

@app.get("/api/info")
def info():
    return {
        "name": "نظامي",
        "articles_count": len(_dataset),
    }


@app.post("/api/ask")
def ask(payload: AskRequest):
    try:
        answer, sources, contexts = ask_rag(payload.question)
        return {
            "answer": answer,
            "sources": sources,
        }
    except Exception as e:
        # مؤقت للتشخيص: يطبع الخطأ الكامل بلوق Render
        # ويرجعه بالرد نفسه علشان تشوفينه من الفرونت إند
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"error": f"{type(e).__name__}: {e}"},
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8501)
