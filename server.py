from fastapi import FastAPI, HTTPException, UploadFile, File, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from multimodal_rag_system import MultimodalRAGSystem
import uvicorn
import io
from PIL import Image


app = FastAPI(title="Multimodal RAG API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

rag_system = MultimodalRAGSystem()

# API 모델 정의


class ProductResult(BaseModel):
    id: str
    product_name: Optional[str] = None
    category: Optional[str] = None
    image_url: Optional[str] = None
    product_address: Optional[str] = None
    last_updated: Optional[str] = None
    is_emb: Optional[str] = None
    similarity_score: Optional[float] = None


class SearchResponse(BaseModel):
    results: List[ProductResult]


# 유틸 함수


async def read_imagefile(file: UploadFile) -> Image.Image:
    image_bytes = await file.read()
    return Image.open(io.BytesIO(image_bytes)).convert("RGB")


def safe_remove_embedding(results: List[dict]):
    for item in results:
        item.pop("embedding", None)


# ===================== 🔍 벡터 검색 API ===================== #


@app.post("/search/text", response_model=SearchResponse)
async def search_products(query: str = Form(...), top_k: int = Form(30)):
    try:
        results = rag_system.search_by_text(query, top_k)
        safe_remove_embedding(results)
        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/search/image", response_model=SearchResponse)
async def search_by_image(file: UploadFile = File(...), top_k: int = Form(30)):
    try:
        image = await read_imagefile(file)
        results = rag_system.search_by_image(image, limit=top_k)
        safe_remove_embedding(results)
        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Image search failed: {str(e)}")


@app.post("/search/multimodal", response_model=SearchResponse)
async def search_multimodal(
    query: str = Form(...),
    file: UploadFile = File(...),
    alpha: float = Form(0.7),
    top_k: int = Form(30),
):
    try:
        image = await read_imagefile(file)
        results = rag_system.search_multimodal(query, image, limit=top_k, alpha=alpha)
        safe_remove_embedding(results)
        return {"results": results}
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Multimodal search failed: {str(e)}"
        )


# ===================== 🧠 Firestore 인덱싱 API ===================== #


@app.get("/start-index")
def start_indexing(background_tasks: BackgroundTasks):
    try:
        background_tasks.add_task(
            rag_system.vector_db.store_products(rag_system.embedding_model)
        )
        return {"status": "인덱싱이 시작되었습니다"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ================== ✅ 헬스 체크 ================== #


@app.get("/health")
async def health_check():
    return {"status": "ok"}


if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
