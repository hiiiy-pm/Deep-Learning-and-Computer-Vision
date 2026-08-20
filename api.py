from __future__ import annotations

import os
from functools import lru_cache
from io import BytesIO

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from src.inference import CornDiseasePredictor
from src.settings import DEFAULT_DEVICE, DEFAULT_TOPK


app = FastAPI(title="Corn Leaf Disease Classification API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"^http://(localhost|127\.0\.0\.1):\d+$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@lru_cache(maxsize=1)
def get_predictor() -> CornDiseasePredictor:
    checkpoint = os.getenv("CORN_CHECKPOINT") or None
    device = os.getenv("CORN_DEVICE", DEFAULT_DEVICE)
    eval_crops = int(os.getenv("CORN_EVAL_CROPS", "1"))
    return CornDiseasePredictor(checkpoint=checkpoint, device=device, eval_crops=eval_crops)


@app.get("/health")
def health() -> dict[str, str | bool]:
    predictor = get_predictor()
    return {"ok": True, "checkpoint": str(predictor.checkpoint_path)}


@app.get("/metadata")
def metadata() -> dict:
    return get_predictor().metadata()


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    topk: int = Query(DEFAULT_TOPK, ge=1, le=10),
) -> dict:
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="请上传图片文件。")

    payload = await file.read()
    try:
        image = Image.open(BytesIO(payload)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail="图片无法读取或格式不受支持。") from exc

    result = get_predictor().predict_image(image, image_path=file.filename, topk=topk)
    return result.to_dict()
