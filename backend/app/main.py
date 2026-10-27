from __future__ import annotations

import logging
import shutil
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone

import cv2
import numpy as np
from fastapi import BackgroundTasks, Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from . import config
from .db import SessionLocal, get_db, init_db
from .engine import engine
from .models import Analysis, Region
from .pipeline.analyze import analyze_image, heat_map, overall_score, render_heatmap_png, resize_max_side
from .schemas import AnalysisOut, AnalysisSummary, Health, LabelIn, LabelStats, RegionOut

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("api")

MAX_UPLOAD_BYTES = 25 * 1024 * 1024


def analysis_dir(analysis_id: uuid.UUID):
    return config.RESULT_DIR / str(analysis_id)


@asynccontextmanager
async def lifespan(_: FastAPI):
    config.RESULT_DIR.mkdir(parents=True, exist_ok=True)
    init_db()
    engine.load_classifier()
    yield


app = FastAPI(title="AI Edit Detector", lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)
app.add_middleware(
    CORSMiddleware, allow_origins=config.CORS_ORIGINS, allow_methods=["*"], allow_headers=["*"]
)


def run_analysis(analysis_id: uuid.UUID):
    db = SessionLocal()
    try:
        a = db.get(Analysis, analysis_id)
        a.status = "processing"
        db.commit()

        out = analysis_dir(analysis_id)
        img = cv2.imread(str(out / "image.png"), cv2.IMREAD_COLOR)
        with engine.lock:
            segmenter = engine.get_segmenter()
            results = analyze_image(img, segmenter, engine.classifier, config.CLASSIFIER_WEIGHT)
            version = engine.model_version

        mask_dir = out / "masks"
        mask_dir.mkdir(exist_ok=True)
        for i, r in enumerate(results):
            cv2.imwrite(str(mask_dir / f"{i}.png"), r.mask.astype(np.uint8) * 255)
            db.add(Region(
                analysis_id=analysis_id, index=i, bbox=r.bbox, polygons=r.polygons,
                area_frac=r.area_frac, noise_score=r.noise_score,
                classifier_score=r.classifier_score, score=r.score, details=r.extra,
            ))
        cv2.imwrite(str(out / "heatmap.png"), render_heatmap_png(heat_map(img.shape[:2], results)))

        a.overall_score = overall_score(results)
        a.model_version = version or "noise-only"
        a.status = "done"
        db.commit()
    except Exception as exc:
        log.exception("Analysis %s failed", analysis_id)
        db.rollback()
        a = db.get(Analysis, analysis_id)
        if a:
            a.status, a.error = "failed", str(exc)[:2000]
            db.commit()
    finally:
        db.close()


@app.get("/api/health", response_model=Health)
def health():
    return Health(
        status="ok", sam_loaded=engine.segmenter is not None,
        classifier_loaded=engine.classifier is not None, model_version=engine.model_version,
    )


@app.post("/api/analyses", response_model=AnalysisSummary, status_code=201)
async def create_analysis(background: BackgroundTasks, file: UploadFile = File(...),
                          db: Session = Depends(get_db)):
    data = await file.read()
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "Image is larger than 25 MB")
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "Could not read that file as an image")
    img = resize_max_side(img, config.MAX_SIDE)

    a = Analysis(filename=(file.filename or "upload")[:255], height=img.shape[0], width=img.shape[1])
    db.add(a)
    db.commit()

    out = analysis_dir(a.id)
    out.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out / "image.png"), img)
    background.add_task(run_analysis, a.id)
    return a


@app.get("/api/analyses", response_model=list[AnalysisSummary])
def list_analyses(limit: int = 50, db: Session = Depends(get_db)):
    return db.scalars(select(Analysis).order_by(Analysis.created_at.desc()).limit(min(limit, 200))).all()


@app.get("/api/analyses/{analysis_id}", response_model=AnalysisOut)
def get_analysis(analysis_id: uuid.UUID, db: Session = Depends(get_db)):
    a = db.get(Analysis, analysis_id)
    if not a:
        raise HTTPException(404, "Analysis not found")
    return a


@app.get("/api/analyses/{analysis_id}/image")
def get_image(analysis_id: uuid.UUID):
    path = analysis_dir(analysis_id) / "image.png"
    if not path.exists():
        raise HTTPException(404)
    return FileResponse(path, media_type="image/png")


@app.get("/api/analyses/{analysis_id}/heatmap")
def get_heatmap(analysis_id: uuid.UUID):
    path = analysis_dir(analysis_id) / "heatmap.png"
    if not path.exists():
        raise HTTPException(404)
    return FileResponse(path, media_type="image/png")


@app.delete("/api/analyses/{analysis_id}", status_code=204)
def delete_analysis(analysis_id: uuid.UUID, db: Session = Depends(get_db)):
    a = db.get(Analysis, analysis_id)
    if not a:
        raise HTTPException(404)
    db.delete(a)
    db.commit()
    shutil.rmtree(analysis_dir(analysis_id), ignore_errors=True)


@app.put("/api/regions/{region_id}/label", response_model=RegionOut)
def label_region(region_id: int, body: LabelIn, db: Session = Depends(get_db)):
    r = db.get(Region, region_id)
    if not r:
        raise HTTPException(404, "Region not found")
    r.label = body.label
    r.labeled_at = datetime.now(timezone.utc) if body.label else None
    db.commit()
    return r


@app.get("/api/labels/stats", response_model=LabelStats)
def label_stats(db: Session = Depends(get_db)):
    counts = dict(db.execute(select(Region.label, func.count()).group_by(Region.label)).all())
    return LabelStats(real=counts.get("real", 0), ai=counts.get("ai", 0), unlabeled=counts.get(None, 0))


@app.post("/api/model/reload", response_model=Health)
def reload_model():
    with engine.lock:
        engine.load_classifier()
    return health()
