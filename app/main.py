"""
Main FastAPI Application Entrypoint for DFU Live Segmentation System.
"""

from __future__ import annotations

import logging
import os
import urllib.request
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import API_TITLE, API_VERSION, CORS_ORIGINS, MODELS_DIR
from app.api.routes import router as api_router
from app.model_registry.registry import model_registry

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("dfu_ui.main")

MODEL_FILENAME = "dfu_unet_boundary_seed42_512x384_bs1.pt2"
MODEL_PATH = MODELS_DIR / MODEL_FILENAME
MODEL_CDN_URL = f"https://huggingface.co/spaces/praveen202/Diabetic_Foot_Ulcer_Segmentation/resolve/main/models/{MODEL_FILENAME}"


def ensure_model_downloaded():
    """Ensures Stage 16 certified PT2 model is downloaded from Hugging Face CDN if not present."""
    if not MODEL_PATH.exists():
        logger.info(f"Model file not found locally. Downloading Stage 16 certified model from {MODEL_CDN_URL}...")
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(MODEL_CDN_URL, str(MODEL_PATH))
        logger.info(f"Model successfully downloaded ({os.path.getsize(MODEL_PATH) / (1024*1024):.2f} MB).")
    else:
        logger.info(f"Model file exists locally at {MODEL_PATH}.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Pre-flight startup: Verify registry and preload default certified model
    logger.info("Initializing DFU Live Segmentation System Backend...")
    try:
        ensure_model_downloaded()
    except Exception as e:
        logger.error(f"Failed to download certified model: {e}", exc_info=True)

    device_status = model_registry.get_device_status()
    logger.info(f"Hardware compute environment: {device_status['device_name']} ({device_status['device_type']})")
    try:
        active_id = model_registry.active_model_id
        logger.info(f"Pre-warming default certified model: {active_id}")
        model_registry.load_model(active_id)
        logger.info(f"Default certified model {active_id} loaded successfully and ready.")
    except Exception as e:
        logger.error(f"Failed to pre-warm default model: {e}", exc_info=True)
    yield
    logger.info("Shutting down DFU Live Segmentation System Backend...")


app = FastAPI(
    title=API_TITLE,
    version=API_VERSION,
    description="Engineering deployment layer for frozen Diabetic Foot Ulcer live segmentation.",
    lifespan=lifespan
)

# Configure Cross-Origin Resource Sharing (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register API Router
app.include_router(api_router, prefix="/api")


@app.get("/")
async def root():
    return {
        "system": API_TITLE,
        "version": API_VERSION,
        "status": "online",
        "active_model": model_registry.active_model_id,
        "docs_url": "/docs"
    }


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app.main:app", host="0.0.0.0", port=port, reload=False)
