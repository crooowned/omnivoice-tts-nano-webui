import logging
import os
import threading
from contextlib import asynccontextmanager
from logging.handlers import RotatingFileHandler

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from routes.tts import router as tts_router
from routes.voices import router as voices_router
from routes.models import router as models_router
from routes.speech import router as speech_router
from routes.transcriptions import router as transcriptions_router

LOG_FILE = os.environ.get("LOG_FILE", "/app/logs/backend.log")
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

fmt = logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s")

file_handler = RotatingFileHandler(LOG_FILE, maxBytes=10 * 1024 * 1024, backupCount=3)
file_handler.setFormatter(fmt)

stream_handler = logging.StreamHandler()
stream_handler.setFormatter(fmt)

root = logging.getLogger()
root.setLevel(logging.DEBUG)
root.propagate = False
if not root.handlers:
    root.addHandler(file_handler)
    root.addHandler(stream_handler)

# reduce noise from noisy libs
for noisy in (
    "httpx",
    "httpcore",
    "urllib3",
    "huggingface_hub",
    "librosa",
    "numba",
    "matplotlib",
    "PIL",
    "soundfile",
):
    logging.getLogger(noisy).setLevel(logging.WARNING)

@asynccontextmanager
async def lifespan(_app: FastAPI):
    logger = logging.getLogger("startup")

    def _load():
        logger.info("Pre-loading OmniVoice model on startup...")
        from model_manager import get_model

        get_model()
        logger.info("Model ready.")

    threading.Thread(target=_load, daemon=True).start()
    yield


app = FastAPI(title="OmniVoice API", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(tts_router)
app.include_router(voices_router)
app.include_router(models_router)
app.include_router(speech_router)
app.include_router(transcriptions_router)


@app.get("/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    uvicorn.run("serve:app", host="0.0.0.0", port=8883, reload=False)
