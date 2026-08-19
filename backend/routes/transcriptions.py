import logging
import time
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse, PlainTextResponse

from asr_manager import asr_status, get_asr_pipeline, transcribe, unload_asr_model
from config import ASR_MODEL

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/v1", tags=["openai-compat", "stt"])

ASR_ALIASES = {ASR_MODEL, "parakeet-tdt-0.6b-v3", "parakeet"}
SUPPORTED_FORMATS = {"json", "verbose_json", "text"}


@router.post("/audio/transcriptions")
def openai_transcription(
    file: UploadFile = File(...),
    model: str = Form("parakeet-tdt-0.6b-v3"),
    response_format: str = Form("json"),
    language: Optional[str] = Form(None),
):
    if model not in ASR_ALIASES:
        raise HTTPException(404, f"Unknown transcription model: {model}")
    if response_format not in SUPPORTED_FORMATS:
        raise HTTPException(400, "Supported response_format values are: json, verbose_json, text")

    started_at = time.monotonic()
    try:
        text = transcribe(file.file.read(), file.content_type)
    except Exception as exc:
        logger.exception("Parakeet transcription failed")
        raise HTTPException(500, f"Transcription failed: {exc}") from exc

    elapsed = time.monotonic() - started_at
    headers = {"X-Total-Time-S": f"{elapsed:.2f}"}
    if response_format == "text":
        return PlainTextResponse(text, headers=headers)
    if response_format == "verbose_json":
        return JSONResponse(
            {"text": text, "language": language, "duration": None, "segments": []},
            headers=headers,
        )
    return JSONResponse({"text": text}, headers=headers)


@router.get("/asr/status")
def get_asr_status():
    return asr_status()


@router.post("/asr/load")
def load_asr():
    get_asr_pipeline()
    return asr_status()


@router.post("/asr/unload")
def unload_asr():
    return {"unloaded": unload_asr_model()}
