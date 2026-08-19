"""Lazy Parakeet TDT inference using the existing PyTorch/Transformers stack."""

from __future__ import annotations

import gc
import logging
import threading
import time

import numpy as np
import torch
from pydub import AudioSegment

from config import (
    ASR_CPU_FALLBACK,
    ASR_DEVICE,
    ASR_MIN_FREE_GPU_GB,
    ASR_MODEL,
    ASR_MODEL_TTL_SECONDS,
)

logger = logging.getLogger(__name__)

_pipeline = None
_active_device = None
_last_used = 0.0
_load_lock = threading.Lock()
_inference_lock = threading.Lock()
_ttl_thread = None


def _gpu_memory_gb() -> tuple[float | None, float | None]:
    if not torch.cuda.is_available():
        return None, None
    try:
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        gib = 1024**3
        return free_bytes / gib, total_bytes / gib
    except (RuntimeError, TypeError):
        return None, None


def _device() -> str:
    if ASR_DEVICE.startswith("cuda") and not torch.cuda.is_available():
        logger.warning("ASR_DEVICE=%s requested but no CUDA/HIP device is available; using CPU", ASR_DEVICE)
        return "cpu"
    if ASR_DEVICE.startswith("cuda") and ASR_CPU_FALLBACK:
        free_gb, _ = _gpu_memory_gb()
        if free_gb is not None and free_gb < ASR_MIN_FREE_GPU_GB:
            return "cpu"
    return ASR_DEVICE


def _dtype(device: str):
    return torch.float16 if device.startswith("cuda") else torch.float32


def _ensure_ttl_thread():
    global _ttl_thread
    if _ttl_thread is None or not _ttl_thread.is_alive():
        _ttl_thread = threading.Thread(target=_unload_loop, daemon=True)
        _ttl_thread.start()


def _unload_loop():
    while True:
        time.sleep(30)
        if _pipeline is not None and time.time() - _last_used > ASR_MODEL_TTL_SECONDS:
            unload_asr_model()


def _clear_accelerator_cache():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _load_pipeline(device: str):
    from transformers import AutoModelForTDT, AutoProcessor, pipeline

    dtype = _dtype(device)
    logger.info("Loading Parakeet ASR model %s on %s as %s", ASR_MODEL, device, dtype)
    model = None
    processor = None
    try:
        processor = AutoProcessor.from_pretrained(ASR_MODEL)
        load_kwargs = {"dtype": dtype, "low_cpu_mem_usage": True}
        if device.startswith("cuda"):
            load_kwargs["device_map"] = device
        model = AutoModelForTDT.from_pretrained(ASR_MODEL, **load_kwargs)
        return pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
        )
    except Exception:
        del model, processor
        _clear_accelerator_cache()
        raise


def get_asr_pipeline():
    global _pipeline, _active_device, _last_used
    _ensure_ttl_thread()
    with _load_lock:
        if _pipeline is None:
            device = _device()
            if device == "cpu" and ASR_DEVICE.startswith("cuda"):
                free_gb, total_gb = _gpu_memory_gb()
                logger.warning(
                    "Only %.2fGB of %.2fGB GPU/HIP memory is free (%.2fGB required reserve); "
                    "loading Parakeet on CPU",
                    free_gb or 0.0,
                    total_gb or 0.0,
                    ASR_MIN_FREE_GPU_GB,
                )
            try:
                _pipeline = _load_pipeline(device)
            except torch.OutOfMemoryError:
                if not device.startswith("cuda") or not ASR_CPU_FALLBACK:
                    raise
                logger.warning("Parakeet exhausted GPU/HIP memory while loading; retrying on CPU")
                _pipeline = _load_pipeline("cpu")
                device = "cpu"
            except Exception:
                _pipeline = None
                _active_device = None
                _clear_accelerator_cache()
                raise
            _active_device = device
            logger.info("Parakeet ASR model ready on %s", device)
        _last_used = time.time()
        return _pipeline


def _decode_audio(audio_bytes: bytes, content_type: str | None = None) -> np.ndarray:
    """Decode uploaded audio to mono float32 at Parakeet's expected sample rate."""
    import io

    media_type = (content_type or "").split(";", 1)[0].lower()
    format_hint = {
        "audio/wav": "wav",
        "audio/x-wav": "wav",
        "audio/mpeg": "mp3",
        "audio/mp4": "mp4",
        "audio/x-m4a": "m4a",
        "audio/flac": "flac",
        "audio/ogg": "ogg",
        "audio/webm": "webm",
    }.get(media_type)

    pipe = get_asr_pipeline()
    sampling_rate = int(pipe.feature_extractor.sampling_rate)
    segment = AudioSegment.from_file(io.BytesIO(audio_bytes), format=format_hint)
    segment = segment.set_channels(1).set_frame_rate(sampling_rate)
    samples = np.asarray(segment.get_array_of_samples(), dtype=np.float32)
    if samples.size == 0:
        raise ValueError("Decoded audio is empty")
    scale = float(1 << (8 * segment.sample_width - 1))
    return samples / scale


def transcribe(audio_bytes: bytes, content_type: str | None = None) -> str:
    global _last_used
    if not audio_bytes:
        raise ValueError("Uploaded audio is empty")

    pipe = get_asr_pipeline()
    waveform = _decode_audio(audio_bytes, content_type)
    with _inference_lock, torch.inference_mode():
        result = pipe(waveform)
    _last_used = time.time()
    if isinstance(result, dict):
        return str(result.get("text") or "").strip()
    return str(result).strip()


def unload_asr_model() -> bool:
    global _pipeline, _active_device
    with _inference_lock:
        with _load_lock:
            if _pipeline is None:
                return False
            logger.info("Unloading idle Parakeet ASR model")
            _pipeline = None
            _active_device = None
            _clear_accelerator_cache()
            return True


def asr_status() -> dict:
    free_gb, total_gb = _gpu_memory_gb()
    return {
        "model": ASR_MODEL,
        "device": _active_device or _device(),
        "configured_device": ASR_DEVICE,
        "loaded": _pipeline is not None,
        "ttl_seconds": ASR_MODEL_TTL_SECONDS,
        "idle_seconds": round(time.time() - _last_used, 1) if _pipeline is not None else None,
        "gpu_free_gb": round(free_gb, 2) if free_gb is not None else None,
        "gpu_total_gb": round(total_gb, 2) if total_gb is not None else None,
    }
