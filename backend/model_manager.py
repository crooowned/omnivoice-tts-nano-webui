import gc
import logging
import os
import time
import threading
from typing import Optional

import torch
from omnivoice import OmniVoice

from config import (
    AUDIO_TOKENIZER_DEVICE,
    CHECKPOINT,
    CPU_OFFLOAD,
    CPU_OFFLOAD_GB,
    DEVICE,
    DTYPE,
    LM_QUANT,
    LOAD_ASR,
    MAX_VRAM_GB,
    MODEL_TTL_SECONDS,
    OFFLOAD_DIR,
)

logger = logging.getLogger(__name__)

_model: Optional[OmniVoice] = None
_last_used: float = 0.0
_lock = threading.Lock()
_ttl_thread: Optional[threading.Thread] = None


def _dtype():
    if DTYPE == "bfloat16":
        return torch.bfloat16
    if DTYPE == "float16":
        return torch.float16
    if DTYPE in ("fp8", "float8", "fp4", "nf4"):
        logger.warning(
            f"DTYPE={DTYPE!r} is not a valid load dtype — falling back to bfloat16. "
            f"For sub-16-bit weights use LM_QUANT instead."
        )
        return torch.bfloat16
    logger.warning(f"Unknown DTYPE={DTYPE!r} — falling back to bfloat16")
    return torch.bfloat16


def _quantize_llm_in_place(model):
    """Replace nn.Linear layers inside model.llm with bnb quantized versions.

    Preserves OmniVoice's fine-tuned weights — we just store them in fewer bits.
    """
    if LM_QUANT not in ("nf4", "fp4", "int8"):
        return

    import bitsandbytes as bnb
    import torch.nn as nn

    llm = model.llm
    target_modules = []
    for name, module in llm.named_modules():
        if isinstance(module, nn.Linear):
            target_modules.append(name)

    logger.info(f"Quantizing {len(target_modules)} Linear layers in LM with {LM_QUANT}...")

    is_4bit = LM_QUANT in ("nf4", "fp4")

    for name in target_modules:
        parent = llm
        parts = name.split(".")
        for p in parts[:-1]:
            parent = getattr(parent, p)
        leaf_name = parts[-1]
        old: nn.Linear = getattr(parent, leaf_name)

        if is_4bit:
            new = bnb.nn.Linear4bit(
                old.in_features,
                old.out_features,
                bias=old.bias is not None,
                compute_dtype=_dtype(),
                quant_type=LM_QUANT,
                quant_storage=torch.uint8,
            )
            new.weight = bnb.nn.Params4bit(
                old.weight.data.clone(), requires_grad=False, quant_type=LM_QUANT
            )
        else:  # int8
            new = bnb.nn.Linear8bitLt(
                old.in_features,
                old.out_features,
                bias=old.bias is not None,
                has_fp16_weights=False,
                threshold=6.0,
            )
            new.weight = bnb.nn.Int8Params(
                old.weight.data.clone(), requires_grad=False, has_fp16_weights=False
            )

        if old.bias is not None:
            new.bias = nn.Parameter(old.bias.data.clone())
        new = new.to(DEVICE)

        setattr(parent, leaf_name, new)
        del old

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info(f"LM quantization done — VRAM: {_vram_info()}")


def _unload_loop():
    while True:
        time.sleep(30)
        with _lock:
            global _model, _last_used
            if _model is not None and (time.time() - _last_used) > MODEL_TTL_SECONDS:
                logger.info("TTL expired — unloading OmniVoice model")
                del _model
                _model = None
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()


def _ensure_ttl_thread():
    global _ttl_thread
    if _ttl_thread is None or not _ttl_thread.is_alive():
        _ttl_thread = threading.Thread(target=_unload_loop, daemon=True)
        _ttl_thread.start()


def _vram_info() -> str:
    if not torch.cuda.is_available():
        return "cpu"
    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    return f"allocated={allocated:.2f}GB reserved={reserved:.2f}GB"


def reset_peak_vram():
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def peak_vram_gb() -> float:
    """Peak reserved VRAM (closer to what nvidia-smi shows than max_memory_allocated)."""
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_reserved() / 1024**3


def _apply_vram_limit():
    if CPU_OFFLOAD:
        return
    if MAX_VRAM_GB > 0 and torch.cuda.is_available():
        total_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        fraction = MAX_VRAM_GB / total_gb
        fraction = max(0.05, min(fraction, 1.0))
        torch.cuda.set_per_process_memory_fraction(fraction)
        logger.info(f"VRAM limit: {MAX_VRAM_GB:.2f}GB ({fraction*100:.1f}% of {total_gb:.2f}GB)")


def _load_kwargs() -> dict:
    kwargs = dict(dtype=_dtype(), load_asr=LOAD_ASR)

    if CPU_OFFLOAD:
        os.makedirs(OFFLOAD_DIR, exist_ok=True)
        max_memory = {"cpu": f"{CPU_OFFLOAD_GB:.0f}GiB"}
        if torch.cuda.is_available() and MAX_VRAM_GB > 0:
            max_memory[0] = f"{MAX_VRAM_GB:.0f}GiB"
        kwargs.update(
            device_map="auto",
            max_memory=max_memory,
            offload_folder=OFFLOAD_DIR,
            offload_state_dict=True,
            low_cpu_mem_usage=True,
        )
        logger.info(
            "CPU offload enabled: device_map=auto max_memory=%s offload_folder=%s",
            max_memory,
            OFFLOAD_DIR,
        )
    else:
        # Stream safetensor shards directly to the target device instead of
        # materializing a complete CPU model before moving it to CUDA/HIP.
        kwargs.update(
            device_map=DEVICE,
            low_cpu_mem_usage=True,
        )

    return kwargs


def _move_audio_tokenizer(model):
    target = AUDIO_TOKENIZER_DEVICE
    if not target:
        return
    if target == "cuda" and not torch.cuda.is_available():
        logger.warning("AUDIO_TOKENIZER_DEVICE=cuda requested but CUDA is unavailable")
        return
    if not hasattr(model, "audio_tokenizer") or model.audio_tokenizer is None:
        return
    logger.info("Moving audio tokenizer to %s", target)
    model.audio_tokenizer.to(target)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info("Audio tokenizer moved — VRAM: %s", _vram_info())


def get_model() -> OmniVoice:
    global _model, _last_used
    _ensure_ttl_thread()
    with _lock:
        if _model is None:
            _apply_vram_limit()
            logger.info(f"Loading OmniVoice from {CHECKPOINT} on {DEVICE} as {DTYPE} (lm_quant={LM_QUANT})")
            logger.info(f"VRAM before load: {_vram_info()}")

            loaded_model = None
            try:
                loaded_model = OmniVoice.from_pretrained(
                    CHECKPOINT,
                    **_load_kwargs(),
                )
                _model = loaded_model
            except Exception:
                _model = None
                del loaded_model
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                logger.exception("OmniVoice loading failed; released partial model state")
                raise
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info(f"OmniVoice loaded — VRAM after load: {_vram_info()}")
            _move_audio_tokenizer(_model)

            if CPU_OFFLOAD and LM_QUANT in ("nf4", "fp4", "int8"):
                logger.warning(
                    "CPU_OFFLOAD=true with LM_QUANT=%s keeps the quantized LLM "
                    "layers on DEVICE. This may use more VRAM than pure CPU "
                    "offload, but is usually much smaller than unquantized weights.",
                    LM_QUANT,
                )
                _quantize_llm_in_place(_model)
            elif LM_QUANT in ("nf4", "int8"):
                _quantize_llm_in_place(_model)
        _last_used = time.time()
        return _model


def is_loaded() -> bool:
    return _model is not None
