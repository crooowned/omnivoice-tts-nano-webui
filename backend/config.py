import os
from pathlib import Path

MODELS_DIR = Path(os.environ.get("MODELS_DIR", "/app/models"))
VOICES_DIR = Path(os.environ.get("VOICES_DIR", "/app/voices"))
CHECKPOINT = os.environ.get("OMNIVOICE_MODEL", "k2-fsa/OmniVoice")
DEVICE = os.environ.get("DEVICE", "cuda")
DTYPE = os.environ.get("DTYPE", "float16")  # float16 or bfloat16
LM_QUANT = os.environ.get("LM_QUANT", "none").lower()  # none, nf4, int8
LOAD_ASR = os.environ.get("LOAD_ASR", "false").lower() == "true"
ASR_MODEL = os.environ.get("ASR_MODEL", "nvidia/parakeet-tdt-0.6b-v3")
ASR_DEVICE = os.environ.get("ASR_DEVICE", DEVICE).strip() or DEVICE
ASR_MODEL_TTL_SECONDS = int(os.environ.get("ASR_MODEL_TTL_SECONDS", "300"))
ASR_CPU_FALLBACK = os.environ.get("ASR_CPU_FALLBACK", "true").lower() == "true"
ASR_MIN_FREE_GPU_GB = float(os.environ.get("ASR_MIN_FREE_GPU_GB", "2"))
MODEL_TTL_SECONDS = int(os.environ.get("MODEL_TTL_SECONDS", "3600"))
MAX_VRAM_GB = float(os.environ.get("MAX_VRAM_GB", "0"))  # 0 = no limit
CPU_OFFLOAD = os.environ.get("CPU_OFFLOAD", "false").lower() == "true"
CPU_OFFLOAD_GB = float(os.environ.get("CPU_OFFLOAD_GB", "8"))
OFFLOAD_DIR = os.environ.get("OFFLOAD_DIR", "/app/offload")
AUDIO_TOKENIZER_DEVICE = os.environ.get("AUDIO_TOKENIZER_DEVICE", "").strip().lower()
SAMPLING_RATE = 24000
