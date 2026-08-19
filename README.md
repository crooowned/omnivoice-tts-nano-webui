# OmniVoice Low-VRAM WebUI API

*NEW*: Added a mixed CPU/GPU offload compose profile for very tight VRAM budgets!
This enables running with less than 1.5GB of GPU VRAM, by offloading around 2-3GB to the CPU Ram, while remaining high speeds!

Docker Compose setup for running an OmniVoice backend with a Gradio WebUI.
The main goal of this project is to make OmniVoice practical on machines with
limited VRAM by keeping model loading, quantization, chunking, and cleanup
configurable.

This repo is a WebUI/API wrapper and VRAM-efficiency setup around the original
OmniVoice model by k2-fsa:
[k2-fsa/OmniVoice on Hugging Face](https://huggingface.co/k2-fsa/OmniVoice).

I have tested multiple TTS Models and though the "best" in terms of efficiency might be kokoro-tts, but it does not support German for example.
And I first tested many others, chatterbox, qwen3tts, piper, xttsv2, vibevoice...
But in the end omnivoice has really thrown me off the chair.
Generated Voice in good enough quality and 5s of voice generated in just 0.6s on a RTX 5060Ti and 0.2s on my RTX 4080.

I did some, or a lot, of tweaking to squeeze it as much as I could.
It defaults the backbone Qwen3 0.6B Model to use nf4 quants, could be done better, but good enough.

HINT: I set the maximum VRAM to 3GB, it works for me, but if you use a sample voice that is longer than 3.5s, it will not be enough! *edit* default is 4GB now as I ran into OOMs.
I will try to bring it down further while running stable.


It supports:

- voice cloning from short reference audio
- voice design with OmniVoice speaker tags
- saved voice profiles
- optional STT-assisted reference transcription and trimming
- low-VRAM GPU settings such as `LM_QUANT`, `MAX_VRAM_GB`, chunking, and TTL
- CPU, NVIDIA GPU, and AMD ROCm compose profiles

## Requirements

- Docker and Docker Compose
- For NVIDIA GPU mode: NVIDIA Container Toolkit and a CUDA-capable GPU
- For AMD GPU mode: Linux, a ROCm-compatible `amdgpu` host driver, and access to
  `/dev/kfd` plus `/dev/dri`

## Quick Start

GPU:

```bash
cp .env-example .env
./start-gpu.sh -d
```

CPU:

```bash
cp .env-example .env
./start-cpu.sh -d
```

AMD ROCm (Ryzen AI Max+ 395 / `gfx1151`):

```bash
cp .env-example .env
./start-rocm.sh -d
```

Open the WebUI at:

```text
http://localhost:7863
```

The backend API listens on:

```text
http://localhost:8883
```

## Configuration

Use `.env-example` as the public template:

```bash
cp .env-example .env
```

`.env` is ignored by git and is meant for local/private settings.

Common variables:

- `FRONTEND_PORT`: host port for the Gradio UI
- `UI_LANG`: UI language, currently `en` or `de`
- `STT_URL`: optional OpenAI-compatible transcription service URL
- `OMNIVOICE_MODEL`: Hugging Face model id or local model path
- `DEVICE`: `cpu` or `cuda`
- `DTYPE`: `float16` or `bfloat16`
- `LM_QUANT`: `none`, `nf4`, or `int8`
- `MAX_VRAM_GB`: GPU memory guard, `0` disables the limit
- `CPU_OFFLOAD`: experimental Accelerate CPU offload mode
- `CPU_OFFLOAD_GB`: CPU RAM budget for offloaded model weights
- `OFFLOAD_DIR`: folder used by Accelerate for offload state
- `AUDIO_TOKENIZER_DEVICE`: optional `cpu` or `cuda` override for OmniVoice's audio tokenizer
- `CHUNK_CHARS`: split long text into smaller generation chunks
- `MODEL_TTL_SECONDS`: unload idle backend model state after this many seconds

Compose uses the internal backend URL between containers. `API_URL` in `.env-example`
is mainly useful when running the frontend directly on the host.

## Low-VRAM Notes

This repository is tuned around keeping VRAM usage as low and predictable as
possible:

- `LM_QUANT=nf4` is the default GPU compose setting for the language-model part.
- `MAX_VRAM_GB` can stop generation when measured peak VRAM grows beyond your
  chosen budget.
- `MODEL_TTL_SECONDS` allows the backend to unload idle model state.
- `CHUNK_CHARS` avoids pushing long prompts through generation as one large item.
- The backend reports peak VRAM and total generation time in response headers.

For the lowest memory footprint, start with the GPU defaults in `.env-example`
and reduce `CHUNK_CHARS` before increasing model precision or step count.

`float16` is the recommended GPU dtype. `bfloat16` can fail in OmniVoice post
processing because parts of the upstream model convert tensors to NumPy, which
does not support PyTorch `BFloat16` tensors directly.

### AMD ROCm / Ryzen AI Max

The Ryzen AI Max+ 395 GPU target is `gfx1151` (not `gfx1511`). The ROCm profile
uses AMD's official `rocm/pytorch` image with ROCm 7.2.4 and passes `/dev/kfd`
and `/dev/dri` into the backend container. No `HSA_OVERRIDE_GFX_VERSION` is
needed because this target is supported natively.

PyTorch intentionally still uses `DEVICE=cuda`: its Python device API keeps the
`cuda` name when PyTorch is built with HIP/ROCm. You can verify the running
container after startup with:

```bash
docker compose -f docker-compose.rocm.yaml exec omnivoice-backend \
  python -c "import torch; print(torch.version.hip, torch.cuda.get_device_name(0), torch.cuda.get_device_properties(0).gcnArchName)"
```

The ROCm profile defaults to `LM_QUANT=none`. The existing `nf4`/`int8` path
uses bitsandbytes and is not part of this tested ROCm setup. Since the APU uses
shared memory, the profile also defaults `MAX_VRAM_GB=0`. To select another AMD
base image without editing files, set `ROCM_PYTORCH_IMAGE` in `.env`.

### Experimental CPU Offload

For very tight VRAM budgets, set:

```env
CPU_OFFLOAD=true
MAX_VRAM_GB=3
CPU_OFFLOAD_GB=8
AUDIO_TOKENIZER_DEVICE=cpu
```

or use the dedicated mixed CPU/GPU offload compose profile:

```bash
./start-mixed-offload.sh -d
```

This uses Hugging Face Accelerate with `device_map="auto"` and `max_memory`,
similar in spirit to layer offload. It can reduce resident GPU memory, but it is
slower. The mixed-offload compose profile still uses `LM_QUANT=nf4`, because
unquantized weights are usually worse for this project than pure CPU offload.

The mixed-offload compose profile also defaults `AUDIO_TOKENIZER_DEVICE=cpu`.
That may matter more than LLM offload for OmniVoice because the upstream loader
creates the audio tokenizer separately after loading the main model.

## Optional STT

STT is optional and intentionally lives in a separate project:
[stt-nano-webui](https://github.com/Wladastic/stt-nano-webui).
This OmniVoice repo only needs an OpenAI-compatible transcription endpoint at:

```text
POST /v1/audio/transcriptions
```

If `STT_URL` is empty, transcription is disabled and auto-trim falls back to
silence detection.

For a local STT service on the host:

```env
STT_URL=http://localhost:8882
```

For a service on another machine, set that URL only in your local `.env`.

The companion `stt-nano-webui` project provides:

- WebUI: `http://localhost:7861`
- backend API: `http://localhost:8882`
- OpenAI-style transcription: `POST /v1/audio/transcriptions`
- lightweight default model: `parakeet-onnx-int8`
- optional `whisper-1` alias for OpenAI-compatible clients

Together, `omnivoice-tts-nano-webui` and `stt-nano-webui` can be used as local
speech services for tools such as OpenWebUI: this repo covers TTS/voice cloning,
while `stt-nano-webui` covers speech-to-text.

## Voice Design Tags

OmniVoice voice design does not accept free-form prompts. Use comma-separated
speaker tags, for example:

```text
female, young adult, low pitch, british accent
```

Supported English tags include:

```text
american accent, australian accent, british accent, canadian accent, child,
chinese accent, elderly, female, high pitch, indian accent, japanese accent,
korean accent, low pitch, male, middle-aged, moderate pitch, portuguese accent,
russian accent, teenager, very high pitch, very low pitch, whisper, young adult
```

## Local Data

These directories/files are intentionally ignored:

- `.env`
- `logs/`
- `models/`
- `voices/`
- `__pycache__/`

This keeps generated audio, model cache, logs, and private configuration out of git.
