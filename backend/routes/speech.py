import io
import logging
import os
import inspect
import tempfile
import time
import re
from typing import Iterator, Optional

import numpy as np
from fastapi import APIRouter, HTTPException, Request, UploadFile
from fastapi.responses import StreamingResponse
from omnivoice import OmniVoiceGenerationConfig
from pydub import AudioSegment

from model_manager import get_model
from model_manager import reset_peak_vram
from routes.tts import (
    _audio_to_wav_bytes,
    _build_kwargs,
    _chunk_text,
    _free_cuda_cache,
    CHUNK_CHARS,
    CHUNK_GAP_MS,
)
from voice_store import list_voices as list_saved_voices, load_voice
from config import ASR_MODEL

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/v1", tags=["openai-compat"])

TURBO_STEPS = int(os.environ.get("TURBO_STEPS", "4"))
DYNAMIC_STEPS_MIN = int(os.environ.get("DYNAMIC_STEPS_MIN", "4"))
DYNAMIC_STEPS_MAX = int(os.environ.get("DYNAMIC_STEPS_MAX", "64"))
DYNAMIC_DESIRED_SPEED = float(os.environ.get("DYNAMIC_DESIRED_SPEED", "4.0"))
DYNAMIC_STEP_DELTA = int(os.environ.get("DYNAMIC_STEP_DELTA", "4"))

# tts-1 -> fast, tts-1-hd -> quality
MODEL_STEPS = {
    "turbo": TURBO_STEPS,
    "tts-1": 16,
    "tts-1-hd": 32,
    "gpt-4o-mini-tts": 32,
    "gpt-4o-tts": 32,
}
DEFAULT_STEPS = 16
SUPPORTED_RESPONSE_FORMATS = {"wav", "pcm", "mp3", "opus"}


def _truthy(value) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


DEFAULT_CLEAN_MARKDOWN = _truthy(os.environ.get("CLEAN_MARKDOWN", "true"))
DEFAULT_STRIP_EMOJI = _truthy(os.environ.get("STRIP_EMOJI", "true"))
DEFAULT_NORMALIZE_SPOKEN_TEXT = _truthy(os.environ.get("NORMALIZE_SPOKEN_TEXT", "false"))
DEFAULT_SPOKEN_TEXT_LANGUAGE = os.environ.get("SPOKEN_TEXT_LANGUAGE", "de").strip().lower() or "de"
PARAGRAPH_PAUSE_MS = int(os.environ.get("PARAGRAPH_PAUSE_MS", "200"))


def _float_value(data: dict, key: str, default: float) -> float:
    value = data.get(key)
    if value is None or value == "":
        return default
    return float(value)


def _int_value(data: dict, key: str, default: int) -> int:
    value = data.get(key)
    if value is None or value == "":
        return default
    return int(value)


def _clamp_int(value: int, minimum: int, maximum: int) -> int:
    return max(minimum, min(maximum, value))


def _dynamic_step_state(data: dict, model_name: str, base_config: OmniVoiceGenerationConfig) -> Optional[dict]:
    dynamic_enabled = (
        model_name == "dynamic"
        or _truthy(data.get("dynamic_steps"))
        or _truthy(data.get("dynamic_inference_steps"))
    )
    if not dynamic_enabled:
        return None

    min_steps = _int_value(data, "dynamic_min_steps", DYNAMIC_STEPS_MIN)
    max_steps = _int_value(data, "dynamic_max_steps", DYNAMIC_STEPS_MAX)
    min_steps = max(1, min_steps)
    max_steps = max(1, max_steps)
    if min_steps > max_steps:
        min_steps, max_steps = max_steps, min_steps

    desired_speed = max(0.1, _float_value(data, "dynamic_desired_speed", DYNAMIC_DESIRED_SPEED))
    step_delta = max(1, _int_value(data, "dynamic_step_delta", DYNAMIC_STEP_DELTA))
    current_steps = _clamp_int(_int_value(data, "dynamic_start_steps", max_steps), min_steps, max_steps)

    return {
        "base_config": base_config,
        "current_steps": current_steps,
        "min_steps": min_steps,
        "max_steps": max_steps,
        "desired_speed": desired_speed,
        "step_delta": step_delta,
    }


def _config_with_steps(base_config: OmniVoiceGenerationConfig, num_step: int) -> OmniVoiceGenerationConfig:
    return OmniVoiceGenerationConfig(
        num_step=num_step,
        guidance_scale=base_config.guidance_scale,
        denoise=base_config.denoise,
        preprocess_prompt=base_config.preprocess_prompt,
        postprocess_output=base_config.postprocess_output,
    )


def _kwargs_for_chunk(kw_base: dict, chunk: str, dynamic_state: Optional[dict] = None) -> dict:
    kw = dict(kw_base, text=chunk)
    if dynamic_state is not None:
        kw["generation_config"] = _config_with_steps(
            dynamic_state["base_config"],
            dynamic_state["current_steps"],
        )
    return kw


def _chunk_steps(kw_base: dict, dynamic_state: Optional[dict] = None) -> int:
    if dynamic_state is not None:
        return dynamic_state["current_steps"]
    config = kw_base.get("generation_config")
    return getattr(config, "num_step", 0)


def _update_dynamic_steps(dynamic_state: Optional[dict], audio: np.ndarray, sampling_rate: int, elapsed_s: float) -> None:
    if dynamic_state is None or elapsed_s <= 0:
        return

    audio_s = len(audio) / sampling_rate
    if audio_s <= 0:
        return

    generation_speed = audio_s / elapsed_s
    old_steps = dynamic_state["current_steps"]
    new_steps = old_steps
    if generation_speed < dynamic_state["desired_speed"]:
        new_steps = max(dynamic_state["min_steps"], old_steps - dynamic_state["step_delta"])
    elif generation_speed > dynamic_state["desired_speed"] * 1.35:
        new_steps = min(dynamic_state["max_steps"], old_steps + dynamic_state["step_delta"])

    if new_steps != old_steps:
        logger.info(
            "[openai] dynamic steps %d -> %d (chunk %.2fx, target %.2fx)",
            old_steps,
            new_steps,
            generation_speed,
            dynamic_state["desired_speed"],
        )
        dynamic_state["current_steps"] = new_steps
    else:
        logger.debug(
            "[openai] dynamic steps stay %d (chunk %.2fx, target %.2fx)",
            old_steps,
            generation_speed,
            dynamic_state["desired_speed"],
        )


def _is_emoji_char(char: str) -> bool:
    codepoint = ord(char)
    return (
        0x1F000 <= codepoint <= 0x1FAFF
        or 0x2600 <= codepoint <= 0x27BF
        or codepoint in {0x200D, 0xFE0E, 0xFE0F, 0x20E3}
    )


def _strip_emoji_for_tts(text: str) -> str:
    return "".join(" " if _is_emoji_char(char) else char for char in text)


_URL_RE = re.compile(r"\b(?:https?://|www\.)\S+|\b\S+@\S+\.\S+\b", re.IGNORECASE)

_DE_UNITS = [
    "null", "eins", "zwei", "drei", "vier", "fünf", "sechs", "sieben", "acht", "neun",
    "zehn", "elf", "zwölf", "dreizehn", "vierzehn", "fünfzehn", "sechzehn", "siebzehn", "achtzehn", "neunzehn",
]
_DE_TENS = {
    20: "zwanzig", 30: "dreißig", 40: "vierzig", 50: "fünfzig",
    60: "sechzig", 70: "siebzig", 80: "achtzig", 90: "neunzig",
}
_EN_UNITS = [
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
    "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen",
]
_EN_TENS = {
    20: "twenty", 30: "thirty", 40: "forty", 50: "fifty",
    60: "sixty", 70: "seventy", 80: "eighty", 90: "ninety",
}
_DE_MONTHS = [
    "", "Januar", "Februar", "März", "April", "Mai", "Juni",
    "Juli", "August", "September", "Oktober", "November", "Dezember",
]
_EN_MONTHS = [
    "", "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]


def _spoken_language(language: Optional[str]) -> str:
    value = (language or DEFAULT_SPOKEN_TEXT_LANGUAGE).strip().lower()
    if value in {"de", "ger", "german", "deutsch"} or value.startswith("german"):
        return "de"
    if value in {"en", "eng", "english"} or value.startswith("english"):
        return "en"
    return DEFAULT_SPOKEN_TEXT_LANGUAGE if DEFAULT_SPOKEN_TEXT_LANGUAGE in {"de", "en"} else "de"


def _de_number(n: int) -> str:
    if n < 0:
        return "minus " + _de_number(abs(n))
    if n < 20:
        return _DE_UNITS[n]
    if n < 100:
        tens = n // 10 * 10
        rest = n % 10
        if rest == 0:
            return _DE_TENS[tens]
        unit = "ein" if rest == 1 else _DE_UNITS[rest]
        return unit + "und" + _DE_TENS[tens]
    if n < 1000:
        hundreds = n // 100
        rest = n % 100
        prefix = ("ein" if hundreds == 1 else _DE_UNITS[hundreds]) + "hundert"
        return prefix if rest == 0 else prefix + _de_number(rest)
    if n < 1_000_000:
        thousands = n // 1000
        rest = n % 1000
        prefix = ("ein" if thousands == 1 else _de_number(thousands)) + "tausend"
        return prefix if rest == 0 else prefix + _de_number(rest)
    return f"{n:,}".replace(",", " ")


def _en_number(n: int) -> str:
    if n < 0:
        return "minus " + _en_number(abs(n))
    if n < 20:
        return _EN_UNITS[n]
    if n < 100:
        tens = n // 10 * 10
        rest = n % 10
        return _EN_TENS[tens] if rest == 0 else f"{_EN_TENS[tens]} {_EN_UNITS[rest]}"
    if n < 1000:
        hundreds = n // 100
        rest = n % 100
        prefix = _EN_UNITS[hundreds] + " hundred"
        return prefix if rest == 0 else prefix + " " + _en_number(rest)
    if n < 1_000_000:
        thousands = n // 1000
        rest = n % 1000
        prefix = _en_number(thousands) + " thousand"
        return prefix if rest == 0 else prefix + " " + _en_number(rest)
    return f"{n:,}".replace(",", " ")


def _number_words(n: int, lang: str) -> str:
    return _en_number(n) if lang == "en" else _de_number(n)


def _ordinal_words(n: int, lang: str) -> str:
    if lang == "en":
        special = {1: "first", 2: "second", 3: "third", 4: "fourth", 5: "fifth", 8: "eighth", 9: "ninth", 12: "twelfth"}
        if n in special:
            return special[n]
        return _en_number(n) + "th"
    special = {1: "ersten", 3: "dritten", 7: "siebten", 8: "achten"}
    if n in special:
        return special[n]
    return _de_number(n) + "ten" if n < 20 else _de_number(n) + "sten"


def _normalize_time(match: re.Match, lang: str) -> str:
    hour = int(match.group(1))
    minute = int(match.group(2))
    if hour > 23 or minute > 59:
        return match.group(0)
    if lang == "en":
        if minute == 0:
            return f"{_number_words(hour, lang)} o'clock"
        minute_words = "oh " + _number_words(minute, lang) if minute < 10 else _number_words(minute, lang)
        return f"{_number_words(hour, lang)} {minute_words}"
    if minute == 0:
        return f"{_number_words(hour, lang)} Uhr"
    minute_words = "null " + _number_words(minute, lang) if minute < 10 else _number_words(minute, lang)
    return f"{_number_words(hour, lang)} Uhr {minute_words}"


def _normalize_date(match: re.Match, lang: str) -> str:
    if match.group("iso"):
        year = int(match.group("year"))
        month = int(match.group("month"))
        day = int(match.group("day"))
    else:
        day = int(match.group("day2"))
        month = int(match.group("month2"))
        year = int(match.group("year2")) if match.group("year2") else None
    if not (1 <= day <= 31 and 1 <= month <= 12):
        return match.group(0)
    if lang == "en":
        result = f"{_EN_MONTHS[month]} {_ordinal_words(day, lang)}"
        return result if year is None else f"{result} {_number_words(year, lang)}"
    result = f"{_ordinal_words(day, lang)} {_DE_MONTHS[month]}"
    return result if year is None else f"{result} {_number_words(year, lang)}"


def _normalize_number(match: re.Match, lang: str) -> str:
    token = match.group(0)
    try:
        return _number_words(int(token.replace(".", "").replace(",", "")), lang)
    except ValueError:
        return token


def _normalize_spoken_text_for_tts(text: str, language: Optional[str]) -> str:
    lang = _spoken_language(language)
    text = _URL_RE.sub(" ", text)
    text = re.sub(
        r"(?P<iso>(?P<year>\d{4})-(?P<month>\d{1,2})-(?P<day>\d{1,2}))|(?P<day2>\d{1,2})\.(?P<month2>\d{1,2})\.(?P<year2>\d{2,4})?",
        lambda match: _normalize_date(match, lang),
        text,
    )
    text = re.sub(r"\b([01]?\d|2[0-3]):([0-5]\d)\b", lambda match: _normalize_time(match, lang), text)
    text = re.sub(r"(?<![\w.-])-?\d{1,6}(?![\w.-])", lambda match: _normalize_number(match, lang), text)
    return text


def _clean_markdown_paragraph(text: str) -> str:
    text = re.sub(r"```[\s\S]*?```", " ", text)
    text = re.sub(r"`([^`]+)`", r"\1", text)
    text = re.sub(r"!\[([^\]]*)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"^\s{0,3}#{1,6}\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s{0,3}>\s?", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*[-*+]\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*\d+[.)]\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*[-*_]{3,}\s*$", " ", text, flags=re.MULTILINE)
    text = re.sub(r"[*_~]{1,3}([^*_~]+)[*_~]{1,3}", r"\1", text)
    text = text.replace(":", ".")
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+([,.!?;。！？])", r"\1", text)
    return text.strip()


def _ensure_terminal_punctuation(text: str) -> str:
    if not text:
        return text
    if re.search(r"[.!?。！？…]$", text):
        return text
    return text + "."


def _clean_markdown_for_tts(text: str) -> str:
    return " ".join(_clean_markdown_paragraphs(text))


def _clean_markdown_paragraphs(text: str) -> list[str]:
    text = re.sub(r"\r\n?", "\n", text)
    paragraphs = re.split(r"\n\s*\n+", text)
    cleaned = [_clean_markdown_paragraph(p) for p in paragraphs]
    return [_ensure_terminal_punctuation(p) for p in cleaned if p]


def _text_chunks_for_speech(
    text: str,
    clean_markdown: bool,
    strip_emoji: bool = True,
    normalize_spoken_text: bool = False,
    language: Optional[str] = None,
) -> list[Optional[str]]:
    if strip_emoji:
        text = _strip_emoji_for_tts(text)
    if normalize_spoken_text:
        text = _normalize_spoken_text_for_tts(text, language)

    if clean_markdown:
        paragraphs = _clean_markdown_paragraphs(text)
    else:
        paragraphs = [text.strip()]

    chunks: list[Optional[str]] = []
    for paragraph in paragraphs:
        if chunks and PARAGRAPH_PAUSE_MS > 0:
            chunks.append(None)
        chunks.extend(_chunk_text(paragraph, CHUNK_CHARS))
    return chunks


def _audio_to_pcm_bytes(audio: np.ndarray) -> bytes:
    waveform = np.clip(audio, -1.0, 1.0)
    return (waveform * 32767).astype(np.int16).tobytes()


def _wav_stream_header(sampling_rate: int) -> bytes:
    # Chunked HTTP has no fixed length up front, so use the largest valid RIFF sizes.
    # Most clients accept this for live-ish WAV streams.
    byte_rate = sampling_rate * 2
    block_align = 2
    return (
        b"RIFF"
        + (0xFFFFFFFF).to_bytes(4, "little")
        + b"WAVEfmt "
        + (16).to_bytes(4, "little")
        + (1).to_bytes(2, "little")
        + (1).to_bytes(2, "little")
        + sampling_rate.to_bytes(4, "little")
        + byte_rate.to_bytes(4, "little")
        + block_align.to_bytes(2, "little")
        + (16).to_bytes(2, "little")
        + b"data"
        + (0xFFFFFFFF).to_bytes(4, "little")
    )


def _format_audio_response(wav_bytes: bytes, response_format: str) -> tuple[bytes, str]:
    if response_format == "wav":
        return wav_bytes, "audio/wav"
    if response_format == "pcm":
        audio = AudioSegment.from_file(io.BytesIO(wav_bytes), format="wav")
        return audio.raw_data, "audio/pcm"
    if response_format == "mp3":
        audio = AudioSegment.from_file(io.BytesIO(wav_bytes), format="wav")
        buf = io.BytesIO()
        audio.export(buf, format="mp3")
        buf.seek(0)
        return buf.read(), "audio/mpeg"
    if response_format == "opus":
        audio = AudioSegment.from_file(io.BytesIO(wav_bytes), format="wav")
        buf = io.BytesIO()
        audio.export(buf, format="opus", codec="libopus")
        buf.seek(0)
        return buf.read(), "audio/opus"
    raise HTTPException(400, f"Unsupported response_format={response_format!r}")


def _stream_media_type(response_format: str) -> str:
    if response_format == "mp3":
        return "audio/mpeg"
    if response_format == "opus":
        return "audio/opus"
    if response_format == "pcm":
        return "audio/pcm"
    return "audio/wav"


def _encode_stream_chunk(audio: np.ndarray, sampling_rate: int, response_format: str) -> bytes:
    if response_format == "pcm":
        return _audio_to_pcm_bytes(audio)
    if response_format == "mp3":
        wav_bytes = _audio_to_wav_bytes(audio, sampling_rate)
        segment = AudioSegment.from_file(io.BytesIO(wav_bytes), format="wav")
        buf = io.BytesIO()
        segment.export(buf, format="mp3")
        buf.seek(0)
        return buf.read()
    if response_format == "opus":
        wav_bytes = _audio_to_wav_bytes(audio, sampling_rate)
        segment = AudioSegment.from_file(io.BytesIO(wav_bytes), format="wav")
        buf = io.BytesIO()
        segment.export(buf, format="opus", codec="libopus")
        buf.seek(0)
        return buf.read()
    return _audio_to_pcm_bytes(audio)


def _create_voice_clone_prompt(model, ref_audio, ref_text=None, x_vector_only_mode=False):
    create_prompt = model.create_voice_clone_prompt
    if x_vector_only_mode:
        logger.info("[openai] voice conditioning: speaker embedding only requested")
        try:
            sig = inspect.signature(create_prompt)
        except (TypeError, ValueError):
            sig = None

        if sig is None or "x_vector_only_mode" in sig.parameters:
            logger.info("[openai] voice conditioning: using native x_vector_only_mode")
            return create_prompt(
                ref_audio=ref_audio,
                ref_text=ref_text,
                x_vector_only_mode=True,
            )

        prompt = create_prompt(ref_audio=ref_audio, ref_text=ref_text)
        items = prompt if isinstance(prompt, list) else [prompt]
        embeddings = [
            getattr(item, "ref_spk_embedding")
            for item in items
            if getattr(item, "ref_spk_embedding", None) is not None
        ]
        if embeddings:
            logger.warning(
                "create_voice_clone_prompt does not expose x_vector_only_mode; "
                "using extracted speaker embeddings only"
            )
            return {"ref_spk_embedding": embeddings}
        logger.warning(
            "create_voice_clone_prompt does not expose x_vector_only_mode and "
            "returned no speaker embeddings; falling back to full prompt"
        )
        return prompt

    logger.info("[openai] voice conditioning: full reference prompt")
    return create_prompt(ref_audio=ref_audio, ref_text=ref_text)


def _generate_audio_stream(
    model,
    kw_base: dict,
    chunks: list[Optional[str]],
    response_format: str,
    cleanup_path: Optional[str] = None,
    dynamic_state: Optional[dict] = None,
) -> Iterator[bytes]:
    gap = np.zeros(int(model.sampling_rate * CHUNK_GAP_MS / 1000), dtype=np.float32)
    paragraph_pause = np.zeros(int(model.sampling_rate * PARAGRAPH_PAUSE_MS / 1000), dtype=np.float32)
    text_chunks = [chunk for chunk in chunks if chunk is not None]
    logger.info(
        "[openai] streaming %d text chunk%s with %d paragraph pause%s",
        len(text_chunks),
        "" if len(text_chunks) == 1 else "s",
        len(chunks) - len(text_chunks),
        "" if len(chunks) - len(text_chunks) == 1 else "s",
    )

    try:
        if response_format == "wav":
            yield _wav_stream_header(model.sampling_rate)

        need_gap = False
        text_i = 0
        for chunk in chunks:
            if chunk is None:
                if PARAGRAPH_PAUSE_MS > 0:
                    yield _encode_stream_chunk(paragraph_pause, model.sampling_rate, response_format)
                need_gap = False
                continue

            text_i += 1
            logger.debug("[openai] stream chunk %d/%d: %r", text_i, len(text_chunks), chunk)
            if need_gap and CHUNK_GAP_MS > 0:
                yield _encode_stream_chunk(gap, model.sampling_rate, response_format)

            logger.info(
                "[openai] stream chunk %d/%d using %d inference step%s%s",
                text_i,
                len(text_chunks),
                _chunk_steps(kw_base, dynamic_state),
                "" if _chunk_steps(kw_base, dynamic_state) == 1 else "s",
                " (dynamic)" if dynamic_state is not None else " (fixed)",
            )
            started_at = time.monotonic()
            out = model.generate(**_kwargs_for_chunk(kw_base, chunk, dynamic_state))
            _update_dynamic_steps(dynamic_state, out[0], model.sampling_rate, time.monotonic() - started_at)
            yield _encode_stream_chunk(out[0], model.sampling_rate, response_format)
            need_gap = True
    except Exception:
        logger.exception("OpenAI speech stream generation failed")
        raise
    finally:
        if cleanup_path:
            try:
                os.unlink(cleanup_path)
            except FileNotFoundError:
                pass
        _free_cuda_cache()


def _generate_audio_sequence(
    model,
    kw_base: dict,
    chunks: list[Optional[str]],
    dynamic_state: Optional[dict] = None,
) -> tuple[np.ndarray, int]:
    gap = np.zeros(int(model.sampling_rate * CHUNK_GAP_MS / 1000), dtype=np.float32)
    paragraph_pause = np.zeros(int(model.sampling_rate * PARAGRAPH_PAUSE_MS / 1000), dtype=np.float32)
    parts = []
    n_text_chunks = 0
    need_gap = False

    for chunk in chunks:
        if chunk is None:
            if PARAGRAPH_PAUSE_MS > 0:
                parts.append(paragraph_pause)
            need_gap = False
            continue

        if need_gap and CHUNK_GAP_MS > 0:
            parts.append(gap)
        started_at = time.monotonic()
        logger.info(
            "[openai] sequence chunk %d using %d inference step%s%s",
            n_text_chunks + 1,
            _chunk_steps(kw_base, dynamic_state),
            "" if _chunk_steps(kw_base, dynamic_state) == 1 else "s",
            " (dynamic)" if dynamic_state is not None else " (fixed)",
        )
        out = model.generate(**_kwargs_for_chunk(kw_base, chunk, dynamic_state))
        _update_dynamic_steps(dynamic_state, out[0], model.sampling_rate, time.monotonic() - started_at)
        parts.append(out[0])
        n_text_chunks += 1
        need_gap = True

    if not parts:
        return np.array([], dtype=np.float32), 0
    return np.concatenate(parts), n_text_chunks


@router.post("/audio/speech")
async def openai_speech(request: Request):
    content_type = request.headers.get("content-type", "")
    ref_audio: Optional[UploadFile] = None

    if "multipart/form-data" in content_type or "application/x-www-form-urlencoded" in content_type:
        form = await request.form()
        data = dict(form)
        uploaded = data.get("ref_audio")
        if isinstance(uploaded, UploadFile) or hasattr(uploaded, "read"):
            ref_audio = uploaded
    else:
        try:
            data = await request.json()
        except Exception:
            raise HTTPException(400, "Expected JSON or form data body")

    model = str(data.get("model") or "tts-1-hd")
    input = str(data.get("input") or "").strip()
    if not input:
        raise HTTPException(422, "Field 'input' is required")
    response_format = str(data.get("response_format") or "wav").lower()
    if response_format not in SUPPORTED_RESPONSE_FORMATS:
        raise HTTPException(
            400,
            "Supported response_format values are: "
            + ", ".join(sorted(SUPPORTED_RESPONSE_FORMATS)),
        )
    speed = float(data.get("speed") or 1.0)
    voice = data.get("voice")
    language = data.get("language")
    instruct = data.get("instruct")
    guidance_scale = float(data.get("guidance_scale") or 2.0)
    ref_text = data.get("ref_text")
    stream = _truthy(data.get("stream"))
    x_vector_only_mode = _truthy(data.get("x_vector_only_mode"))
    clean_markdown = _truthy(data.get("clean_markdown", DEFAULT_CLEAN_MARKDOWN))
    strip_emoji = _truthy(data.get("strip_emoji", DEFAULT_STRIP_EMOJI))
    normalize_spoken_text = _truthy(data.get("normalize_spoken_text", DEFAULT_NORMALIZE_SPOKEN_TEXT))
    text_chunks = _text_chunks_for_speech(input, clean_markdown, strip_emoji, normalize_spoken_text, language)
    if not any(chunk for chunk in text_chunks if chunk is not None):
        raise HTTPException(422, "Input contains no speakable text")
    input = " ".join(chunk for chunk in text_chunks if chunk is not None)

    requested_model = model
    if model.startswith("voice:"):
        voice = model.split(":", 1)[1]
        model = "tts-1-hd"

    dynamic_enabled = (
        model == "dynamic"
        or _truthy(data.get("dynamic_steps"))
        or _truthy(data.get("dynamic_inference_steps"))
    )
    if dynamic_enabled:
        num_step = _int_value(data, "num_step", DYNAMIC_STEPS_MAX)
    else:
        num_step = _int_value(data, "num_step", MODEL_STEPS.get(model, DEFAULT_STEPS))
    logger.info(
        "[openai] request model=%r voice=%r response_format=%r stream=%s text_len=%d "
        "dynamic=%s dynamic_raw=%r x_vector_only=%s strip_emoji=%s normalize_spoken_text=%s raw=%r",
        requested_model,
        voice,
        response_format,
        stream,
        len(input),
        dynamic_enabled,
        data.get("dynamic_steps") or data.get("dynamic_inference_steps"),
        x_vector_only_mode,
        strip_emoji,
        normalize_spoken_text,
        data.get("x_vector_only_mode"),
    )
    reset_peak_vram()
    omni = get_model()

    class _Req:
        pass
    req = _Req()
    req.text = input
    req.language = language
    req.speed = speed
    req.duration = _float_value(data, "duration", 0.0)
    req.num_step = num_step
    req.guidance_scale = guidance_scale
    req.denoise = _truthy(data.get("denoise", True))
    req.preprocess_prompt = _truthy(data.get("preprocess_prompt", True))
    req.postprocess_output = _truthy(data.get("postprocess_output", True))

    gen_config = OmniVoiceGenerationConfig(
        num_step=num_step,
        guidance_scale=guidance_scale,
        denoise=req.denoise,
        preprocess_prompt=req.preprocess_prompt,
        postprocess_output=req.postprocess_output,
    )
    dynamic_state = _dynamic_step_state(data, model, gen_config)
    if dynamic_state is not None:
        req.num_step = dynamic_state["current_steps"]
        gen_config = _config_with_steps(gen_config, dynamic_state["current_steps"])
        logger.info(
            "[openai] dynamic steps enabled min=%d max=%d current=%d target=%.2fx",
            dynamic_state["min_steps"],
            dynamic_state["max_steps"],
            dynamic_state["current_steps"],
            dynamic_state["desired_speed"],
        )
    else:
        logger.info("[openai] dynamic steps disabled; fixed inference steps=%d", req.num_step)
    kw = _build_kwargs(req, gen_config)

    tmp_path = None
    try:
        if ref_audio is not None:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp.write(ref_audio.file.read())
                tmp_path = tmp.name
            try:
                kw["voice_clone_prompt"] = _create_voice_clone_prompt(
                    omni,
                    ref_audio=tmp_path,
                    ref_text=ref_text or None,
                    x_vector_only_mode=x_vector_only_mode,
                )
            except Exception as e:
                raise HTTPException(400, f"Failed to create voice clone prompt: {e}")
        elif voice and str(voice).strip() and str(voice).strip() not in ("alloy", "echo", "fable", "onyx", "nova", "shimmer"):
            voice_id = str(voice).strip()
            saved_voice = load_voice(voice_id)
            if saved_voice is None:
                raise HTTPException(404, f"Voice '{voice_id}' not found")
            ref_audio_path = saved_voice.get("ref_audio_path")
            if not ref_audio_path:
                raise HTTPException(400, f"Voice '{voice_id}' has no reference audio")
            try:
                kw["voice_clone_prompt"] = _create_voice_clone_prompt(
                    omni,
                    ref_audio=ref_audio_path,
                    ref_text=ref_text or saved_voice.get("ref_text"),
                    x_vector_only_mode=x_vector_only_mode,
                )
                logger.info("[openai] using saved voice %s", voice_id)
            except Exception as e:
                raise HTTPException(400, f"Failed to load voice '{voice_id}': {e}")

        if instruct and str(instruct).strip():
            kw["instruct"] = str(instruct).strip()

        if stream:
            cleanup_path = tmp_path
            tmp_path = None
            return StreamingResponse(
                _generate_audio_stream(omni, kw, text_chunks, response_format, cleanup_path, dynamic_state),
                media_type=_stream_media_type(response_format),
            )

        try:
            audio, n_chunks = _generate_audio_sequence(omni, kw, text_chunks, dynamic_state)
        except Exception as e:
            logger.exception("OpenAI speech generation failed")
            raise HTTPException(500, f"Generation failed: {e}")

    finally:
        if tmp_path:
            os.unlink(tmp_path)
        _free_cuda_cache()

    logger.info(f"[openai] {model} done — {n_chunks} chunk{'s' if n_chunks != 1 else ''}")
    wav_bytes = _audio_to_wav_bytes(audio, omni.sampling_rate)
    response_bytes, media_type = _format_audio_response(wav_bytes, response_format)
    return StreamingResponse(io.BytesIO(response_bytes), media_type=media_type)


@router.get("/models")
def list_models():
    voice_models = [
        {
            "id": f"voice:{voice['id']}",
            "object": "model",
            "description": f"OmniVoice saved voice: {voice.get('name', voice['id'])}",
        }
        for voice in list_saved_voices()
        if voice.get("has_audio")
    ]
    return {
        "object": "list",
        "data": [
            {"id": "parakeet-tdt-0.6b-v3", "object": "model", "owned_by": "nvidia", "source": ASR_MODEL},
            {"id": "turbo", "object": "model", "description": f"OmniVoice turbo ({TURBO_STEPS} steps)"},
            {"id": "dynamic", "object": "model", "description": "OmniVoice adaptive inference steps"},
            {"id": "tts-1", "object": "model", "description": "OmniVoice fast (16 steps)"},
            {"id": "tts-1-hd", "object": "model", "description": "OmniVoice quality (32 steps)"},
            {"id": "gpt-4o-mini-tts", "object": "model", "description": "OmniVoice quality (32 steps)"},
            {"id": "gpt-4o-tts", "object": "model", "description": "OmniVoice quality (32 steps)"},
        ] + voice_models,
    }
