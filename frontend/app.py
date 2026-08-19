import os
import re
import tempfile
import time
from datetime import datetime
from pathlib import Path

import gradio as gr
import httpx
import librosa
import numpy as np
import soundfile as sf

from i18n import t

API_URL = os.environ.get("API_URL", "http://localhost:8883")
STT_URL = (os.environ.get("STT_URL", "").strip() or API_URL).rstrip("/")
FRONTEND_PORT = int(os.environ.get("FRONTEND_PORT", "7861"))
STREAM_CHUNK_SIZE = int(os.environ.get("STREAM_CHUNK_SIZE", "65536"))
STREAM_SAMPLE_RATE = int(os.environ.get("STREAM_SAMPLE_RATE", "24000"))
STREAM_START_BUFFER_SECONDS = float(os.environ.get("STREAM_START_BUFFER_SECONDS", "4.0"))
OUTPUTS_DIR = Path(os.environ.get("OUTPUTS_DIR", "outputs"))

LANGUAGES = [
    "Auto", "English", "German", "French", "Spanish", "Italian", "Portuguese",
    "Dutch", "Polish", "Russian", "Chinese", "Japanese", "Korean", "Arabic",
    "Hindi", "Turkish", "Swedish", "Norwegian", "Danish", "Finnish",
]

MAX_REF_SECONDS = 3.5
TRIM_TRAILING_SILENCE_SECONDS = 0.15
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)


def _save_response_wav(response_bytes: bytes) -> str:
    fd, path = tempfile.mkstemp(suffix=".wav")
    with os.fdopen(fd, "wb") as f:
        f.write(response_bytes)
    return path


def _safe_output_name(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip().lower())
    return value.strip("-") or "tts"


def _save_output_audio(samples: np.ndarray, label: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    path = OUTPUTS_DIR / f"{timestamp}-{_safe_output_name(label)}.wav"
    sf.write(path, samples.astype(np.int16), STREAM_SAMPLE_RATE, subtype="PCM_16", format="WAV")
    os.chmod(path, 0o644)
    return str(path)


def _done_status(output_path: str | None = None) -> str:
    if output_path:
        return f"{t('done')} {t('saved_to', path=output_path)}"
    return t("done")


def _stt_url() -> str | None:
    return STT_URL or None


def _status_from_response(r) -> str:
    peak = r.headers.get("X-Peak-VRAM-GB")
    secs = r.headers.get("X-Total-Time-S")
    extras = []
    if secs:
        extras.append(f"{secs}s")
    if peak:
        extras.append(f"peak {peak}GB VRAM")
    return f"{t('done')} ({', '.join(extras)})" if extras else t("done")


def set_audio_autoplay(enabled):
    return gr.update(autoplay=bool(enabled))


def cancel_status():
    return t("cancelled")


def _post_wav(endpoint: str, payload: dict) -> tuple:
    try:
        r = httpx.post(f"{API_URL}{endpoint}", json=payload, timeout=120)
        r.raise_for_status()
        return _save_response_wav(r.content), _status_from_response(r)
    except httpx.HTTPStatusError as e:
        return None, f"{t('error_prefix')} {e.response.status_code}: {e.response.text}"
    except Exception as e:
        return None, f"{t('error_prefix')}: {e}"


def _stream_status(total_bytes: int) -> str:
    return f"{t('streaming')} ({total_bytes // 1024} KiB)"


def _empty_metrics():
    return ""


def _current_peak_vram() -> float | None:
    try:
        r = httpx.get(f"{API_URL}/models/status", timeout=5)
        r.raise_for_status()
        value = r.json().get("peak_vram_gb")
        return float(value) if value is not None else None
    except Exception:
        return None


CHARS_PER_AUDIO_SECOND = float(os.environ.get("CHARS_PER_AUDIO_SECOND", "14.0"))


def _format_eta(seconds: float) -> str:
    seconds = max(0.0, seconds)
    if seconds >= 60:
        m, s = divmod(int(round(seconds)), 60)
        return f"{m}m{s:02d}s"
    return f"{seconds:.1f}s"


def _progress_estimate(generated_s: float, elapsed: float, text_length: int, done: bool):
    if text_length <= 0:
        return None
    estimated_total_s = max(1.0, text_length / max(1.0, CHARS_PER_AUDIO_SECOND))
    if done:
        return 1.0, 0.0
    progress = generated_s / estimated_total_s
    progress = min(0.99, max(0.0, progress))
    ratio = generated_s / elapsed if elapsed > 0 else 0.0
    if ratio > 0 and progress < 1.0:
        eta = (estimated_total_s - generated_s) / ratio
    else:
        eta = 0.0
    return progress, eta


def _metrics_html(
    generated_samples: int,
    started_at: float,
    peak_vram: float | None = None,
    text_length: int = 0,
    done: bool = False,
) -> str:
    elapsed = max(0.001, time.monotonic() - started_at)
    generated_s = generated_samples / STREAM_SAMPLE_RATE
    ratio = generated_s / elapsed
    if ratio >= 1.25:
        color = "#16a34a"
        label = "faster than playback"
    elif ratio >= 0.90:
        color = "#ca8a04"
        label = "near realtime"
    else:
        color = "#6b7280"
        label = "behind playback"
    peak = f" · peak {peak_vram:.2f}GB VRAM" if peak_vram is not None else ""

    progress_block = ""
    est = _progress_estimate(generated_s, elapsed, text_length, done)
    if est is not None:
        progress, eta = est
        pct = progress * 100.0
        bar_color = "#16a34a" if done else "#3b82f6"
        eta_text = "done" if done else f"ETA {_format_eta(eta)}"
        progress_block = (
            "<div style='margin-top:0.35rem;'>"
            "<div style='background:#e5e7eb;border-radius:999px;height:0.55rem;overflow:hidden;'>"
            f"<div style='width:{pct:.1f}%;background:{bar_color};height:100%;transition:width 0.2s;'></div>"
            "</div>"
            "<div style='font-size:0.82rem;color:#374151;margin-top:0.2rem;'>"
            f"<strong>{pct:.1f}%</strong> · {eta_text}"
            "</div>"
            "</div>"
        )

    return (
        "<div style='font-family: system-ui, sans-serif; font-size: 0.92rem;'>"
        f"<span style='display:inline-block;width:0.75rem;height:0.75rem;border-radius:999px;background:{color};margin-right:0.45rem;'></span>"
        f"<strong>{ratio:.2f}x</strong> {label}"
        f" · audio {generated_s:.1f}s · elapsed {elapsed:.1f}s{peak}"
        f"{progress_block}"
        "</div>"
    )


def _buffer_status(samples: int) -> str:
    seconds = samples / STREAM_SAMPLE_RATE
    return f"{t('buffering')} ({seconds:.1f}s)"


def _iter_buffered_audio(samples, state):
    if samples.size == 0:
        return []

    state["generated_samples"] += samples.size

    if not state["started"]:
        state["buffered"].append(samples.copy())
        state["buffered_samples"] += samples.size
        if state["buffered_samples"] < state["start_samples"]:
            return [(
                gr.skip(),
                _buffer_status(state["buffered_samples"]),
                _metrics_html(state["generated_samples"], state["started_at"], text_length=state["text_length"]),
            )]
        samples = np.concatenate(state["buffered"])
        state["buffered"] = []
        state["started"] = True

    outputs = []
    if state["pending"] is not None:
        state["saved_parts"].append(state["pending"].copy())
        outputs.append((
            (STREAM_SAMPLE_RATE, state["pending"]),
            _stream_status(state["total_bytes"]),
            _metrics_html(state["generated_samples"], state["started_at"], text_length=state["text_length"]),
        ))
    state["pending"] = samples
    return outputs


def _finish_buffered_audio(state):
    outputs = []
    if state["buffered"]:
        samples = np.concatenate(state["buffered"])
        state["buffered"] = []
        state["started"] = True
        if state["pending"] is not None:
            outputs.append((
                (STREAM_SAMPLE_RATE, state["pending"]),
                _stream_status(state["total_bytes"]),
                _metrics_html(state["generated_samples"], state["started_at"], text_length=state["text_length"]),
            ))
        state["pending"] = samples
    if state["pending"] is not None:
        state["saved_parts"].append(state["pending"].copy())
        saved_path = _save_output_audio(np.concatenate(state["saved_parts"]), state["output_label"])
        peak_vram = _current_peak_vram()
        outputs.append((
            (STREAM_SAMPLE_RATE, state["pending"]),
            _done_status(saved_path),
            _metrics_html(state["generated_samples"], state["started_at"], peak_vram, text_length=state["text_length"], done=True),
        ))
    else:
        outputs.append((gr.skip(), t("done"), _empty_metrics()))
    return outputs


def _new_stream_buffer_state(text_length: int = 0):
    start_samples = int(STREAM_SAMPLE_RATE * STREAM_START_BUFFER_SECONDS)
    return {
        "started": start_samples <= 0,
        "start_samples": start_samples,
        "buffered": [],
        "buffered_samples": 0,
        "pending": None,
        "saved_parts": [],
        "output_label": "tts",
        "generated_samples": 0,
        "started_at": time.monotonic(),
        "total_bytes": 0,
        "text_length": int(text_length or 0),
    }


def _speech_stream_fields(
    text, language, instruct,
    num_step, guidance_scale, denoise, speed, duration,
    preprocess_prompt, postprocess_output,
    speaker_embedding_only=False,
    clean_markdown=True,
    normalize_spoken_text=False,
    stream_audio=True,
    dynamic_steps=False,
    dynamic_min_steps=4,
    dynamic_max_steps=64,
    dynamic_desired_speed=4.0,
):
    fields = {
        "model": "tts-1-hd",
        "input": text,
        "response_format": "pcm",
        "stream": str(bool(stream_audio)).lower(),
        "num_step": str(int(num_step)),
        "guidance_scale": str(float(guidance_scale)),
        "denoise": str(bool(denoise)).lower(),
        "speed": str(float(speed)),
        "duration": str(float(duration)),
        "preprocess_prompt": str(bool(preprocess_prompt)).lower(),
        "postprocess_output": str(bool(postprocess_output)).lower(),
        "clean_markdown": str(bool(clean_markdown)).lower(),
        "normalize_spoken_text": str(bool(normalize_spoken_text)).lower(),
        "x_vector_only_mode": str(bool(speaker_embedding_only)).lower(),
    }
    if dynamic_steps:
        fields["dynamic_steps"] = "true"
        fields["dynamic_min_steps"] = str(int(dynamic_min_steps))
        fields["dynamic_max_steps"] = str(int(dynamic_max_steps))
        fields["dynamic_desired_speed"] = str(float(dynamic_desired_speed))
    if language and language != "Auto":
        fields["language"] = language
    if instruct:
        fields["instruct"] = instruct
    return fields


def _log_speech_request(source: str, payload: dict):
    dynamic_enabled = payload.get("dynamic_steps") == "true" or payload.get("model") == "dynamic"
    print(
        "[frontend] speech request "
        f"source={source} stream={payload.get('stream')} "
        f"steps={payload.get('num_step')} "
        f"dynamic={dynamic_enabled} "
        f"dynamic_min={payload.get('dynamic_min_steps', '-')} "
        f"dynamic_max={payload.get('dynamic_max_steps', '-')} "
        f"dynamic_speed={payload.get('dynamic_desired_speed', '-')} "
        f"speaker_embedding_only={payload.get('x_vector_only_mode')} "
        f"voice={payload.get('voice', '-')}",
        flush=True,
    )


def _pcm_bytes_to_audio(response_bytes: bytes):
    even_len = len(response_bytes) - (len(response_bytes) % 2)
    samples = np.frombuffer(response_bytes[:even_len], dtype="<i2")
    return (STREAM_SAMPLE_RATE, samples.copy())


def _buffered_speech_json(payload: dict, output_label: str):
    started_at = time.monotonic()
    yield gr.skip(), t("streaming"), _empty_metrics()
    try:
        r = httpx.post(f"{API_URL}/v1/audio/speech", json=payload, timeout=None)
        r.raise_for_status()
        audio = _pcm_bytes_to_audio(r.content)
        saved_path = _save_output_audio(audio[1], output_label)
        yield audio, _done_status(saved_path), _metrics_html(len(audio[1]), started_at, _current_peak_vram())
    except httpx.HTTPStatusError as e:
        yield gr.skip(), f"{t('error_prefix')} {e.response.status_code}: {e.response.text}", _empty_metrics()
    except Exception as e:
        yield gr.skip(), f"{t('error_prefix')}: {e}", _empty_metrics()


def _buffered_speech_multipart(fields: dict, ref_audio: str, output_label: str):
    started_at = time.monotonic()
    yield gr.skip(), t("streaming"), _empty_metrics()
    try:
        with open(ref_audio, "rb") as f:
            r = httpx.post(
                f"{API_URL}/v1/audio/speech",
                data=fields,
                files={"ref_audio": ("ref.wav", f, "audio/wav")},
                timeout=None,
            )
        r.raise_for_status()
        audio = _pcm_bytes_to_audio(r.content)
        saved_path = _save_output_audio(audio[1], output_label)
        yield audio, _done_status(saved_path), _metrics_html(len(audio[1]), started_at, _current_peak_vram())
    except httpx.HTTPStatusError as e:
        yield gr.skip(), f"{t('error_prefix')} {e.response.status_code}: {e.response.text}", _empty_metrics()
    except Exception as e:
        yield gr.skip(), f"{t('error_prefix')}: {e}", _empty_metrics()


def _stream_speech_json(payload: dict, output_label: str):
    yield gr.skip(), t("streaming"), _empty_metrics()
    remainder = b""
    state = _new_stream_buffer_state(text_length=len(str(payload.get("input", ""))))
    state["output_label"] = output_label
    try:
        with httpx.stream(
            "POST",
            f"{API_URL}/v1/audio/speech",
            json=payload,
            timeout=None,
        ) as r:
            if r.is_error:
                error_text = r.read().decode("utf-8", errors="replace")
                yield gr.skip(), f"{t('error_prefix')} {r.status_code}: {error_text}", _empty_metrics()
                return
            for chunk in r.iter_bytes(chunk_size=STREAM_CHUNK_SIZE):
                if not chunk:
                    continue
                state["total_bytes"] += len(chunk)
                chunk = remainder + chunk
                even_len = len(chunk) - (len(chunk) % 2)
                remainder = chunk[even_len:]
                samples = np.frombuffer(chunk[:even_len], dtype="<i2")
                for audio, status, metrics in _iter_buffered_audio(samples, state):
                    yield audio, status, metrics
        for audio, status, metrics in _finish_buffered_audio(state):
            yield audio, status, metrics
    except httpx.HTTPStatusError as e:
        yield gr.skip(), f"{t('error_prefix')} {e.response.status_code}: {e.response.text}", _empty_metrics()
    except Exception as e:
        yield gr.skip(), f"{t('error_prefix')}: {e}", _empty_metrics()


def _stream_speech_multipart(fields: dict, ref_audio: str, output_label: str):
    yield gr.skip(), t("streaming"), _empty_metrics()
    remainder = b""
    state = _new_stream_buffer_state(text_length=len(str(fields.get("input", ""))))
    state["output_label"] = output_label
    try:
        with open(ref_audio, "rb") as f:
            with httpx.stream(
                "POST",
                f"{API_URL}/v1/audio/speech",
                data=fields,
                files={"ref_audio": ("ref.wav", f, "audio/wav")},
                timeout=None,
            ) as r:
                if r.is_error:
                    error_text = r.read().decode("utf-8", errors="replace")
                    yield gr.skip(), f"{t('error_prefix')} {r.status_code}: {error_text}", _empty_metrics()
                    return
                for chunk in r.iter_bytes(chunk_size=STREAM_CHUNK_SIZE):
                    if not chunk:
                        continue
                    state["total_bytes"] += len(chunk)
                    chunk = remainder + chunk
                    even_len = len(chunk) - (len(chunk) % 2)
                    remainder = chunk[even_len:]
                    samples = np.frombuffer(chunk[:even_len], dtype="<i2")
                    for audio, status, metrics in _iter_buffered_audio(samples, state):
                        yield audio, status, metrics
        for audio, status, metrics in _finish_buffered_audio(state):
            yield audio, status, metrics
    except httpx.HTTPStatusError as e:
        yield gr.skip(), f"{t('error_prefix')} {e.response.status_code}: {e.response.text}", _empty_metrics()
    except Exception as e:
        yield gr.skip(), f"{t('error_prefix')}: {e}", _empty_metrics()


def _stt_word_segments(audio_path: str) -> list[dict]:
    stt_url = _stt_url()
    if not stt_url:
        print("[trim] STT_URL is not set; using silence fallback")
        return []
    try:
        with open(audio_path, "rb") as f:
            r = httpx.post(
                f"{stt_url}/v1/audio/transcriptions",
                files={"file": ("ref.wav", f, "audio/wav")},
                data={"model": "whisper-turbo", "response_format": "verbose_json"},
                timeout=600,
            )
        r.raise_for_status()
        data = r.json()
        print(f"[trim] STT response: text={data.get('text', '')!r} duration={data.get('duration')}")
        segments = data.get("segments", []) or []
        for i, s in enumerate(segments):
            print(f"[trim]   seg {i}: start={s.get('start')} end={s.get('end')} text={s.get('text', '')!r}")
        valid = [s for s in segments if s.get("end", 0) > 0]
        if not valid:
            print("[trim] STT returned segments without usable timestamps")
        return valid
    except Exception as e:
        print(f"[trim] STT segments failed: {e}")
        return []


def trim_audio(audio_path):
    if not audio_path:
        return None, t("err_no_audio")
    data, sr = sf.read(audio_path)
    if data.ndim > 1:
        data = data.mean(axis=1)
    data = data.astype(np.float32)

    audio_len_s = len(data) / sr
    if audio_len_s <= MAX_REF_SECONDS:
        print(f"[trim] audio is {audio_len_s:.2f}s, no trim needed")
        return audio_path, t("trim_not_needed", seconds=f"{audio_len_s:.2f}")

    soft_limit = MAX_REF_SECONDS  # 3.5s ideal
    hard_limit = 4.0  # absolute cap so we always stay under 4s

    cut_s = None
    status = None
    segments = _stt_word_segments(audio_path)
    if segments:
        in_window = [s["end"] for s in segments
                     if soft_limit <= s.get("end", 0) <= hard_limit]
        under = [s["end"] for s in segments if s.get("end", 0) <= soft_limit]
        if in_window:
            cut_s = in_window[-1]
            status = t("trim_stt_cut", seconds=f"{cut_s:.2f}")
            print(f"[trim] STT cut at {cut_s:.2f}s (within {soft_limit}-{hard_limit}s window)")
        elif under:
            cut_s = under[-1]
            status = t("trim_stt_cut", seconds=f"{cut_s:.2f}")
            print(f"[trim] STT cut at {cut_s:.2f}s (latest before soft limit)")

    if cut_s is None:
        search_end = min(len(data), int(hard_limit * sr))
        for top_db, min_gap_s in ((35, 0.25), (30, 0.20), (25, 0.15), (20, 0.10)):
            intervals = librosa.effects.split(data[:search_end], top_db=top_db,
                                               frame_length=2048, hop_length=512)
            if len(intervals) == 0:
                continue
            candidates = []
            for i, (_, end) in enumerate(intervals):
                if end / sr > hard_limit:
                    break
                next_start = intervals[i + 1][0] if i + 1 < len(intervals) else search_end
                gap = (next_start - end) / sr
                if gap >= min_gap_s:
                    candidates.append(end / sr)
            if candidates:
                cut_s = candidates[-1]
                status = t("trim_silence_cut", seconds=f"{cut_s:.2f}")
                print(f"[trim] silence cut at {cut_s:.2f}s (top_db={top_db}, gap≥{min_gap_s*1000:.0f}ms)")
                break

    if cut_s is None:
        cut_s = hard_limit
        status = t("trim_hard_cut", seconds=f"{hard_limit:.2f}")
        print(f"[trim] no boundary found — hard cut at {hard_limit}s")

    cut_s = max(1.0, min(cut_s, audio_len_s))
    cut = int(cut_s * sr)
    cut = max(sr, min(cut, len(data)))
    print(f"[trim] final cut: {cut} samples ({cut/sr:.2f}s) of {len(data)} ({audio_len_s:.2f}s)")

    fd, new_path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    trailing = np.zeros(int(sr * TRIM_TRAILING_SILENCE_SECONDS), dtype=np.float32)
    sf.write(new_path, np.concatenate([data[:cut], trailing]), sr, subtype="PCM_16")
    return new_path, status or t("trim_done", seconds=f"{cut_s:.2f}")


def transcribe_audio(audio_path, fallback_path=None):
    audio_path = audio_path or fallback_path
    if not audio_path:
        return t("err_no_audio")
    stt_url = _stt_url()
    if not stt_url:
        return t("stt_unconfigured")
    try:
        with open(audio_path, "rb") as f:
            r = httpx.post(
                f"{stt_url}/v1/audio/transcriptions",
                files={"file": ("ref.wav", f, "audio/wav")},
                data={"model": "parakeet-tdt-0.6b-v3", "response_format": "json"},
                timeout=600,
            )
        r.raise_for_status()
        data = r.json()
        text = data.get("text", "") if isinstance(data, dict) else str(data)
        return text or ""
    except Exception as e:
        return f"{t('stt_error')}: {e}"


def generate_clone(
    text, language, ref_audio, ref_trimmed, ref_text, instruct,
    num_step, guidance_scale, denoise, speed, duration,
    preprocess_prompt, postprocess_output, speaker_embedding_only, clean_markdown, normalize_spoken_text, stream_audio,
    dynamic_steps, dynamic_min_steps, dynamic_max_steps, dynamic_desired_speed,
):
    if not text or not text.strip():
        yield gr.skip(), t("err_text_empty"), _empty_metrics()
        return
    ref_audio = ref_trimmed or ref_audio
    if not ref_audio:
        yield gr.skip(), t("err_no_ref_audio"), _empty_metrics()
        return

    fields = _speech_stream_fields(
        text, language, instruct,
        num_step, guidance_scale, denoise, speed, duration,
        preprocess_prompt, postprocess_output, speaker_embedding_only, clean_markdown, normalize_spoken_text, stream_audio,
        dynamic_steps, dynamic_min_steps, dynamic_max_steps, dynamic_desired_speed,
    )
    if ref_text:
        fields["ref_text"] = ref_text
    _log_speech_request("clone", fields)
    if stream_audio:
        yield from _stream_speech_multipart(fields, ref_audio, "clone")
    else:
        yield from _buffered_speech_multipart(fields, ref_audio, "clone")


def generate_design(
    text, language, instruct,
    num_step, guidance_scale, denoise, speed, duration,
    preprocess_prompt, postprocess_output, clean_markdown, normalize_spoken_text, stream_audio,
    dynamic_steps, dynamic_min_steps, dynamic_max_steps, dynamic_desired_speed,
):
    if not text or not text.strip():
        yield gr.skip(), t("err_text_empty"), _empty_metrics()
        return
    if not instruct or not instruct.strip():
        yield gr.skip(), t("err_no_instruct"), _empty_metrics()
        return

    payload = _speech_stream_fields(
        text, language, instruct,
        num_step, guidance_scale, denoise, speed, duration,
        preprocess_prompt, postprocess_output, False, clean_markdown, normalize_spoken_text, stream_audio,
        dynamic_steps, dynamic_min_steps, dynamic_max_steps, dynamic_desired_speed,
    )
    _log_speech_request("design", payload)
    if stream_audio:
        yield from _stream_speech_json(payload, "design")
    else:
        yield from _buffered_speech_json(payload, "design")


def generate_voice(
    voice_id, text, language, instruct,
    num_step, guidance_scale, denoise, speed, duration,
    preprocess_prompt, postprocess_output, speaker_embedding_only, clean_markdown, normalize_spoken_text, stream_audio,
    dynamic_steps, dynamic_min_steps, dynamic_max_steps, dynamic_desired_speed,
):
    if not voice_id or voice_id == "none":
        yield gr.skip(), t("err_no_voice_selected"), _empty_metrics()
        return
    if not text or not text.strip():
        yield gr.skip(), t("err_text_empty"), _empty_metrics()
        return

    payload = _speech_stream_fields(
        text, language, instruct,
        num_step, guidance_scale, denoise, speed, duration,
        preprocess_prompt, postprocess_output, speaker_embedding_only, clean_markdown, normalize_spoken_text, stream_audio,
        dynamic_steps, dynamic_min_steps, dynamic_max_steps, dynamic_desired_speed,
    )
    payload["voice"] = voice_id
    _log_speech_request("saved_voice", payload)
    output_label = f"voice-{voice_id}"
    if stream_audio:
        yield from _stream_speech_json(payload, output_label)
    else:
        yield from _buffered_speech_json(payload, output_label)


def list_voices(selected_voice=None):
    try:
        r = httpx.get(f"{API_URL}/voices", timeout=10)
        r.raise_for_status()
        voices = r.json()
        choices = [
            (f"{v['name']} ({t('audio_present') if v['has_audio'] else t('audio_missing')})", v["id"])
            for v in voices
        ]
        voice_ids = [choice[1] for choice in choices]
        value = selected_voice if selected_voice in voice_ids else (choices[0][1] if choices else None)
        return gr.update(choices=choices, value=value)
    except Exception:
        return gr.update(choices=[], value=None)


def clear_audio_output():
    return None, t("streaming"), _empty_metrics()


def list_outputs():
    try:
        files = sorted(OUTPUTS_DIR.glob("*.wav"), key=lambda p: p.stat().st_mtime, reverse=True)
        choices = [(p.name, str(p)) for p in files]
        return gr.update(choices=choices, value=choices[0][1] if choices else None)
    except Exception:
        return gr.update(choices=[], value=None)


def load_output(output_path):
    if not output_path:
        return None, t("err_no_audio")
    path = Path(output_path)
    if not path.exists():
        return None, t("err_no_audio")
    return str(path), t("loaded_output", name=path.name)


def load_voice_for_edit(voice_id):
    if not voice_id:
        return "", "", "", None, t("err_no_voice_selected")
    try:
        r = httpx.get(f"{API_URL}/voices/{voice_id}", timeout=10)
        r.raise_for_status()
        voice = r.json()
        audio_path = None
        if voice.get("has_audio"):
            r2 = httpx.get(f"{API_URL}/voices/{voice_id}/audio", timeout=30)
            r2.raise_for_status()
            audio_path = _save_response_wav(r2.content)
        return (
            voice.get("name", ""),
            voice.get("ref_text", ""),
            voice.get("description", "") or "",
            audio_path,
            t("voice_loaded", name=voice.get("name", voice_id), id=voice_id),
        )
    except httpx.HTTPStatusError as e:
        return "", "", "", None, f"{t('error_prefix')}: {e.response.text}"
    except Exception as e:
        return "", "", "", None, f"{t('error_prefix')}: {e}"


def create_voice(voice_id, name, ref_text, description, ref_audio, ref_trimmed=None):
    ref_audio = ref_trimmed or ref_audio
    if not voice_id or not name or not ref_text:
        return t("err_id_name_text")
    try:
        r = httpx.post(
            f"{API_URL}/voices/{voice_id}",
            json={"name": name, "ref_text": ref_text, "description": description or ""},
            timeout=10,
        )
        r.raise_for_status()
    except httpx.HTTPStatusError as e:
        return f"{t('error_prefix')}: {e.response.text}"

    if ref_audio:
        with open(ref_audio, "rb") as f:
            r2 = httpx.put(
                f"{API_URL}/voices/{voice_id}/audio",
                files={"file": ("ref.wav", f, "audio/wav")},
                timeout=30,
            )
        if not r2.is_success:
            return t("voice_audio_upload_failed", detail=r2.text)

    return t("voice_saved", name=name, id=voice_id)


def update_saved_voice(voice_id, name, ref_text, description, ref_audio, ref_trimmed=None):
    ref_audio = ref_trimmed or ref_audio
    if not voice_id or not name or not ref_text:
        return t("err_id_name_text")
    try:
        r = httpx.put(
            f"{API_URL}/voices/{voice_id}",
            json={"name": name, "ref_text": ref_text, "description": description or ""},
            timeout=10,
        )
        r.raise_for_status()
    except httpx.HTTPStatusError as e:
        return f"{t('error_prefix')}: {e.response.text}"
    except Exception as e:
        return f"{t('error_prefix')}: {e}"

    if ref_audio:
        try:
            with open(ref_audio, "rb") as f:
                r2 = httpx.put(
                    f"{API_URL}/voices/{voice_id}/audio",
                    files={"file": ("ref.wav", f, "audio/wav")},
                    timeout=30,
                )
            r2.raise_for_status()
        except httpx.HTTPStatusError as e:
            return t("voice_audio_upload_failed", detail=e.response.text)
        except Exception as e:
            return f"{t('error_prefix')}: {e}"

    return t("voice_updated", name=name, id=voice_id)


def delete_voice(voice_id):
    if not voice_id:
        return t("err_no_voice_to_delete")
    try:
        r = httpx.delete(f"{API_URL}/voices/{voice_id}", timeout=10)
        r.raise_for_status()
        return t("voice_deleted", id=voice_id)
    except httpx.HTTPStatusError as e:
        return f"{t('error_prefix')}: {e.response.text}"


def model_status():
    try:
        r = httpx.get(f"{API_URL}/models/status", timeout=5)
        d = r.json()
        return t("model_loaded") if d.get("loaded") else t("model_not_loaded")
    except Exception:
        return t("backend_unreachable")


def load_model():
    try:
        r = httpx.post(f"{API_URL}/models/load", timeout=300)
        r.raise_for_status()
        return t("model_loaded_done")
    except Exception as e:
        return f"{t('error_prefix')}: {e}"


CTRL_ENTER_JS = """
() => {
    if (window.__omniVoiceUiBound) {
        return;
    }
    window.__omniVoiceUiBound = true;

    const storagePrefix = "omniVoiceWebui.";
    const defaultTab = "voices";
    const tabLabels = {
        clone: ["Voice Cloning", "Stimmen-Klonen"],
        design: ["Voice Design", "Stimmen-Design"],
        voices: ["Saved Voices", "Gespeicherte Stimmen"],
        create: ["Create Voice", "Stimme erstellen"],
        manage: ["Manage Voices", "Stimmen verwalten"],
        history: ["History", "Verlauf"],
    };
    const generateButtonIds = {
        voices: ["voice-generate-btn", "voice-generate-top-btn"],
        clone: ["clone-generate-btn", "clone-generate-top-btn"],
        design: ["design-generate-btn", "design-generate-top-btn"],
    };
    const pasteButtonTargets = {
        "voice-paste-btn": "voice-text",
        "clone-paste-btn": "clone-text",
        "design-paste-btn": "design-text",
    };
    const cancelButtonIds = {
        voices: "voice-cancel-btn",
        clone: "clone-cancel-btn",
        design: "design-cancel-btn",
    };
    const persistedFields = {
        "voice-text": "voiceText",
        "voice-instruct": "voiceInstruct",
        "clone-text": "cloneText",
        "clone-ref-text": "cloneRefText",
        "clone-instruct": "cloneInstruct",
        "design-text": "designText",
        "design-instruct": "designInstruct",
    };

    const storageGet = (key) => {
        try {
            return window.localStorage.getItem(storagePrefix + key);
        } catch (_) {
            return null;
        }
    };
    const storageSet = (key, value) => {
        try {
            window.localStorage.setItem(storagePrefix + key, value ?? "");
        } catch (_) {}
    };

    const isVisible = (element) => {
        if (!element) {
            return false;
        }
        const rect = element.getBoundingClientRect();
        const style = window.getComputedStyle(element);
        return rect.width > 0 && rect.height > 0 && style.visibility !== "hidden" && style.display !== "none";
    };

    const nativeSetValue = (element, value) => {
        const prototype = element instanceof HTMLTextAreaElement
            ? HTMLTextAreaElement.prototype
            : HTMLInputElement.prototype;
        const descriptor = Object.getOwnPropertyDescriptor(prototype, "value");
        if (descriptor?.set) {
            descriptor.set.call(element, value);
        } else {
            element.value = value;
        }
        element.dispatchEvent(new Event("input", { bubbles: true }));
        element.dispatchEvent(new Event("change", { bubbles: true }));
    };

    const editableIn = (root) => root?.querySelector("textarea, input:not([type='hidden'])");

    const bindPersistedFields = () => {
        for (const [id, key] of Object.entries(persistedFields)) {
            const root = document.getElementById(id);
            const input = editableIn(root);
            if (!input || input.dataset.omniPersistBound === "true") {
                continue;
            }
            input.dataset.omniPersistBound = "true";
            const saved = storageGet(key);
            if (saved != null && saved !== "" && input.value !== saved) {
                nativeSetValue(input, saved);
            }
            input.addEventListener("input", () => storageSet(key, input.value));
            input.addEventListener("change", () => storageSet(key, input.value));
        }
    };

    const stopAllAudio = () => {
        document.querySelectorAll("audio").forEach((audio) => {
            try {
                audio.pause();
                audio.currentTime = 0;
                audio.removeAttribute("src");
                audio.load();
            } catch (_) {}
        });
    };

    const pasteIntoTextbox = async (targetId) => {
        const input = editableIn(document.getElementById(targetId));
        if (!input || !navigator.clipboard?.readText) {
            return;
        }
        try {
            const text = await navigator.clipboard.readText();
            nativeSetValue(input, text);
            input.focus();
        } catch (_) {}
    };

    const clickButtonRoot = (id) => {
        const root = document.getElementById(id);
        const button = root?.querySelector("button") || root;
        if (button && !button.disabled) {
            button.click();
        }
    };

    const cancelOtherGenerations = (activeTab) => {
        for (const [tab, id] of Object.entries(cancelButtonIds)) {
            if (tab !== activeTab) {
                clickButtonRoot(id);
            }
        }
    };

    const buttonRoots = (ids) => (Array.isArray(ids) ? ids : [ids])
        .map((id) => document.getElementById(id))
        .filter(Boolean);

    const activeGenerate = () => {
        for (const [tab, ids] of Object.entries(generateButtonIds)) {
            for (const root of buttonRoots(ids)) {
                if (!isVisible(root)) {
                    continue;
                }
                return { tab, button: root.querySelector("button") || root };
            }
        }
        return null;
    };

    const allTabButtons = () => Array.from(document.querySelectorAll("[role='tab'], button"));
    const findTabButton = (tabId) => {
        const labels = tabLabels[tabId] || [];
        return allTabButtons().find((button) => {
            const text = (button.textContent || "").trim();
            return labels.some((label) => text === label || text.includes(label));
        });
    };

    const currentTabFromButton = (button) => {
        const text = (button?.textContent || "").trim();
        for (const [tab, labels] of Object.entries(tabLabels)) {
            if (labels.some((label) => text === label || text.includes(label))) {
                return tab;
            }
        }
        return null;
    };

    const activateInitialTab = () => {
        if (window.__omniVoiceInitialTabDone) {
            return;
        }
        const target = storageGet("activeTab") || defaultTab;
        const button = findTabButton(target) || findTabButton(defaultTab);
        if (!button) {
            return;
        }
        window.__omniVoiceInitialTabDone = true;
        button.click();
    };

    const bindGenerateButtons = () => {
        for (const [tab, ids] of Object.entries(generateButtonIds)) {
            for (const root of buttonRoots(ids)) {
                const button = root.querySelector("button") || root;
                if (!button || button.dataset.omniGenerateBound === "true") {
                    continue;
                }
                button.dataset.omniGenerateBound = "true";
                button.addEventListener("click", () => {
                    stopAllAudio();
                    cancelOtherGenerations(tab);
                    storageSet("activeTab", tab);
                }, true);
            }
        }
    };

    const bindPasteButtons = () => {
        for (const [buttonId, targetId] of Object.entries(pasteButtonTargets)) {
            const root = document.getElementById(buttonId);
            const button = root?.querySelector("button") || root;
            if (!button || button.dataset.omniPasteBound === "true") {
                continue;
            }
            button.dataset.omniPasteBound = "true";
            button.addEventListener("click", (event) => {
                event.preventDefault();
                pasteIntoTextbox(targetId);
            }, true);
        }
    };

    const bindTabButtons = () => {
        for (const button of allTabButtons()) {
            if (button.dataset.omniTabBound === "true") {
                continue;
            }
            const tab = currentTabFromButton(button);
            if (!tab) {
                continue;
            }
            button.dataset.omniTabBound = "true";
            button.addEventListener("click", () => {
                storageSet("activeTab", tab);
                stopAllAudio();
                cancelOtherGenerations(tab);
            }, true);
        }
    };

    const refreshBindings = () => {
        bindPersistedFields();
        bindGenerateButtons();
        bindPasteButtons();
        bindTabButtons();
        activateInitialTab();
    };

    document.addEventListener("keydown", (event) => {
        if (!(event.ctrlKey || event.metaKey) || event.key !== "Enter") {
            return;
        }
        const active = activeGenerate();
        if (!active?.button || active.button.disabled) {
            return;
        }
        event.preventDefault();
        stopAllAudio();
        cancelOtherGenerations(active.tab);
        storageSet("activeTab", active.tab);
        active.button.click();
    }, true);

    const observer = new MutationObserver(refreshBindings);
    observer.observe(document.body, { childList: true, subtree: true });
    window.setTimeout(refreshBindings, 250);
    window.setTimeout(refreshBindings, 1000);
    window.setTimeout(refreshBindings, 2500);
}
"""


with gr.Blocks(title=t("title"), theme=gr.themes.Soft(), js=CTRL_ENTER_JS) as demo:
    gr.Markdown(f"# {t('title')}\n{t('subtitle')}")

    with gr.Row():
        status_box = gr.Textbox(label=t("model_status"), interactive=False, scale=3)
        refresh_btn = gr.Button(t("refresh_status"), scale=1)
        load_btn = gr.Button(t("load_model"), scale=1)

    refresh_btn.click(model_status, outputs=status_box)
    load_btn.click(load_model, outputs=status_box)
    demo.load(model_status, outputs=status_box)

    with gr.Tabs(selected="voices"):

        # --- Voice Clone tab ---
        with gr.Tab(t("tab_clone"), id="clone", render_children=True):
            with gr.Row():
                with gr.Column():
                    clone_text = gr.Textbox(label=t("text"), lines=4, placeholder=t("text_placeholder"), elem_id="clone-text")
                    with gr.Row():
                        clone_text_paste_btn = gr.Button(t("paste_text"), size="sm", elem_id="clone-paste-btn")
                        clone_text_clear_btn = gr.Button(t("clear_text"), size="sm")
                    clone_text_clear_btn.click(lambda: "", outputs=clone_text)
                    clone_lang = gr.Dropdown(LANGUAGES, value="Auto", label=t("language"))
                    clone_ref_audio = gr.Audio(
                        label=t("ref_audio"),
                        type="filepath",
                        sources=["upload", "microphone"],
                    )
                    clone_trim_btn = gr.Button(t("trim_clone"), size="sm")
                    clone_ref_trimmed = gr.Audio(label=t("trimmed"), type="filepath")
                    clone_transcribe_btn = gr.Button(t("transcribe"), size="sm")
                    clone_ref_text = gr.Textbox(label=t("ref_text_optional"), lines=2, elem_id="clone-ref-text")
                    clone_instruct = gr.Textbox(label=t("instruct_clone"), lines=1, elem_id="clone-instruct")
                    gr.Markdown(t("advanced_options"))
                    clone_num_step = gr.Slider(1, 200, value=32, step=1, label=t("diffusion_steps"))
                    with gr.Row():
                        clone_dynamic_steps = gr.Checkbox(
                            value=False,
                            label=t("dynamic_steps"),
                            info=t("dynamic_steps_hint"),
                        )
                        clone_dynamic_min_steps = gr.Slider(1, 200, value=4, step=1, label=t("dynamic_min_steps"))
                        clone_dynamic_max_steps = gr.Slider(1, 200, value=64, step=1, label=t("dynamic_max_steps"))
                    clone_dynamic_desired_speed = gr.Number(value=4.0, label=t("dynamic_desired_speed"))
                    clone_guidance = gr.Slider(0.1, 10.0, value=2.0, step=0.1, label=t("guidance_scale"))
                    with gr.Row():
                        clone_denoise = gr.Checkbox(value=True, label=t("denoise"))
                        clone_preprocess = gr.Checkbox(value=True, label=t("preprocess_prompt"))
                        clone_postprocess = gr.Checkbox(value=True, label=t("postprocess_output"))
                        clone_speaker_embedding_only = gr.Checkbox(
                            value=False,
                            label=t("speaker_embedding_only"),
                        )
                        clone_clean_markdown = gr.Checkbox(value=True, label=t("clean_markdown"))
                        clone_normalize_spoken_text = gr.Checkbox(value=False, label=t("normalize_spoken_text"), info=t("normalize_spoken_text_hint"))
                    clone_speed = gr.Slider(0.1, 3.0, value=1.0, step=0.05, label=t("speed"))
                    clone_duration = gr.Number(value=0.0, label=t("duration"))
                    with gr.Row():
                        clone_btn = gr.Button(t("synthesize"), variant="primary", elem_id="clone-generate-btn")
                        clone_cancel_btn = gr.Button(t("cancel"), variant="stop", elem_id="clone-cancel-btn")
                with gr.Column():
                    with gr.Row():
                        clone_btn_top = gr.Button(t("synthesize"), variant="primary", elem_id="clone-generate-top-btn")
                        clone_cancel_btn_top = gr.Button(t("cancel"), variant="stop")
                    with gr.Row():
                        clone_stream = gr.Checkbox(value=True, label=t("stream_audio"))
                        clone_autoplay = gr.Checkbox(value=True, label=t("autoplay"))
                    clone_audio_out = gr.Audio(
                        label=t("output"),
                        streaming=True,
                        autoplay=True,
                        format="wav",
                    )
                    clone_status = gr.Textbox(label=t("status"), interactive=False)
                    clone_metrics = gr.HTML(label=t("metrics"))

            clone_trim_btn.click(trim_audio, inputs=clone_ref_audio, outputs=[clone_ref_trimmed, clone_status])
            clone_autoplay.change(set_audio_autoplay, inputs=clone_autoplay, outputs=clone_audio_out)
            clone_transcribe_btn.click(
                transcribe_audio,
                inputs=[clone_ref_trimmed, clone_ref_audio],
                outputs=clone_ref_text,
            )

            clone_inputs = [
                clone_text, clone_lang, clone_ref_audio, clone_ref_trimmed, clone_ref_text, clone_instruct,
                clone_num_step, clone_guidance, clone_denoise, clone_speed, clone_duration,
                clone_preprocess, clone_postprocess, clone_speaker_embedding_only, clone_clean_markdown,
                clone_normalize_spoken_text, clone_stream, clone_dynamic_steps, clone_dynamic_min_steps, clone_dynamic_max_steps,
                clone_dynamic_desired_speed,
            ]
            clone_top_event = clone_btn_top.click(
                clear_audio_output,
                outputs=[clone_audio_out, clone_status, clone_metrics],
                show_api=False,
            ).then(
                generate_clone,
                inputs=clone_inputs,
                outputs=[clone_audio_out, clone_status, clone_metrics],
            )
            clone_event = clone_btn.click(
                clear_audio_output,
                outputs=[clone_audio_out, clone_status, clone_metrics],
                show_api=False,
            ).then(
                generate_clone,
                inputs=clone_inputs,
                outputs=[clone_audio_out, clone_status, clone_metrics],
            )
            clone_cancel_btn_top.click(
                cancel_status,
                cancels=[clone_top_event, clone_event],
                outputs=clone_status,
                show_api=False,
            )
            clone_cancel_btn.click(
                cancel_status,
                cancels=[clone_top_event, clone_event],
                outputs=clone_status,
                show_api=False,
            )

        # --- Voice Design tab ---
        with gr.Tab(t("tab_design"), id="design", render_children=True):
            with gr.Row():
                with gr.Column():
                    design_text = gr.Textbox(label=t("text"), lines=4, placeholder=t("text_placeholder"), elem_id="design-text")
                    with gr.Row():
                        design_text_paste_btn = gr.Button(t("paste_text"), size="sm", elem_id="design-paste-btn")
                        design_text_clear_btn = gr.Button(t("clear_text"), size="sm")
                    design_text_clear_btn.click(lambda: "", outputs=design_text)
                    design_lang = gr.Dropdown(LANGUAGES, value="Auto", label=t("language"))
                    design_instruct = gr.Textbox(
                        label=t("speaker_instruct"),
                        lines=3,
                        elem_id="design-instruct",
                        placeholder=t("speaker_instruct_placeholder"),
                    )
                    gr.Markdown(t("advanced_options"))
                    design_num_step = gr.Slider(1, 200, value=32, step=1, label=t("diffusion_steps"))
                    with gr.Row():
                        design_dynamic_steps = gr.Checkbox(
                            value=False,
                            label=t("dynamic_steps"),
                            info=t("dynamic_steps_hint"),
                        )
                        design_dynamic_min_steps = gr.Slider(1, 200, value=4, step=1, label=t("dynamic_min_steps"))
                        design_dynamic_max_steps = gr.Slider(1, 200, value=64, step=1, label=t("dynamic_max_steps"))
                    design_dynamic_desired_speed = gr.Number(value=4.0, label=t("dynamic_desired_speed"))
                    design_guidance = gr.Slider(0.1, 10.0, value=2.0, step=0.1, label=t("guidance_scale"))
                    with gr.Row():
                        design_denoise = gr.Checkbox(value=True, label=t("denoise"))
                        design_preprocess = gr.Checkbox(value=True, label=t("preprocess_prompt"))
                        design_postprocess = gr.Checkbox(value=True, label=t("postprocess_output"))
                        design_clean_markdown = gr.Checkbox(value=True, label=t("clean_markdown"))
                        design_normalize_spoken_text = gr.Checkbox(value=False, label=t("normalize_spoken_text"), info=t("normalize_spoken_text_hint"))
                    design_speed = gr.Slider(0.1, 3.0, value=1.0, step=0.05, label=t("speed"))
                    design_duration = gr.Number(value=0.0, label=t("duration"))
                    with gr.Row():
                        design_btn = gr.Button(t("synthesize"), variant="primary", elem_id="design-generate-btn")
                        design_cancel_btn = gr.Button(t("cancel"), variant="stop", elem_id="design-cancel-btn")
                with gr.Column():
                    with gr.Row():
                        design_btn_top = gr.Button(t("synthesize"), variant="primary", elem_id="design-generate-top-btn")
                        design_cancel_btn_top = gr.Button(t("cancel"), variant="stop")
                    with gr.Row():
                        design_stream = gr.Checkbox(value=True, label=t("stream_audio"))
                        design_autoplay = gr.Checkbox(value=True, label=t("autoplay"))
                    design_audio_out = gr.Audio(
                        label=t("output"),
                        streaming=True,
                        autoplay=True,
                        format="wav",
                    )
                    design_status = gr.Textbox(label=t("status"), interactive=False)
                    design_metrics = gr.HTML(label=t("metrics"))

            design_autoplay.change(set_audio_autoplay, inputs=design_autoplay, outputs=design_audio_out)
            design_inputs = [
                design_text, design_lang, design_instruct,
                design_num_step, design_guidance, design_denoise, design_speed, design_duration,
                design_preprocess, design_postprocess, design_clean_markdown, design_normalize_spoken_text, design_stream,
                design_dynamic_steps, design_dynamic_min_steps, design_dynamic_max_steps,
                design_dynamic_desired_speed,
            ]
            design_top_event = design_btn_top.click(
                clear_audio_output,
                outputs=[design_audio_out, design_status, design_metrics],
                show_api=False,
            ).then(
                generate_design,
                inputs=design_inputs,
                outputs=[design_audio_out, design_status, design_metrics],
            )
            design_event = design_btn.click(
                clear_audio_output,
                outputs=[design_audio_out, design_status, design_metrics],
                show_api=False,
            ).then(
                generate_design,
                inputs=design_inputs,
                outputs=[design_audio_out, design_status, design_metrics],
            )
            design_cancel_btn_top.click(
                cancel_status,
                cancels=[design_top_event, design_event],
                outputs=design_status,
                show_api=False,
            )
            design_cancel_btn.click(
                cancel_status,
                cancels=[design_top_event, design_event],
                outputs=design_status,
                show_api=False,
            )

        # --- Saved Voices tab ---
        with gr.Tab(t("tab_voices"), id="voices", render_children=True):
            voice_selected_state = gr.BrowserState(default_value="none", storage_key="omnivoice_selected_voice")
            with gr.Row():
                with gr.Column():
                    voice_dropdown = gr.Dropdown(
                        choices=[("(click refresh)", "none")],
                        value="none",
                        label=t("voice"),
                        interactive=True,
                        elem_id="voice-dropdown",
                    )
                    voices_refresh_btn = gr.Button(t("refresh_voices"))
                    voice_text = gr.Textbox(label=t("text"), lines=4, elem_id="voice-text")
                    with gr.Row():
                        voice_text_paste_btn = gr.Button(t("paste_text"), size="sm", elem_id="voice-paste-btn")
                        voice_text_clear_btn = gr.Button(t("clear_text"), size="sm")
                    voice_text_clear_btn.click(lambda: "", outputs=voice_text)
                    voice_lang = gr.Dropdown(LANGUAGES, value="Auto", label=t("language"))
                    voice_instruct = gr.Textbox(label=t("instruct_optional"), lines=1, elem_id="voice-instruct")
                    gr.Markdown(t("advanced_options"))
                    voice_num_step = gr.Slider(1, 200, value=32, step=1, label=t("diffusion_steps"))
                    with gr.Row():
                        voice_dynamic_steps = gr.Checkbox(
                            value=False,
                            label=t("dynamic_steps"),
                            info=t("dynamic_steps_hint"),
                        )
                        voice_dynamic_min_steps = gr.Slider(1, 200, value=4, step=1, label=t("dynamic_min_steps"))
                        voice_dynamic_max_steps = gr.Slider(1, 200, value=64, step=1, label=t("dynamic_max_steps"))
                    voice_dynamic_desired_speed = gr.Number(value=4.0, label=t("dynamic_desired_speed"))
                    voice_guidance = gr.Slider(0.1, 10.0, value=2.0, step=0.1, label=t("guidance_scale"))
                    with gr.Row():
                        voice_denoise = gr.Checkbox(value=True, label=t("denoise"))
                        voice_preprocess = gr.Checkbox(value=True, label=t("preprocess_prompt"))
                        voice_postprocess = gr.Checkbox(value=True, label=t("postprocess_output"))
                        voice_speaker_embedding_only = gr.Checkbox(
                            value=False,
                            label=t("speaker_embedding_only"),
                        )
                        voice_clean_markdown = gr.Checkbox(value=True, label=t("clean_markdown"))
                        voice_normalize_spoken_text = gr.Checkbox(value=False, label=t("normalize_spoken_text"), info=t("normalize_spoken_text_hint"))
                    voice_speed = gr.Slider(0.1, 3.0, value=1.0, step=0.05, label=t("speed"))
                    voice_duration = gr.Number(value=0.0, label=t("duration"))
                    with gr.Row():
                        voice_btn = gr.Button(t("synthesize"), variant="primary", elem_id="voice-generate-btn")
                        voice_cancel_btn = gr.Button(t("cancel"), variant="stop", elem_id="voice-cancel-btn")
                with gr.Column():
                    with gr.Row():
                        voice_btn_top = gr.Button(t("synthesize"), variant="primary", elem_id="voice-generate-top-btn")
                        voice_cancel_btn_top = gr.Button(t("cancel"), variant="stop")
                    with gr.Row():
                        voice_stream = gr.Checkbox(value=True, label=t("stream_audio"))
                        voice_autoplay = gr.Checkbox(value=True, label=t("autoplay"))
                    voice_audio_out = gr.Audio(
                        label=t("output"),
                        streaming=True,
                        autoplay=True,
                        format="wav",
                    )
                    voice_status = gr.Textbox(label=t("status"), interactive=False)
                    voice_metrics = gr.HTML(label=t("metrics"))

            voices_refresh_btn.click(list_voices, inputs=voice_selected_state, outputs=voice_dropdown)
            demo.load(list_voices, inputs=voice_selected_state, outputs=voice_dropdown)
            voice_dropdown.change(lambda voice_id: voice_id, inputs=voice_dropdown, outputs=voice_selected_state)
            voice_autoplay.change(set_audio_autoplay, inputs=voice_autoplay, outputs=voice_audio_out)
            voice_inputs = [
                voice_dropdown, voice_text, voice_lang, voice_instruct,
                voice_num_step, voice_guidance, voice_denoise, voice_speed, voice_duration,
                voice_preprocess, voice_postprocess, voice_speaker_embedding_only, voice_clean_markdown,
                voice_normalize_spoken_text, voice_stream, voice_dynamic_steps, voice_dynamic_min_steps, voice_dynamic_max_steps,
                voice_dynamic_desired_speed,
            ]
            voice_top_event = voice_btn_top.click(
                clear_audio_output,
                outputs=[voice_audio_out, voice_status, voice_metrics],
                show_api=False,
            ).then(
                generate_voice,
                inputs=voice_inputs,
                outputs=[voice_audio_out, voice_status, voice_metrics],
            )
            voice_event = voice_btn.click(
                clear_audio_output,
                outputs=[voice_audio_out, voice_status, voice_metrics],
                show_api=False,
            ).then(
                generate_voice,
                inputs=voice_inputs,
                outputs=[voice_audio_out, voice_status, voice_metrics],
            )
            voice_cancel_btn_top.click(
                cancel_status,
                cancels=[voice_top_event, voice_event],
                outputs=voice_status,
                show_api=False,
            )
            voice_cancel_btn.click(
                cancel_status,
                cancels=[voice_top_event, voice_event],
                outputs=voice_status,
                show_api=False,
            )

        # --- Create Voice tab ---
        with gr.Tab(t("tab_create_voice"), id="create", render_children=True):
            with gr.Row():
                with gr.Column():
                    gr.Markdown(t("create_voice_header"))
                    new_voice_id = gr.Textbox(label=t("voice_id"), placeholder=t("voice_id_placeholder"))
                    new_voice_name = gr.Textbox(label=t("display_name"), placeholder=t("display_name_placeholder"))
                    new_voice_ref_text = gr.Textbox(label=t("ref_text"), lines=2)
                    new_voice_desc = gr.Textbox(label=t("description_optional"), lines=1)
                    new_voice_audio = gr.Audio(
                        label=t("ref_audio"),
                        type="filepath",
                        sources=["upload", "microphone"],
                    )
                    new_voice_trim_btn = gr.Button(t("trim_clone"), size="sm")
                    new_voice_trimmed = gr.Audio(label=t("trimmed"), type="filepath")
                    new_voice_transcribe_btn = gr.Button(t("transcribe"), size="sm")
                    create_btn = gr.Button(t("create_voice_btn"), variant="primary")
                with gr.Column():
                    create_status = gr.Textbox(label=t("status"), interactive=False)

            new_voice_trim_btn.click(trim_audio, inputs=new_voice_audio, outputs=[new_voice_trimmed, create_status])
            new_voice_transcribe_btn.click(
                transcribe_audio,
                inputs=[new_voice_trimmed, new_voice_audio],
                outputs=new_voice_ref_text,
            )

            create_btn.click(
                create_voice,
                inputs=[new_voice_id, new_voice_name, new_voice_ref_text, new_voice_desc, new_voice_audio, new_voice_trimmed],
                outputs=create_status,
            )

        # --- Manage Voices tab ---
        with gr.Tab(t("tab_manage"), id="manage", render_children=True):
            with gr.Row():
                with gr.Column():
                    gr.Markdown(t("edit_voice_header"))
                    edit_voice_dropdown = gr.Dropdown(label=t("voice"), interactive=True)
                    with gr.Row():
                        edit_refresh_btn = gr.Button(t("refresh_voices"))
                        edit_load_btn = gr.Button(t("load_voice_btn"))
                    edit_voice_name = gr.Textbox(label=t("display_name"), placeholder=t("display_name_placeholder"))
                    edit_voice_ref_text = gr.Textbox(label=t("ref_text"), lines=2)
                    edit_voice_desc = gr.Textbox(label=t("description_optional"), lines=1)
                    edit_voice_audio = gr.Audio(
                        label=t("ref_audio"),
                        type="filepath",
                        sources=["upload", "microphone"],
                    )
                    edit_voice_trim_btn = gr.Button(t("trim_clone"), size="sm")
                    edit_voice_trimmed = gr.Audio(label=t("trimmed"), type="filepath")
                    edit_voice_transcribe_btn = gr.Button(t("transcribe"), size="sm")
                    edit_save_btn = gr.Button(t("save_voice_btn"), variant="primary")
                with gr.Column():
                    manage_status = gr.Textbox(label=t("status"), interactive=False)
                    gr.Markdown(t("delete_voice_header"))
                    del_voice_id = gr.Textbox(label=t("delete_voice_id"))
                    del_btn = gr.Button(t("delete_voice_btn"), variant="stop")

            edit_refresh_btn.click(list_voices, outputs=edit_voice_dropdown)
            demo.load(list_voices, outputs=edit_voice_dropdown)
            edit_load_btn.click(
                load_voice_for_edit,
                inputs=edit_voice_dropdown,
                outputs=[edit_voice_name, edit_voice_ref_text, edit_voice_desc, edit_voice_audio, manage_status],
            )
            edit_voice_trim_btn.click(
                trim_audio,
                inputs=edit_voice_audio,
                outputs=[edit_voice_trimmed, manage_status],
            )
            edit_voice_transcribe_btn.click(
                transcribe_audio,
                inputs=[edit_voice_trimmed, edit_voice_audio],
                outputs=edit_voice_ref_text,
            )
            edit_save_btn.click(
                update_saved_voice,
                inputs=[
                    edit_voice_dropdown, edit_voice_name, edit_voice_ref_text, edit_voice_desc,
                    edit_voice_audio, edit_voice_trimmed,
                ],
                outputs=manage_status,
            )
            del_btn.click(delete_voice, inputs=del_voice_id, outputs=manage_status)

        # --- History tab ---
        with gr.Tab(t("tab_history"), id="history", render_children=True):
            with gr.Row():
                with gr.Column():
                    history_dropdown = gr.Dropdown(label=t("output_file"), interactive=True)
                    with gr.Row():
                        history_refresh_btn = gr.Button(t("refresh_outputs"))
                        history_load_btn = gr.Button(t("load_output"), variant="primary")
                with gr.Column():
                    history_audio = gr.Audio(label=t("output"), type="filepath")
                    history_status = gr.Textbox(label=t("status"), interactive=False)

            history_refresh_btn.click(list_outputs, outputs=history_dropdown)
            demo.load(list_outputs, outputs=history_dropdown)
            history_load_btn.click(
                load_output,
                inputs=history_dropdown,
                outputs=[history_audio, history_status],
            )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=FRONTEND_PORT)
