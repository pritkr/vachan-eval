"""Mock ASR (+ optional real faster-whisper) — privacy-first, offline by default.

Mock mode is deterministic given (text, seed): it injects word-level noise
(substitute / delete / insert) weighted by difficulty so the eval harness
produces realistic-but-reproducible WER numbers without downloading models.

Real engine: if `faster_whisper` is importable AND a model is cached
(`VACHAN_MODEL` env or default tiny), we use it. Otherwise mock.
No network call is ever made from this module.

v3: a *streaming* STT adapter (`stream_transcribe`) that slices the very same
mock hypothesis into 1-second audio chunks so the eval can measure
**time-to-first-token (TTFT)**, **inter-chunk gap** and **end-to-end** latency
on a real voice-UX clock. Every millisecond it charges comes from an injected
clock, so `eval/latency.py`'s `FakeClock` makes the whole trace deterministic
and instantaneous.
"""

from __future__ import annotations

import hashlib
import math
import os
import random
import time

# per-difficulty word error probabilities (tuned so mean WER lands ~0.12-0.20
# on the golden set — under the 0.25 CI gate, but dialect/noise items fail)
DIFFICULTY_WER = {"easy": 0.04, "medium": 0.12, "hard": 0.28}

# small pool of plausible Hindi ASR confusions (matra / anusvara / code-mix)
CONFUSIONS = {
    "में": ["मे", "मैं"],
    "है": ["हे", "हैं"],
    "कैसे": ["कइसे", "कैसे"],
    "राशन": ["रासन", "राशन"],
    "छात्रवृत्ति": ["छात्रव्रत्ति", "स्कॉलरशिप"],
    "स्कॉलरशिप": ["स्कालरशिप", "छात्रवृत्ति"],
    "आवेदन": ["आवेदान", "अवेदन"],
    "करे": ["करें", "करे"],
    "करें": ["करे", "करे"],
    "हमार": ["हमारा", "हमार"],
    "ना": ["न", "ना"],
    "है": ["हे", "है"],
    "eKYC": ["इकेवाइसी", "eKYC"],
    "मोबाइल": ["मोबाईल", "मोबाइल"],
    "स्कूल": ["इसकूल", "स्कूल"],
    "पैसा": ["पइसा", "पैसा"],
}

FILLERS = ["अ", "हम्म", "जी"]


def _seeded_rng(text: str, seed: int) -> random.Random:
    h = hashlib.sha256(f"{seed}::{text}".encode("utf-8")).hexdigest()
    return random.Random(int(h[:16], 16))


def mock_transcribe(expected: str, difficulty: str = "medium", seed: int = 7) -> tuple[str, float]:
    """Return (hypothesis, simulated_latency_ms). Deterministic in (expected, seed)."""
    rng = _seeded_rng(expected, seed)
    p = DIFFICULTY_WER.get(difficulty, 0.12)
    words = expected.split()
    out: list[str] = []
    for w in words:
        r = rng.random()
        if r < p * 0.45:
            continue  # deletion
        elif r < p:
            out.append(rng.choice(CONFUSIONS.get(w, [w + "ह", w])))  # substitution-ish
        else:
            out.append(w)
        if rng.random() < p * 0.12:
            out.append(rng.choice(FILLERS))  # insertion
    # simulated latency: base 120ms + 18ms/word + jitter, harder = slower
    mult = {"easy": 1.0, "medium": 1.35, "hard": 1.9}.get(difficulty, 1.35)
    latency = (120 + 18 * len(words)) * mult + rng.uniform(0, 60)
    return " ".join(out), round(latency, 2)


def real_engine_available() -> bool:
    try:
        import faster_whisper  # noqa: F401
    except Exception:
        return False
    # require explicit opt-in so `pip install` alone never downloads weights
    return os.getenv("VACHAN_REAL_ASR", "0") == "1"


def transcribe_file(path: str, difficulty: str = "medium", seed: int = 7) -> dict:
    """Transcribe an audio file. Real faster-whisper if opted in, else mock-note.

    Mock has no audio to work from, so it returns engine=mock with an empty
    hypothesis and measured (tiny) latency — callers that want scored evals
    should use /eval with text or eval/run.py. Kept honest: we never pretend
    mock listened to the waveform.
    """
    started = time.perf_counter()
    if real_engine_available():
        try:
            from faster_whisper import WhisperModel

            model_name = os.getenv("VACHAN_MODEL", "tiny")
            model = WhisperModel(model_name, device="cpu", compute_type="int8")
            segments, _info = model.transcribe(path, language="hi")
            text = " ".join(s.text.strip() for s in segments).strip()
            ms = (time.perf_counter() - started) * 1000
            return {"transcript": text, "latency_ms": round(ms, 2), "engine": f"faster-whisper:{model_name}"}
        except Exception as e:  # fall through to mock, but report why
            ms = (time.perf_counter() - started) * 1000
            return {
                "transcript": "",
                "latency_ms": round(ms, 2),
                "engine": "mock",
                "warning": f"real ASR failed ({type(e).__name__}: {e}), fell back to mock",
            }
    ms = (time.perf_counter() - started) * 1000
    # estimate "audio seconds" from file size so cost model has an input
    try:
        audio_sec = max(0.5, os.path.getsize(path) / 32000.0)
    except OSError:
        audio_sec = 3.0
    return {
        "transcript": "",
        "latency_ms": round(ms + audio_sec * 2, 2),
        "engine": "mock",
        "warning": "mock mode: no acoustic model loaded; use /eval with text for scored evals",
    }


def engine_name() -> str:
    return "faster-whisper" if real_engine_available() else "mock"


# ==========================================================================
# v3 — streaming STT adapter (SPEC-019 latency lab)
# ==========================================================================
# Audio is delivered in 1s chunks, exactly like a browser MediaRecorder →
# server WebSocket hop. The cost model below is a *layer* model: every number
# charged is attributed to a named layer so the CI budget gate can say WHICH
# layer blew the budget instead of just "too slow".
#
# Voice-UX convention used here (documented so the numbers are comparable):
#   TTFT = end-of-speech (endpointed) → first agent audio byte
#        = endpoint + asr_first + (retrieval | cache) + tts_first
#   e2e  = TTFT + the remaining streaming decode (asr_stream)
# The retrieval round-trip is deliberately the most expensive layer: that is
# the VoiceAgentRAG (arXiv 2026) finding this lab exists to demonstrate.
#
# Two timelines run at once, like a real streaming agent:
#   * decode timeline  — chunk k lands at endpoint+asr_first+Σ(chunk costs);
#     inter-chunk gap is measured here because "did the stream stall" is a
#     property of the decoder, not of the retrieval the agent does meanwhile.
#   * agent timeline   — TTFT and e2e are charged serially through the clock
#     so the layer costs sum exactly to e2e.
CHUNK_MS = 1000.0
RETRIEVAL_TOP_K = 4

# per-difficulty layer costs in ms: (endpoint, asr_first, asr_chunk, tts_first)
STREAM_LAYER_MS: dict[str, tuple[float, float, float, float]] = {
    "easy": (55.0, 65.0, 55.0, 80.0),
    "medium": (70.0, 82.0, 72.0, 95.0),
    "hard": (88.0, 105.0, 95.0, 115.0),
    "noise": (105.0, 125.0, 112.0, 135.0),
}
# doc-embedding + ANN search + rerank round trip (CPU, warm index)
RETRIEVAL_BASE_MS = 360.0
RETRIEVAL_PER_TOPK_MS = 14.0
# cache fast path: embed the query + scan the index (no retrieval round trip)
CACHE_LOOKUP_MS = 10.0
# realistic dial/network jitter applied to every layer
STREAM_JITTER_MS = 14.0
# Hindi conversational speech: ~0.30s of audio per spoken word
WORD_AUDIO_SEC = 0.30
AUDIO_PAD_SEC = 0.45
MIN_AUDIO_SEC = 1.2
MAX_AUDIO_SEC = 5.0


class MonotonicClock:
    """Real wall clock. Same 2-method interface as `eval.latency.FakeClock`,
    so any streaming trace is reproducible under a fake clock."""

    def t_ms(self) -> float:
        return time.perf_counter() * 1000.0

    def advance(self, ms: float) -> None:
        if ms > 0:
            time.sleep(ms / 1000.0)


def audio_seconds(text: str) -> float:
    """Simulated spoken duration of `text` in seconds (deterministic)."""
    words = max(1, len(text.split()))
    return round(min(MAX_AUDIO_SEC, max(MIN_AUDIO_SEC, WORD_AUDIO_SEC * words + AUDIO_PAD_SEC)), 3)


def stream_plan(text: str, chunk_ms: float = CHUNK_MS) -> tuple[int, list[int]]:
    """(#chunks, words per chunk) for a streaming decode of `text`.

    Words are spread as evenly as the chunk grid allows; the first chunk gets
    the remainder so no words are invented or dropped. A streaming decoder
    really does commit to a *partial* hypothesis on chunk 1, so the uneven
    first chunk is the honest shape.
    """
    words = max(1, len(text.split()))
    n_chunks = max(1, math.ceil(audio_seconds(text) / (chunk_ms / 1000.0)))
    base, extra = divmod(words, n_chunks)
    per_chunk = [base + (1 if i < extra else 0) for i in range(n_chunks)]
    return n_chunks, per_chunk


def stream_transcribe(
    expected: str,
    difficulty: str = "medium",
    seed: int = 7,
    *,
    clock=None,
    chunk_ms: float = CHUNK_MS,
    top_k: int = RETRIEVAL_TOP_K,
    cache_hit: bool = False,
) -> dict:
    """Streaming STT trace: chunked hypothesis + TTFT / inter-chunk / e2e.

    The hypothesis is byte-identical to `mock_transcribe(expected, difficulty,
    seed)` — only the *timing* is new — so WER/CER and the v2 baseline are
    untouched by enabling stream mode.

    `cache_hit=True` takes the **fast path**: the near-duplicate was answered
    from `app/cache.py` so the retrieval round trip is skipped and replaced by
    the (10ms) cache lookup. `cache_hit=False` is the **slow path**: full
    stream + retrieval. Never blocks when `clock` is a FakeClock.
    """
    clock = clock or MonotonicClock()
    hyp, _ = mock_transcribe(expected, difficulty, seed)
    rng = _seeded_rng(f"stream::{expected}", seed)
    endpoint_ms, asr_first_ms, asr_chunk_ms, tts_first_ms = STREAM_LAYER_MS.get(
        difficulty, STREAM_LAYER_MS["medium"]
    )
    jit = lambda: round(rng.uniform(0.0, STREAM_JITTER_MS), 2)  # noqa: E731

    n_chunks, per_chunk = stream_plan(hyp, chunk_ms)
    words = hyp.split()
    groups, idx = [], 0
    for count in per_chunk:
        groups.append(words[idx : idx + count])
        idx += count

    t0 = clock.t_ms()
    chunks: list[dict] = []

    # 1. VAD / endpointing pre-roll: you must hear a beat of silence to know
    #    the speaker stopped before you can commit to an answer.
    endpoint = round(endpoint_ms + jit(), 2)
    clock.advance(endpoint)
    # 2. first partial decode → first text token out
    asr_first = round(asr_first_ms + jit(), 2)
    clock.advance(asr_first)
    # Per-chunk decode costs for the rest of the audio. Computed up front so
    # the *decode timeline* (chunk cadence) is independent of the agent
    # timeline (retrieval + synthesis): a real decoder keeps emitting while
    # the agent is busy retrieving and speaking.
    tail_costs = [round(asr_chunk_ms + jit(), 2) for _ in groups[1:]]
    decode_t = [round(endpoint + asr_first, 2)]
    for cost in tail_costs:
        decode_t.append(round(decode_t[-1] + cost, 2))
    chunks.append(
        {"index": 0, "text": " ".join(groups[0]), "t_ms": decode_t[0], "words": len(groups[0])}
    )
    for i, group in enumerate(groups[1:], start=1):
        chunks.append(
            {"index": i, "text": " ".join(group), "t_ms": decode_t[i], "words": len(group)}
        )
    # 3. answer path: cache fast path (10ms lookup) vs retrieval slow path
    if cache_hit:
        answer_ms = round(CACHE_LOOKUP_MS + jit() * 0.4, 2)
        layer = "cache"
    else:
        answer_ms = round(RETRIEVAL_BASE_MS + RETRIEVAL_PER_TOPK_MS * top_k + jit(), 2)
        layer = "retrieval"
    clock.advance(answer_ms)
    # 4. synthesizer first audio byte — TTFT lands here
    tts_first = round(tts_first_ms + jit(), 2)
    clock.advance(tts_first)
    ttft_ms = round(clock.t_ms() - t0, 2)
    # 5. the rest of the audio streams in while the agent talks
    asr_stream = 0.0
    for cost in tail_costs:
        clock.advance(cost)
        asr_stream = round(asr_stream + cost, 2)
    e2e_ms = round(clock.t_ms() - t0, 2)

    inter_chunk = [round(decode_t[i] - decode_t[i - 1], 2) for i in range(1, len(chunks))]
    return {
        "transcript": hyp,
        "chunks": chunks,
        "n_chunks": len(chunks),
        "audio_sec": audio_seconds(hyp),
        "ttft_ms": ttft_ms,
        "e2e_ms": e2e_ms,
        # last text token decoded (decode timeline) — the stream itself is done
        # here even though the agent is still retrieving / speaking.
        "decode_end_ms": decode_t[-1],
        "inter_chunk_ms": inter_chunk,
        "path": "fast" if cache_hit else "slow",
        "cache_hit": cache_hit,
        "top_k": top_k,
        "layers": {
            "endpoint": endpoint,
            "asr_first": asr_first,
            layer: answer_ms,
            "tts_first": tts_first,
            "asr_stream": asr_stream,
        },
        "engine": "mock-stream",
    }
