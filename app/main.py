"""Vachan Eval API — FastAPI backend, mock-by-default, no API keys."""

from __future__ import annotations

import base64
import io
import json
import math
import os
import struct
import sys
import tempfile
import time

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from . import asr as asr_mod
from . import metrics as M
from . import store

GOLDEN_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "golden.json")
HISTORY_PATH = os.path.join(os.path.dirname(__file__), "..", "eval", "history.jsonl")
RESULTS_PATH = os.path.join(os.path.dirname(__file__), "..", "eval", "results.json")

app = FastAPI(title="Vachan Eval", version="0.2.0", description="Hindi voice-AI eval harness (mock by default)")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)


def load_golden() -> list[dict]:
    with open(GOLDEN_PATH, encoding="utf-8") as f:
        return json.load(f)


def _latency_lab():
    """Lazy import of the eval latency lab (`eval/latency.py`).

    `eval/` is a sibling harness, not a package the API depends on, so it is
    resolved by path at call time instead of at import time (keeps `app`
    importable from anywhere). Returns None if it cannot be loaded.
    """
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if root not in sys.path:
        sys.path.insert(0, root)
    try:
        from eval import latency  # noqa: PLC0415

        return latency
    except Exception:  # pragma: no cover - only on a broken checkout
        return None


def _latency_from_artifact() -> dict | None:
    """Last committed `eval/run.py --stream` numbers (offline fallback)."""
    try:
        with open(RESULTS_PATH, encoding="utf-8") as f:
            s = json.load(f).get("summary", {})
    except (OSError, json.JSONDecodeError):
        return None
    if not s.get("stream"):
        return None
    return {
        "stream": s["stream"],
        "cache": s.get("cache"),
        "dual_path": s.get("dual_path"),
        "budget": s.get("budget"),
        "source": "eval/results.json (last `eval/run.py --stream` run)",
    }


# ---------- models ----------
class EvalItem(BaseModel):
    expected: str
    hypothesis: str = ""
    difficulty: str = "medium"
    id: str = ""
    lang: str = "hi"
    seed: int = 7


class EvalRequest(BaseModel):
    items: list[EvalItem] = Field(default_factory=list)
    seed: int = 7
    engine: str = "mock"
    # v3: attach the realtime latency report (TTFT / inter-chunk / e2e) and the
    # semantic-cache hit rate. Costs one extra fake-clock pass, never sleeps.
    stream: bool = False
    top_k: int = 4


class TTSRequest(BaseModel):
    text: str
    lang: str = "hi"
    seed: int = 7


# ---------- helpers ----------
def _sine_wav(text: str, seed: int = 7) -> bytes:
    """Deterministic placeholder TTS: beep-per-word WAV, 8kHz 16-bit mono.
    Honest mock — lets the frontend exercise audio playback + latency without
    pretending to be a real voice model."""
    import hashlib
    import random

    rng = random.Random(int(hashlib.sha256(f"{seed}::{text}".encode()).hexdigest()[:8], 16))
    sr = 8000
    words = max(1, len(text.split()))
    dur = min(8.0, 0.35 * words + rng.uniform(0, 0.15))
    n = int(sr * dur)
    freq = 220 + (len(text) % 7) * 40
    frames = bytearray()
    for i in range(n):
        t = i / sr
        env = 0.6 + 0.4 * math.sin(2 * math.pi * 2.2 * t)  # syllable-ish warble
        s = int(9000 * env * math.sin(2 * math.pi * freq * t))
        frames += struct.pack("<h", max(-32768, min(32767, s)))
    buf = io.BytesIO()
    buf.write(b"RIFF" + struct.pack("<I", 36 + len(frames)) + b"WAVEfmt ")
    buf.write(struct.pack("<IHHIIHH", 16, 1, 1, sr, sr * 2, 2, 16))
    buf.write(b"data" + struct.pack("<I", len(frames)))
    buf.write(bytes(frames))
    return buf.getvalue()


def score_items(items: list[dict], seed: int, engine: str) -> tuple[list[dict], dict]:
    results = []
    for it in items:
        expected = it.get("expected", "")
        hyp = it.get("hypothesis", "")
        diff = it.get("difficulty", "medium")
        if not hyp:
            hyp, lat = asr_mod.mock_transcribe(expected, diff, seed)
        else:
            # caller-supplied hypothesis: still charge a nominal scoring latency
            lat = 2.0
        results.append(
            {
                "id": it.get("id", ""),
                "lang": it.get("lang", "hi"),
                "domain": it.get("domain", ""),
                "difficulty": diff,
                "expected": expected,
                "hypothesis": hyp,
                "wer": round(M.wer(expected, hyp), 4),
                "cer": round(M.cer(expected, hyp), 4),
                "latency_ms": lat,
            }
        )
    summary = M.aggregate(results)
    summary["engine"] = engine
    summary["seed"] = seed
    # cost: assume ~4s audio per utterance for real engines
    audio_sec = 4.0 * len(results)
    summary["cost_usd"] = M.cost_usd(audio_sec, engine)
    summary["cost_per_eval_usd"] = round(summary["cost_usd"] / max(1, len(results)), 6)
    return results, summary


# ---------- history helpers (v2) ----------
def load_history(limit: int = 50) -> list[dict]:
    """Read eval/history.jsonl (one JSON object per run, oldest first)."""
    if not os.path.exists(HISTORY_PATH):
        return []
    entries: list[dict] = []
    try:
        with open(HISTORY_PATH, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except OSError:
        return []
    return entries[-limit:]


def history_delta(entries: list[dict]) -> dict | None:
    """Delta of latest vs previous history run (negative WER delta = improvement)."""
    if len(entries) < 2:
        return None
    prev, latest = entries[-2], entries[-1]
    out = {}
    for k in ("mean_wer", "mean_cer", "p95_ms"):
        if prev.get(k) is not None and latest.get(k) is not None:
            out[k] = round(float(latest[k]) - float(prev[k]), 4)
    return out


# ---------- routes ----------
@app.get("/health")
def health():
    return {"ok": True, "engine": asr_mod.engine_name(), "golden_n": len(load_golden())}


@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    suffix = os.path.splitext(file.filename or "audio.wav")[1] or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        path = tmp.name
    try:
        return asr_mod.transcribe_file(path)
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


@app.post("/tts")
def tts(req: TTSRequest):
    t0 = time.perf_counter()
    wav = _sine_wav(req.text, req.seed)
    ms = (time.perf_counter() - t0) * 1000
    return {
        "audio_b64": base64.b64encode(wav).decode(),
        "mime": "audio/wav",
        "chars": len(req.text),
        "latency_ms": round(ms, 2),
        "engine": "mock-sine",
        "note": "placeholder beep-per-word audio; swap with coqui/piper for real voice",
    }


@app.post("/eval")
def eval_endpoint(req: EvalRequest):
    items = [it.model_dump() for it in req.items] if req.items else [
        {**g, "hypothesis": ""} for g in load_golden()
    ]
    t0 = time.perf_counter()
    results, summary = score_items(items, req.seed, req.engine or "mock")
    summary["wall_ms"] = round((time.perf_counter() - t0) * 1000, 2)
    if req.stream:
        lab = _latency_lab()
        if lab is not None:
            out = lab.run_latency_lab(load_golden(), req.seed, top_k=req.top_k)
            summary["stream"] = out["stream"]
            summary["cache"] = out["cache"]
            summary["dual_path"] = out["stream"]["dual_path"]
            summary["budget"] = out["budget"]
            by_id = {t["id"]: t for t in out["traces"]}
            for r in results:
                t = by_id.get(r["id"])
                if t:
                    r["path"] = t["path"]
                    r["ttft_ms"] = t["ttft_ms"]
                    r["e2e_ms"] = t["e2e_ms"]
                    r["n_chunks"] = t["n_chunks"]
        else:  # pragma: no cover - only on a broken checkout
            summary["stream_error"] = "eval/latency.py not importable"
    run_id = store.save_run(summary["engine"], req.seed, summary, results)
    return {"run_id": run_id, "summary": summary, "results": results}


@app.get("/latency")
def latency_endpoint():
    """v3 realtime report for the dashboard's latency section.

    Runs the lab live on a fake clock (no sleeping, no GPU, no key). If the
    eval harness cannot be imported it falls back to the last committed
    `eval/run.py --stream` artifact and says which one it served.
    """
    lab = _latency_lab()
    if lab is not None:
        out = lab.run_latency_lab(load_golden(), seed=7)
        return {
            "stream": out["stream"],
            "cache": out["cache"],
            "dual_path": out["stream"]["dual_path"],
            "budget": out["budget"],
            "traces": out["traces"],
            "source": "live (eval/latency.py, fake clock)",
        }
    art = _latency_from_artifact()
    if art is not None:
        return art
    return JSONResponse({"error": "no latency data: run `python eval/run.py --stream`"}, status_code=503)


@app.get("/eval/history")
def history(limit: int = 20):
    return {"runs": store.list_runs(limit)}


@app.get("/leaderboard")
def leaderboard(limit: int = 20):
    """Static leaderboard: mock seeds vs documented real-model reference points.

    Real-model rows are literature/reference values (clearly labelled), not
    measured here — included so reviewers can see where the mock sits.

    v2: also serves `history` (every eval/run.py run, oldest first) and
    `delta` (latest vs previous run; negative WER delta = improvement)."""
    runs = store.list_runs(50)
    rows: list[dict] = []
    for r in runs:
        s = r.get("summary", {})
        rows.append(
            {
                "name": f"mock run #{r.get('id')} (seed {r.get('seed')})",
                "mean_wer": s.get("mean_wer"),
                "p95_ms": (s.get("latency") or {}).get("p95_ms"),
                "source": "measured-here",
            }
        )
    rows += [
        {"name": "faster-whisper small (ref, Hindi clean)", "mean_wer": 0.18, "p95_ms": 900, "source": "reference"},
        {"name": "faster-whisper base (ref, Hindi noisy)", "mean_wer": 0.31, "p95_ms": 600, "source": "reference"},
        {"name": "mock-easy (ref)", "mean_wer": 0.05, "p95_ms": 350, "source": "reference"},
    ]
    history = load_history(limit)
    return {"rows": rows, "history": history, "delta": history_delta(history)}


@app.get("/")
def root():
    return JSONResponse(
        {
            "name": "vachan-eval",
            "docs": "/docs",
            "endpoints": [
                "/health",
                "/transcribe",
                "/tts",
                "/eval",
                "/eval/history",
                "/latency",
                "/leaderboard",
            ],
        }
    )
