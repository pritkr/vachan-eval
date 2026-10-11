"""ASR mock + API + eval-gate + store tests."""
import json
import os

from fastapi.testclient import TestClient

from app import asr
from app.main import app
from eval.run import run, WER_GATE, P95_GATE_MS

client = TestClient(app)
GOLDEN = os.path.join(os.path.dirname(__file__), "..", "data", "golden.json")


def test_mock_deterministic():
    a, _ = asr.mock_transcribe("राशन कार्ड में नाम जुड़वाऊं", "medium", seed=7)
    b, _ = asr.mock_transcribe("राशन कार्ड में नाम जुड़वाऊं", "medium", seed=7)
    assert a == b


def test_mock_seed_varies_or_harder_worse():
    # different seeds may differ; hard should be no better than easy on average
    import statistics

    with open(GOLDEN, encoding="utf-8") as f:
        golden = json.load(f)
    from app import metrics as M

    wers = {}
    for diff in ("easy", "hard"):
        ws = []
        for g in [x for x in golden if x["difficulty"] == diff][:5]:
            h, _ = asr.mock_transcribe(g["expected"], diff, seed=7)
            ws.append(M.wer(g["expected"], h))
        wers[diff] = statistics.mean(ws)
    assert wers["hard"] >= wers["easy"]


def test_engine_defaults_mock(monkeypatch):
    monkeypatch.delenv("VACHAN_REAL_ASR", raising=False)
    assert asr.engine_name() == "mock"
    assert not asr.real_engine_available()


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["golden_n"] == 30


def test_eval_endpoint_full_golden():
    r = client.post("/eval", json={"seed": 7, "engine": "mock"})
    assert r.status_code == 200
    body = r.json()
    assert body["summary"]["n"] == 30
    assert len(body["results"]) == 30
    assert body["summary"]["mean_wer"] <= 0.6  # sanity upper bound


def test_eval_single_item_honest_wer():
    r = client.post(
        "/eval",
        json={"items": [{"id": "t1", "expected": "राम घर गया", "hypothesis": "राम घर गया"}], "seed": 7},
    )
    assert r.json()["results"][0]["wer"] == 0.0


def test_tts_returns_wav():
    r = client.post("/tts", json={"text": "राशन कार्ड", "lang": "hi"})
    assert r.status_code == 200
    import base64

    raw = base64.b64decode(r.json()["audio_b64"])
    assert raw[:4] == b"RIFF"


def test_transcribe_mock_never_crashes():
    r = client.post("/transcribe", files={"file": ("a.wav", b"\x00" * 3200, "audio/wav")})
    assert r.status_code == 200
    assert r.json()["engine"] == "mock"


def test_eval_gate_passes_default_seed():
    with open(GOLDEN, encoding="utf-8") as f:
        golden = json.load(f)
    _results, summary = run(golden, seed=7)
    assert summary["mean_wer"] <= WER_GATE
    assert summary["latency"]["p95_ms"] <= P95_GATE_MS
    assert summary["pass"] is True


def test_leaderboard_shape():
    r = client.get("/leaderboard")
    assert r.status_code == 200
    assert "rows" in r.json()


def test_history_records_runs():
    client.post("/eval", json={"seed": 7, "engine": "mock"})
    r = client.get("/eval/history?limit=3")
    assert r.status_code == 200
    assert len(r.json()["runs"]) >= 1


def test_golden_has_30_and_all_categories():
    with open(GOLDEN, encoding="utf-8") as f:
        golden = json.load(f)
    assert len(golden) == 30
    langs = {g["lang"] for g in golden}
    assert {"hi", "bho", "mag"} <= langs
    domains = {g["domain"] for g in golden}
    assert {"ration", "scholarship", "rte-admission"} <= domains
    from collections import Counter
    cats = Counter(g["category"] for g in golden)
    assert cats["adversarial"] == 5
    assert set(cats) == {"easy", "medium", "hard", "noise", "adversarial"}
