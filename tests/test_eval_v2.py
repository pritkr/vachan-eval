"""SPEC-014 v2: category gates, baseline regression, leaderboard history."""
import json
import os

from fastapi.testclient import TestClient

import app.main as main_mod
from app.main import app
from eval.run import append_history, category_of, check_baseline, run

client = TestClient(app)
GOLDEN = os.path.join(os.path.dirname(__file__), "..", "data", "golden.json")
BASELINE = os.path.join(os.path.dirname(__file__), "..", "eval", "baseline.json")


def _golden():
    with open(GOLDEN, encoding="utf-8") as f:
        return json.load(f)


def test_category_of_explicit_and_fallback():
    assert category_of({"id": "x", "difficulty": "easy", "category": "noise"}) == "noise"
    assert category_of({"id": "noise-9", "difficulty": "hard"}) == "noise"
    assert category_of({"id": "adv-9", "difficulty": "hard"}) == "adversarial"
    assert category_of({"id": "ration-1", "difficulty": "easy"}) == "easy"


def test_category_gates_pass_default_seed():
    results, summary = run(_golden(), seed=7)
    assert set(summary["categories"]) == {"easy", "medium", "hard", "noise", "adversarial"}
    assert summary["categories"]["adversarial"]["gate_wer_max"] is None  # report-only
    assert summary["category_pass"] is True
    assert summary["pass"] is True
    assert all("category" in r for r in results)


def test_category_gate_names_failing_category(monkeypatch):
    from app import asr

    monkeypatch.setattr(asr, "mock_transcribe", lambda *a, **k: ("बिल्कुल गलत परिकल्पना पाठ", 100.0))
    monkeypatch.setattr("eval.run.asr_mod", asr)
    _results, summary = run(_golden(), seed=7)
    assert summary["pass"] is False
    assert summary["categories"]["easy"]["pass"] is False  # failing category is named
    assert summary["categories"]["adversarial"]["pass"] is True  # report-only never fails


def test_baseline_check_passes_current():
    _results, summary = run(_golden(), seed=7)
    ok, msgs = check_baseline(summary, BASELINE)
    assert ok is True
    assert any("mean_wer" in m for m in msgs)


def test_baseline_regression_detected():
    _results, summary = run(_golden(), seed=7)
    summary["mean_wer"] = round(summary["mean_wer"] + 0.05, 4)
    ok, msgs = check_baseline(summary, BASELINE)
    assert ok is False
    assert any("REGRESSED mean_wer" in m for m in msgs)


def test_append_history_writes_jsonl(tmp_path):
    _results, summary = run(_golden(), seed=7)
    p = str(tmp_path / "history.jsonl")
    append_history(summary, p)
    lines = open(p, encoding="utf-8").read().strip().split("\n")
    assert len(lines) == 1
    entry = json.loads(lines[0])
    assert entry["model"] == "mock" and entry["mean_wer"] == summary["mean_wer"]
    assert entry["categories"]["easy"] == summary["categories"]["easy"]["mean_wer"]
    assert entry["p95_ms"] == summary["latency"]["p95_ms"]


def test_leaderboard_serves_history_and_delta(tmp_path, monkeypatch):
    p = tmp_path / "history.jsonl"
    rows = [
        {"ts": 1.0, "model": "mock", "seed": 7, "n": 30, "mean_wer": 0.17,
         "mean_cer": 0.12, "p95_ms": 600.0, "categories": {"easy": 0.03}, "pass": True},
        {"ts": 2.0, "model": "mock", "seed": 7, "n": 30, "mean_wer": 0.16,
         "mean_cer": 0.12, "p95_ms": 595.0, "categories": {"easy": 0.03}, "pass": True},
    ]
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    monkeypatch.setattr(main_mod, "HISTORY_PATH", str(p))
    r = client.get("/leaderboard")
    assert r.status_code == 200
    body = r.json()
    assert "rows" in body and len(body["history"]) == 2
    assert body["delta"]["mean_wer"] == round(0.16 - 0.17, 4)  # negative = improvement
