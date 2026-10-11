"""SPEC-019 v3 — streaming STT adapter, fake clocks, latency budget gate.

Every latency assertion runs on `FakeClock`, so the 30-utterance trace
(≈42s of virtual voice time) completes in milliseconds and the numbers are
byte-identical on every machine. Nothing here sleeps.
"""
import json
import os
import subprocess
import sys

import pytest

from app import asr as asr_mod
from app import metrics as M
from app.cache import SemanticCache
from eval.latency import (
    E2E_BUDGET_MS,
    TTFT_BUDGET_MS,
    FakeClock,
    budget_report,
    build_cache,
    cache_lab,
    check_budget,
    latency_report,
    run_latency_lab,
    stream_run,
)
from eval.run import category_of

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
GOLDEN = os.path.join(ROOT, "data", "golden.json")


def _golden():
    with open(GOLDEN, encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def golden():
    return _golden()


@pytest.fixture(scope="module")
def lab(golden):
    return run_latency_lab(golden, seed=7)


# ---------------------------------------------------------------- clocks --
def test_fake_clock_advances_and_records_charges():
    c = FakeClock(start_ms=500.0)
    assert c.t_ms() == 500.0
    c.advance(120)
    c.advance(80.5)
    assert c.t_ms() == 700.5
    assert c.charges == [120, 80.5]
    assert c.virtual_elapsed_ms == 200.5
    assert c.charged_ms == 200.5  # every advance is attributed


def test_fake_clock_never_sleeps_and_rejects_backwards_time():
    import time

    c = FakeClock()
    for _ in range(5000):
        c.advance(37)
    t0 = time.perf_counter()
    c2 = FakeClock()
    for _ in range(5000):
        c2.advance(37)
    assert (time.perf_counter() - t0) < 0.05  # 185s of virtual time, instant
    with pytest.raises(ValueError):
        c2.advance(-1)


# ------------------------------------------------------- streaming adapter --
def test_stream_plan_covers_every_word_exactly_once():
    for g in _golden():
        hyp, _ = asr_mod.mock_transcribe(g["expected"], g["difficulty"], seed=7)
        n_chunks, per_chunk = asr_mod.stream_plan(hyp)
        assert sum(per_chunk) == len(hyp.split())
        assert all(n > 0 for n in per_chunk)
        assert len(per_chunk) == n_chunks == asr_mod.stream_transcribe(
            g["expected"], g["difficulty"], seed=7, clock=FakeClock()
        )["n_chunks"]


def test_stream_transcribe_hypothesis_is_identical_to_batch_mock(golden):
    """The streamed text is the v1 mock hypothesis, so WER/baseline never move."""
    for g in golden:
        batch, _ = asr_mod.mock_transcribe(g["expected"], g["difficulty"], seed=7)
        s = asr_mod.stream_transcribe(g["expected"], g["difficulty"], seed=7, clock=FakeClock())
        assert s["transcript"] == batch
        assert " ".join(c["text"] for c in s["chunks"]) == batch
        assert M.wer(g["expected"], s["transcript"]) == M.wer(g["expected"], batch)


def test_chunk_timestamps_and_inter_chunk_gaps_are_monotonic(golden):
    g = golden[0]
    s = asr_mod.stream_transcribe(g["expected"], g["difficulty"], seed=7, clock=FakeClock())
    ts = [c["t_ms"] for c in s["chunks"]]
    assert ts == sorted(ts)
    assert ts[0] == pytest.approx(s["layers"]["endpoint"] + s["layers"]["asr_first"], abs=0.01)
    assert len(s["inter_chunk_ms"]) == s["n_chunks"] - 1
    assert all(gap > 0 for gap in s["inter_chunk_ms"])
    assert ts[-1] == pytest.approx(s["decode_end_ms"], abs=0.01)
    # inter-chunk cadence is the decoder's, not the retrieval the agent does
    for gap in s["inter_chunk_ms"]:
        assert gap < s["layers"]["retrieval"]


def test_layers_sum_exactly_to_e2e_and_ttft(lab):
    for t in lab["traces"]:
        assert sum(t["layers"].values()) == pytest.approx(t["e2e_ms"], abs=0.05)
        ttft_layers = {k: v for k, v in t["layers"].items() if k != "asr_stream"}
        assert sum(ttft_layers.values()) == pytest.approx(t["ttft_ms"], abs=0.05)
        assert t["ttft_ms"] <= t["e2e_ms"]
        # the path decides which answer layer was paid
        assert ("retrieval" in t["layers"]) == (t["path"] == "slow")
        assert ("cache" in t["layers"]) == (t["path"] == "fast")


def test_fast_path_skips_retrieval_and_is_faster_on_both_metrics(lab):
    for t in lab["traces"]:
        assert t["fast_e2e_ms"] < t["slow_e2e_ms"]
        assert t["fast_ttft_ms"] < t["slow_ttft_ms"]
        # the whole delta is the retrieval round trip the cache removes
        assert t["slow_e2e_ms"] - t["fast_e2e_ms"] == pytest.approx(
            t["slow_ttft_ms"] - t["fast_ttft_ms"], abs=0.05
        )
    d = lab["stream"]["dual_path"]
    assert d["fast_hits"] + d["slow_misses"] == lab["stream"]["n"]
    assert d["speedup"] > 1.0 and d["ttft_speedup"] > 1.0


def test_stream_run_is_deterministic_across_fake_clocks(golden):
    a, _, ca = stream_run(golden, seed=7, cache=build_cache(golden))
    b, _, cb = stream_run(golden, seed=7, cache=build_cache(golden))
    ra, rb = latency_report(a), latency_report(b)
    assert ra["ttft_ms"] == rb["ttft_ms"]
    assert ra["e2e_ms"] == rb["e2e_ms"]
    assert ra["inter_chunk_ms"] == rb["inter_chunk_ms"]
    # 60 traces x ~700ms of virtual time, zero wall time
    assert sum(c.virtual_elapsed_ms for c in ca) > 30_000
    assert sum(c.charged_ms for c in ca) == pytest.approx(sum(c.charged_ms for c in cb))


# --------------------------------------------------------------- reporting --
def test_report_has_ttft_inter_chunk_and_e2e_percentiles(lab):
    s = lab["stream"]
    for key in ("ttft_ms", "ttft_fast_ms", "ttft_slow_ms", "inter_chunk_ms", "e2e_ms", "e2e_fast_ms", "e2e_slow_ms"):
        st = s[key]
        assert {"p50_ms", "p95_ms", "mean_ms", "max_ms"} <= set(st), key
        assert st["p50_ms"] <= st["p95_ms"] <= st["max_ms"]
    assert s["inter_chunk_ms"]["p95_ms"] < s["ttft_slow_ms"]["p95_ms"]  # stream does not stall
    assert set(s["by_category"]) == {"easy", "medium", "hard", "noise", "adversarial"}
    for cat, v in s["by_category"].items():
        assert v["n"] == sum(1 for g in _golden() if category_of(g) == cat)
        assert v["e2e_p95_ms"] >= v["ttft_p95_ms"]


def test_adversarial_never_takes_the_fast_path(lab):
    for cat, v in lab["stream"]["by_category"].items():
        if cat == "adversarial":
            assert v["hit_rate"] == 0.0
    slow_adversarial = [t for t in lab["traces"] if t["category"] == "adversarial"]
    assert len(slow_adversarial) == 5
    assert all(t["path"] == "slow" for t in slow_adversarial)


# ----------------------------------------------------------------- budget --
def test_budget_passes_on_the_shipped_dual_path(lab):
    ok, msgs = check_budget(lab["stream"])
    assert ok is True
    assert any("ttft_p95_fast" in m and "PASS" in m for m in msgs)
    assert any("e2e_p95_served" in m and "PASS" in m for m in msgs)
    assert lab["budget"]["breached"] == []


def test_budget_failure_names_the_breaching_layer(lab):
    ok, msgs = check_budget(lab["stream"], ttft_budget_ms=250.0)
    assert ok is False
    fail = [m for m in msgs if m.startswith("BUDGET FAIL")]
    assert fail and "ttft_p95_fast" in fail[0]
    assert "layer " in fail[0]  # names the layer, not just "too slow"
    bud = budget_report(lab["stream"], ttft_budget_ms=250.0)
    assert bud["breached"] == ["ttft_p95_fast"]
    assert bud["breached_layers"] == ["tts_first"]


def test_budget_names_retrieval_as_the_slow_path_breach(lab):
    """The slow path's TTFT breach must be attributed to the retrieval layer."""
    bud = budget_report(lab["stream"])
    slow = [c for c in bud["checks"] if c["name"] == "ttft_p95_slow"][0]
    assert slow["gated"] is False
    assert slow["pass"] is False
    assert slow["value_ms"] > TTFT_BUDGET_MS
    assert slow["dominant_layer"] == "retrieval"
    assert slow["dominant_layer_p95_ms"] == pytest.approx(
        lab["stream"]["layers_slow_p95_ms"]["retrieval"], abs=0.01
    )


def test_budget_e2e_breach_names_a_streaming_layer(lab):
    ok, msgs = check_budget(lab["stream"], e2e_budget_ms=800.0)
    assert ok is False
    e2e = [c for c in budget_report(lab["stream"], e2e_budget_ms=800.0)["checks"] if c["name"] == "e2e_p95_served"][0]
    assert e2e["pass"] is False
    assert e2e["dominant_layer"] in {"retrieval", "asr_stream"}
    assert any("e2e_p95_served" in m for m in msgs)


def test_budget_constants_are_the_spec_values():
    assert (TTFT_BUDGET_MS, E2E_BUDGET_MS) == (400.0, 1200.0)


# ------------------------------------------------------------------- cache --
def test_cache_lab_measures_hit_rate_with_negative_controls(golden):
    out = cache_lab(golden, seed=7, cache=build_cache(golden))
    assert out["classes"]["repeat"]["hit_rate"] == 1.0
    assert out["classes"]["novel"]["hit_rate"] == 0.0  # control: never over-serve
    assert 0.0 < out["classes"]["rephrase"]["hit_rate"] < 1.0
    assert out["hit_rate"] == pytest.approx(
        (out["classes"]["repeat"]["hits"] + out["classes"]["rephrase"]["hits"] + out["classes"]["novel"]["hits"])
        / out["eligible_probes"],
        abs=1e-4,
    )
    assert out["unsafe_served"] == 0


def test_cache_never_serves_adversarial_even_on_exact_match(golden):
    cache = build_cache(golden)
    for g in golden:
        if category_of(g) != "adversarial":
            continue
        d = cache.lookup(g["expected"], g["expected"], "adversarial")
        assert d.hit is False
        assert d.reason == "adversarial"
        assert d.path == "slow"
        assert d.payload is None
    assert cache.stats()["bypassed_adversarial"] == 5


def test_stream_run_routes_through_the_cache(golden):
    traces, stats, clocks = stream_run(golden, seed=7, cache=build_cache(golden))
    assert stats["entries"] == len(golden)
    for t in traces:
        if t.category == "adversarial":
            assert t.path == "slow"
        else:
            assert t.cache["reason"] in {"hit", "below-threshold"}
            assert t.path == ("fast" if t.cache["reason"] == "hit" else "slow")


def test_empty_cache_and_degenerate_text_never_raise():
    empty = SemanticCache()
    d = empty.lookup("", "", "easy")
    assert d.hit is False and d.path == "slow" and d.reason == "no-entry"
    c = SemanticCache()
    assert c.put("", {}, "", "easy") is None
    d = c.lookup("कुछ भी", "कुछ भी", "easy")
    assert d.hit is False and d.reason == "no-entry"


# --------------------------------------------------------------------- CLI --
def test_run_py_stream_cli_exits_zero_and_prints_ttft_and_cache(tmp_path):
    out = tmp_path / "results.json"
    proc = subprocess.run(
        [sys.executable, os.path.join(ROOT, "eval", "run.py"), "--stream", "--budget", "--no-history"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "TTFT fast" in proc.stdout and "TTFT served" in proc.stdout
    assert "inter-chunk" in proc.stdout
    assert "hit_rate" in proc.stdout and "semantic cache" in proc.stdout
    assert "BUDGET PASS" in proc.stdout and "OVERALL PASS" in proc.stdout
    assert "GATE PASS" in proc.stdout and "BASELINE OK mean_wer" in proc.stdout
    blob = json.load(open(os.path.join(ROOT, "eval", "results.json"), encoding="utf-8"))
    assert blob["summary"]["stream"]["ttft_fast_ms"]["p95_ms"] <= TTFT_BUDGET_MS
    assert blob["summary"]["cache"]["hit_rate"] > 0
    assert all("ttft_ms" in r for r in blob["results"])


def test_run_py_stream_does_not_change_wer_or_baseline(tmp_path):
    """Stream mode is additive: the v1/v2 gates and baseline stay identical."""
    base = subprocess.run(
        [sys.executable, os.path.join(ROOT, "eval", "run.py"), "--no-history"],
        cwd=ROOT, capture_output=True, text=True, timeout=300,
    )
    streamed = subprocess.run(
        [sys.executable, os.path.join(ROOT, "eval", "run.py"), "--stream", "--no-history"],
        cwd=ROOT, capture_output=True, text=True, timeout=300,
    )
    assert base.returncode == 0 and streamed.returncode == 0
    grab = lambda out: [ln for ln in out.splitlines() if ln.startswith("mean_wer=")][0]
    assert grab(base.stdout) == grab(streamed.stdout)
    assert "OVERALL PASS (gates + baseline)" in base.stdout
    assert "OVERALL PASS (gates + baseline)" in streamed.stdout
