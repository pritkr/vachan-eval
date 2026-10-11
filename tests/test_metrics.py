"""Metrics unit tests."""
from app import metrics as M


def test_wer_identical_is_zero():
    assert M.wer("राशन कार्ड बनवाएं", "राशन कार्ड बनवाएं") == 0.0


def test_wer_one_substitution():
    # 1 of 3 words wrong -> 1/3
    assert abs(M.wer("a b c", "a x c") - 1 / 3) < 1e-9


def test_wer_empty_both():
    assert M.wer("", "") == 0.0


def test_wer_empty_ref_nonempty_hyp():
    assert M.wer("", "कुछ") == 1.0


def test_cer_identical_is_zero():
    assert M.cer("राम", "राम") == 0.0


def test_cer_single_char():
    assert M.cer("abc", "axc") > 0


def test_percentile_empty():
    assert M.percentile([], 95) == 0.0


def test_percentile_p50():
    assert M.percentile([1, 2, 3, 4], 50) == 2.5


def test_latency_stats_shape():
    s = M.latency_stats([100, 200, 300])
    assert s["count"] == 3 and s["p50_ms"] > 0 and s["p95_ms"] >= s["p50_ms"]


def test_cost_mock_is_zero():
    assert M.cost_usd(999, "mock") == 0.0


def test_cost_real_positive():
    assert M.cost_usd(100, "faster-whisper:small") > 0


def test_aggregate_counts():
    rows = [
        {"wer": 0.0, "cer": 0.0, "latency_ms": 100},
        {"wer": 0.5, "cer": 0.4, "latency_ms": 300},
    ]
    agg = M.aggregate(rows)
    assert agg["n"] == 2
    assert agg["mean_wer"] == 0.25
    assert agg["latency"]["count"] == 2
