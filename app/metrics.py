"""WER / CER / latency metrics. Pure stdlib, no deps."""

from __future__ import annotations

from .normalize import normalize


def _levenshtein(a: list | str, b: list | str) -> int:
    # classic DP on sequences (words or chars)
    n, m = len(a), len(b)
    if n == 0:
        return m
    if m == 0:
        return n
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        cur = [i] + [0] * m
        ai = a[i - 1]
        for j in range(1, m + 1):
            cost = 0 if ai == b[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[m]


def wer(reference: str, hypothesis: str) -> float:
    """Word Error Rate on normalized text (see app.normalize).

    Returns 0.0 for two empty strings, 1.0 if ref empty but hyp not."""
    ref_w = normalize(reference).split()
    hyp_w = normalize(hypothesis).split()
    if not ref_w:
        return 0.0 if not hyp_w else 1.0
    return _levenshtein(ref_w, hyp_w) / len(ref_w)


def cer(reference: str, hypothesis: str) -> float:
    """Character Error Rate on normalized text (whitespace-insensitive splice: compare raw strings)."""
    r = normalize(reference)
    h = normalize(hypothesis)
    if not r:
        return 0.0 if not h else 1.0
    # char-level incl. spaces gives a harsher, honest number; use it
    return _levenshtein(list(r), list(h)) / max(1, len(list(r)))


def percentile(values: list[float], p: float) -> float:
    """Nearest-rank percentile. p in [0,100]. Returns 0.0 on empty."""
    if not values:
        return 0.0
    s = sorted(values)
    if len(s) == 1:
        return float(s[0])
    k = (len(s) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(s) - 1)
    frac = k - f
    return float(s[f] + (s[c] - s[f]) * frac)


def latency_stats(latencies_ms: list[float]) -> dict:
    if not latencies_ms:
        return {"count": 0, "p50_ms": 0.0, "p95_ms": 0.0, "mean_ms": 0.0, "max_ms": 0.0}
    return {
        "count": len(latencies_ms),
        "p50_ms": round(percentile(latencies_ms, 50), 2),
        "p95_ms": round(percentile(latencies_ms, 95), 2),
        "mean_ms": round(sum(latencies_ms) / len(latencies_ms), 2),
        "max_ms": round(max(latencies_ms), 2),
    }


# ---- cost model (documented, mock-friendly) ----
# faster-whisper-small on CPU ~ real-time; we model $0.00006/sec
# mock mode costs $0 by definition (no GPU).
COST_PER_SEC_USD = 0.00006


def cost_usd(total_audio_sec: float, engine: str = "mock") -> float:
    if engine == "mock":
        return 0.0
    return round(total_audio_sec * COST_PER_SEC_USD, 6)


def aggregate(results: list[dict]) -> dict:
    """Aggregate per-utterance {wer, cer, latency_ms} dicts into a summary."""
    wers = [r["wer"] for r in results]
    cers = [r["cer"] for r in results]
    lats = [r["latency_ms"] for r in results]
    n = len(results)
    return {
        "n": n,
        "mean_wer": round(sum(wers) / n, 4) if n else 0.0,
        "mean_cer": round(sum(cers) / n, 4) if n else 0.0,
        "p50_wer": round(percentile(wers, 50), 4) if n else 0.0,
        "p95_wer": round(percentile(wers, 95), 4) if n else 0.0,
        "fail_rate_wer_gt_05": round(sum(1 for w in wers if w > 0.5) / n, 4) if n else 0.0,
        "latency": latency_stats(lats),
    }
