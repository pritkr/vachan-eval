"""SPEC-019 v3 — latency lab: fake clocks, streaming STT metrics, budget gate.

Three jobs, in order of importance:

1. **Measure the realtime path honestly.** A voice turn is not one number. We
   report time-to-first-token (TTFT, the metric a user actually feels),
   inter-chunk gap (jitter/stalling between streamed tokens) and end-to-end
   latency, per category, on a **layer** model: `endpoint` (VAD pre-roll),
   `asr_first` (first partial decode), `retrieval` | `cache` (the answer path),
   `tts_first` (first audio byte) and `asr_stream` (the rest of the audio).
2. **Run the dual path.** `app/cache.py` serves near-duplicate utterances on a
   *fast path* (no retrieval round trip); everything else pays the *slow path*.
   `run_latency_lab` measures both, on identical inputs, so the speedup is a
   measurement and not a claim.
3. **Fail CI on a budget breach, naming the layer.** `check_budget` gates
   fast-path TTFT p95 ≤ 400ms and served e2e p95 ≤ 1200ms. When something
   breaches it prints the layer that breached, not just "too slow".

Every millisecond is charged to an injected clock, so `FakeClock` makes the
whole 30-utterance trace deterministic *and* instantaneous — no test ever
sleeps and no CI number depends on machine load.
"""

from __future__ import annotations

import os
import statistics
import sys
from dataclasses import dataclass, field

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app import asr as asr_mod  # noqa: E402
from app import metrics as M  # noqa: E402
from app.cache import SemanticCache, cosine, embed, equivalence_key, paraphrase  # noqa: E402

# v3 voice budget. TTFT is the interactive budget (what a caller perceives);
# e2e is the hard ceiling for any single utterance.
TTFT_BUDGET_MS = 400.0
E2E_BUDGET_MS = 1200.0

# layers that sit in front of the first audio byte (i.e. inside TTFT)
TTFT_LAYERS = ("endpoint", "asr_first", "cache", "retrieval", "tts_first")
# every layer, in causal order — this order *is* the trace
LAYER_ORDER = ("endpoint", "asr_first", "cache", "retrieval", "tts_first", "asr_stream")

#: Negative-control query set: fresh intents that are in no golden item, so a
#: correct cache must miss on every one of them. A cache that "hits" here is
#: over-serving, which is the same failure as the adversarial guard.
NOVEL_QUERIES: tuple[str, ...] = (
    "मेरी मम्मी का पुराना पेंशन कहाँ जमा होता है",
    "बच्चे को कौन सा वैक्सीन लगवाना चाहिए",
    "आधार कार्ड में आंख का नाम जोड़ना है",
    "किसान का खाता बैंक में जुड़ा है या नहीं",
    "बेटी की शिक्षा के लिए छात्रवृत्ति कैसे मिलेगी",
    "आरटीई वाली फीस का पैसा कब भरना है",
    "गांव की पानी की टंकी कहाँ बनवाएं",
    "आधार कार्ड गुम हो गया तो क्या करें",
    "पेंशन वाला फॉर्म किस ऑफिस में भरें",
    "मासूल भुगतान में देरी क्यों होती है",
)


# ==========================================================================
# clocks
# ==========================================================================
class FakeClock:
    """Deterministic, non-sleeping clock with the same 2-method interface as
    `app.asr.MonotonicClock`. `advance` moves virtual time and records the
    charge, so a test can assert both the trace and *where* the time went.
    """

    def __init__(self, start_ms: float = 1_000_000.0) -> None:
        self._t = float(start_ms)
        self.start_ms = float(start_ms)
        self.charges: list[float] = []

    def t_ms(self) -> float:
        return round(self._t, 4)

    def advance(self, ms: float) -> None:
        ms = float(ms)
        if ms < 0:
            raise ValueError("clock cannot go backwards (negative advance)")
        self._t += ms
        self.charges.append(round(ms, 4))

    @property
    def virtual_elapsed_ms(self) -> float:
        return round(self._t - self.start_ms, 4)

    @property
    def charged_ms(self) -> float:
        return round(sum(self.charges), 4)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"FakeClock(t={self.t_ms()}ms, +{self.virtual_elapsed_ms}ms virtual)"


# ==========================================================================
# traces
# ==========================================================================
@dataclass
class Trace:
    """One streamed utterance, both paths measured on identical input."""

    id: str
    category: str
    difficulty: str
    lang: str
    domain: str
    expected: str
    hypothesis: str
    path: str  # "fast" (cache) | "slow" (retrieval)
    cache: dict = field(default_factory=dict)  # CacheDecision.as_dict()
    fast: dict = field(default_factory=dict)
    slow: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        served = self.slow if self.path == "slow" else self.fast
        return {
            "id": self.id,
            "category": self.category,
            "difficulty": self.difficulty,
            "lang": self.lang,
            "domain": self.domain,
            "expected": self.expected,
            "hypothesis": self.hypothesis,
            "path": self.path,
            "cache": self.cache,
            "ttft_ms": served["ttft_ms"],
            "e2e_ms": served["e2e_ms"],
            "n_chunks": served["n_chunks"],
            "audio_sec": served["audio_sec"],
            "inter_chunk_p95_ms": M.percentile(served["inter_chunk_ms"], 95) if served["inter_chunk_ms"] else 0.0,
            "fast_ttft_ms": self.fast["ttft_ms"],
            "fast_e2e_ms": self.fast["e2e_ms"],
            "slow_ttft_ms": self.slow["ttft_ms"],
            "slow_e2e_ms": self.slow["e2e_ms"],
            "saved_ms": round(self.slow["e2e_ms"] - self.fast["e2e_ms"], 2),
            "layers": served["layers"],
        }


def _category_of(g: dict) -> str:
    """Same category rule as `eval/run.py` (kept local: no import cycle)."""
    if g.get("category"):
        return str(g["category"])
    gid = str(g.get("id", ""))
    if gid.startswith("noise-"):
        return "noise"
    if gid.startswith("adv-"):
        return "adversarial"
    return str(g.get("difficulty", "medium"))


def _payload(g: dict) -> dict:
    return {
        "id": g.get("id", ""),
        "answer": f"[cached answer] {g.get('domain', '—')}: {g.get('expected', '')}",
        "source": "semantic-cache",
    }


# ==========================================================================
# cache lab
# ==========================================================================
def build_cache(golden: list[dict], threshold: float | None = None) -> SemanticCache:
    """Warm the cache with every golden intent (a warm production cache).

    Key = the intent's transcript-equivalence key, surface = the utterance as
    first heard. Lookups then ask "has this user asked something equivalent
    before?", which is the only question a voice agent can cheaply answer.
    """
    cache = SemanticCache() if threshold is None else SemanticCache(threshold=threshold)
    for g in golden:
        cache.put(g["expected"], _payload(g), g["expected"], _category_of(g))
    return cache


def cache_lab(golden: list[dict], seed: int = 7, cache: SemanticCache | None = None) -> dict:
    """Measure hit rate, guard activity and the fast/slow speedup.

    Four probe classes, because a hit rate is meaningless without its
    negative controls:

    - **repeat**  — the same utterance again (should hit)
    - **rephrase**— a politeness-wrapped near-duplicate, strength cycled 1..3
      (should mostly hit; weak rephrasings fall below threshold)
    - **novel**   — a fresh intent absent from the cache (must miss every time)
    - **trap**    — a near-duplicate *surface* carrying a *different* intent
      (cosine says hit; the equivalence guard must refuse)

    Adversarial items are probed too, and must be bypassed even on an exact
    surface match — that is the invariant the report asserts.
    """
    cache = cache if cache is not None else build_cache(golden)
    cache.reset_stats()

    by_cat = [_category_of(g) for g in golden]
    eligible = [i for i, c in enumerate(by_cat) if c != "adversarial"]
    adversarial_idx = [i for i, c in enumerate(by_cat) if c == "adversarial"]

    classes: dict[str, dict] = {
        "repeat": {"n": 0, "hits": 0},
        "rephrase": {"n": 0, "hits": 0},
        "novel": {"n": 0, "hits": 0},
        "trap": {"n": 0, "hits": 0},
    }
    misroutes: list[dict] = []

    # repeat + rephrase on every non-adversarial intent
    for rank, i in enumerate(eligible):
        g = golden[i]
        d = cache.lookup(g["expected"], g["expected"], by_cat[i])
        classes["repeat"]["n"] += 1
        classes["repeat"]["hits"] += int(d.hit)
        strength = 1 + (rank % 3)
        probe = paraphrase(g["expected"], seed, strength)
        d = cache.lookup(probe, g["expected"], by_cat[i])
        classes["rephrase"]["n"] += 1
        classes["rephrase"]["hits"] += int(d.hit)

    # negative control: fresh intents must never be served
    for q in NOVEL_QUERIES:
        d = cache.lookup(q, q, "easy")
        classes["novel"]["n"] += 1
        classes["novel"]["hits"] += int(d.hit)
        if d.hit:
            misroutes.append({"probe": q, "served_key": d.key})

    # trap: near-duplicate surface, different intent -> guard must refuse
    cosine_would_misroute = 0
    for k, i in enumerate(eligible):
        j = eligible[(k + 1) % len(eligible)]
        a, b = golden[i], golden[j]
        probe = paraphrase(a["expected"], seed, 1)
        d = cache.lookup(probe, b["expected"], by_cat[j])
        classes["trap"]["n"] += 1
        classes["trap"]["hits"] += int(d.hit)
        if d.hit:
            misroutes.append({"probe": probe, "asked": b["expected"], "served_key": d.key})
        # counterfactual: on the first 10 traps, would raw cosine have served
        # a *different* intent? If yes, the guard is doing real work.
        if k < 10:
            v = embed(probe)
            best = max(cache.entries, key=lambda e: cosine(v, e.vec))
            if cosine(v, best.vec) >= cache.threshold and best.key != equivalence_key(b["expected"]):
                cosine_would_misroute += 1

    # adversarial: exact surface match must still be refused
    for i in adversarial_idx:
        g = golden[i]
        d = cache.lookup(g["expected"], g["expected"], by_cat[i])
        if d.hit:
            misroutes.append({"probe": g["expected"], "served_key": d.key, "note": "ADVERSARIAL SERVED"})

    stats = cache.stats()
    served_n = classes["repeat"]["n"] + classes["rephrase"]["n"] + classes["novel"]["n"]
    served_hits = classes["repeat"]["hits"] + classes["rephrase"]["hits"] + classes["novel"]["hits"]
    return {
        "threshold": stats["threshold"],
        "dim": stats["dim"],
        "entries": stats["entries"],
        "novel_queries": len(NOVEL_QUERIES),
        "classes": {k: {**v, "hit_rate": round(v["hits"] / v["n"], 4) if v["n"] else 0.0} for k, v in classes.items()},
        "eligible_probes": served_n,
        "hits": served_hits,
        "hit_rate": round(served_hits / served_n, 4) if served_n else 0.0,
        "adversarial_probes": len(adversarial_idx),
        "bypassed_adversarial": stats["bypassed_adversarial"],
        "guard_rejects": stats["guard_rejects"],
        "cosine_would_misroute": cosine_would_misroute,
        "unsafe_served": len(misroutes),
        "misroutes": misroutes[:10],
        "counters": stats,
    }


# ==========================================================================
# streaming run
# ==========================================================================
def stream_run(
    golden: list[dict],
    seed: int = 7,
    cache: SemanticCache | None = None,
    top_k: int = asr_mod.RETRIEVAL_TOP_K,
    chunk_ms: float = asr_mod.CHUNK_MS,
) -> tuple[list[Trace], dict, list[FakeClock]]:
    """Stream every utterance on both paths, using fake clocks.

    The cache probe uses a *paraphrased* surface form while the hypothesis
    under test is the unmodified golden utterance. That is deliberate: it
    stands in for "the user rephrased a question they already asked" without
    perturbing the hypothesis, so stream mode leaves v2 WER/CER and the
    baseline check bit-for-bit identical.
    """
    cache = cache if cache is not None else build_cache(golden)
    traces: list[Trace] = []
    clocks: list[FakeClock] = []
    for rank, g in enumerate(golden):
        cat = _category_of(g)
        probe = paraphrase(g["expected"], seed, 1 + (rank % 3))
        decision = cache.lookup(probe, g["expected"], cat)

        fast_clock, slow_clock = FakeClock(), FakeClock()
        fast = asr_mod.stream_transcribe(
            g["expected"], g.get("difficulty", "medium"), seed,
            clock=fast_clock, chunk_ms=chunk_ms, top_k=top_k, cache_hit=True,
        )
        slow = asr_mod.stream_transcribe(
            g["expected"], g.get("difficulty", "medium"), seed,
            clock=slow_clock, chunk_ms=chunk_ms, top_k=top_k, cache_hit=False,
        )
        clocks += [fast_clock, slow_clock]
        traces.append(
            Trace(
                id=g.get("id", ""),
                category=cat,
                difficulty=g.get("difficulty", "medium"),
                lang=g.get("lang", "hi"),
                domain=g.get("domain", ""),
                expected=g["expected"],
                hypothesis=fast["transcript"],
                path=decision.path,
                cache=decision.as_dict(),
                fast=fast,
                slow=slow,
            )
        )
    return traces, cache.stats(), clocks


# ==========================================================================
# report
# ==========================================================================
def _stats(values: list[float]) -> dict:
    if not values:
        return {"count": 0, "p50_ms": 0.0, "p95_ms": 0.0, "mean_ms": 0.0, "max_ms": 0.0}
    return {
        "count": len(values),
        "p50_ms": round(M.percentile(values, 50), 2),
        "p95_ms": round(M.percentile(values, 95), 2),
        "mean_ms": round(statistics.mean(values), 2),
        "max_ms": round(max(values), 2),
    }


def _served(t: Trace) -> dict:
    return t.slow if t.path == "slow" else t.fast


def latency_report(traces: list[Trace], cache_stats: dict | None = None) -> dict:
    """Aggregate streamed traces into the latency report the CI gate reads.

    Per-layer p95 is the union over **both** paths on purpose: `retrieval` is
    paid by the misses and `cache` by the hits, so a p95 over "whoever paid it"
    is the honest per-layer cost. The breakdown is verified to sum to e2e in
    `tests/test_latency_v3.py`.
    """
    if not traces:
        return {"mode": "stream", "n": 0}

    served = [_served(t) for t in traces]
    fast = [t.fast for t in traces if t.path == "fast"]
    slow = [t.slow for t in traces if t.path == "slow"]
    gaps = [g for s in served for g in s["inter_chunk_ms"]]

    layer_p95: dict[str, float] = {}
    for layer in LAYER_ORDER:
        vals = [r[layer] for r in (s["layers"] for s in served) if layer in r]
        vals += [r[layer] for r in (t.fast["layers"] for t in traces) if layer in r]
        vals += [r[layer] for r in (t.slow["layers"] for t in traces) if layer in r]
        if vals:
            layer_p95[layer] = round(M.percentile(vals, 95), 2)

    def _path_layers(rows: list[dict]) -> dict[str, float]:
        out: dict[str, float] = {}
        for layer in LAYER_ORDER:
            vals = [r[layer] for r in (s["layers"] for s in rows) if layer in r]
            if vals:
                out[layer] = round(M.percentile(vals, 95), 2)
        return out

    layers_fast = _path_layers(fast)
    layers_slow = _path_layers(slow)

    by_category: dict[str, dict] = {}
    for cat in sorted({t.category for t in traces}):
        rows = [t for t in traces if t.category == cat]
        sv = [_served(t) for t in rows]
        by_category[cat] = {
            "n": len(rows),
            "mean_wer": round(statistics.mean(M.wer(t.expected, t.hypothesis) for t in rows), 4),
            "hit_rate": round(sum(1 for t in rows if t.path == "fast") / len(rows), 4),
            "ttft_p95_ms": round(M.percentile([s["ttft_ms"] for s in sv], 95), 2),
            "e2e_p95_ms": round(M.percentile([s["e2e_ms"] for s in sv], 95), 2),
            "mean_chunks": round(statistics.mean(s["n_chunks"] for s in sv), 2),
        }

    fast_e2e = [t.fast["e2e_ms"] for t in traces]
    slow_e2e = [t.slow["e2e_ms"] for t in traces]
    report = {
        "mode": "stream",
        "engine": asr_mod.engine_name() + "-stream",
        "chunk_ms": asr_mod.CHUNK_MS,
        "n": len(traces),
        "chunks": {
            "mean": round(statistics.mean(s["n_chunks"] for s in served), 2),
            "p95": round(M.percentile([s["n_chunks"] for s in served], 95), 2),
        },
        "ttft_ms": _stats([s["ttft_ms"] for s in served]),
        "ttft_fast_ms": _stats([s["ttft_ms"] for s in fast]),
        "ttft_slow_ms": _stats([s["ttft_ms"] for s in slow]),
        "inter_chunk_ms": _stats(gaps),
        "e2e_ms": _stats([s["e2e_ms"] for s in served]),
        "e2e_fast_ms": _stats([s["e2e_ms"] for s in fast]),
        "e2e_slow_ms": _stats([s["e2e_ms"] for s in slow]),
        "layers_p95_ms": layer_p95,
        "layers_fast_p95_ms": layers_fast,
        "layers_slow_p95_ms": layers_slow,
        "by_category": by_category,
        "dual_path": {
            "fast_hits": len(fast),
            "slow_misses": len(slow),
            "hit_rate": round(len(fast) / len(traces), 4),
            "fast_mean_ms": round(statistics.mean(fast_e2e), 2),
            "slow_mean_ms": round(statistics.mean(slow_e2e), 2),
            "speedup": round(statistics.mean(slow_e2e) / statistics.mean(fast_e2e), 2),
            "ttft_speedup": round(
                statistics.mean(s["ttft_ms"] for s in slow) / statistics.mean(s["ttft_ms"] for s in fast), 2
            )
            if fast
            else 0.0,
            "saved_mean_ms": round(statistics.mean(slow_e2e) - statistics.mean(fast_e2e), 2),
        },
    }
    if cache_stats:
        report["cache"] = cache_stats
    return report


# ==========================================================================
# budget
# ==========================================================================
def _breaching_layer(report: dict, layers: tuple[str, ...], pool_key: str = "layers_p95_ms") -> tuple[str, float]:
    """Name the layer with the worst p95 among `layers` (the cost driver)."""
    p95 = report.get(pool_key) or {}
    pool = {k: v for k, v in p95.items() if k in layers and v > 0}
    if not pool:
        return "unknown", 0.0
    name = max(pool, key=lambda k: pool[k])
    return name, pool[name]


def budget_report(
    report: dict,
    ttft_budget_ms: float = TTFT_BUDGET_MS,
    e2e_budget_ms: float = E2E_BUDGET_MS,
) -> dict:
    """Gate the shipped dual path against the voice budget.

    Gated (a breach fails CI):
      * `ttft_p95_fast` — the interactive budget, measured on the cache fast
        path. This is the number a caller perceives on a repeat question.
        Its breaching layer is drawn from the *fast* path's own layers, so the
        report never blames `retrieval` for a path that never pays it.
      * `e2e_p95_served` — the hard ceiling, measured over *every* utterance
        the system serves (fast and slow).
      * `e2e_p95_fast` — the fast path must also clear the e2e ceiling.

    Reported but not gated:
      * `ttft_p95_slow` — the latency the cache removes. It breaches the 400ms
        TTFT budget by design (that is the whole reason the fast path exists),
        so it is printed with its dominant layer instead of failing the run.
    """
    checks: list[dict] = []
    fast_ttft = report.get("ttft_fast_ms", {}).get("p95_ms", 0.0)
    served_e2e = report.get("e2e_ms", {}).get("p95_ms", 0.0)
    fast_e2e = report.get("e2e_fast_ms", {}).get("p95_ms", 0.0)
    slow_ttft = report.get("ttft_slow_ms", {}).get("p95_ms", 0.0)

    for name, value, budget, layers, pool, gated in (
        ("ttft_p95_fast", fast_ttft, ttft_budget_ms, TTFT_LAYERS, "layers_fast_p95_ms", True),
        ("e2e_p95_served", served_e2e, e2e_budget_ms, LAYER_ORDER, "layers_p95_ms", True),
        ("e2e_p95_fast", fast_e2e, e2e_budget_ms, LAYER_ORDER, "layers_fast_p95_ms", True),
        ("ttft_p95_slow", slow_ttft, ttft_budget_ms, TTFT_LAYERS, "layers_slow_p95_ms", False),
    ):
        if not report.get("n") or (value == 0.0 and not gated):
            continue
        ok = bool(value <= budget)
        layer, layer_p95 = _breaching_layer(report, layers, pool)
        checks.append(
            {
                "name": name,
                "value_ms": value,
                "budget_ms": budget,
                "pass": ok,
                "gated": gated,
                "dominant_layer": layer,
                "dominant_layer_p95_ms": layer_p95,
            }
        )

    breached = [c for c in checks if c["gated"] and not c["pass"]]
    return {
        "ttft_budget_ms": ttft_budget_ms,
        "e2e_budget_ms": e2e_budget_ms,
        "checks": checks,
        "breached": [c["name"] for c in breached],
        "breached_layers": sorted({c["dominant_layer"] for c in breached}),
        "pass": not breached,
    }


def check_budget(
    report: dict,
    ttft_budget_ms: float = TTFT_BUDGET_MS,
    e2e_budget_ms: float = E2E_BUDGET_MS,
) -> tuple[bool, list[str]]:
    """(ok, messages). Every failure names the layer that breached."""
    bud = budget_report(report, ttft_budget_ms, e2e_budget_ms)
    msgs: list[str] = []
    for c in bud["checks"]:
        head = "BUDGET" if c["gated"] else "NOTE"
        verdict = "PASS" if c["pass"] else ("FAIL" if c["gated"] else "over budget by design")
        if c["gated"]:
            line = (
                f"{head} {verdict} {c['name']}={c['value_ms']}ms "
                f"budget={c['budget_ms']}ms (layer {c['dominant_layer']} p95={c['dominant_layer_p95_ms']}ms)"
            )
        else:
            line = (
                f"{head} {c['name']}={c['value_ms']}ms > {c['budget_ms']}ms — breaching layer "
                f"{c['dominant_layer']} p95={c['dominant_layer_p95_ms']}ms; the cache fast path is what "
                f"holds the gated TTFT budget"
            )
        msgs.append(line)
    return bud["pass"], msgs


# ==========================================================================
# single entry point
# ==========================================================================
def run_latency_lab(
    golden: list[dict],
    seed: int = 7,
    top_k: int = asr_mod.RETRIEVAL_TOP_K,
    chunk_ms: float = asr_mod.CHUNK_MS,
) -> dict:
    """Full v3 latency lab: cache behaviour, streamed traces, budget verdict.

    Returns `{"stream": report, "cache": cache_lab_out, "budget": budget, ...}`.
    Pure and deterministic — same golden set + seed always gives the same
    numbers, with no wall-clock dependency and no sleeping.
    """
    cache = build_cache(golden)
    traces, cache_stats, clocks = stream_run(golden, seed, cache=cache, top_k=top_k, chunk_ms=chunk_ms)
    report = latency_report(traces, cache_stats)
    report["seed"] = seed
    report["top_k"] = top_k
    report["virtual_ms"] = round(sum(c.virtual_elapsed_ms for c in clocks), 2)
    lab = cache_lab(golden, seed, cache=build_cache(golden))
    budget = budget_report(report)
    return {
        "stream": report,
        "cache": lab,
        "budget": budget,
        "traces": [t.as_dict() for t in traces],
        "traces_fast": [
            {"id": t.id, "category": t.category, "ttft_ms": t.fast["ttft_ms"], "e2e_ms": t.fast["e2e_ms"]}
            for t in traces
        ],
        "traces_slow": [
            {"id": t.id, "category": t.category, "ttft_ms": t.slow["ttft_ms"], "e2e_ms": t.slow["e2e_ms"]}
            for t in traces
        ],
    }
