"""Eval-gated CI runner: `python eval/run.py [--seed 7] [--golden data/golden.json]`.

Fails (exit 1) if:
  - mean WER > 0.25 or latency p95 > 1200ms (global v1 gates), or
  - any category mean WER exceeds its threshold (v2 category gates;
    `adversarial` is report-only and never fails), or
  - any WER/CER metric regresses > 0.03 absolute vs eval/baseline.json, or
  - `--budget` is set and the v3 voice budget is breached (stream mode: TTFT
    p95 <= 400ms on the fast path, e2e p95 <= 1200ms served; the failing
    check names the layer that breached).

v3 adds the realtime path:
  --stream   run the streaming STT adapter on a fake clock and report TTFT,
             inter-chunk cadence, e2e per category, and the semantic-cache
             hit rate / speedup. WER/CER are untouched (the streamed
             hypothesis is the same mock hypothesis), so the v1/v2 gates and
             the baseline check stay bit-for-bit identical.
  --budget   gate the v3 voice budget and fail the run on a breach.

Writes eval/results.json + eval/results.md for the dashboard / CI artifacts,
and appends one line per run to eval/history.jsonl (leaderboard history).
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app import asr as asr_mod  # noqa: E402
from app import metrics as M  # noqa: E402
from eval import latency as latency_mod  # noqa: E402

WER_GATE = 0.25
P95_GATE_MS = 1200.0

# v3 voice budget (stream mode). TTFT is the interactive budget a caller
# perceives on the cache fast path; e2e is the ceiling for any utterance.
TTFT_BUDGET_MS = latency_mod.TTFT_BUDGET_MS
E2E_BUDGET_MS = latency_mod.E2E_BUDGET_MS

# v2: per-category WER gates. `adversarial` is report-only (threshold None).
CATEGORY_GATES: dict[str, float | None] = {
    "easy": 0.10,
    "medium": 0.20,
    "hard": 0.30,
    "noise": 0.35,
    "adversarial": None,
}

# v2: absolute regression tolerance vs baseline.json (WER/CER scale).
BASELINE_TOL = 0.03

EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
HISTORY_PATH = os.path.join(EVAL_DIR, "history.jsonl")
BASELINE_PATH = os.path.join(EVAL_DIR, "baseline.json")


def category_of(g: dict) -> str:
    """Golden category: explicit `category` field, else id-prefix fallback,
    else the mock `difficulty` (easy/medium/hard)."""
    if g.get("category"):
        return str(g["category"])
    gid = str(g.get("id", ""))
    if gid.startswith("noise-"):
        return "noise"
    if gid.startswith("adv-"):
        return "adversarial"
    return str(g.get("difficulty", "medium"))


def run(golden: list[dict], seed: int, engine: str = "mock") -> tuple[list[dict], dict]:
    results = []
    for g in golden:
        hyp, lat = asr_mod.mock_transcribe(g["expected"], g.get("difficulty", "medium"), seed)
        results.append(
            {
                "id": g["id"],
                "lang": g.get("lang", "hi"),
                "domain": g.get("domain", ""),
                "difficulty": g.get("difficulty", "medium"),
                "category": category_of(g),
                "expected": g["expected"],
                "hypothesis": hyp,
                "wer": round(M.wer(g["expected"], hyp), 4),
                "cer": round(M.cer(g["expected"], hyp), 4),
                "latency_ms": lat,
            }
        )
    summary = M.aggregate(results)
    summary["engine"] = engine
    summary["seed"] = seed
    summary["gates"] = {"wer_max": WER_GATE, "p95_max_ms": P95_GATE_MS}
    # v2: per-category means + gates
    categories: dict[str, dict] = {}
    for cat in sorted({r["category"] for r in results}):
        cwers = [r["wer"] for r in results if r["category"] == cat]
        ccers = [r["cer"] for r in results if r["category"] == cat]
        gate = CATEGORY_GATES.get(cat)
        mean_wer = round(statistics.mean(cwers), 4)
        categories[cat] = {
            "n": len(cwers),
            "mean_wer": mean_wer,
            "mean_cer": round(statistics.mean(ccers), 4),
            "gate_wer_max": gate,  # None = report-only
            "pass": True if gate is None else bool(mean_wer <= gate),
        }
    summary["categories"] = categories
    summary["category_pass"] = all(c["pass"] for c in categories.values())
    summary["pass"] = bool(
        summary["mean_wer"] <= WER_GATE
        and summary["latency"]["p95_ms"] <= P95_GATE_MS
        and summary["category_pass"]
    )
    audio_sec = 4.0 * len(results)
    summary["cost_usd"] = M.cost_usd(audio_sec, engine)
    return results, summary


def append_history(summary: dict, path: str = HISTORY_PATH) -> None:
    """Append one JSON line per run (never raises — history must not fail eval)."""
    try:
        entry = {
            "ts": time.time(),
            "model": summary.get("engine", "mock"),
            "seed": summary.get("seed"),
            "n": summary.get("n"),
            "mean_wer": summary.get("mean_wer"),
            "mean_cer": summary.get("mean_cer"),
            "p95_ms": (summary.get("latency") or {}).get("p95_ms"),
            "categories": {c: v["mean_wer"] for c, v in (summary.get("categories") or {}).items()},
            "pass": summary.get("pass"),
        }
        # v3: realtime numbers ride along when the run was in stream mode
        stream = summary.get("stream") or {}
        if stream:
            entry["ttft_p95_fast_ms"] = (stream.get("ttft_fast_ms") or {}).get("p95_ms")
            entry["ttft_p95_slow_ms"] = (stream.get("ttft_slow_ms") or {}).get("p95_ms")
            entry["e2e_p95_ms"] = (stream.get("e2e_ms") or {}).get("p95_ms")
        cache = summary.get("cache") or {}
        if cache:
            entry["cache_hit_rate"] = cache.get("hit_rate")
            entry["cache_speedup"] = (stream.get("dual_path") or {}).get("speedup")
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except OSError:
        pass


def check_baseline(summary: dict, path: str = BASELINE_PATH, tol: float = BASELINE_TOL) -> tuple[bool, list[str]]:
    """Compare WER/CER metrics vs baseline.json. Returns (ok, messages).

    Missing baseline file is a pass-with-warning (fresh checkout); create it
    with `python eval/run.py --update-baseline`.
    """
    if not os.path.exists(path):
        return True, [f"BASELINE missing ({os.path.basename(path)}) — run with --update-baseline to create"]
    with open(path, encoding="utf-8") as f:
        base = json.load(f)
    msgs: list[str] = []
    ok = True
    checks = [("mean_wer", summary.get("mean_wer")), ("mean_cer", summary.get("mean_cer"))]
    for cat, c in (summary.get("categories") or {}).items():
        if cat == "adversarial":
            continue  # report-only: tracked, never gated
        checks.append((f"cat:{cat}", c.get("mean_wer")))
    for name, cur in checks:
        ref = (base.get("metrics") or {}).get(name)
        if ref is None or cur is None:
            msgs.append(f"BASELINE SKIP {name} (no reference)")
            continue
        delta = float(cur) - float(ref)
        status = "OK" if delta <= tol else "REGRESSED"
        if delta > tol:
            ok = False
        msgs.append(f"BASELINE {status} {name}: current={cur} baseline={ref} delta={delta:+.4f} (tol +{tol})")
    return ok, msgs


def write_baseline(summary: dict, path: str = BASELINE_PATH) -> None:
    base = {
        "model": summary.get("engine", "mock"),
        "seed": summary.get("seed"),
        "n": summary.get("n"),
        "metrics": {
            "mean_wer": summary.get("mean_wer"),
            "mean_cer": summary.get("mean_cer"),
            **{f"cat:{c}": v["mean_wer"] for c, v in (summary.get("categories") or {}).items() if c != "adversarial"},
        },
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(base, f, ensure_ascii=False, indent=2)
        f.write("\n")


# --------------------------------------------------------------------------
# v3: streaming (realtime) pass
# --------------------------------------------------------------------------
def run_stream(
    golden: list[dict],
    seed: int,
    top_k: int = asr_mod.RETRIEVAL_TOP_K,
    chunk_ms: float = asr_mod.CHUNK_MS,
) -> dict:
    """Run the v3 latency lab and fold it into an eval summary.

    WER/CER/latency are *not* recomputed: `stream_transcribe` emits the same
    mock hypothesis as `mock_transcribe`, so v1/v2 gates and the baseline are
    unaffected. This only attaches `stream` / `cache` / `dual_path` / `budget`
    to the summary and stamps per-result realtime numbers.
    """
    lab = latency_mod.run_latency_lab(golden, seed, top_k=top_k, chunk_ms=chunk_ms)
    summary = {
        "stream": lab["stream"],
        "cache": lab["cache"],
        "dual_path": lab["stream"]["dual_path"],
        "budget": lab["budget"],
    }
    return {"summary_extra": summary, "stamps": {t["id"]: t for t in lab["traces"]}}


def attach_stream(results: list[dict], stamps: dict) -> None:
    """Copy realtime numbers onto the existing per-result rows (in place)."""
    for r in results:
        t = stamps.get(r["id"])
        if not t:
            continue
        r["path"] = t["path"]
        r["ttft_ms"] = t["ttft_ms"]
        r["e2e_ms"] = t["e2e_ms"]
        r["n_chunks"] = t["n_chunks"]
        r["fast_ttft_ms"] = t["fast_ttft_ms"]
        r["fast_e2e_ms"] = t["fast_e2e_ms"]
        r["slow_ttft_ms"] = t["slow_ttft_ms"]
        r["slow_e2e_ms"] = t["slow_e2e_ms"]
        r["saved_ms"] = t["saved_ms"]


def stream_lines(stream: dict, cache: dict, dual: dict) -> list[str]:
    """Human-readable v3 block: TTFT / inter-chunk / e2e + cache hit rate."""
    t = stream.get("ttft_ms", {})
    tf = stream.get("ttft_fast_ms", {})
    ts = stream.get("ttft_slow_ms", {})
    ic = stream.get("inter_chunk_ms", {})
    e2e = stream.get("e2e_ms", {})
    lines = [
        "",
        f"realtime (stream={stream.get('chunk_ms')}ms chunks, engine={stream.get('engine')}):",
        f"  TTFT served  p50={t.get('p50_ms')}ms p95={t.get('p95_ms')}ms max={t.get('max_ms')}ms",
        f"  TTFT fast    p50={tf.get('p50_ms')}ms p95={tf.get('p95_ms')}ms (cache hit, n={tf.get('count')})",
        f"  TTFT slow    p50={ts.get('p50_ms')}ms p95={ts.get('p95_ms')}ms (retrieval, n={ts.get('count')})",
        f"  inter-chunk  p50={ic.get('p50_ms')}ms p95={ic.get('p95_ms')}ms over {ic.get('count')} gaps"
        f" · {stream.get('chunks', {}).get('mean')} chunks/utt avg",
        f"  e2e served   p50={e2e.get('p50_ms')}ms p95={e2e.get('p95_ms')}ms max={e2e.get('max_ms')}ms",
        f"  dual path    {dual.get('fast_hits')} fast / {dual.get('slow_misses')} slow"
        f" · hit_rate={dual.get('hit_rate')} · e2e speedup={dual.get('speedup')}x"
        f" · ttft speedup={dual.get('ttft_speedup')}x · saved {dual.get('saved_mean_ms')}ms mean",
    ]
    for layer, p95 in (stream.get("layers_p95_ms") or {}).items():
        lines.append(f"  layer p95    {layer:<10s} {p95}ms")
    lines += [
        "",
        f"semantic cache (threshold={cache.get('threshold')}, dim={cache.get('dim')},"
        f" entries={cache.get('entries')}):",
        f"  hit_rate     {cache.get('hit_rate')} over {cache.get('eligible_probes')} eligible probes"
        f" (repeat {cache.get('classes', {}).get('repeat', {}).get('hit_rate')},"
        f" rephrase {cache.get('classes', {}).get('rephrase', {}).get('hit_rate')},"
        f" novel {cache.get('classes', {}).get('novel', {}).get('hit_rate')})",
        f"  guard        {cache.get('bypassed_adversarial')}/{cache.get('adversarial_probes')}"
        f" adversarial bypassed · {cache.get('guard_rejects')} equivalence rejects"
        f" · {cache.get('cosine_would_misroute')} of those would have hit on cosine alone"
        f" · unsafe_served={cache.get('unsafe_served')}",
        "",
        "| category | n | cache hit rate | ttft p95 | e2e p95 | chunks/utt |",
        "|---|---|---|---|---|---|",
    ]
    for cat, v in (stream.get("by_category") or {}).items():
        lines.append(
            f"| {cat} | {v['n']} | {v['hit_rate']} |"
            f" {v['ttft_p95_ms']}ms | {v['e2e_p95_ms']}ms | {v['mean_chunks']} |"
        )
    return lines


def batch_budget_report(summary: dict) -> dict:
    """Budget report for a non-stream run.

    TTFT does not exist in batch mode (nothing is streamed), so only the e2e
    ceiling is checked and the TTFT budget is reported as skipped rather than
    silently passed.
    """
    p95 = float((summary.get("latency") or {}).get("p95_ms") or 0.0)
    return {
        "ttft_budget_ms": TTFT_BUDGET_MS,
        "e2e_budget_ms": E2E_BUDGET_MS,
        "checks": [
            {
                "name": "e2e_p95_batch",
                "value_ms": p95,
                "budget_ms": E2E_BUDGET_MS,
                "pass": bool(p95 <= E2E_BUDGET_MS),
                "gated": True,
                "dominant_layer": "asr_batch",
                "dominant_layer_p95_ms": p95,
            }
        ],
        "breached": [] if p95 <= E2E_BUDGET_MS else ["e2e_p95_batch"],
        "breached_layers": [] if p95 <= E2E_BUDGET_MS else ["asr_batch"],
        "ttft_skipped": True,
        "pass": bool(p95 <= E2E_BUDGET_MS),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--golden", default=os.path.join(os.path.dirname(__file__), "..", "data", "golden.json"))
    ap.add_argument("--engine", default="mock")
    ap.add_argument("--baseline", default=BASELINE_PATH)
    ap.add_argument("--update-baseline", action="store_true", help="write current numbers to baseline.json and exit 0")
    ap.add_argument("--no-history", action="store_true", help="skip appending to history.jsonl")
    ap.add_argument(
        "--stream",
        action="store_true",
        help="v3: stream the golden set (fake clock) and report TTFT / inter-chunk / e2e + cache hit rate",
    )
    ap.add_argument(
        "--budget",
        action="store_true",
        help=f"v3: gate the voice budget (TTFT p95<={TTFT_BUDGET_MS}ms in stream mode, e2e p95<={E2E_BUDGET_MS}ms)",
    )
    ap.add_argument("--top-k", type=int, default=asr_mod.RETRIEVAL_TOP_K, help="retrieval top-k (latency model)")
    ap.add_argument("--chunk-ms", type=float, default=asr_mod.CHUNK_MS, help="streaming audio chunk size (ms)")
    args = ap.parse_args()

    with open(args.golden, encoding="utf-8") as f:
        golden = json.load(f)

    results, summary = run(golden, args.seed, args.engine)

    # v3 realtime pass — attaches to (never rewrites) the v1/v2 numbers
    stream_report: dict | None = None
    if args.stream:
        streamed = run_stream(golden, args.seed, top_k=args.top_k, chunk_ms=args.chunk_ms)
        attach_stream(results, streamed["stamps"])
        summary.update(streamed["summary_extra"])
        stream_report = streamed["summary_extra"]["stream"]

    outdir = os.path.join(os.path.dirname(__file__))
    with open(os.path.join(outdir, "results.json"), "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "results": results}, f, ensure_ascii=False, indent=2)

    if args.update_baseline:
        write_baseline(summary, args.baseline)
        print(f"baseline written to {args.baseline}")

    if not args.no_history:
        append_history(summary)

    base_ok, base_msgs = check_baseline(summary, args.baseline)
    gate_ok = bool(summary["pass"])
    overall = gate_ok and base_ok

    budget_msgs: list[str] = []
    budget_ok = True
    if args.budget:
        if stream_report is not None:
            budget_ok, budget_msgs = latency_mod.check_budget(stream_report, TTFT_BUDGET_MS, E2E_BUDGET_MS)
        else:
            bud = batch_budget_report(summary)
            summary["budget"] = bud
            budget_ok = bool(bud["pass"])
            budget_msgs = [
                f"BUDGET {'PASS' if budget_ok else 'FAIL'} e2e_p95_batch="
                f"{(summary.get('latency') or {}).get('p95_ms')}ms budget={E2E_BUDGET_MS}ms"
                f" (breaching layer: {'asr_batch' if not budget_ok else 'none'})",
                f"NOTE ttft skipped — batch mode does not stream; add --stream for the TTFT budget",
            ]
        overall = overall and budget_ok

    cat_rows = [
        f"| {c} | {v['n']} | {v['mean_wer']} | "
        f"{'report-only' if v['gate_wer_max'] is None else v['gate_wer_max']} | "
        f"{'PASS' if v['pass'] else 'FAIL'} |"
        for c, v in summary["categories"].items()
    ]
    lines = [
        "# vachan-eval results",
        "",
        f"engine={summary['engine']} seed={summary['seed']} n={summary['n']}",
        f"mean_wer={summary['mean_wer']} mean_cer={summary['mean_cer']} "
        f"p50={summary['latency']['p50_ms']}ms p95={summary['latency']['p95_ms']}ms "
        f"cost=${summary['cost_usd']}",
        f"GATE {'PASS' if gate_ok else 'FAIL'} (wer<={WER_GATE}, p95<={P95_GATE_MS}ms)",
        *base_msgs,
    ]
    if stream_report is not None:
        lines += stream_lines(stream_report, summary["cache"], summary["dual_path"])
    if args.budget:
        lines += ["", *budget_msgs]
    lines += [
        f"OVERALL {'PASS' if overall else 'FAIL'}"
        + (" (gates + baseline + budget)" if args.budget else " (gates + baseline)"),
        "",
        "per-category:",
        "| category | n | mean_wer | gate_wer_max | status |",
        "|---|---|---|---|---|",
        *cat_rows,
    ]
    if stream_report is not None:
        lines += [
            "",
            "| id | path | ttft_ms | fast_ttft | slow_ttft | e2e_ms | saved_ms |",
            "|---|---|---|---|---|---|---|",
        ]
        for r in results:
            lines.append(
                f"| {r['id']} | {r['path']} | {r['ttft_ms']} | {r['fast_ttft_ms']} |"
                f" {r['slow_ttft_ms']} | {r['e2e_ms']} | {r['saved_ms']} |"
            )
    else:
        lines += [
            "",
            "| id | lang | diff | wer | cer | lat_ms |",
            "|---|---|---|---|---|---|",
        ]
        for r in results:
            lines.append(
                f"| {r['id']} | {r['lang']} | {r['difficulty']} | {r['wer']} | {r['cer']} | {r['latency_ms']} |"
            )
    with open(os.path.join(outdir, "results.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print("\n".join(lines))
    return 0 if overall else 1


if __name__ == "__main__":
    raise SystemExit(main())
