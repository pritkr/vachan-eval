# vachan-eval — Hindi Voice-AI Eval Harness

Mock-first eval harness for Hindi / Bhojpuri / Magahi voice AI: a 30-utterance
golden set (ration, scholarship, RTE-admission, pension, health + 5 adversarial),
input normalization, WER / CER / latency scoring with per-category gates, a
baseline-regression check, an eval-gated CI, a FastAPI backend, and a React dashboard.
**v3 adds the realtime path**: a streaming STT adapter with TTFT / inter-chunk /
end-to-end latency on deterministic fake clocks, a semantic response cache with a
correctness guard, and a `--budget` CI gate that names the layer that breached.
No API key, no model download, no network needed for the demo.

## Problem

Voice bots for welfare schemes (ration, scholarship, RTE admission) fail
silently on dialect speech — Bhojpuri / Magahi verb forms, code-mixed
Hindi-English ("eKYC", "status check"), and noisy phone audio. Teams ship
without a regression gate, so a model swap quietly breaks the users who need
it most. vachan-eval gives that gate: a hand-written golden set, deterministic
mock STT for offline CI, WER + p95-latency thresholds that block regressions,
and a swap-in path to real faster-whisper when hardware is available.

## Architecture

```
┌─────────────┐     ┌──────────────────┐     ┌────────────────────────────┐
│ web/ (Vite  │ api/ (FastAPI)   │ eval/ + data/              │
│ React dash) │ HTTP /transcribe │ golden.json (30)           │
│ upload/paste│  /tts /eval      │ run.py   → WER + gates     │
│ dashboard)  │ /latency /health │ latency.py → TTFT/dual path│
│             │  + /eval /health │ cache.py  → hit rate+guard│
└─────────────┘     └────────┬─────────┘     └────────────────────────────┘
                             │ SQLite (default) / Postgres+pgvector (stretch)
                             ▼
                              audit.log JSONL (utt_id, WER, CER, p50/p95, cost)
```

v3 realtime path — one user turn, dual fast/slow (measured p95 layer costs):

```
 speech ─▶ endpoint ─▶ asr_first ─┬─ retrieval 427.7ms ─▶ tts_first ─▶ asr_stream
  (1s      101.3ms     116.7ms    │                            ▲             │
  chunks)                          └─ cache 15.1ms ────────────┤ TTFT         │ e2e
                                    ▲                         │              │
                                    └── FAST 347 / 632 ms ─────┴── SLOW 764 / 1067 ms
```

- **Mock-first, no key required:** `TRANSCRIBE_MODE=mock|faster-whisper`.
  Mock uses a deterministic seeded noisy transform (`app/asr.py`) so evals run
  offline in CI. Real STT plugs in via the `transcribe_file()` interface
  (faster-whisper, opt-in with `VACHAN_REAL_ASR=1` — never downloads weights
  unprompted).
- **Metrics:** WER / CER via own Levenshtein implementation (`app/metrics.py`,
  stdlib only), latency p50/p95, cost/run (mock = $0 by definition).
- **v3 realtime:** `stream_transcribe()` slices the *same* mock hypothesis into
  1-second audio chunks and charges a per-layer cost model (VAD endpoint → first
  partial → retrieval | cache → first TTS byte → rest of the stream) to an
  injected clock. `FakeClock` makes ~42 s of virtual voice time cost ~40 ms of
  wall time, so latency tests are instant and machine-independent.
- **v3 semantic cache:** `app/cache.py` — deterministic hash embeddings (word
  unigrams/bigrams + char trigrams, L2-normalized, no model) over
  near-duplicate utterances, with a two-rule correctness guard.
- **Frontend:** single React page — paste text or run the full 30-utt eval,
  SVG latency histogram (no chart lib), leaderboard table, and a **v3 latency
  section** (TTFT vs e2e bars per category + cache hit-rate polyline + budget
  guides, pure SVG). Works off `file://` against the bundled
  `web/public/mock_eval.json` fallback when the API is unreachable.
- **Security/privacy:** audio never leaves the machine in mock mode; mock
  `/transcribe` never pretends to hear (returns `engine=mock` + honest
  warning); TTS is a labelled beep-per-word placeholder, not a fake voice.

## Run

```bash
pip install -r requirements.txt

# 1. eval gate (exit 1 if mean WER > 0.25 or p95 > 1200ms)
python3 eval/run.py --seed 7

# 1b. v3 realtime gate: streaming STT (fake clock) + semantic cache +
#     voice budget (TTFT p95 <= 400ms fast path, e2e p95 <= 1200ms).
#     Exits 1 and prints the breaching layer on a breach.
python3 eval/run.py --seed 7 --stream --budget    # or: make eval-stream

# 2. backend
uvicorn app.main:app --port 8000        # docs at /docs, latency at /latency

# 3. tests
python3 -m pytest -q

# 4. dashboard — either:
cd web && npm install && npm run dev    # Vite dev (proxies /eval, /latency → :8000)
# or open web/index.html directly (static build, mock-JSON tolerant)

# 5. docker
docker compose up --build               # api on :8000
docker compose --profile postgres up    # opt-in pgvector backend
```

Real ASR (optional, needs weights + CPU):

```bash
VACHAN_REAL_ASR=1 VACHAN_MODEL=tiny uvicorn app.main:app --port 8000
```

## API

| Method | Path | Body | Returns |
|---|---|---|---|
| `GET` | `/` | — | service index + endpoint list |
| `GET` | `/health` | — | `{ok, engine, golden_n}` |
| `POST` | `/transcribe` | multipart `file` (audio) | `{transcript, latency_ms, engine}` (+ `warning` in mock mode) |
| `POST` | `/tts` | `{text, lang, seed}` | `{audio_b64, mime, chars, latency_ms, engine}` (placeholder beeps) |
| `POST` | `/eval` | `{seed, engine, items?, stream?}` (empty = full golden set) | `{run_id, summary, results[]}` — `stream:true` adds `summary.stream` / `cache` / `dual_path` / `budget` |
| `GET` | `/latency` | — | `{stream, cache, dual_path, budget, traces[], source}` — v3 realtime report (live on a fake clock, or the last `eval/run.py --stream` artifact) |
| `GET` | `/eval/history?limit=` | — | `{runs[]}` (SQLite / Postgres / in-memory) |
| `GET` | `/leaderboard` | `?limit=` | `{rows[], history[], delta}` — measured mock runs + labelled literature reference rows + every `eval/run.py` run with delta vs previous run |

## Eval results (real, `python3 eval/run.py --seed 7`)

```
engine=mock seed=7 n=30
mean_wer=0.1617 mean_cer=0.1181 p50=438.94ms p95=594.46ms cost=$0.0
GATE PASS (wer<=0.25, p95<=1200.0ms)
BASELINE OK ×6 (mean + per-category, tol +0.03)
OVERALL PASS (gates + baseline)
```

| Aggregate | Value |
|---|---|
| mean WER | **0.1617** (gate ≤ 0.25 ✅) |
| mean CER | 0.1181 |
| p50 / p95 WER | 0.134 / 0.4132 |
| WER > 0.5 fail rate | 0.0 |
| latency p50 / p95 / mean / max | 438.94 / **594.46** (gate ≤ 1200ms ✅) / 452.83 / 605.15 ms |
| cost / 30 evals | $0.0 (mock) |

Per-category gates (v2):

| category | n | mean WER | gate | status |
|---|---|---|---|---|
| easy | 5 | 0.0286 | ≤ 0.10 | PASS |
| medium | 10 | 0.123 | ≤ 0.20 | PASS |
| hard | 8 | 0.2029 | ≤ 0.30 | PASS |
| noise | 2 | 0.1111 | ≤ 0.35 | PASS |
| adversarial | 5 | 0.3267 | report-only | — |

Per-utterance (worst first):

| id | lang | diff | wer | cer | lat_ms |
|---|---|---|---|---|---|
| adv-04 | bho | hard | 0.5 | 0.4286 | 536.76 |
| rte-05 | mag | hard | 0.4444 | 0.2791 | 572.64 |
| pension-03 | hi | medium | 0.375 | 0.2432 | 382.67 |
| adv-01 | hi | hard | 0.3333 | 0.1905 | 563.36 |
| adv-03 | hi | hard | 0.3333 | 0.4074 | 450.95 |
| scholar-03 | bho | hard | 0.3 | 0.1087 | 598.77 |
| adv-02 | hi | hard | 0.3 | 0.1277 | 589.2 |
| ration-03 | bho | hard | 0.25 | 0.1622 | 524.5 |
| scholar-05 | mag | hard | 0.25 | 0.2632 | 547.99 |
| ration-05 | hi | medium | 0.2222 | 0.1163 | 409.53 |
| adv-05 | hi | hard | 0.1667 | 0.069 | 464.39 |
| pension-01 | hi | easy | 0.1429 | 0.1579 | 301.1 |
| scholar-02 | hi | medium | 0.1429 | 0.0278 | 333.39 |
| rte-04 | hi | medium | 0.1429 | 0.0882 | 348.64 |
| ration-04 | mag | hard | 0.1429 | 0.1389 | 509.51 |
| health-02 | bho | hard | 0.125 | 0.1351 | 559.2 |
| mixed-02 | hi | medium | 0.125 | 0.1818 | 363.79 |
| noise-01 | hi | hard | 0.1111 | 0.0222 | 564.76 |
| noise-02 | bho | hard | 0.1111 | 0.0714 | 570.55 |
| pension-02 | bho | hard | 0.1111 | 0.1579 | 563.15 |
| rte-02 | hi | medium | 0.1111 | 0.1429 | 411.66 |
| mixed-01 | hi | medium | 0.1111 | 0.0233 | 422.45 |
| ration-01 | hi | easy | 0.0 | 0.0 | 258.62 |
| ration-02 | hi | medium | 0.0 | 0.0 | 397.65 |
| scholar-01 | hi | easy | 0.0 | 0.0 | 338.4 |
| scholar-04 | hi | medium | 0.0 | 0.0 | 426.94 |
| rte-01 | hi | easy | 0.0 | 0.0 | 343.41 |
| rte-03 | bho | hard | 0.0 | 0.0 | 605.15 |
| health-01 | hi | easy | 0.0 | 0.0 | 214.96 |
| mixed-03 | hi | medium | 0.0 | 0.0 | 410.75 |

## Failure analysis

Grouped means tell the story: **hi 0.1309 / bho 0.1996 / mag 0.2791** by
language, **easy 0.0286 / medium 0.123 / hard 0.2029 / noise 0.1111 /
adversarial 0.3267 (report-only)** by category — monotonic in difficulty as
designed, with Magahi the systematic weak spot (all 3 Magahi items are `hard`),
and the adversarial set (homophones, dense Hinglish, truncated audio) the
hardest by construction.

Worst cases, and why:

1. **`rte-05` (mag, WER 0.4444)** — `इसकूल में नाम लिखावे ला…` →
   `इसकूल नाम लाह…`: a deletion plus the `आवेदन→अवेदन` matra confusion
   stacked on a 9-word utterance. Magahi verb morphology (`परतौ`,
   `लिखावे ला`) and the epenthetic `इसकूल` ("school") have no entries in the
   mock `CONFUSIONS` table, so once seeded noise hits a content word there is
   no recovery — exactly the gap a real dialect fine-tune would need to close.
2. **`pension-03` (hi medium, WER 0.375)** — filler insertion (`हम्म`) +
   suffix substitution (`कार्ड→कार्डह`) on a short 8-word utterance. Short
   utterances are WER-volatile: two word errors = 0.375. Lesson: per-utterance
   WER needs the aggregate gate, not item-level asserts.
3. **`scholar-03` (bho hard, WER 0.30)** — code-mixed `स्कॉलरशिप` survives,
   but Bhojpuri function words (`के→केह`) plus a filler insertion sink it.
   Function-word confusions dominate dialect WER, not the English loanwords.
4. **Batch latency is not the problem — the retrieval round trip is.** Batch p95
   is 593ms vs the 1200ms gate, ~2× headroom, reserved for swapping mock →
   faster-whisper-small (~900ms reference, see `/leaderboard`). But the batch
   number hides the realtime one: v3 streams the same 30 utterances and finds
   that **slow-path TTFT is 764ms, and 427.7ms of it is the retrieval round
   trip** — see [v3](#v3--the-latency-lab-ttft-inter-chunk-and-the-dual-path).

Honest limits: the mock is word-level seeded noise, not acoustics — it cannot
reveal real STT failure modes (matra/anusvara errors, code-mix segmentation).
The harness measures the *pipeline* (scoring, gating, latency tracking);
acoustic fidelity arrives with `VACHAN_REAL_ASR=1`.

## v2 — production-grade eval discipline

v2 applies production-RAG eval discipline (categorized golden sets,
regression-vs-baseline gates, input normalization) to voice:

1. **Input normalization (`app/normalize.py`)** — applied to hypothesis AND
   reference before WER/CER: Unicode NFC, chandrabindra→anusvara folding,
   nukta folding (क़ == क), punctuation/danda strip + whitespace collapse,
   Devanagari/ASCII digits → Hindi words (५ / 5 → पाँच), ASCII lowercasing
   for code-mixed Hinglish. 13 unit tests in `tests/test_normalize.py`.
2. **Category gates (`eval/run.py`)** — golden items carry a `category`
   (`easy` 5 / `medium` 10 / `hard` 8 / `noise` 2 / `adversarial` 5).
   Thresholds: easy ≤ 0.10, medium ≤ 0.20, hard ≤ 0.30, noise ≤ 0.35;
   `adversarial` (homophones दीन/दिन + समान/सामान, dense Hinglish
   EPFO/UAN/eKYC, truncated utterances) is **report-only** and never fails
   the gate. A failing gate names its category.
3. **Leaderboard history** — every `eval/run.py` run appends one JSON line to
   `eval/history.jsonl` (model, seed, mean WER/CER, p95, per-category means).
   `GET /leaderboard` serves `{rows, history, delta}` where `delta` is
   latest-vs-previous (negative WER = improvement); the dashboard renders
   ▼/▲ arrows per run.
4. **Regression-vs-baseline (`eval/baseline.json`)** — `run.py` exits 1 if
   mean WER/CER or any gated category mean regresses **> 0.03 absolute** vs
   baseline. Workflow:

```bash
python3 eval/run.py --seed 7                 # gates + baseline check, exit 0
python3 eval/run.py --seed 7 --update-baseline  # bless new numbers after an intended change
python3 eval/run.py --seed 7 --no-history       # dry run without appending history
```

## v3 — the latency lab: TTFT, inter-chunk, and the dual path

Everything above measures a **batch** number: one latency per utterance, after
the whole utterance is done. A voice turn is not one number, and the batch
number hides the one a user actually feels. v3 measures the realtime path, on
**deterministic fake clocks**, with a **budget gate** in CI.

### The problem this addresses

VoiceAgentRAG (arXiv 2026) measured that for voice agents the *retrieval
round-trip* — not the model — eats the entire latency budget. A streaming
decode plus a vector round trip plus a synthesizer does not fit in 400 ms of
time-to-first-audio. The architecture that does fit is **dual-path**: answer a
near-duplicate question from a cache on a fast path, and only pay full
stream + retrieval on the slow path. That is the single design decision v3
measures, and the one the budget gate protects.

### 1. Streaming STT adapter (`app/asr.py` + `eval/latency.py`)

`stream_transcribe()` slices the *same* mock hypothesis into 1-second audio
chunks — the byte-identical text, so **WER/CER and the v2 baseline are
untouched by stream mode** (asserted in `tests/test_latency_v3.py`). What is
new is the clock, charged per **layer**:

| layer | what it is | p95 (measured) |
|---|---|---|
| `endpoint` | VAD pre-roll — you must hear a beat of silence before you can commit | 101.3 ms |
| `asr_first` | first partial decode → first text token out | 116.7 ms |
| `retrieval` | doc-embedding + ANN + rerank round trip (top-k 4) | **427.7 ms** |
| `cache` | fast path: embed the query + scan the index | **15.1 ms** |
| `tts_first` | synthesizer first audio byte | 127.3 ms |
| `asr_stream` | the rest of the audio streaming in | 310.1 ms |

Two conventions, stated so the numbers are comparable:

- **TTFT** = end-of-speech (endpointed) → first agent audio byte =
  `endpoint + asr_first + (retrieval | cache) + tts_first`.
- **e2e** = TTFT + `asr_stream`. The layer costs sum to e2e **exactly**
  (asserted to 0.05 ms for all 30 traces).
- **inter-chunk** is measured on the *decode* timeline (chunk *k* lands at
  `endpoint + asr_first + Σ chunk costs`), because "did the stream stall" is a
  property of the decoder, not of the retrieval the agent does meanwhile.

### 2. Measured numbers (`python3 eval/run.py --seed 7 --stream --budget`)

```
realtime (stream=1000.0ms chunks, engine=mock-stream):
  TTFT served  p50=335.43ms p95=761.35ms max=764.37ms
  TTFT fast    p50=280.02ms p95=347.16ms (cache hit, n=21)
  TTFT slow    p50=746.07ms p95=764.34ms (retrieval, n=9)
  inter-chunk  p50=95.01ms p95=106.34ms over 67 gaps · 3.23 chunks/utt avg
  e2e served   p50=540.96ms p95=1052.58ms max=1074.43ms
  dual path    21 fast / 9 slow · hit_rate=0.7 · e2e speedup=1.82x
               · ttft speedup=2.52x · saved 409.77ms mean

semantic cache (threshold=0.88, dim=96, entries=30):
  hit_rate     0.7667 over 60 eligible probes
               (repeat 1.0, rephrase 0.84, novel 0.0)
  guard        5/5 adversarial bypassed · 25 equivalence rejects
               · 10 of those would have hit on cosine alone · unsafe_served=0

BUDGET PASS ttft_p95_fast=347.16ms budget=400.0ms (layer tts_first p95=127.32ms)
BUDGET PASS e2e_p95_served=1052.58ms budget=1200.0ms (layer retrieval p95=427.73ms)
BUDGET PASS e2e_p95_fast=632.36ms budget=1200.0ms (layer asr_stream p95=293.29ms)
NOTE ttft_p95_slow=764.34ms > 400.0ms — breaching layer retrieval p95=426.57ms;
     the cache fast path is what holds the gated TTFT budget
```

Headline numbers:

| metric | value | budget | verdict |
|---|---|---|---|
| **TTFT p95, fast path (cache hit)** | **347.16 ms** | ≤ 400 ms | **PASS** (53 ms headroom) |
| **TTFT p95, slow path (retrieval)** | **764.34 ms** | ≤ 400 ms | over by design — layer `retrieval` at 427.7 ms |
| **e2e p95, served (all 30)** | **1052.58 ms** | ≤ 1200 ms | **PASS** (147 ms headroom) |
| **e2e p95, fast path** | 632.36 ms | ≤ 1200 ms | PASS |
| **cache hit rate** (60 eligible probes) | **0.7667** | — | repeat 1.0 / rephrase 0.84 / novel 0.0 |
| **cache e2e speedup** | **1.82×** (TTFT 2.52×) | — | 409.77 ms saved per repeat question |
| inter-chunk p50 / p95 | 95.01 / 106.34 ms | — | the stream does not stall |
| unsafe cache serves | **0** | must be 0 | 5/5 adversarial bypassed, 25 guard rejects |

Per category, as rendered in the dashboard's latency section:

| category | n | cache hit rate | TTFT p95 | e2e p95 | chunks/utt |
|---|---|---|---|---|---|
| easy | 5 | 1.0 | 241.39 ms | 427.66 ms | 3.2 |
| medium | 10 | 0.8 | 684.63 ms | 886.10 ms | 3.2 |
| hard | 8 | 0.875 | 615.41 ms | 843.32 ms | 3.25 |
| noise | 2 | 0.5 | 716.95 ms | 1023.13 ms | 3.5 |
| adversarial | 5 | **0.0** | 764.35 ms | 1070.72 ms | 3.2 |

The spread is the point: `easy` is fully cache-served and lands at 241 ms TTFT
with 50% e2e headroom, while `adversarial` is *never* cache-served and pays the
full 764 ms. Same 30 questions, one is 3× faster than the other purely because
of routing — and the category that must not be cached is the one that pays full
price. That is the design working as intended, not a failure.

### 3. The dual path, spelled out

```
                       ┌─ near-duplicate? ── adversarial? ── yes ─┐
user speaks ──▶ STT ───┤                                            ├─▶ SLOW PATH
             (stream)  └─ no ─▶ hit ── equivalent intent? ── yes ──┘   endpoint → asr_first
                        │                                              → RETRIEVAL 427.7 ms
                        └── hit but intent differs ──▶ refuse         → tts_first  → asr_stream
                                                                          e2e p95 1067 ms, TTFT p95 764 ms
             FAST PATH: endpoint → asr_first → CACHE 15.1 ms
                        → tts_first → asr_stream
                        e2e p95 632 ms, TTFT p95 347 ms
```

- **Fast path** (21/30 here): the query embeds, the index is scanned, the cached
  answer is served. The 427.7 ms retrieval round trip is replaced by a 15.1 ms
  cache lookup — **the entire speedup is that one swap**, which is why the
  measured `saved_mean_ms` (409.77 ms) tracks the layer p95 (427.7 ms).
- **Slow path** (9/30): full streaming decode + retrieval + synthesis. It
  breaches the 400 ms TTFT budget by 364 ms, and the gate names the layer:
  `retrieval`.
- The `1.82×` (e2e) and `2.52×` (TTFT) speedups are **mean-over-30** ratios;
  the 347 / 632 vs 764 / 1067 ms figures above are **p95**.

### 4. Semantic cache correctness guard (`app/cache.py`)

Embeddings are deterministic hashes of word unigrams, word bigrams and character
trigrams of the normalized utterance, L2-normalized to 96 dims. Same text →
same vector on every machine, forever; no key, no model, no download. Cosine
≥ 0.88 marks a *candidate* near-duplicate. **Similarity alone is never enough
to serve**, because two rules outrank it:

1. **`adversarial` is never served** — not even on an exact surface match.
   Measured: 5/5 bypassed, 0 hits. Homophone pairs (दीन/दिन, समान/सामान),
   dense Hinglish acronyms (EPFO/UAN/eKYC) and truncated audio are exactly the
   cases where "looks the same" is a lie.
2. **Transcript equivalence** — a candidate is served only when the normalized
   expected text equals the cached intent's. Measured: 25 traps rejected, and
   **10 of the 10 audited traps scored above the cosine threshold**, i.e. raw
   similarity *would* have served a wrong answer. `adv-03`
   ("मेरा राशन कार्ड में नाम जुड़", truncated) is the canonical case: it is a
   near-duplicate surface of `ration-01` ("…जुड़वाऊं") with a different intent.

The hit rate is measured against **negative controls**, because a hit rate
without them is a vanity metric:

| probe class | n | hit rate | meaning |
|---|---|---|---|
| repeat | 25 | 1.00 | exact re-ask must hit |
| rephrase | 25 | 0.84 | politeness-wrapper near-dup, strength cycled 1..3 |
| novel | 10 | **0.00** | fresh intents absent from the cache must all miss |
| trap | 25 | **0.00** | near-dup surface + different intent must all be refused |

Overall **0.7667** over the 60 eligible (non-adversarial, non-trap) probes, and
`unsafe_served = 0`.

One honest limit: rule 1 keys off a caller-supplied `category` label (the eval
harness reads it off the golden item; production would pass a risk classifier's
verdict). A caller that *mislabels* an adversarial utterance would get it
served — which is precisely why rule 2 exists as an independent backstop, and
why both rules are tested separately.

### 5. The `--budget` gate

`eval/run.py --budget` (with `--stream`) exits 1 when the voice budget is
breached, and every failure **names the layer that breached** rather than just
"too slow":

```
$ python3 eval/run.py --stream --budget          # budget tightened to 300ms
BUDGET FAIL ttft_p95_fast=347.16ms budget=300ms (layer tts_first p95=127.32ms)
BUDGET PASS e2e_p95_served=1052.58ms budget=1200ms (layer retrieval p95=427.73ms)
NOTE ttft_p95_slow=764.34ms > 300ms — breaching layer retrieval p95=426.57ms; …
```

- **Gated:** `ttft_p95_fast` ≤ 400 ms (drawn from the *fast* path's own layers,
  so the report never blames `retrieval` for a path that never pays it),
  `e2e_p95_served` ≤ 1200 ms over *every* utterance, `e2e_p95_fast` ≤ 1200 ms.
- **Reported, not gated:** `ttft_p95_slow`. It breaches by design — that is the
  latency the cache removes — so it is printed with its dominant layer instead
  of failing the run.
- Without `--stream`, `--budget` checks the batch e2e p95 against the same
  1200 ms ceiling and says `NOTE ttft skipped — batch mode does not stream`
  rather than silently passing a TTFT check that does not exist.
- Wired into `.github/workflows/eval-gate.yml` as a second gate after the v1/v2
  run, and into `make eval-stream`.

### 6. Web

The dashboard gains a **Latency** section in both frontends (`web/index.html`
static and `web/src/App.jsx` Vite): grouped **TTFT vs e2e bars per category**,
a **cache hit-rate polyline** on a right-hand 0–100% axis, the 400 ms / 1200 ms
budget rules drawn as dashed guides, plus a per-layer p95 table with each
layer's share of e2e and the budget verdict per check. All pure SVG — no chart
library, consistent with the rest of the dashboard. `GET /latency` serves it
live; `web/public/mock_eval.json` carries the same numbers for the offline
`file://` demo. The per-utterance table now shows the served `path`
(fast/slow) and TTFT/e2e next to WER/CER.

### 7. Why stream mode is additive

`--stream` never rewrites the v1/v2 numbers. `stream_transcribe` emits the same
hypothesis as `mock_transcribe`, so `mean_wer=0.1617`, `mean_cer=0.1181` and
every category mean are bit-identical with and without it
(`test_run_py_stream_does_not_change_wer_or_baseline` runs both and compares).
`stream` adds keys to the summary and stamps `path` / `ttft_ms` / `e2e_ms` onto
the existing result rows; it never touches `latency_ms`, the WER gates, or
`baseline.json`. The cache probe applies a paraphrased surface form to the
*lookup* only — it stands in for "the user rephrased a question they already
asked" without perturbing the hypothesis under test.

### 8. Honest limits

- The latency model is a **layer cost model, not a stopwatch**. It reproduces
  the shape of a streaming voice turn (endpointing, partial decode, retrieval
  round trip, synthesis) and its *relative* costs are argued from the
  literature, but the absolute milliseconds are synthetic. Swapping in real
  faster-whisper means replacing `stream_transcribe`'s cost table — the metrics,
  the budget gate, and the guard around them do not change.
- There is no WebSocket and no real audio. Chunking is simulated at 1 s, which
  is the granularity a browser `MediaRecorder` actually produces, but nothing
  is decoded.
- Cache hit rate is measured against a **synthetic** rephrase distribution
  (politeness wrappers at three strengths) plus a fixed 10-query novel control
  set. Real traffic has a long tail of partial restarts and mid-sentence
  corrections that would shift the number; the guard, not the hit rate, is the
  part built to survive that.
- Adversarial TTFT (764 ms) and e2e (1071 ms) are the worst in the set and are
  never improved by the cache. That is by design — they are the traffic the
  cache must refuse to touch.

## Who this impresses

- **Karya (Research Intern, AI Evals)** — dialect-stratified golden set
  (hi/bho/mag), per-group WER breakdown, eval-gated CI. This is their daily
  vocabulary.
- **SuperKalam (Voice-First)** — Hindi voice pipeline thinking: mock STT →
  real-model swap path, **TTFT / inter-chunk / e2e on deterministic fake
  clocks**, a **dual fast/slow path** where the cache is the fast path and
  retrieval is the slow path, and a **budget gate that names the breaching
  layer** instead of saying "too slow".
- **Wadhwani AI** — social-sector domains throughout (ration, scholarship,
  RTE, pension, health), privacy-first mock mode (audio never leaves the
  machine), $0 eval cost, and a cache guard that refuses to serve the
  adversarial dialect users rather than optimising them away.

## Repo-split note

This folder is self-contained — `app/`, `eval/`, `data/`, `web/`, `tests/`,
`Dockerfile`, `docker-compose.yml`, `Makefile`, `.github/` — with zero imports
from the parent portfolio repo (`eval/run.py` only adds its own parent dir to
`sys.path`). The API reaches the latency lab through a lazy, path-resolved
import of `eval/latency.py` (`app/main.py::_latency_lab`) so `app/` never
hard-depends on the harness, and falls back to the committed
`eval/results.json` artifact if that import is unavailable. To publish as
standalone `pritkr/vachan-eval`:

```bash
git subtree split -P projects/vachan-eval -b vachan-eval-split
# then push that branch to the new repo (LICENSE is MIT, author Prit Kumar)
```

## License

MIT — see [LICENSE](LICENSE).
