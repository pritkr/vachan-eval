import React from 'react'

const API = ''
async function post(path, body) {
  const r = await fetch(API + path, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) })
  return r.json()
}

function LatChart({ results, summary }) {
  if (!results.length) return <p>…</p>
  const max = Math.max(...results.map(r => r.latency_ms), 1)
  const buckets = new Array(12).fill(0)
  results.forEach(r => { buckets[Math.min(11, Math.floor(r.latency_ms / max * 12))]++ })
  const bmax = Math.max(...buckets, 1), W = 520, H = 140, bw = W / 12
  const x = v => (v / max) * W
  return (
    <svg width="100%" viewBox={`0 0 ${W} ${H}`} role="img" aria-label="latency histogram">
      {buckets.map((b, i) => { const h = (b / bmax) * (H - 30)
        return <rect key={i} x={i * bw + 3} y={H - 20 - h} width={bw - 6} height={h} fill="#58a6ff" opacity={0.85} rx={2} /> })}
      <line x1={x(summary.latency.p50_ms)} x2={x(summary.latency.p50_ms)} y1={0} y2={H - 20} stroke="#3fb950" strokeWidth={2} />
      <line x1={x(summary.latency.p95_ms)} x2={x(summary.latency.p95_ms)} y1={0} y2={H - 20} stroke="#f85149" strokeWidth={2} strokeDasharray="4 3" />
    </svg>
  )
}

/* ---------------------------------------------------------------------------
 * v3 latency section — TTFT vs e2e per category + cache hit rate.
 * Pure SVG: grouped bars for the two latency percentiles, a polyline for the
 * cache hit rate, plus the budget rules (TTFT 400ms / e2e 1200ms) as dashed
 * guides so you can see headroom without reading a table.
 * ------------------------------------------------------------------------- */
const TTFT_BUDGET_MS = 400
const E2E_BUDGET_MS = 1200
const LAYERS = [
  ['endpoint', 'VAD endpoint'], ['asr_first', 'ASR 1st partial'],
  ['retrieval', 'retrieval'], ['cache', 'cache lookup'],
  ['tts_first', 'TTS 1st byte'], ['asr_stream', 'ASR rest'],
]

function LatencySection({ data }) {
  if (!data || data.error) {
    return <p>{data && data.error ? data.error : 'No latency report yet — '}<code>python eval/run.py --stream</code></p>
  }
  if (!data.stream || !data.stream.n) {
    return <p>No latency report yet — <code>python eval/run.py --stream</code> or start the backend for <code>/latency</code>.</p>
  }
  const s = data.stream
  const cats = Object.keys(s.by_category || {})
  const W = 720, H = 340, padL = 78, padB = 54, padT = 16
  const plotH = H - padB - padT, plotW = W - padL - 16
  const max = Math.max(E2E_BUDGET_MS, ...cats.map(c => s.by_category[c].e2e_p95_ms)) * 1.05
  const y = v => padT + plotH - (v / max) * plotH
  const groupW = plotW / Math.max(1, cats.length)
  const barW = Math.min(30, groupW / 3.2)
  const budget = data.budget || {}
  const cached = (data.cache && data.cache.hit_rate) || 0
  const dual = data.dual_path || s.dual_path || {}
  const layers = s.layers_p95_ms || {}

  return (
    <div>
      <p className="mut" style={{ fontSize: 12.5 }}>
        {s.source ? s.source + ' · ' : ''}streamed on a deterministic fake clock — 1s chunks, no sleeping, no GPU, no API key.
        Budget: <b>TTFT ≤ {TTFT_BUDGET_MS}ms</b> (fast path), <b>e2e ≤ {E2E_BUDGET_MS}ms</b>.
      </p>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4,1fr)', gap: 8, margin: '10px 0' }}>
        <Kpi v={s.ttft_fast_ms.p95_ms} u="ms" label="TTFT p95 · fast path" good={s.ttft_fast_ms.p95_ms <= TTFT_BUDGET_MS} />
        <Kpi v={s.ttft_slow_ms.p95_ms} u="ms" label="TTFT p95 · slow path" good={s.ttft_slow_ms.p95_ms <= TTFT_BUDGET_MS} />
        <Kpi v={s.e2e_ms.p95_ms} u="ms" label="e2e p95 · served" good={s.e2e_ms.p95_ms <= E2E_BUDGET_MS} />
        <Kpi v={dual.speedup} u="×" label="cache e2e speedup" good={dual.speedup >= 1} />
      </div>
      <svg width="100%" viewBox={`0 0 ${W} ${H}`} role="img" aria-label="TTFT versus end-to-end latency by category, with cache hit rate">
        {/* y grid + labels */}
        {[0, 0.25, 0.5, 0.75, 1].map(f => {
          const v = Math.round(max * f)
          return <g key={f}>
            <line x1={padL} x2={W - 16} y1={y(v)} y2={y(v)} stroke="#30363d" strokeWidth={1} />
            <text x={padL - 8} y={y(v) + 4} fill="#8b949e" fontSize={10} textAnchor="end">{v}</text>
          </g>
        })}
        <text x={padL - 60} y={padT + 8} fill="#8b949e" fontSize={10}>ms</text>
        {/* budget rules */}
        <line x1={padL} x2={W - 16} y1={y(TTFT_BUDGET_MS)} y2={y(TTFT_BUDGET_MS)} stroke="#d29922" strokeWidth={1.5} strokeDasharray="6 4" />
        <text x={W - 16} y={y(TTFT_BUDGET_MS) - 5} fill="#d29922" fontSize={10} textAnchor="end">TTFT budget {TTFT_BUDGET_MS}ms</text>
        <line x1={padL} x2={W - 16} y1={y(E2E_BUDGET_MS)} y2={y(E2E_BUDGET_MS)} stroke="#f85149" strokeWidth={1.5} strokeDasharray="6 4" />
        <text x={W - 16} y={y(E2E_BUDGET_MS) - 5} fill="#f85149" fontSize={10} textAnchor="end">e2e budget {E2E_BUDGET_MS}ms</text>
        {/* grouped bars: TTFT p95 (blue) vs e2e p95 (purple), cache hit-rate line (green, right axis) */}
        {cats.map((c, i) => {
          const gx = padL + i * groupW
          const v = s.by_category[c]
          const cx = gx + groupW / 2
          return <g key={c}>
            <rect x={cx - barW - 3} y={y(v.ttft_p95_ms)} width={barW} height={Math.max(1, y(0) - y(v.ttft_p95_ms))} fill="#58a6ff" rx={2} opacity={0.9} />
            <rect x={cx + 3} y={y(v.e2e_p95_ms)} width={barW} height={Math.max(1, y(0) - y(v.e2e_p95_ms))} fill="#a371f7" rx={2} opacity={0.9} />
            <text x={cx - barW / 2} y={y(v.ttft_p95_ms) - 4} fill="#58a6ff" fontSize={9.5} textAnchor="middle">{v.ttft_p95_ms}</text>
            <text x={cx + barW / 2} y={y(v.e2e_p95_ms) - 4} fill="#a371f7" fontSize={9.5} textAnchor="middle">{v.e2e_p95_ms}</text>
            <text x={cx} y={H - padB + 16} fill="#e6edf3" fontSize={11} textAnchor="middle">{c}</text>
            <text x={cx} y={H - padB + 29} fill="#8b949e" fontSize={9.5} textAnchor="middle">n={v.n} · hit {Math.round(v.hit_rate * 100)}%</text>
            {/* cache hit-rate polyline point (right axis 0..1) */}
            <circle cx={cx} cy={padT + plotH - v.hit_rate * plotH} r={3.5} fill="#3fb950" />
          </g>
        })}
        <polyline fill="none" stroke="#3fb950" strokeWidth={2}
          points={cats.map((c, i) => `${padL + i * groupW + groupW / 2},${padT + plotH - s.by_category[c].hit_rate * plotH}`).join(' ')} />
        {/* legend */}
        <g>
          <rect x={padL} y={H - 14} width={10} height={10} fill="#58a6ff" /><text x={padL + 14} y={H - 5} fill="#8b949e" fontSize={10}>TTFT p95</text>
          <rect x={padL + 84} y={H - 14} width={10} height={10} fill="#a371f7" /><text x={padL + 98} y={H - 5} fill="#8b949e" fontSize={10}>e2e p95</text>
          <line x1={padL + 160} x2={padL + 176} y1={H - 9} y2={H - 9} stroke="#3fb950" strokeWidth={2} />
          <text x={padL + 180} y={H - 5} fill="#8b949e" fontSize={10}>cache hit rate (right axis, overall {(cached * 100).toFixed(0)}%)</text>
        </g>
      </svg>
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12, marginTop: 8 }}>
        <div>
          <h3 style={{ fontSize: 13, margin: '0 0 6px' }}>Where the time goes (layer p95)</h3>
          <table>
            <thead><tr><th>layer</th><th>p95</th><th>share of e2e</th></tr></thead>
            <tbody>
              {LAYERS.filter(([k]) => layers[k] !== undefined).map(([k, n]) => {
                const v = layers[k]
                const share = s.e2e_ms.p95_ms ? Math.round((v / s.e2e_ms.p95_ms) * 100) : 0
                return <tr key={k}><td>{n} <code>{k}</code></td>
                    <td style={{ color: k === 'retrieval' ? '#f85149' : 'inherit' }}>{v}ms</td>
                    <td>{share}%</td></tr>
              })}
            </tbody>
          </table>
        </div>
        <div>
          <h3 style={{ fontSize: 13, margin: '0 0 6px' }}>Budget verdict</h3>
          <table>
            <thead><tr><th>check</th><th>value</th><th>budget</th><th>dominant layer</th><th></th></tr></thead>
            <tbody>
              {(budget.checks || []).map(c => <tr key={c.name}>
                <td><code>{c.name}</code>{c.gated ? null : <span className="mut"> (note)</span>}</td>
                <td>{c.value_ms}ms</td>
                <td>≤ {c.budget_ms}ms</td>
                <td style={{ color: c.pass ? 'inherit' : '#f85149' }}>{c.dominant_layer}</td>
                <td style={{ color: c.pass ? '#3fb950' : '#f85149' }}>{c.pass ? 'PASS' : (c.gated ? 'FAIL' : 'note')}</td>
              </tr>)}
            </tbody>
          </table>
          <p className="mut" style={{ fontSize: 12 }}>
            Dual path: {dual.fast_hits} fast / {dual.slow_misses} slow · TTFT speedup {dual.ttft_speedup}× ·
            mean e2e saved {dual.saved_mean_ms}ms. Adversarial utterances are never served from cache
            ({(data.cache && data.cache.bypassed_adversarial) || 0} bypassed,
            {' '}{(data.cache && data.cache.unsafe_served) === 0 ? '0 unsafe serves' : 'UNSAFE SERVES'}).
          </p>
        </div>
      </div>
    </div>
  )
}

function Kpi({ v, u, label, good }) {
  return <div style={{ background: '#0d1117', border: '1px solid #30363d', borderRadius: 8, padding: 8, textAlign: 'center' }}>
    <b style={{ display: 'block', fontSize: 18, color: good ? '#3fb950' : '#f85149' }}>{v}{u}</b>
    <span style={{ fontSize: 11, color: '#8b949e' }}>{label}</span>
  </div>
}

export default function App() {
  const [summary, setSummary] = React.useState(null)
  const [results, setResults] = React.useState([])
  const [text, setText] = React.useState('मेरा राशन कार्ड में नाम कैसे जुड़वाऊं')
  const [busy, setBusy] = React.useState(false)
  const [offline, setOffline] = React.useState(false)
  const [hist, setHist] = React.useState([])
  const [lat, setLat] = React.useState(null)
  const loadBoard = async () => {
    try {
      const b = await fetch('/leaderboard').then(r => r.json())
      setHist(b.history || [])
    } catch (e) { /* backend down — eval section still works offline */ }
  }
  const loadLatency = async () => {
    // live /latency first; fall back to the bundled `eval/run.py --stream`
    // snapshot so the latency section is never blank when the API is down.
    try {
      const r = await fetch('/latency')
      if (!r.ok) throw new Error('status ' + r.status)
      setLat(await r.json())
    } catch (e) {
      try {
        const m = await fetch('mock_eval.json').then(r => r.json())
        if (m.summary && m.summary.stream) {
          setLat({
            stream: m.summary.stream, cache: m.summary.cache, dual_path: m.summary.dual_path,
            budget: m.summary.budget, source: 'offline snapshot (mock_eval.json) — backend unreachable',
          })
        } else { setLat({ error: 'no latency data: run `python eval/run.py --stream`' }) }
      } catch (e2) { setLat({ error: 'no latency data: run `python eval/run.py --stream`' }) }
    }
  }
  React.useEffect(() => { loadBoard(); loadLatency() }, [])
  const dArrow = d => {
    if (d === null || d === undefined || d === 0) return <span>—</span>
    const better = d < 0
    return <span style={{ color: better ? '#3fb950' : '#f85149' }}>{better ? '▼ ' : '▲ '}{d > 0 ? '+' : ''}{d}</span>
  }
  const runEval = async () => {
    setBusy(true); setOffline(false)
    try {
      const d = await post('/eval', { seed: 7, engine: 'mock', stream: true })
      setSummary(d.summary); setResults(d.results)
      if (d.summary.stream) setLat({ stream: d.summary.stream, cache: d.summary.cache, dual_path: d.summary.dual_path, budget: d.summary.budget, source: 'live (/eval stream:true)' })
      loadBoard()
    } catch (e) {
      // API unreachable (e.g. opened via file:// or backend down):
      // fall back to the committed snapshot generated by `python eval/run.py --stream`.
      const m = await fetch('mock_eval.json').then(r => r.json())
      setSummary({ ...m.summary, note: 'offline snapshot (mock_eval.json) — backend unreachable' })
      setResults(m.results); setOffline(true)
    } finally { setBusy(false) }
  }
  return (
    <main className="vapp" style={{ maxWidth: 1100, margin: '0 auto', fontFamily: 'system-ui' }}>
      <h1>वचन Eval (Vite + React DX)</h1>
      <p>Same harness as <code>web/index.html</code> static build — this is the Vite dev entry (<code>npm run dev</code>).</p>
      <textarea rows={3} style={{ width: '100%' }} value={text} onChange={e => setText(e.target.value)} />
      <div><button disabled={busy} onClick={runEval}>▶ Run full eval (stream)</button></div>
      {offline && <p>Offline mode — showing bundled <code>mock_eval.json</code> snapshot (seed 7, stream). Start the backend for live scores.</p>}
      {summary && summary.note && <p>{summary.note}</p>}
      {summary && <p>mean WER {summary.mean_wer} · p95 {summary.latency.p95_ms}ms</p>}
      {summary && <LatChart results={results} summary={summary} />}
      <h2 style={{ gridColumn: '1/-1' }}>Latency (v3) — TTFT vs e2e per category + cache hit rate</h2>
      <div style={{ gridColumn: '1/-1' }}>{lat ? <LatencySection data={lat} /> : <p>Loading latency report…</p>}</div>
      {results.length > 0 && (
        <table><thead><tr><th>id</th><th>path</th><th>wer</th><th>ttft ms</th><th>e2e ms</th><th>hypothesis</th></tr></thead>
          <tbody>{results.map(r => <tr key={r.id}><td>{r.id}</td><td>{r.path || '—'}</td><td>{r.wer}</td><td>{r.ttft_ms ?? '—'}</td><td>{r.e2e_ms ?? '—'}</td><td>{r.hypothesis}</td></tr>)}</tbody></table>)}
      <h2>Run history (Δ vs previous run)</h2>
      {hist.length > 0 ? (
        <table><thead><tr><th>run</th><th>model</th><th>mean WER Δ</th><th>p95 ms Δ</th><th>gate</th></tr></thead>
          <tbody>{hist.map((h, i) => {
            const p = i > 0 ? hist[i - 1] : null
            const d = k => (p && h[k] !== undefined && p[k] !== undefined) ? Math.round((h[k] - p[k]) * 10000) / 10000 : null
            return <tr key={i}><td>{h.ts ? new Date(h.ts * 1000).toLocaleString() : '#' + i}</td>
              <td>{h.model} s{h.seed}</td>
              <td>{h.mean_wer} {dArrow(d('mean_wer'))}</td>
              <td>{h.p95_ms} {dArrow(d('p95_ms'))}</td>
              <td>{h.pass ? 'PASS' : 'FAIL'}</td></tr>
          })}</tbody></table>
      ) : <p>No runs yet — <code>python eval/run.py --seed 7</code> appends here.</p>}
    </main>
  )
}
