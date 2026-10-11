"""Result store: Postgres when DATABASE_URL is set (and driver present),
else SQLite (stdlib), else in-memory. No hard dependency on any driver."""

from __future__ import annotations

import json
import os
import sqlite3
import threading
import time

_SCHEMA = """
CREATE TABLE IF NOT EXISTS eval_runs (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  ts REAL NOT NULL,
  engine TEXT NOT NULL,
  seed INTEGER NOT NULL,
  summary TEXT NOT NULL
);
"""

_lock = threading.Lock()
_mem_runs: list[dict] = []


def _pg_url() -> str | None:
    return os.getenv("DATABASE_URL")


def _sqlite_path() -> str:
    return os.getenv("VACHAN_DB", os.path.join(os.path.dirname(__file__), "..", "vachan.db"))


def save_run(engine: str, seed: int, summary: dict, results: list[dict]) -> int:
    payload = json.dumps({"summary": summary, "results": results}, ensure_ascii=False)
    ts = time.time()
    pg = _pg_url()
    if pg:
        try:
            import psycopg  # psycopg3, optional

            with psycopg.connect(pg) as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "CREATE TABLE IF NOT EXISTS eval_runs "
                        "(id SERIAL PRIMARY KEY, ts DOUBLE PRECISION, engine TEXT, seed INT, summary JSONB)"
                    )
                    cur.execute(
                        "INSERT INTO eval_runs (ts, engine, seed, summary) VALUES (%s,%s,%s,%s) RETURNING id",
                        (ts, engine, seed, payload),
                    )
                    row = cur.fetchone()
                    return int(row[0]) if row else -1
        except Exception:
            pass  # fall through to sqlite — postgres is best-effort
    try:
        with _lock:
            con = sqlite3.connect(_sqlite_path())
            con.execute(_SCHEMA)
            cur = con.execute(
                "INSERT INTO eval_runs (ts, engine, seed, summary) VALUES (?,?,?,?)",
                (ts, engine, seed, payload),
            )
            con.commit()
            rid = int(cur.lastrowid or 0)
            con.close()
            return rid
    except Exception:
        with _lock:
            _mem_runs.append({"ts": ts, "engine": engine, "seed": seed, "summary": summary, "results": results})
            return len(_mem_runs)


def list_runs(limit: int = 20) -> list[dict]:
    pg = _pg_url()
    if pg:
        try:
            import psycopg

            with psycopg.connect(pg) as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT id, ts, engine, seed, summary FROM eval_runs ORDER BY id DESC LIMIT %s", (limit,))
                    out = []
                    for _id, ts, eng, seed, summ in cur.fetchall():
                        data = json.loads(summ) if isinstance(summ, str) else (summ or {})
                        out.append({"id": _id, "ts": ts, "engine": eng, "seed": seed, **data})
                    return out
        except Exception:
            pass
    try:
        with _lock:
            con = sqlite3.connect(_sqlite_path())
            con.execute(_SCHEMA)
            rows = con.execute(
                "SELECT id, ts, engine, seed, summary FROM eval_runs ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
            con.close()
            out = []
            for _id, ts, eng, seed, summ in rows:
                data = json.loads(summ)
                out.append({"id": _id, "ts": ts, "engine": eng, "seed": seed, **data})
            # include in-memory runs on top
            for i, m in enumerate(reversed(_mem_runs[-limit:])):
                out.append({"id": f"mem-{i}", **m})
            return out[:limit]
    except Exception:
        return list(reversed(_mem_runs[-limit:]))
