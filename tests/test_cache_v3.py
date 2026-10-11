"""SPEC-019 v3 — semantic cache: embeddings, guard, and the safety invariant.

The guard is the interesting part. A near-duplicate cache that serves the
wrong answer is worse than no cache, so these tests pin down the two rules
that make it safe:

  1. `adversarial` utterances are NEVER served, not even on an exact match;
  2. any other near-duplicate is served only when the normalized expected text
     is transcript-equivalent to the cached intent.
"""
import json
import os

import pytest

from app.cache import (
    DEFAULT_THRESHOLD,
    EMBED_DIM,
    NEVER_CACHE_CATEGORIES,
    CacheDecision,
    SemanticCache,
    cosine,
    embed,
    equivalence_key,
    paraphrase,
    speedup,
)
from eval.run import category_of

GOLDEN = os.path.join(os.path.dirname(__file__), "..", "data", "golden.json")
RATION = "मेरा राशन कार्ड में नाम कैसे जुड़वाऊं"
RATION_CUT = "मेरा राशन कार्ड में नाम जुड़"  # adv-03: truncated near-duplicate
ADVERSARIAL = "दीन दयाल योजना में दिन में आवेदन कैसे करें"  # homophone दीन/दिन


def _golden():
    with open(GOLDEN, encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def golden():
    return _golden()


def _warm(items):
    c = SemanticCache()
    for text, cat in items:
        c.put(text, {"answer": f"answer for {text}"}, text, cat)
    return c


# ------------------------------------------------------------- embeddings --
def test_embed_is_deterministic_and_l2_normalized():
    a, b = embed(RATION), embed(RATION)
    assert a == b  # same text -> same vector, forever, no key/model
    assert len(a) == EMBED_DIM
    assert abs(sum(x * x for x in a) ** 0.5 - 1.0) < 1e-9
    assert embed("") == [0.0] * EMBED_DIM


def test_cosine_bounds_and_mismatch_safety():
    assert cosine(embed(RATION), embed(RATION)) == pytest.approx(1.0, abs=1e-6)
    assert cosine([1.0, 0.0], [0.0, 1.0]) == 0.0
    assert cosine([], []) == 0.0
    assert cosine([1.0], [1.0, 2.0]) == 0.0  # length mismatch -> miss, not a crash


def test_near_duplicate_scores_above_threshold_and_unrelated_below(golden):
    assert cosine(embed(paraphrase(RATION, 7, 1)), embed(RATION)) >= DEFAULT_THRESHOLD
    unrelated = "आधार कार्ड गुम हो गया तो क्या करें"
    assert cosine(embed(unrelated), embed(RATION)) < DEFAULT_THRESHOLD


def test_equivalence_key_normalizes_orthographic_variants():
    assert equivalence_key("राशन  कार्ड, में।") == equivalence_key("राशन कार्ड में")
    # devanagari + ascii digits fold to the same hindi word as the literal
    assert equivalence_key("५") == equivalence_key("पाँच") == equivalence_key("पांच")
    assert equivalence_key("eKYC") == equivalence_key("ekyc")
    assert equivalence_key(RATION) != equivalence_key(RATION_CUT)


# ------------------------------------------------------------------ lookup --
def test_exact_repeat_hits_and_takes_the_fast_path():
    c = _warm([(RATION, "easy")])
    d = c.lookup(RATION, RATION, "easy")
    assert isinstance(d, CacheDecision)
    assert (d.hit, d.path, d.reason) == (True, "fast", "hit")
    assert d.payload["answer"] == f"answer for {RATION}"
    assert c.stats()["hits"] == 1


def test_near_duplicate_serves_the_cached_answer():
    c = _warm([(RATION, "easy")])
    d = c.lookup(paraphrase(RATION, 7, 1), RATION, "easy")
    assert d.hit is True and d.similarity >= DEFAULT_THRESHOLD
    assert d.payload == {"answer": f"answer for {RATION}"}


def test_below_threshold_query_misses_slowly():
    c = _warm([(RATION, "easy")])
    d = c.lookup("आधार कार्ड गुम हो गया तो क्या करें", "आधार कार्ड गुम हो गया तो क्या करें", "easy")
    assert (d.hit, d.path, d.reason) == (False, "slow", "below-threshold")


def test_empty_cache_reports_no_entry():
    d = SemanticCache().lookup(RATION, RATION, "easy")
    assert (d.hit, d.reason, d.path) == (False, "no-entry", "slow")


# ------------------------------------------------------------------- guard --
def test_rule1_adversarial_is_never_served_even_on_exact_match():
    c = _warm([(ADVERSARIAL, "adversarial")])
    d = c.lookup(ADVERSARIAL, ADVERSARIAL, "adversarial")
    assert (d.hit, d.path, d.reason, d.payload) == (False, "slow", "adversarial", None)
    assert c.stats()["bypassed_adversarial"] == 1
    assert c.stats()["hits"] == 0
    assert NEVER_CACHE_CATEGORIES == {"adversarial"}
    # bypassed traffic is not a miss: the hit-rate denominator stays honest
    assert c.stats()["misses"] == 0
    # the rule keys off the caller-supplied category, and never mutates state
    d2 = c.lookup(ADVERSARIAL, ADVERSARIAL, "adversarial")
    assert d2.reason == "adversarial" and d2.payload is None
    assert c.stats()["bypassed_adversarial"] == 2


def test_rule2_equivalence_guard_rejects_near_duplicate_with_other_intent():
    """The trap: cosine says "hit", the intent says "no"."""
    c = _warm([(RATION, "easy")])
    probe = paraphrase(RATION, 7, 1)
    sim = cosine(embed(probe), embed(RATION))
    assert sim >= DEFAULT_THRESHOLD  # the similarity gate alone WOULD have hit
    d = c.lookup(probe, RATION_CUT, "easy")  # ...but the intent is a different one
    assert d.hit is False
    assert d.reason == "equivalence-guard"
    assert d.path == "slow"
    assert d.payload is None
    assert c.stats()["guard_rejects"] == 1
    assert c.stats()["hits"] == 0


def test_truncated_adversarial_is_double_protected(golden):
    """adv-03 is both adversarial and an intent-different near-dup of ration-01."""
    by_id = {g["id"]: g for g in golden}
    c = SemanticCache()
    c.put(by_id["ration-01"]["expected"], {"a": 1}, by_id["ration-01"]["expected"], "easy")
    cut = by_id["adv-03"]
    assert category_of(cut) == "adversarial"
    d = c.lookup(paraphrase(cut["expected"], 7, 1), cut["expected"], category_of(cut))
    assert d.hit is False and d.reason == "adversarial"
    # strip the adversarial rule and the equivalence guard alone still refuses
    d = c.lookup(paraphrase(cut["expected"], 7, 1), cut["expected"], "easy")
    assert d.hit is False and d.reason in {"equivalence-guard", "below-threshold"}


def test_put_dedupes_by_intent_and_keeps_last_payload():
    c = _warm([(RATION, "easy")])
    c.put(RATION, {"answer": "v2"}, RATION, "easy")
    assert len(c) == 1
    assert c.lookup(RATION, RATION, "easy").payload == {"answer": "v2"}


def test_max_entries_is_bounded():
    c = SemanticCache(max_entries=5)
    for i in range(20):
        c.put(f"सवाल {i}", {"i": i}, f"सवाल {i}", "easy")
    assert len(c) == 5


# ------------------------------------------------------------------ stats --
def test_stats_math_and_reset():
    c = _warm([(RATION, "easy"), (ADVERSARIAL, "adversarial")])
    c.lookup(RATION, RATION, "easy")  # hit
    c.lookup("कुछ और बिल्कुल नया सवाल", "कुछ और बिल्कुल नया सवाल", "easy")  # miss
    c.lookup(ADVERSARIAL, ADVERSARIAL, "adversarial")  # bypassed
    s = c.stats()
    assert (s["hits"], s["misses"], s["bypassed_adversarial"]) == (1, 1, 1)
    assert s["hit_rate"] == 0.5  # bypassed traffic is not counted as a miss
    assert s["entries"] == 2 and s["threshold"] == DEFAULT_THRESHOLD
    c.reset_stats()
    assert c.stats()["hit_rate"] == 0.0 and c.stats()["lookups"] == 0


def test_speedup_helper():
    assert speedup(1000.0, 500.0) == 2.0
    assert speedup(1000.0, 0.0) == 0.0
    assert speedup(0.0, 100.0) == 0.0


def test_paraphrase_is_deterministic_and_semantically_stable():
    for strength in (1, 2, 3):
        p = paraphrase(RATION, 7, strength)
        assert p == paraphrase(RATION, 7, strength)
        assert equivalence_key(p) != ""
    # strength 0 is the exact-repeat probe; >=1 adds a wrapper
    assert paraphrase(RATION, 7, 0) == RATION
    assert paraphrase(RATION, 7, 1) != RATION
    assert equivalence_key(paraphrase(RATION, 7, 1)).endswith(equivalence_key(RATION))
    assert paraphrase(RATION, 7, 1) == paraphrase(RATION, 7, 1)


def test_payload_returned_is_a_copy_not_the_stored_dict():
    c = _warm([(RATION, "easy")])
    d = c.lookup(RATION, RATION, "easy")
    d.payload["answer"] = "mutated by caller"
    assert c.lookup(RATION, RATION, "easy").payload["answer"] == f"answer for {RATION}"
