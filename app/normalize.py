"""Input normalization for Hindi-voice scoring (SPEC-014 v2).

Applied to BOTH hypothesis and reference before WER/CER so the harness does
not penalize orthographic variants a real STT engine cannot control:

- Unicode NFC (canonical composition)
- chandrabindra (U+0981) folded to anusvara (U+0902)
- nukta folded away (NFD, strip U+093C, NFC) so क़ == क
- punctuation / danda stripped, whitespace collapsed
- Devanagari + ASCII digits mapped to Hindi words (५ / 5 -> पाँच)
- ASCII lowercased (code-mixed Hinglish: eKYC == ekyc)
- zero-width / BOM chars removed

Stdlib only.
"""

from __future__ import annotations

import re
import unicodedata

CHANDRABINDRA = "\u0901"  # Devanagari sign candrabindu (U+0981 Bengali equiv. also folded)
CHANDRABINDRA_ALIASES = ("\u0901", "\u0981")
ANUSVARA = "\u0902"
NUKTA = "\u093c"

# Devanagari + ASCII digits -> Hindi word (one token per digit; multi-digit
# runs become space-separated words, e.g. 2024 -> "दो शून्य दो चार").
DIGIT_WORDS = {
    "०": "शून्य", "१": "एक", "२": "दो", "३": "तीन", "४": "चार",
    "५": "पाँच", "६": "छह", "७": "सात", "८": "आठ", "९": "नौ",
    "0": "शून्य", "1": "एक", "2": "दो", "3": "तीन", "4": "चार",
    "5": "पाँच", "6": "छह", "7": "सात", "8": "आठ", "9": "नौ",
}

# Punctuation stripped before scoring (incl. Devanagari danda/double-danda).
_PUNCT = set("।॥,.!?;:\"'‘’“”()[]{}-–—…/\\|@#$%^&*~`+=<>")
_PUNCT_RE = re.compile("[" + re.escape("".join(sorted(_PUNCT))) + "]")
_WS_RE = re.compile(r"\s+")
_ZW_RE = re.compile("[\u200b\u200c\u200d\ufeff]")


def normalize(text: str) -> str:
    """Normalize Hindi/ASR text for scoring. Idempotent; never raises."""
    if not text:
        return ""
    s = unicodedata.normalize("NFC", text)
    s = _ZW_RE.sub("", s)
    # chandrabindra -> anusvara (nasalization marks fold together)
    for _cb in CHANDRABINDRA_ALIASES:
        s = s.replace(_cb, ANUSVARA)
    # nukta folding: क़ (U+0958) -> क, via canonical decomposition
    s = unicodedata.normalize("NFD", s)
    s = s.replace(NUKTA, "")
    s = unicodedata.normalize("NFC", s)
    # digits -> Hindi words, one token per digit
    s = "".join(f" {DIGIT_WORDS[ch]} " if ch in DIGIT_WORDS else ch for ch in s)
    # ASCII lowercase for code-mixed stability (Devanagari unaffected)
    s = s.lower()
    s = _PUNCT_RE.sub(" ", s)
    s = _WS_RE.sub(" ", s).strip()
    return s
