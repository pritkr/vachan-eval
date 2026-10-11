"""SPEC-014 v2: input-normalization unit tests (bho/mag variants, noisy punct)."""
from app import metrics as M
from app.normalize import normalize


def test_nfc_composes():
    assert normalize("क" + "ा") == "का"


def test_chandrabindra_folds_to_anusvara():
    assert normalize("चाँद") == normalize("चांद")


def test_nukta_folds_to_base():
    assert normalize("क़ानून") == normalize("कानून")


def test_punct_and_danda_stripped():
    assert normalize("राम, घर गया।") == "राम घर गया"


def test_whitespace_collapsed():
    assert normalize("राम   घर\nगया") == "राम घर गया"


def test_devanagari_digit_to_word():
    assert normalize("कमरा ५") == "कमरा पाँच"


def test_ascii_digit_to_word():
    assert normalize("कमरा 5") == "कमरा पाँच"


def test_ascii_lowercased_codmix():
    assert normalize("EPFO पोर्टल") == "epfo पोर्टल"


def test_empty_and_zero_width():
    assert normalize("") == ""
    assert normalize("राम\u200b घर") == "राम घर"


def test_bho_variant_noisy_punct():
    assert normalize("हमार राशन, कार्ड।") == "हमार राशन कार्ड"


def test_mag_variant_anusvara_ellipsis():
    assert normalize("हथिन…") == "हथिन"


def test_wer_invariant_to_punct():
    assert M.wer("राम, घर गया।", "राम घर गया") == 0.0


def test_wer_invariant_to_digit_form():
    assert M.wer("कमरा ५ नंबर", "कमरा 5 नंबर") == 0.0
