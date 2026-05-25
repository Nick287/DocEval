import pytest

from intsig_eval.core.tokens import iter_token_matches, strip_markdown


def _surfaces(text: str, *, relaxed: bool = False) -> set[str]:
    return {m.surface for m in iter_token_matches(text, relaxed=relaxed)}


def test_long_number():
    assert "1234567" in _surfaces("invoice 1234567 due")


def test_alnum_id_requires_both():
    out = _surfaces("ABC123 ABCDEF 123456")
    assert "ABC123" in out
    # ABCDEF has no digit, must not match alnum_id (still no number rules match it)
    assert "ABCDEF" not in out


def test_mixed_id():
    assert "AB-12-CD" in _surfaces("doc AB-12-CD") or "AB12CD" in _surfaces("doc AB12CD")


def test_currency():
    out = _surfaces("total $1,234.50 paid")
    assert any("1,234.50" in s for s in out)


@pytest.mark.parametrize(
    "txt, hit",
    [
        ("3JAN2024", "3JAN2024"),
        ("01/02/2024", "01/02/2024"),
        ("2024-01-02", "2024-01-02"),
        ("01-Jan-2024", "01-Jan-2024"),
    ],
)
def test_dates(txt, hit):
    assert hit in _surfaces(txt)


def test_strip_markdown_kills_images_and_code():
    raw = "![](x.png) `inline` plain 12345 ```fence 99999 ```"
    cleaned = strip_markdown(raw)
    assert "12345" in cleaned
    # 99999 lived inside a fence — should be gone
    assert "99999" not in cleaned


def test_relaxed_mode_finds_concatenated():
    # ``\b`` would normally not fire at boundaries between digit and letter,
    # but the relaxed mode strips word boundaries — easy way to check it loads.
    out = _surfaces("X9999Y", relaxed=True)
    assert any("9999" in s for s in out)
