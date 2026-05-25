import pytest

from intsig_eval.core.normalize import edit_distance, normalize


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("AB-1234", "AB-1234"),
        ("ab 12 34", "AB1234"),
        ("5,555,777", "5555777"),
        ("ＡＢ１２３", "AB123"),   # full-width
        (" 12345. ", "12345"),
        ("/2024-01-01;", "2024-01-01"),
    ],
)
def test_normalize_basic(raw, expected):
    assert normalize(raw) == expected


def test_edit_distance_exact():
    assert edit_distance("ABC123", "ABC123") == 0


def test_edit_distance_one_sub():
    assert edit_distance("ABC123", "ABC124") == 1


def test_edit_distance_capped():
    # 5 differences > cap=2 → must return 3
    assert edit_distance("AAAAA", "BBBBB", cap=2) == 3


def test_edit_distance_length_diff_capped():
    assert edit_distance("AB", "ABCDEFGH", cap=2) == 3
