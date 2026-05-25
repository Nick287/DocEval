from doceval.consensus.clustering import build_clusters
from doceval.core import TokenHit


def _hit(source: str, norm: str) -> TokenHit:
    return TokenHit(source=source, surface=norm, norm=norm)


def test_identical_norm_clusters_together():
    hits = [_hit("ocr", "ABC123"), _hit("gemini", "ABC123"), _hit("gpt", "ABC123")]
    clusters = build_clusters(hits)
    assert len(clusters) == 1
    assert clusters[0].sources == {"ocr", "gemini", "gpt"}


def test_one_edit_merges_across_sources():
    hits = [_hit("ocr", "ABC123"), _hit("gemini", "ABC124")]
    clusters = build_clusters(hits, max_distance=1)
    assert len(clusters) == 1


def test_one_edit_does_not_merge_within_same_source():
    hits = [_hit("gemini", "ABC123"), _hit("gemini", "ABC124")]
    clusters = build_clusters(hits, max_distance=1)
    assert len(clusters) == 2  # same source, near-miss → distinct


def test_two_edits_do_not_merge_by_default():
    hits = [_hit("ocr", "ABC123"), _hit("gemini", "ABC456")]
    clusters = build_clusters(hits, max_distance=1)
    assert len(clusters) == 2


def test_two_edits_merge_when_distance_bumped():
    hits = [_hit("ocr", "ABC123"), _hit("gemini", "ABC456")]
    clusters = build_clusters(hits, max_distance=3)
    assert len(clusters) == 1


def test_empty_hits():
    assert build_clusters([]) == []
