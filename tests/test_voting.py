from doceval.consensus.clustering import build_clusters
from doceval.consensus.voting import apply_vision_verdict, vote
from doceval.core import BBox, TokenHit


def _hit(
    source: str,
    norm: str,
    surface: str | None = None,
    bbox: BBox | None = None,
) -> TokenHit:
    return TokenHit(source=source, surface=surface or norm, norm=norm, bbox=bbox)


def test_full_agreement_is_correct():
    hits = [
        _hit("ocr", "ABC123", bbox=(0.1, 0.1, 0.2, 0.2)),
        _hit("gemini", "ABC123"),
        _hit("gpt", "ABC123"),
    ]
    clusters = build_clusters(hits)
    _, judgements = vote(clusters, ["ocr", "gemini", "gpt"])
    assert {j.verdict for j in judgements} == {"correct"}
    assert clusters[0].canonical_norm == "ABC123"
    # OCR's bbox propagated to the cluster
    assert clusters[0].bbox == (0.1, 0.1, 0.2, 0.2)


def test_typo_is_flagged_with_distance():
    hits = [
        _hit("ocr", "ABC123"),
        _hit("gemini", "ABC123"),
        _hit("gpt", "ABC124"),  # one-edit typo
    ]
    clusters = build_clusters(hits, max_distance=1)
    _, judgements = vote(clusters, ["ocr", "gemini", "gpt"])
    by_src = {j.source: j for j in judgements}
    assert by_src["ocr"].verdict == "correct"
    assert by_src["gemini"].verdict == "correct"
    assert by_src["gpt"].verdict == "typo"
    assert by_src["gpt"].distance == 1


def test_omission_when_source_missing_from_consensus():
    hits = [_hit("ocr", "ABC123"), _hit("gemini", "ABC123")]
    clusters = build_clusters(hits)
    _, judgements = vote(clusters, ["ocr", "gemini", "gpt"])
    by_src = {j.source: j for j in judgements}
    assert by_src["gpt"].verdict == "omission"
    assert by_src["gpt"].surface_observed is None


def test_singleton_is_hallucination_only_for_that_source():
    hits = [_hit("gemini", "RANDOM999")]
    clusters = build_clusters(hits)
    _, judgements = vote(clusters, ["ocr", "gemini", "gpt"])
    # singleton: only one judgement (for the source that emitted it)
    assert len(judgements) == 1
    assert judgements[0].source == "gemini"
    assert judgements[0].verdict == "hallucination"


def test_ocr_wins_tie_for_canonical():
    # one vote each for two variants
    hits = [_hit("ocr", "ABC123"), _hit("gemini", "ABC124")]
    clusters = build_clusters(hits, max_distance=1)
    _, _ = vote(clusters, ["ocr", "gemini"])
    assert clusters[0].canonical_norm == "ABC123"  # OCR wins the tie


def test_apply_vision_verdict_promotes_hallucination():
    hits = [_hit("gemini", "RARE-ID-9999")]
    clusters = build_clusters(hits)
    _, judgements = vote(clusters, ["ocr", "gemini", "gpt"])
    j = judgements[0]
    apply_vision_verdict(j, visible=True, evidence="seen in image")
    assert j.verdict == "correct"
    assert j.evidence == "seen in image"


def test_apply_vision_verdict_keeps_hallucination_when_not_visible():
    hits = [_hit("gemini", "RARE-ID-9999")]
    clusters = build_clusters(hits)
    _, judgements = vote(clusters, ["ocr", "gemini", "gpt"])
    j = judgements[0]
    apply_vision_verdict(j, visible=False, evidence="not present")
    assert j.verdict == "hallucination"
    assert j.evidence == "not present"
