import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sentinel import pattern_extractor


def test_extract_slots_drilling_derrick_no_harness():
    text = "Worker observed tripping pipe on derrick floor without harness, no fall protection used."
    result = pattern_extractor.extract_slots(text)
    assert "Tripping Pipe" in result.activities
    assert any("Derrick" in loc or "Drill Floor" in loc for loc in result.locations)
    assert any("Harness" in bf or "Fall Protection" in bf for bf in result.barrier_failures)
    assert result.has_full_triad


def test_extract_slots_welding_no_permit():
    text = "Hot work welding near gas line at gathering station without permit."
    result = pattern_extractor.extract_slots(text)
    assert "Welding" in result.activities
    assert "Gathering Station" in result.locations
    assert any("Permit" in bf for bf in result.barrier_failures)


def test_extract_slots_empty_text():
    result = pattern_extractor.extract_slots("Housekeeping issue in office area.")
    # Should return empty or minimal slots without crashing
    assert isinstance(result.activities, list)
    assert isinstance(result.locations, list)
    assert isinstance(result.barrier_failures, list)


def test_rank_patterns_batch():
    reports = [
        "Drilling on rig floor without harness",
        "Drilling on rig floor without harness, no fall protection",
        "Welding at gathering station without permit",
        "Drilling on drill floor no fall protection",
    ]
    ranked = pattern_extractor.rank_patterns(reports, top_k=5)
    assert len(ranked) > 0
    assert ranked[0].count >= 1
    # The most frequent triad should involve drilling
    assert any("Drilling" in t.activity for t in ranked)


def test_pattern_result_to_dict():
    result = pattern_extractor.extract_slots(
        "Crane lifting at tank farm with corroded sling"
    )
    d = result.to_dict()
    assert "activities" in d
    assert "locations" in d
    assert "barrier_failures" in d
    assert "has_full_triad" in d
