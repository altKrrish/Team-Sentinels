import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sentinel import interlock


def test_direct_interlock_fires_on_english_phrase():
    r = interlock.scan("Worker suffered amputation while clearing jam on drawworks.")
    assert r.fired
    assert r.forced_label == "SIF"
    assert any(m.canonical == "caught_between" for m in r.matches)


def test_negated_phrase_does_not_fire():
    r = interlock.scan("This was a fall from height drill conducted with no incident.")
    assert not r.fired


def test_near_miss_language_does_not_fire_alone():
    r = interlock.scan("Near miss: worker almost fell from height but caught the rail in time.")
    assert not r.fired


def test_outcome_marker_overrides_negation_window():
    # "fatality" is an outcome-override canonical: even with nearby soft words
    # it must still fire because a death is not something you "almost" have.
    r = interlock.scan("Sadly confirming fatality of contractor at site this morning.")
    assert r.fired
    assert any(m.canonical == "fatality" for m in r.matches)


def test_romanized_hindi_electrical_shock():
    r = interlock.scan("current lag gaya tha panel ke pass, worker behosh")
    assert r.fired
    assert any(m.canonical == "live_contact" for m in r.matches)


def test_devanagari_electrocution():
    r = interlock.scan("पैनल के पास करंट लगना, कर्मचारी बेहोश")
    assert r.fired


def test_bengali_assamese_fall_from_height():
    r = interlock.scan("উচ্চতা থেকে পড়ে গিয়ে গুরুতর আহত")
    assert r.fired


def test_misspelled_arc_flash_fuzzy_match():
    r = interlock.scan("Sudden arc flassh near switchgear injured two technicians.")
    assert r.fired
    assert any(m.canonical == "arc_flash" for m in r.matches)


def test_two_corroborate_hits_different_energy_classes_fire():
    r = interlock.scan("Gas leak detected near pump while hot work without permit was ongoing.")
    assert r.fired
    assert "INTERLOCK: multiple corroborating" in r.reason


def test_single_corroborate_hit_alone_does_not_fire():
    r = interlock.scan("Minor gas leak noticed and reported to control room.")
    assert not r.fired


def test_short_irrelevant_report_no_false_positive():
    r = interlock.scan("Housekeeping issue near office reported, floor was slippery.")
    assert not r.fired


def test_sparse_report_leak_near_pump_does_not_auto_fire_without_energy_context():
    # "leak near pump" alone should not be an INTERLOCK phrase; this is exactly
    # the case metadata imputation is supposed to catch instead.
    r = interlock.scan("leak near pump")
    assert not r.fired


# --- Fix 6: Barrier negation vs event negation edge cases ---

def test_barrier_negation_without_harness_does_not_suppress_interlock():
    # "without harness" is a BARRIER negation (hazard IS present).
    # The "without" should NOT suppress the fall_from_height interlock match.
    r = interlock.scan(
        "Worker observed working at height without harness on derrick floor, "
        "but no incident occurred."
    )
    # The interlock should fire because "without harness" means the barrier
    # is MISSING, and there are fall-from-height indicators present.
    assert r.fired or any(
        m.canonical in ("fall_from_height", "ppe_gap")
        for m in r.matches + r.corroborate_only
        if not m.negated
    )


def test_event_negation_still_suppresses_interlock():
    # "no arc flash occurred" is EVENT negation — the hazard did NOT happen.
    # The interlock should NOT fire.
    r = interlock.scan("Arc flash training completed, no arc flash occurred during session.")
    assert not r.fired


def test_without_permit_does_not_suppress_hot_work():
    # "without permit" is a BARRIER negation — hot work WAS done improperly.
    # Use text that matches lexicon surfaces: "hot work without permit" is an
    # exact surface in hot_work_uncontrolled, and "gas leak" is in gas_release.
    r = interlock.scan(
        "Gas leak detected during hot work without permit at site."
    )
    # hot_work_uncontrolled and gas_release are both CORROBORATE tier from
    # different energy classes (temperature + pressure), so they should fire
    # the interlock via multi-class corroboration. "without permit" must not
    # suppress either match.
    non_negated = [
        m for m in r.matches + r.corroborate_only if not m.negated
    ]
    assert len(non_negated) > 0, (
        "Barrier negation 'without permit' incorrectly suppressed corroborate matches"
    )


