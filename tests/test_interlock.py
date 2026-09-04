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
