import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sentinel import decision_policy, energy_metadata, interlock


def _empty_interlock():
    return interlock.InterlockResult(fired=False, forced_label=None)


def _empty_metadata():
    return energy_metadata.MetadataAssessment()


def test_interlock_overrides_low_probability():
    il = interlock.scan("Explosion at process unit, one fatality confirmed.")
    meta = _empty_metadata()
    d = decision_policy.decide(0.02, interlock=il, metadata=meta)
    assert d.label == "SIF"
    assert d.route == decision_policy.Route.AUTO


def test_low_confidence_band_routes_to_human():
    d = decision_policy.decide(0.47, interlock=_empty_interlock(), metadata=_empty_metadata())
    assert d.label is None
    assert d.route == decision_policy.Route.HUMAN_REVIEW


def test_low_confidence_band_with_metadata_breach_auto_sif():
    meta = energy_metadata.assess({"voltage_v": 11000})
    d = decision_policy.decide(0.47, interlock=_empty_interlock(), metadata=meta)
    assert d.label == "SIF"
    assert d.route == decision_policy.Route.AUTO


def test_tau_for_asset_differs_by_energy_class():
    assert decision_policy.tau_for_asset(None) == decision_policy.TAU_DEFAULT
    assert decision_policy.tau_for_asset("wellhead") == decision_policy.TAU_HIGH_ENERGY
    assert decision_policy.tau_for_asset("storage_yard") == decision_policy.TAU_DEFAULT


def test_high_energy_asset_flags_sif_where_default_would_still_review_or_clear():
    # A probability comfortably above the high-energy tau but below the
    # default tau's decision band lower edge triggers SIF for a high-energy
    # asset via direct threshold comparison.
    tau_he = decision_policy.TAU_HIGH_ENERGY
    band = decision_policy.LOW_CONF_HALF_WIDTH
    p = tau_he + band + 0.01  # clearly outside the high-energy band, above its tau
    d_wellhead = decision_policy.decide(p, interlock=_empty_interlock(),
                                         metadata=_empty_metadata(), asset_class="wellhead")
    assert d_wellhead.label == "SIF"
    assert d_wellhead.tau_used == tau_he

    d_default = decision_policy.decide(0.05, interlock=_empty_interlock(),
                                        metadata=_empty_metadata(), asset_class=None)
    assert d_default.label == "NOT_SIF"
    assert d_default.tau_used == decision_policy.TAU_DEFAULT


def test_below_threshold_no_metadata_is_not_sif():
    d = decision_policy.decide(0.10, interlock=_empty_interlock(), metadata=_empty_metadata())
    assert d.label == "NOT_SIF"
    assert d.route == decision_policy.Route.AUTO


def test_below_threshold_with_metadata_breach_routes_human_not_silent_not_sif():
    meta = energy_metadata.assess({"working_height_m": 4.0})
    d = decision_policy.decide(0.10, interlock=_empty_interlock(), metadata=meta)
    assert d.label is None
    assert d.route == decision_policy.Route.HUMAN_REVIEW


def test_probability_out_of_range_raises():
    raised = False
    try:
        decision_policy.decide(1.5, interlock=_empty_interlock(), metadata=_empty_metadata())
    except ValueError:
        raised = True
    assert raised
