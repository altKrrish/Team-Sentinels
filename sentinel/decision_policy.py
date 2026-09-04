"""
sentinel.decision_policy
========================
Turns a raw model probability into a final decision, given:
  - the deterministic interlock result (sentinel.interlock)
  - the metadata energy assessment (sentinel.energy_metadata)
  - an asset-class-aware threshold table (not one global tau)

Precedence (highest wins, each documented so an auditor can reconstruct why
a given report got its label):

  1. INTERLOCK fired                     -> forced SIF, label="SIF", route=AUTO
  2. p in low-confidence band            -> label=None, route=HUMAN_REVIEW
  3. metadata energy breach + p >= tau_lo -> label=SIF, route=AUTO (energy-corroborated)
  4. p >= tau(asset_class)                -> label=SIF, route=AUTO
  5. otherwise                            -> label=NOT_SIF, route=AUTO

Why per-asset thresholds instead of one tau in [0.38, 0.42]: a single global
threshold shift trades false negatives for false positives uniformly across
every asset class, including ones where a false positive is cheap (a
storage-yard housekeeping report) and ones where a false negative is
catastrophic (a wellhead or HV switchgear report). Route the threshold by
asset high-energy class instead of by a single flat number.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional

from .energy_metadata import MetadataAssessment
from .interlock import InterlockResult

# Default (non-high-energy) asset threshold. Raise recall modestly vs the
# reported baseline 0.47, not down to a single blanket 0.38-0.42 -- that
# range is reserved for assets tagged as high-energy below.
TAU_DEFAULT = 0.44

# High-energy asset classes get the lower, more sensitive threshold. This
# targets the same '1.45% FN on high-consequence assets' problem without
# dragging every low-energy report's FP rate down with it.
TAU_HIGH_ENERGY = 0.40

HIGH_ENERGY_ASSET_CLASSES = {
    "wellhead", "gathering_station", "flowline", "hv_switchgear",
    "compressor", "crane", "confined_space_vessel", "pipeline",
    "drilling_rig", "workover_rig", "process_train",
}

# Low-confidence band is a window AROUND the asset-specific tau, not a fixed
# absolute range. An earlier version used a fixed [0.40, 0.55] band; because
# that interval fully contained both TAU_DEFAULT (0.44) and TAU_HIGH_ENERGY
# (0.40), it silently swallowed the asset-aware threshold logic below --
# every probability where the two tau values would have disagreed was routed
# to human review before the tau comparison ever ran. Centering the band on
# tau keeps "uncertain near the operating threshold" meaningful per asset
# class while still leaving room above/below it for confident auto-decisions.
LOW_CONF_HALF_WIDTH = 0.06


class Route(str, Enum):
    AUTO = "AUTO"
    HUMAN_REVIEW = "HUMAN_REVIEW"


@dataclass
class Decision:
    label: Optional[str]         # "SIF" | "NOT_SIF" | None (pending human review)
    probability: float
    route: Route
    tau_used: float
    reason: str
    interlock: Optional[InterlockResult] = None
    metadata: Optional[MetadataAssessment] = None

    def to_dict(self) -> Dict:
        return {
            "label": self.label,
            "probability": self.probability,
            "route": self.route.value,
            "tau_used": self.tau_used,
            "reason": self.reason,
            "interlock": self.interlock.to_dict() if self.interlock else None,
            "metadata": self.metadata.to_dict() if self.metadata else None,
        }


def tau_for_asset(asset_class: Optional[str]) -> float:
    if asset_class and asset_class.lower() in HIGH_ENERGY_ASSET_CLASSES:
        return TAU_HIGH_ENERGY
    return TAU_DEFAULT


def decide(
    probability: float,
    *,
    interlock: InterlockResult,
    metadata: MetadataAssessment,
    asset_class: Optional[str] = None,
) -> Decision:
    if not 0.0 <= probability <= 1.0:
        raise ValueError(f"probability out of range: {probability}")

    if metadata is None:
        from .energy_metadata import assess
        metadata = assess({})

    tau = tau_for_asset(asset_class)
    band_low = tau - LOW_CONF_HALF_WIDTH
    band_high = tau + LOW_CONF_HALF_WIDTH

    # 1. Deterministic interlock always wins, before any probability logic.
    if interlock.fired:
        return Decision(
            label="SIF", probability=probability, route=Route.AUTO, tau_used=tau,
            reason=interlock.reason, interlock=interlock, metadata=metadata,
        )

    # 2. Low-confidence band -> human, UNLESS metadata independently confirms
    #    a high-energy condition, in which case we don't want to sit on a
    #    report that structured data already tells us is dangerous.
    if band_low <= probability <= band_high:
        if metadata.any_triggered:
            return Decision(
                label="SIF", probability=probability, route=Route.AUTO, tau_used=tau,
                reason=(
                    "Model probability in low-confidence band, but structured "
                    f"metadata independently confirms high-energy condition(s): "
                    f"{sorted(metadata.energy_classes)}"
                ),
                interlock=interlock, metadata=metadata,
            )
        return Decision(
            label=None, probability=probability, route=Route.HUMAN_REVIEW, tau_used=tau,
            reason=(
                f"Probability {probability:.3f} within low-confidence band "
                f"[{band_low:.3f}, {band_high:.3f}] around tau={tau}; routed to safety officer."
            ),
            interlock=interlock, metadata=metadata,
        )

    # 3 & 4. Threshold decision, asset-aware.
    if probability >= tau:
        reason = f"Model probability {probability:.3f} >= tau({asset_class or 'default'})={tau}"
        if metadata.any_triggered:
            reason += f"; corroborated by metadata energy classes {sorted(metadata.energy_classes)}"
        return Decision(
            label="SIF", probability=probability, route=Route.AUTO, tau_used=tau,
            reason=reason, interlock=interlock, metadata=metadata,
        )

    # 5. Below threshold. Metadata breach on a report the model scored low
    #    is itself a red flag worth a human look, not a silent NOT_SIF.
    if metadata.any_triggered:
        return Decision(
            label=None, probability=probability, route=Route.HUMAN_REVIEW, tau_used=tau,
            reason=(
                f"Model probability {probability:.3f} below tau={tau}, but structured "
                f"metadata flags energy classes {sorted(metadata.energy_classes)}; "
                "routed to safety officer rather than auto-labelled NOT_SIF."
            ),
            interlock=interlock, metadata=metadata,
        )

    return Decision(
        label="NOT_SIF", probability=probability, route=Route.AUTO, tau_used=tau,
        reason=f"Model probability {probability:.3f} below tau={tau}; no interlock or metadata trigger.",
        interlock=interlock, metadata=metadata,
    )


__all__ = ["decide", "Decision", "Route", "tau_for_asset",
           "TAU_DEFAULT", "TAU_HIGH_ENERGY", "LOW_CONF_HALF_WIDTH",
           "HIGH_ENERGY_ASSET_CLASSES"]
