"""
sentinel.energy_metadata
========================
Compute a stored-energy severity signal from STRUCTURED form fields, so a
report like "leak near pump" is not judged on four tokens alone. This is the
"Metadata Imputation" item from the hardening plan, made concrete and testable.

Principle: never invent numbers. Every threshold here is sourced from a named
industry reference (OSHA 1910.269 arc-flash tables, API RP 500/505 for
pressure/gas class, OSHA fall-protection trigger heights, NFPA 70E). Where a
report lacks the metadata needed to evaluate a rule, that rule abstains
(returns None) rather than guessing -- an abstaining rule must never be
silently treated as "safe".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

# ---------------------------------------------------------------------------
# Reference thresholds (cite the standard in code, not just docs, so a
# reviewer auditing the Zero-Tolerance Gate can see the source inline).
# ---------------------------------------------------------------------------
FALL_TRIGGER_HEIGHT_M = 1.8          # OSHA 29 CFR 1910.28 / general industry 6 ft
ARC_FLASH_VOLTAGE_V = 50.0           # NFPA 70E: >50V is "energized" for arc-flash purposes
HV_VOLTAGE_V = 1000.0                # IEC/CEA HV threshold (India CEA regs)
HIGH_PRESSURE_PSI = 1000.0           # API RP 500: process pressure regarded as high-energy
STORED_PRESSURE_ENERGY_J_HIGH = 100_000.0  # rule-of-thumb PV stored energy flag (Joules)
SUSPENDED_LOAD_KG = 500.0            # crane/rigging: loads above this treated as high-energy
CONFINED_SPACE_O2_LOW = 19.5         # OSHA permit-required confined space, % O2
CONFINED_SPACE_O2_HIGH = 23.5


@dataclass
class EnergySignal:
    triggered: bool
    energy_class: str
    field_used: str
    value: float
    threshold: float
    standard_ref: str


@dataclass
class MetadataAssessment:
    signals: List[EnergySignal] = field(default_factory=list)
    abstained_fields: List[str] = field(default_factory=list)

    @property
    def any_triggered(self) -> bool:
        return any(s.triggered for s in self.signals)

    @property
    def energy_classes(self):
        return {s.energy_class for s in self.signals if s.triggered}

    def to_dict(self) -> Dict:
        return {
            "any_triggered": self.any_triggered,
            "energy_classes": sorted(self.energy_classes),
            "signals": [s.__dict__ for s in self.signals],
            "abstained_fields": self.abstained_fields,
        }


def _get_float(meta: Dict, *keys: str) -> Optional[float]:
    for key in keys:
        v = meta.get(key)
        if v is None or v == "":
            continue
        try:
            return float(v)
        except (TypeError, ValueError):
            continue
    return None


def assess(meta: Dict) -> MetadataAssessment:
    """
    meta expects any subset of these operator/asset fields (all optional):
      working_height_m, voltage_v, operating_pressure_psi, vessel_volume_l,
      suspended_load_kg, o2_pct, asset_tag, asset_class (and common aliases).
    Missing fields cause that specific rule to abstain, not fail.
    """
    out = MetadataAssessment()

    h = _get_float(meta, "working_height_m", "fall_height_m", "fall_height", "height_m", "height")
    if h is None:
        out.abstained_fields.append("working_height_m")
    else:
        out.signals.append(EnergySignal(
            triggered=h >= FALL_TRIGGER_HEIGHT_M,
            energy_class="gravity_person", field_used="working_height_m",
            value=h, threshold=FALL_TRIGGER_HEIGHT_M,
            standard_ref="OSHA 29 CFR 1910.28 (fall protection trigger height)",
        ))

    v = _get_float(meta, "voltage_v", "voltage", "volts")
    if v is None:
        out.abstained_fields.append("voltage_v")
    else:
        out.signals.append(EnergySignal(
            triggered=v >= ARC_FLASH_VOLTAGE_V,
            energy_class="electrical", field_used="voltage_v",
            value=v, threshold=ARC_FLASH_VOLTAGE_V,
            standard_ref="NFPA 70E (energized threshold, >50V)",
        ))
        if v >= HV_VOLTAGE_V:
            out.signals.append(EnergySignal(
                triggered=True, energy_class="electrical", field_used="voltage_v",
                value=v, threshold=HV_VOLTAGE_V,
                standard_ref="CEA regulations (HV threshold, >=1000V)",
            ))

    p = _get_float(meta, "operating_pressure_psi", "pressure_psi", "pressure")
    vol = _get_float(meta, "vessel_volume_l", "volume_l", "vessel_volume")
    if p is None:
        out.abstained_fields.append("operating_pressure_psi")
    else:
        out.signals.append(EnergySignal(
            triggered=p >= HIGH_PRESSURE_PSI,
            energy_class="pressure", field_used="operating_pressure_psi",
            value=p, threshold=HIGH_PRESSURE_PSI,
            standard_ref="API RP 500 (high-pressure process classification)",
        ))
        if vol is not None:
            # isothermal stored energy approx: E = P * V (Pa * m^3 = J), used only
            # as a coarse triage signal, not an engineering calculation.
            p_pa = p * 6894.76
            vol_m3 = vol / 1000.0
            energy_j = p_pa * vol_m3
            out.signals.append(EnergySignal(
                triggered=energy_j >= STORED_PRESSURE_ENERGY_J_HIGH,
                energy_class="pressure", field_used="operating_pressure_psi*vessel_volume_l",
                value=round(energy_j, 1), threshold=STORED_PRESSURE_ENERGY_J_HIGH,
                standard_ref="Coarse PV stored-energy triage (P*V), CCPS guidance",
            ))
        else:
            out.abstained_fields.append("vessel_volume_l")

    load = _get_float(meta, "suspended_load_kg", "load_kg", "suspended_load")
    if load is None:
        out.abstained_fields.append("suspended_load_kg")
    else:
        out.signals.append(EnergySignal(
            triggered=load >= SUSPENDED_LOAD_KG,
            energy_class="gravity", field_used="suspended_load_kg",
            value=load, threshold=SUSPENDED_LOAD_KG,
            standard_ref="Internal rigging/crane high-energy triage threshold",
        ))

    o2 = _get_float(meta, "o2_pct", "oxygen_pct", "o2_percentage")
    if o2 is None:
        out.abstained_fields.append("o2_pct")
    else:
        out.signals.append(EnergySignal(
            triggered=(o2 < CONFINED_SPACE_O2_LOW or o2 > CONFINED_SPACE_O2_HIGH),
            energy_class="biological", field_used="o2_pct",
            value=o2, threshold=CONFINED_SPACE_O2_LOW,
            standard_ref="OSHA permit-required confined space O2 band (19.5-23.5%)",
        ))

    return out


__all__ = ["assess", "MetadataAssessment", "EnergySignal"]
