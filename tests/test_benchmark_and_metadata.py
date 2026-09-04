import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sentinel import benchmark, energy_metadata


_GOOD_ROW = (
    "case_id,text,label,energy_class,source_type,source_ref,incident_date,asset_class\n"
)


def _make_csv(n_sif=6, n_not=6):
    rows = [_GOOD_ROW]
    energy_cycle = ["gravity", "motion", "mechanical", "electrical", "pressure",
                     "temperature", "chemical", "radiation", "biological",
                     "gravity_person"]
    for i in range(n_sif):
        ec = energy_cycle[i % len(energy_cycle)]
        rows.append(
            f"C{i},worker suffered serious injury during {ec} incident today,SIF,{ec},"
            f"DGMS_INQUIRY,DGMS/2023/{i:04d},2023-0{(i%9)+1}-01,wellhead\n"
        )
    for i in range(n_not):
        rows.append(
            f"N{i},routine housekeeping inspection completed without incident,NOT_SIF,,"
            f"INTERNAL_INVESTIGATION,INT/2023/{i:04d},2023-0{(i%9)+1}-15,storage_yard\n"
        )
    return "".join(rows)


def test_benchmark_validator_rejects_missing_columns():
    bad = "case_id,text\nC1,too few columns\n"
    report = benchmark.validate_benchmark_csv(bad, min_rows=1)
    assert not report.is_valid


def test_benchmark_validator_flags_missing_source_ref():
    csv_text = (
        _GOOD_ROW +
        "C1,worker suffered arc flash injury at panel,SIF,electrical,DGMS_INQUIRY,,2023-01-01,wellhead\n"
    )
    report = benchmark.validate_benchmark_csv(csv_text, min_rows=1, min_per_energy_class=0)
    assert not report.is_valid
    assert any("source_ref" in e for row in report.row_errors for e in row.errors)


def test_benchmark_validator_enforces_min_rows_and_class_coverage():
    small = _make_csv(n_sif=2, n_not=2)
    report = benchmark.validate_benchmark_csv(small, min_rows=100, min_per_energy_class=5)
    assert not report.is_valid
    dataset_errors = [e for row in report.row_errors for e in row.errors if row.case_id == "<dataset>"]
    assert any("need >=" in e for e in dataset_errors)


def test_benchmark_validator_passes_well_formed_dataset():
    csv_text = _make_csv(n_sif=10, n_not=10)
    report = benchmark.validate_benchmark_csv(csv_text, min_rows=15, min_per_energy_class=1)
    assert report.is_valid, report.row_errors


def test_metadata_abstains_on_missing_fields():
    result = energy_metadata.assess({})
    assert not result.any_triggered
    assert "working_height_m" in result.abstained_fields


def test_metadata_triggers_on_fall_height():
    result = energy_metadata.assess({"working_height_m": 3.5})
    assert result.any_triggered
    assert "gravity_person" in result.energy_classes


def test_metadata_does_not_trigger_below_threshold():
    result = energy_metadata.assess({"working_height_m": 0.5, "voltage_v": 12})
    assert not result.any_triggered
