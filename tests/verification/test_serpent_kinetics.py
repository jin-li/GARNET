"""Global transport-output import, distinct from regional kinetics inputs."""
import hashlib

import pytest

from ndgpu.serpent_kinetics import read_serpent_kinetics


_COUNTER = "if (exist('idx', 'var')); idx = idx + 1; else; idx = 1; end;\n"


def assignment(key, value):
    count = len(value.strip("[]").split()) if value.startswith("[") else 1
    return f"{key}(idx,[1:{count}]) = {value};\n"


def block(beta="[0.00600 0.01 0.00200 0.02 0.00400 0.03]"):
    values = {
        "VERSION": "'Serpent 2.2.2'", "PRECURSOR_GROUPS": "2", "IFP_CHAIN_LENGTH": "15",
        "POP": "20000", "CYCLES": "4000", "SKIP": "800", "TITLE": "'fixture'",
        "INPUT_FILE_NAME": "'input'", "START_DATE": "'start'", "COMPLETE_DATE": "'end'",
        "ADJ_IFP_IMP_BETA_EFF": beta,
        "ADJ_IFP_ANA_BETA_EFF": beta, "FWD_ANA_BETA_ZERO": beta,
        "ADJ_IFP_IMP_LAMBDA": "[0.5 0.01 0.01 0.001 0.1 0.002]",
        "ADJ_IFP_ANA_LAMBDA": "[0.6 0.02 0.02 0.001 0.2 0.002]",
        "FWD_ANA_LAMBDA": "[0.7 0.03 0.03 0.001 0.3 0.002]",
        "ADJ_IFP_GEN_TIME": "[2e-5 0.001 1e-5 0.002 3e-5 0.003]",
        "ADJ_IFP_LIFETIME": "[2.2e-5 0.002 1.1e-5 0.003 3.3e-5 0.004]",
    }
    return _COUNTER + "".join(assignment(key, value) for key, value in values.items())


def read(tmp_path, text):
    path = tmp_path / "input_res.m"
    path.write_text(text)
    return read_serpent_kinetics(path)


def test_repeated_globals_are_immutable_provenance_not_averaged_samples(tmp_path):
    text = block() + block()
    result = read(tmp_path, text)
    assert result.source_sha256 == hashlib.sha256(text.encode()).hexdigest()
    assert result.record_count == 2
    assert result.field_records["ADJ_IFP_IMP_BETA_EFF"] == (0, 1)
    beta = result.estimates["ADJ_IFP_IMP_BETA_EFF"]
    assert beta.mean == (0.006, 0.002, 0.004)
    assert beta.relative_error == (0.01, 0.02, 0.03)
    assert beta.absolute_error == pytest.approx((6e-5, 4e-5, 12e-5))
    assert beta.labels == ("total", "family_1", "family_2")
    assert result.estimates["ADJ_IFP_GEN_TIME"].labels == ("total", "prompt", "delayed")
    assert result.estimates["ADJ_IFP_IMP_LAMBDA"].unit == "s^-1"
    assert result.estimates["ADJ_IFP_GEN_TIME"].unit == "s"
    assert result.estimates["ADJ_IFP_ANA_LAMBDA"].mean[0] == 0.6
    assert "native_score_weighted_arithmetic_lambda:ADJ_IFP_IMP_LAMBDA" in result.flags
    assert "analog cycle" in result.native_definitions["ADJ_IFP_GEN_TIME"]
    assert "unqualified_NDgpu_MC_time_equivalence" in result.flags
    assert "native_score_weighted_arithmetic_lambda:ADJ_IFP_ANA_LAMBDA" in result.flags
    assert "not established" in result.native_definitions["FWD_ANA_LAMBDA"]
    assert not any(flag.startswith("missing:") for flag in result.flags)
    with pytest.raises(TypeError):
        result.metadata["VERSION"] = "changed"
    with pytest.raises(TypeError):
        beta.mean[0] = 0


@pytest.mark.parametrize("old,new", [
    ("0.00600 0.01", "0.00600 0.02"),
    ("Serpent 2.2.2", "Serpent 2.2.1"),
    ("PRECURSOR_GROUPS(idx,[1:1]) = 2", "PRECURSOR_GROUPS(idx,[1:1]) = 3"),
])
def test_conflicting_global_repetitions_rejected(tmp_path, old, new):
    with pytest.raises(ValueError, match="conflicting global repetition"):
        read(tmp_path, block() + block().replace(old, new))


@pytest.mark.parametrize("addition", [
    "ADJ_IFP_GEN_TIME(idx,[1:6]) = system('touch /tmp/kinetics-executed');",
    "ADJ_IFP_GEN_TIME(1,:) = [1 0 1 0 1 0];",
    "ADJ_IFP_GEN_TIME(idx,[1:6]) = [1+2 0 1 0 1 0];",
    "ADJ_IFP_GEN_TIME(idx,[1:6]) = [1 0 1 0 1 0]",
    "ADJ_IFP_GEN_TIME(idx,[1:6]) = [1 0 1 0 1 0];",
])
def test_expressions_bad_indexing_unterminated_or_duplicate_rejected(tmp_path, addition):
    with pytest.raises(ValueError):
        read(tmp_path, block() + addition)


@pytest.mark.parametrize("old,new,match", [
    ("2e-5 0.001", "nan 0.001", "finite"),
    ("2e-5 0.001", "0 0.001", "positive"),
    ("2e-5 0.001", "2e-5 -0.001", "errors"),
    ("0.00600 0.01", "-0.00600 0.01", "nonnegative"),
    ("0.00600 0.01", "0.00602 0.01", "family sum"),
    ("PRECURSOR_GROUPS(idx,[1:1]) = 2", "PRECURSOR_GROUPS(idx,[1:1]) = 2.5", "integer"),
    ("PRECURSOR_GROUPS(idx,[1:1]) = 2", "PRECURSOR_GROUPS(idx,[1:1]) = 3", "families"),
    ("ADJ_IFP_GEN_TIME(idx,[1:6])", "ADJ_IFP_GEN_TIME(idx,[1:4])", "declared length"),
])
def test_invalid_data_rejected(tmp_path, old, new, match):
    with pytest.raises(ValueError, match=match):
        read(tmp_path, block().replace(old, new))


def test_beta_rounding_uses_actual_printed_precision_not_statistical_error(tmp_path):
    result = read(tmp_path, block("[6.000E-3 0.9 2.000E-3 0.9 4.001E-3 0.9]"))
    assert result.estimates["ADJ_IFP_IMP_BETA_EFF"].mean[0] == 0.006
    with pytest.raises(ValueError, match="printed precision"):
        read(tmp_path, block("[6.000E-3 0.9 2.000E-3 0.9 4.010E-3 0.9]"))


def test_missing_metadata_and_estimates_are_flags_without_defaults(tmp_path):
    result = read(tmp_path, assignment("ADJ_IFP_IMP_LAMBDA", "[0.5 0 0.1 0]"))
    assert "VERSION" not in result.metadata
    assert "ADJ_IFP_GEN_TIME" not in result.estimates
    assert "missing:VERSION" in result.flags
    assert "missing:ADJ_IFP_GEN_TIME" in result.flags
    assert "unqualified_family_count:ADJ_IFP_IMP_LAMBDA" in result.flags
    assert "printed_zero_relative_error:ADJ_IFP_IMP_LAMBDA" in result.flags
    assert "unqualified_version" in result.flags


def test_missing_one_repetition_is_flagged_not_silently_complete(tmp_path):
    result = read(tmp_path, block() + _COUNTER + assignment("VERSION", "'Serpent 2.2.2'"))
    assert result.record_count == 2
    assert "incomplete_global_repetition:ADJ_IFP_IMP_BETA_EFF" in result.flags


def test_irrelevant_matlab_is_not_executed_and_does_not_require_xs(tmp_path):
    sentinel = tmp_path / "never_created"
    result = read(tmp_path, block() + f"UNRELATED(idx,1) = system('touch {sentinel}');\n")
    assert not sentinel.exists()
    assert result.metadata["IFP_CHAIN_LENGTH"] == 15


def test_no_relevant_fields_is_error(tmp_path):
    with pytest.raises(ValueError, match="no recognized"):
        read(tmp_path, "UNRELATED(idx,1) = 1;")


def test_write_timestamps_are_per_record_observations_not_global_estimates(tmp_path):
    result = read(tmp_path, block() + block().replace("'end'", "'end plus one second'"))
    assert "COMPLETE_DATE" not in result.metadata
    assert result.record_metadata["COMPLETE_DATE"] == ((0, "end"), (1, "end plus one second"))
    assert result.record_metadata["START_DATE"] == ((0, "start"), (1, "start"))
    assert "per_record_metadata:COMPLETE_DATE" in result.flags
    with pytest.raises(TypeError):
        result.record_metadata["COMPLETE_DATE"] = ()


def test_native_definition_qualification_requires_exact_version(tmp_path):
    result = read(tmp_path, block().replace("Serpent 2.2.2", "Serpent 2.2.3"))
    assert not result.native_definitions
    assert "unqualified_total_lambda_definition:ADJ_IFP_IMP_LAMBDA" in result.flags
    assert "unqualified_time_convention:ADJ_IFP_GEN_TIME" in result.flags
    assert "unqualified_NDgpu_MC_time_equivalence" in result.flags
