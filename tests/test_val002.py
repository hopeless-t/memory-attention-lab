from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from memory_attention_lab.attention.value_sources import (
    factorial_value_cells,
    historical_value_embedding,
)
from memory_attention_lab.experiments.val002 import SpecError, load_spec, run_val002


ROOT = Path(__file__).resolve().parents[1]
SPEC_PATH = ROOT / "specs" / "VAL-002.reference.json"
PARENT_PATH = ROOT / "specs" / "VAL-001.json"


def _loaded():
    return load_spec(SPEC_PATH, PARENT_PATH)


def test_frozen_val002_passes_all_checks():
    spec, parent, spec_sha, parent_sha = _loaded()
    result = run_val002(spec, parent, spec_sha, parent_sha)

    assert result["manifest"]["status"] == "PASS"
    assert all(check["passed"] for check in result["checks"].values())


def test_c11_and_memory_are_anchored_to_val001():
    spec, parent, spec_sha, parent_sha = _loaded()
    result = run_val002(spec, parent, spec_sha, parent_sha)

    assert result["checks"]["parent_C10_content_key_anchor"]["passed"]
    assert result["checks"]["parent_memory_anchor"]["passed"]
    assert result["checks"]["parent_C11_value_anchor"]["passed"]
    assert result["checks"]["parent_rotated_key_anchor"]["passed"]
    assert result["checks"]["parent_fixture_identity"]["passed"]


def test_factorial_memory_deltas_are_identical():
    spec, _, _, _ = _loaded()
    f = spec["fixture"]
    p = spec["parameters"]

    cells = factorial_value_cells(
        np.asarray(f["hidden_states"], dtype=np.float64),
        np.asarray(f["wv"], dtype=np.float64),
        np.asarray(f["wk"], dtype=np.float64),
        np.asarray(f["memory_table"], dtype=np.float64),
        np.asarray(f["token_ids"], dtype=np.int64),
        np.asarray(f["norm_weight"], dtype=np.float64),
        num_kv_heads=p["num_kv_heads"],
        head_dim=p["head_dim"],
        eps=p["eps"],
    )

    np.testing.assert_allclose(cells["C01"] - cells["C00"], cells["memory"])
    np.testing.assert_allclose(cells["C11"] - cells["C10"], cells["memory"])
    np.testing.assert_allclose(
        (cells["C11"] - cells["C10"]) - (cells["C01"] - cells["C00"]),
        0.0,
        atol=1e-15,
    )


def test_historical_lambda_endpoints():
    spec, _, _, _ = _loaded()
    f = spec["fixture"]
    hidden = np.asarray(f["hidden_states"], dtype=np.float64)
    wv = np.asarray(f["wv"], dtype=np.float64)
    table = np.asarray(f["historical_value_table"], dtype=np.float64)
    ids = np.asarray(f["token_ids"], dtype=np.int64)

    projected = hidden @ wv
    np.testing.assert_allclose(
        historical_value_embedding(hidden, wv, table, ids, lamb=0.0),
        projected,
    )
    np.testing.assert_allclose(
        historical_value_embedding(hidden, wv, table, ids, lamb=1.0),
        table[ids],
    )


def test_historical_lane_is_not_clean_additive_control():
    spec, parent, spec_sha, parent_sha = _loaded()
    result = run_val002(spec, parent, spec_sha, parent_sha)

    check = result["checks"]["historical_lane_is_not_clean_additive_control"]
    assert check["passed"]
    assert (
        check["max_abs_delta"]
        >= spec["acceptance"]["min_historical_vs_additive_delta"]
    )


def test_unknown_field_is_invalid(tmp_path: Path):
    raw = json.loads(SPEC_PATH.read_text())
    raw["parameters"]["silent_default"] = 1
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(raw))

    with pytest.raises(SpecError, match="extra"):
        load_spec(bad, PARENT_PATH)


def test_parent_hash_mismatch_is_invalid(tmp_path: Path):
    parent = json.loads(PARENT_PATH.read_text())
    parent["description"] = parent["description"] + " tampered"
    bad_parent = tmp_path / "VAL-001.json"
    bad_parent.write_text(json.dumps(parent))

    with pytest.raises(SpecError, match="SHA-256"):
        load_spec(SPEC_PATH, bad_parent)


def test_malformed_historical_table_is_invalid(tmp_path: Path):
    raw = json.loads(SPEC_PATH.read_text())
    raw["fixture"]["historical_value_table"][0] = [1.0, 2.0]
    bad = tmp_path / "bad-table.json"
    bad.write_text(json.dumps(raw))

    with pytest.raises(SpecError):
        load_spec(bad, PARENT_PATH)


def test_wrong_expected_value_produces_scientific_fail():
    spec, parent, spec_sha, parent_sha = _loaded()
    broken = json.loads(json.dumps(spec))
    broken["expected"]["C11"][0][0] += 0.25

    result = run_val002(broken, parent, spec_sha, parent_sha)

    assert result["manifest"]["status"] == "FAIL"
    assert not result["checks"]["known_answer_C11"]["passed"]


def test_projection_perturbation_guards_are_active():
    spec, parent, spec_sha, parent_sha = _loaded()
    result = run_val002(spec, parent, spec_sha, parent_sha)

    assert result["checks"]["Wv_perturbation_guard"]["passed"]
    assert result["checks"]["Wk_perturbation_guard"]["passed"]
