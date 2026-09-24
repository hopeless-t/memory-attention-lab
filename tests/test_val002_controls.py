from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from memory_attention_lab.attention.value_sources import (
    factorial_effects,
    factorial_value_sources,
    historical_value_embedding,
)
from memory_attention_lab.experiments.val002 import SpecError, load_spec, run_val002


ROOT = Path(__file__).resolve().parents[1]
SPEC_PATH = ROOT / "specs" / "VAL-002.json"
PARENT_PATH = ROOT / "specs" / "VAL-001.json"


def _loaded():
    return load_spec(SPEC_PATH, PARENT_PATH)


def test_frozen_val002_passes_all_checks():
    spec, parent, spec_sha, parent_sha = _loaded()
    result = run_val002(spec, parent, spec_sha, parent_sha)

    assert result["manifest"]["status"] == "PASS"
    assert all(check["passed"] for check in result["checks"].values())


def test_historical_lambda_endpoints():
    spec, _, _, _ = _loaded()
    f = spec["fixture"]
    p = np.asarray(f["projected_values"], dtype=np.float64)
    e = np.asarray(f["historical_token_values"], dtype=np.float64)

    np.testing.assert_allclose(historical_value_embedding(p, e, 0.0), p)
    np.testing.assert_allclose(historical_value_embedding(p, e, 1.0), e)


def test_factorial_identities():
    spec, _, _, _ = _loaded()
    f = spec["fixture"]
    p = np.asarray(f["projected_values"], dtype=np.float64)
    k = np.asarray(f["content_keys"], dtype=np.float64)
    m = np.asarray(f["ma_memory_contribution"], dtype=np.float64)

    cells = factorial_value_sources(p, k, m)
    effects = factorial_effects(cells)

    np.testing.assert_allclose(effects["memory_effect_from_projected"], m)
    np.testing.assert_allclose(effects["memory_effect_from_key"], m)
    np.testing.assert_allclose(effects["source_effect_without_memory"], k - p)
    np.testing.assert_allclose(effects["source_effect_with_memory"], k - p)
    np.testing.assert_allclose(effects["additive_interaction"], 0.0, atol=1e-15)


def test_c11_is_anchored_to_val001_constructed_values():
    spec, parent, _, _ = _loaded()
    f = spec["fixture"]
    cells = factorial_value_sources(
        np.asarray(f["projected_values"], dtype=np.float64),
        np.asarray(f["content_keys"], dtype=np.float64),
        np.asarray(f["ma_memory_contribution"], dtype=np.float64),
    )
    np.testing.assert_allclose(
        cells["c11_key_plus_memory"],
        np.asarray(parent["expected"]["constructed_values"], dtype=np.float64),
    )


def test_historical_lane_is_not_local_additive_control():
    spec, _, _, _ = _loaded()
    f = spec["fixture"]
    historical = historical_value_embedding(
        np.asarray(f["projected_values"], dtype=np.float64),
        np.asarray(f["historical_token_values"], dtype=np.float64),
        spec["parameters"]["historical_lambda"],
    )
    cells = factorial_value_sources(
        np.asarray(f["projected_values"], dtype=np.float64),
        np.asarray(f["content_keys"], dtype=np.float64),
        np.asarray(f["ma_memory_contribution"], dtype=np.float64),
    )
    delta = np.max(np.abs(historical - cells["c01_projected_plus_memory"]))
    assert delta >= spec["acceptance"]["min_historical_additive_separation"]


def test_unknown_field_is_invalid(tmp_path: Path):
    raw = json.loads(SPEC_PATH.read_text())
    raw["parameters"]["silent_default"] = 1
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(raw))

    with pytest.raises(SpecError, match="extra"):
        load_spec(bad, PARENT_PATH)


def test_malformed_shape_is_invalid(tmp_path: Path):
    raw = json.loads(SPEC_PATH.read_text())
    raw["fixture"]["projected_values"][0] = [[1.0, 2.0]]
    bad = tmp_path / "bad-shape.json"
    bad.write_text(json.dumps(raw))

    with pytest.raises(SpecError):
        load_spec(bad, PARENT_PATH)


def test_parent_hash_mismatch_is_invalid(tmp_path: Path):
    parent = json.loads(PARENT_PATH.read_text())
    parent["description"] = parent["description"] + " tampered"
    bad_parent = tmp_path / "VAL-001.json"
    bad_parent.write_text(json.dumps(parent))

    with pytest.raises(SpecError, match="SHA-256"):
        load_spec(SPEC_PATH, bad_parent)


def test_broken_expected_value_produces_fail():
    spec, parent, spec_sha, parent_sha = _loaded()
    broken = json.loads(json.dumps(spec))
    broken["expected"]["historical_value_embedding"][0][0][0] += 0.25

    result = run_val002(broken, parent, spec_sha, parent_sha)

    assert result["manifest"]["status"] == "FAIL"
    assert not result["checks"]["historical_known_answer"]["passed"]
