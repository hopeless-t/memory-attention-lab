from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from memory_attention_lab.attention.reference import (
    construct_values,
    fold_memory_table,
    inverse_rope_neox,
    lookup_memory,
    rope_neox,
)
from memory_attention_lab.experiments.val001 import SpecError, load_spec, run_val001


ROOT = Path(__file__).resolve().parents[1]
SPEC_PATH = ROOT / "specs" / "VAL-001.json"


def _loaded():
    return load_spec(SPEC_PATH)


def test_frozen_reference_passes_all_checks():
    spec, digest = _loaded()
    result = run_val001(spec, digest)

    assert result["manifest"]["status"] == "PASS"
    assert all(check["passed"] for check in result["checks"].values())


def test_repeated_token_has_same_memory_but_context_changes_value():
    spec, _ = _loaded()
    p = spec["parameters"]
    f = spec["fixture"]

    values, normalized = construct_values(
        np.asarray(f["content_keys"], dtype=np.float64),
        np.asarray(f["memory_table"], dtype=np.float64),
        np.asarray(f["token_ids"], dtype=np.int64),
        np.asarray(f["norm_weight"], dtype=np.float64),
        eps=p["eps"],
    )

    assert f["token_ids"][0] == f["token_ids"][-1]
    np.testing.assert_allclose(normalized[0], normalized[-1])
    assert not np.allclose(values[0], values[-1])


def test_folded_inference_table_matches_online_normalization():
    spec, _ = _loaded()
    p = spec["parameters"]
    f = spec["fixture"]

    keys = np.asarray(f["content_keys"], dtype=np.float64)
    ids = np.asarray(f["token_ids"], dtype=np.int64)
    table = np.asarray(f["memory_table"], dtype=np.float64)
    weight = np.asarray(f["norm_weight"], dtype=np.float64)

    values, _ = construct_values(keys, table, ids, weight, eps=p["eps"])
    folded = fold_memory_table(
        table,
        weight,
        num_kv_heads=p["num_kv_heads"],
        head_dim=p["head_dim"],
        eps=p["eps"],
    )
    selected = lookup_memory(
        folded,
        ids,
        num_kv_heads=p["num_kv_heads"],
        head_dim=p["head_dim"],
    )

    np.testing.assert_allclose(
        values,
        keys + selected,
        atol=spec["acceptance"]["atol"],
        rtol=spec["acceptance"]["rtol"],
    )


def test_rope_roundtrip_recovers_content_key():
    spec, _ = _loaded()
    p = spec["parameters"]
    f = spec["fixture"]

    keys = np.asarray(f["content_keys"], dtype=np.float64)
    positions = np.asarray(f["positions"], dtype=np.float64)

    rotated = rope_neox(keys, positions, base=p["rope_base"])
    recovered = inverse_rope_neox(rotated, positions, base=p["rope_base"])

    np.testing.assert_allclose(
        recovered,
        keys,
        atol=spec["acceptance"]["atol"],
        rtol=spec["acceptance"]["rtol"],
    )


def test_post_rope_key_is_detectably_wrong_for_value_construction():
    spec, digest = _loaded()
    result = run_val001(spec, digest)
    guard = result["checks"]["pre_rope_value_guard"]

    assert guard["passed"]
    assert (
        guard["wrong_post_rope_value_max_delta"]
        >= spec["acceptance"]["min_post_rope_guard_delta"]
    )


def test_unknown_top_level_field_is_rejected(tmp_path: Path):
    raw = json.loads(SPEC_PATH.read_text())
    raw["surprise"] = "must fail closed"
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(raw))

    with pytest.raises(SpecError, match="extra"):
        load_spec(bad)


def test_unknown_nested_field_is_rejected(tmp_path: Path):
    raw = json.loads(SPEC_PATH.read_text())
    raw["parameters"]["silent_default"] = 123
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(raw))

    with pytest.raises(SpecError, match="extra"):
        load_spec(bad)


def test_ragged_memory_table_is_invalid_spec(tmp_path: Path):
    raw = json.loads(SPEC_PATH.read_text())
    raw["fixture"]["memory_table"][0] = [1.0, 2.0]
    bad = tmp_path / "bad-ragged.json"
    bad.write_text(json.dumps(raw))

    with pytest.raises(SpecError, match="memory_table"):
        load_spec(bad)


def test_out_of_range_token_id_is_invalid_spec(tmp_path: Path):
    raw = json.loads(SPEC_PATH.read_text())
    raw["fixture"]["token_ids"][1] = 999
    bad = tmp_path / "bad-token.json"
    bad.write_text(json.dumps(raw))

    with pytest.raises(SpecError, match="out-of-range"):
        load_spec(bad)


def test_wrong_expected_value_produces_scientific_fail():
    spec, digest = _loaded()
    broken = json.loads(json.dumps(spec))
    broken["expected"]["constructed_values"][0][0][0] += 0.25

    result = run_val001(broken, digest)

    assert result["manifest"]["status"] == "FAIL"
    assert not result["checks"]["known_answer_constructed_values"]["passed"]
