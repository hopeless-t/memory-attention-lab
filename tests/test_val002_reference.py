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


def _loaded():
    return load_spec(SPEC_PATH)


def _arrays(spec):
    f = spec["fixture"]
    p = spec["parameters"]
    return {
        "hidden": np.asarray(f["hidden_states"], dtype=np.float64),
        "wv": np.asarray(f["wv"], dtype=np.float64),
        "wk": np.asarray(f["wk"], dtype=np.float64),
        "memory_table": np.asarray(f["memory_table"], dtype=np.float64),
        "historical_table": np.asarray(
            f["historical_value_table"], dtype=np.float64
        ),
        "token_ids": np.asarray(f["token_ids"], dtype=np.int64),
        "norm_weight": np.asarray(f["norm_weight"], dtype=np.float64),
        "num_kv_heads": p["num_kv_heads"],
        "head_dim": p["head_dim"],
        "eps": p["eps"],
    }


def test_frozen_reference_passes_all_checks():
    spec, digest = _loaded()
    result = run_val002(spec, digest)

    assert result["manifest"]["status"] == "PASS"
    assert all(check["passed"] for check in result["checks"].values())


def test_factorial_memory_delta_is_identical_for_both_sources():
    spec, _ = _loaded()
    a = _arrays(spec)
    cells = factorial_value_cells(
        a["hidden"],
        a["wv"],
        a["wk"],
        a["memory_table"],
        a["token_ids"],
        a["norm_weight"],
        num_kv_heads=a["num_kv_heads"],
        head_dim=a["head_dim"],
        eps=a["eps"],
    )

    np.testing.assert_allclose(cells["C01"] - cells["C00"], cells["memory"])
    np.testing.assert_allclose(cells["C11"] - cells["C10"], cells["memory"])
    np.testing.assert_allclose(
        (cells["C11"] - cells["C10"]) - (cells["C01"] - cells["C00"]),
        0.0,
    )


def test_historical_lambda_endpoints_are_explicit():
    spec, _ = _loaded()
    a = _arrays(spec)
    cells = factorial_value_cells(
        a["hidden"],
        a["wv"],
        a["wk"],
        a["memory_table"],
        a["token_ids"],
        a["norm_weight"],
        num_kv_heads=a["num_kv_heads"],
        head_dim=a["head_dim"],
        eps=a["eps"],
    )

    hist0 = historical_value_embedding(
        a["hidden"],
        a["wv"],
        a["historical_table"],
        a["token_ids"],
        lamb=0.0,
    )
    hist1 = historical_value_embedding(
        a["hidden"],
        a["wv"],
        a["historical_table"],
        a["token_ids"],
        lamb=1.0,
    )

    np.testing.assert_allclose(hist0, cells["C00"])
    np.testing.assert_allclose(hist1, a["historical_table"][a["token_ids"]])


def test_historical_lane_is_not_the_clean_additive_control():
    spec, digest = _loaded()
    result = run_val002(spec, digest)

    check = result["checks"]["historical_lane_is_not_clean_additive_control"]
    assert check["passed"]
    assert check["max_abs_delta"] >= spec["acceptance"][
        "min_historical_vs_additive_delta"
    ]


def test_wv_perturbation_cannot_leak_into_key_reuse_cells():
    spec, digest = _loaded()
    result = run_val002(spec, digest)
    guard = result["checks"]["Wv_perturbation_guard"]

    assert guard["passed"]
    assert guard["C10_unchanged"]
    assert guard["C11_unchanged"]
    assert guard["C00_changed"]
    assert guard["HIST-VE_changed"]


def test_wk_perturbation_cannot_leak_into_wv_cells():
    spec, digest = _loaded()
    result = run_val002(spec, digest)
    guard = result["checks"]["Wk_perturbation_guard"]

    assert guard["passed"]
    assert guard["C00_unchanged"]
    assert guard["C01_unchanged"]
    assert guard["C10_changed"]
    assert guard["C11_changed"]


def test_post_rope_wrong_path_is_detected():
    spec, digest = _loaded()
    result = run_val002(spec, digest)
    guard = result["checks"]["pre_rope_value_guard"]

    assert guard["passed"]
    assert guard["wrong_post_rope_value_max_delta"] >= spec["acceptance"][
        "min_post_rope_guard_delta"
    ]


def test_unknown_field_is_invalid(tmp_path: Path):
    raw = json.loads(SPEC_PATH.read_text())
    raw["parameters"]["silent_default"] = 123
    bad = tmp_path / "bad-extra.json"
    bad.write_text(json.dumps(raw))

    with pytest.raises(SpecError, match="extra"):
        load_spec(bad)


def test_bad_projection_shape_is_invalid(tmp_path: Path):
    raw = json.loads(SPEC_PATH.read_text())
    raw["fixture"]["wv"][0] = [1.0, 2.0]
    bad = tmp_path / "bad-shape.json"
    bad.write_text(json.dumps(raw))

    with pytest.raises(SpecError, match="wv"):
        load_spec(bad)


def test_numeric_strings_do_not_silently_coerce(tmp_path: Path):
    raw = json.loads(SPEC_PATH.read_text())
    raw["parameters"]["historical_lambda"] = "0.5"
    bad = tmp_path / "bad-type.json"
    bad.write_text(json.dumps(raw))

    with pytest.raises(SpecError, match="JSON number"):
        load_spec(bad)


def test_wrong_expected_value_produces_scientific_fail():
    spec, digest = _loaded()
    broken = json.loads(json.dumps(spec))
    broken["expected"]["C11"][0][0] += 0.25

    result = run_val002(broken, digest)

    assert result["manifest"]["status"] == "FAIL"
    assert not result["checks"]["known_answer_C11"]["passed"]
