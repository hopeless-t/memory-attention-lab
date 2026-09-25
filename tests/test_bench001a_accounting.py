from __future__ import annotations

import json
from pathlib import Path

import pytest

from memory_attention_lab.experiments.bench001a import SpecError, load_spec, run_bench001a
from memory_attention_lab.measurement.residency import (
    AccountingConfigError,
    account_residency,
)


ROOT = Path(__file__).resolve().parents[1]
SPEC_PATH = ROOT / "specs" / "BENCH-001A.reference.json"
DESIGN_PATH = ROOT / "specs" / "BENCH-001.json"


def _loaded():
    return load_spec(SPEC_PATH, DESIGN_PATH)


def test_frozen_reference_cases_pass():
    spec, digest, parent_blob = _loaded()
    result = run_bench001a(spec, digest, parent_blob)
    assert result["manifest"]["status"] == "PASS"
    assert all(check["passed"] for check in result["checks"].values())


def test_group_size_is_clamped_to_layers():
    spec, _, _ = _loaded()
    case = next(x for x in spec["cases"] if x["id"] == "group_clamp_guard")
    result = account_residency(case["config"])
    assert result["derived"]["effective_group"] == case["config"]["num_layers"]
    assert result["derived"]["num_groups"] == 1
    assert result["derived"]["pipeline_slots"] == 1
    assert result["staging"]["pipeline"] == result["staging"]["bulk"]


def test_transfer_payload_is_not_pipeline_working_set():
    spec, _, _ = _loaded()
    case = next(x for x in spec["cases"] if x["id"] == "gqa_mixed_width")
    result = account_residency(case["config"])
    assert result["staging"]["pipeline"]["allocated_elements"] == 1008
    assert result["transfer"]["h2d_payload_elements_per_forward"] == 840
    assert result["staging"]["pipeline"]["allocated_elements"] > result["transfer"][
        "h2d_payload_elements_per_forward"
    ]


def test_parameter_counts_and_bytes_are_distinct():
    spec, _, _ = _loaded()
    case = next(x for x in spec["cases"] if x["id"] == "gqa_mixed_width")
    result = account_residency(case["config"])
    delta = result["placement"]["deltas_vs_standard"]
    assert delta["ma_gpu_gpu_parameters"] > 0
    assert delta["ma_gpu_gpu_parameter_bytes"] < 0


def test_qkv_bias_adds_kv_width_per_layer():
    base = {
        "hidden_size": 32,
        "num_heads": 8,
        "num_kv_heads": 2,
        "num_layers": 5,
        "vocab_size": 64,
        "batch_size": 1,
        "seq_len": 1,
        "group_size": 1,
        "prefetch_depth": 1,
        "qkv_bias": False,
        "weight_element_bytes": 2,
        "memory_element_bytes": 2,
        "kv_cache_element_bytes": 2,
        "kv_cache_capacity": 0,
    }
    without_bias = account_residency(base)
    with_bias = account_residency({**base, "qkv_bias": True})
    kv_width = without_bias["derived"]["kv_width"]
    assert (
        with_bias["parameters"]["standard_wv_params_per_layer"]
        - without_bias["parameters"]["standard_wv_params_per_layer"]
        == kv_width
    )


def test_kv_cache_capacity_zero_is_valid():
    config = {
        "hidden_size": 16,
        "num_heads": 4,
        "num_kv_heads": 4,
        "num_layers": 2,
        "vocab_size": 32,
        "batch_size": 1,
        "seq_len": 1,
        "group_size": 1,
        "prefetch_depth": 1,
        "qkv_bias": False,
        "weight_element_bytes": 2,
        "memory_element_bytes": 2,
        "kv_cache_element_bytes": 2,
        "kv_cache_capacity": 0,
    }
    result = account_residency(config)
    assert result["kv_cache"] == {"elements": 0, "bytes": 0}


def test_invalid_head_divisibility_fails_closed():
    config = {
        "hidden_size": 15,
        "num_heads": 4,
        "num_kv_heads": 2,
        "num_layers": 2,
        "vocab_size": 32,
        "batch_size": 1,
        "seq_len": 1,
        "group_size": 1,
        "prefetch_depth": 1,
        "qkv_bias": False,
        "weight_element_bytes": 2,
        "memory_element_bytes": 2,
        "kv_cache_element_bytes": 2,
        "kv_cache_capacity": 0,
    }
    with pytest.raises(AccountingConfigError, match="hidden_size"):
        account_residency(config)


def test_unknown_config_field_fails_closed():
    spec, _, _ = _loaded()
    config = dict(spec["cases"][0]["config"])
    config["silent_default"] = 1
    with pytest.raises(AccountingConfigError, match="extra"):
        account_residency(config)


def test_tampered_parent_design_is_invalid(tmp_path: Path):
    raw = DESIGN_PATH.read_text() + "\n"
    tampered = tmp_path / "BENCH-001.json"
    tampered.write_text(raw)
    with pytest.raises(SpecError, match="blob SHA-1"):
        load_spec(SPEC_PATH, tampered)


def test_wrong_expected_value_produces_scientific_fail():
    spec, digest, parent_blob = _loaded()
    broken = json.loads(json.dumps(spec))
    broken["cases"][0]["expected"]["derived"]["kv_width"] += 1
    result = run_bench001a(broken, digest, parent_blob)
    assert result["manifest"]["status"] == "FAIL"
    assert not result["checks"]["small_known_answer"]["passed"]


def test_python_integer_accounting_does_not_overflow():
    config = {
        "hidden_size": 131072,
        "num_heads": 128,
        "num_kv_heads": 128,
        "num_layers": 4096,
        "vocab_size": 10000000,
        "batch_size": 1024,
        "seq_len": 1048576,
        "group_size": 4096,
        "prefetch_depth": 4096,
        "qkv_bias": True,
        "weight_element_bytes": 8,
        "memory_element_bytes": 8,
        "kv_cache_element_bytes": 8,
        "kv_cache_capacity": 1048576,
    }
    result = account_residency(config)
    assert result["placement"]["ma_gpu_value_side"]["gpu_parameter_bytes"] > 2**63
    assert result["kv_cache"]["bytes"] > 2**63
