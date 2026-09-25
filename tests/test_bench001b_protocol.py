from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from memory_attention_lab.experiments.bench001b_validate import (
    BundleError,
    validate_bundle,
)


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "specs" / "BENCH-001B.protocol.json"
RESULTS = ROOT / "tests" / "fixtures" / "BENCH-001B" / "synthetic" / "bmk_results.json"


def _hardware(results_path: Path, *, synthetic: bool = True) -> dict:
    digest = hashlib.sha256(results_path.read_bytes()).hexdigest()
    return {
        "schema_version": 1,
        "experiment_id": "BENCH-001B",
        "execution_id": "synthetic-protocol-fixture",
        "synthetic": synthetic,
        "source": {
            "upstream_repository": "Joluck/memory-attention",
            "upstream_commit": "8176f1feaff6724af670e70d75cd9767f8e38223",
            "upstream_benchmark_blob": "9ce6143fe7df1422b2f617996395d238edd2cc1e",
        },
        "hardware": {
            "gpu_name": "SYNTHETIC NVIDIA A100",
            "compute_capability": [8, 0],
            "total_vram_bytes": 42949672960,
            "bf16_supported": True,
        },
        "runtime": {
            "python": "3.12.0",
            "torch": "2.8.0",
            "cuda_runtime": "12.8",
            "driver": "synthetic",
            "flash_attn": "2.8.0",
            "os": "synthetic-linux",
        },
        "execution": {
            "profile": "source_default_latency",
            "command_argv": [
                "python",
                "profile/bmk.py",
                "--mode",
                "both",
                "--variants",
                "standard",
                "ma_gpu",
                "ma_offload",
                "--check-correctness",
                "--json",
                "bmk_results.json",
            ],
            "bmk_results_sha256": digest,
        },
    }


def _write_hardware(tmp_path: Path, payload: dict) -> Path:
    path = tmp_path / "hardware.json"
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return path


def _copy_results(tmp_path: Path) -> Path:
    path = tmp_path / "bmk_results.json"
    path.write_bytes(RESULTS.read_bytes())
    return path


def test_synthetic_protocol_fixture_passes_only_in_test_mode(tmp_path: Path):
    results = _copy_results(tmp_path)
    hardware = _write_hardware(tmp_path, _hardware(results))

    validated = validate_bundle(
        PROTOCOL,
        hardware,
        results,
        allow_synthetic=True,
    )

    assert validated["status"] == "PASS"
    assert validated["performance_authority"] is False
    assert validated["synthetic"] is True


def test_synthetic_bundle_is_rejected_by_default(tmp_path: Path):
    results = _copy_results(tmp_path)
    hardware = _write_hardware(tmp_path, _hardware(results))

    with pytest.raises(BundleError, match="synthetic"):
        validate_bundle(PROTOCOL, hardware, results)


def test_t4_compute_capability_is_rejected(tmp_path: Path):
    results = _copy_results(tmp_path)
    payload = _hardware(results)
    payload["hardware"]["compute_capability"] = [7, 5]
    hardware = _write_hardware(tmp_path, payload)

    with pytest.raises(BundleError, match="compute capability"):
        validate_bundle(PROTOCOL, hardware, results, allow_synthetic=True)


def test_bf16_false_is_rejected(tmp_path: Path):
    results = _copy_results(tmp_path)
    payload = _hardware(results)
    payload["hardware"]["bf16_supported"] = False
    hardware = _write_hardware(tmp_path, payload)

    with pytest.raises(BundleError, match="bf16_supported"):
        validate_bundle(PROTOCOL, hardware, results, allow_synthetic=True)


def test_old_flash_attention_is_rejected(tmp_path: Path):
    results = _copy_results(tmp_path)
    payload = _hardware(results)
    payload["runtime"]["flash_attn"] = "2.0.9"
    hardware = _write_hardware(tmp_path, payload)

    with pytest.raises(BundleError, match="flash-attn"):
        validate_bundle(PROTOCOL, hardware, results, allow_synthetic=True)


def test_result_hash_mismatch_is_rejected(tmp_path: Path):
    results = _copy_results(tmp_path)
    payload = _hardware(results)
    payload["execution"]["bmk_results_sha256"] = "0" * 64
    hardware = _write_hardware(tmp_path, payload)

    with pytest.raises(BundleError, match="SHA-256"):
        validate_bundle(PROTOCOL, hardware, results, allow_synthetic=True)


def test_tampered_median_is_rejected(tmp_path: Path):
    payload = json.loads(RESULTS.read_text())
    payload["results"][0]["median_ms"] += 7.0
    results = tmp_path / "bmk_results.json"
    results.write_text(json.dumps(payload, indent=2) + "\n")
    hardware = _write_hardware(tmp_path, _hardware(results))

    with pytest.raises(BundleError, match="median_ms"):
        validate_bundle(PROTOCOL, hardware, results, allow_synthetic=True)


def test_missing_result_pair_is_rejected(tmp_path: Path):
    payload = json.loads(RESULTS.read_text())
    payload["results"].pop()
    results = tmp_path / "bmk_results.json"
    results.write_text(json.dumps(payload, indent=2) + "\n")
    hardware = _write_hardware(tmp_path, _hardware(results))

    with pytest.raises(BundleError, match="result rows"):
        validate_bundle(PROTOCOL, hardware, results, allow_synthetic=True)


def test_wrong_speedup_is_rejected(tmp_path: Path):
    payload = json.loads(RESULTS.read_text())
    payload["results"][1]["speedup_vs_standard"] = 99.0
    results = tmp_path / "bmk_results.json"
    results.write_text(json.dumps(payload, indent=2) + "\n")
    hardware = _write_hardware(tmp_path, _hardware(results))

    with pytest.raises(BundleError, match="speedup_vs_standard"):
        validate_bundle(PROTOCOL, hardware, results, allow_synthetic=True)


def test_changed_source_default_config_is_rejected(tmp_path: Path):
    payload = json.loads(RESULTS.read_text())
    payload["config"]["batch_size"] = 7
    results = tmp_path / "bmk_results.json"
    results.write_text(json.dumps(payload, indent=2) + "\n")
    hardware = _write_hardware(tmp_path, _hardware(results))

    with pytest.raises(BundleError, match="batch_size"):
        validate_bundle(PROTOCOL, hardware, results, allow_synthetic=True)


def test_wrong_source_pin_is_rejected(tmp_path: Path):
    results = _copy_results(tmp_path)
    payload = _hardware(results)
    payload["source"]["upstream_commit"] = "deadbeef"
    hardware = _write_hardware(tmp_path, payload)

    with pytest.raises(BundleError, match="source pin"):
        validate_bundle(PROTOCOL, hardware, results, allow_synthetic=True)


def test_non_synthetic_structurally_valid_bundle_can_pass_protocol(tmp_path: Path):
    results = _copy_results(tmp_path)
    hardware = _write_hardware(
        tmp_path,
        _hardware(results, synthetic=False),
    )

    validated = validate_bundle(PROTOCOL, hardware, results)

    assert validated["status"] == "PASS"
    assert validated["synthetic"] is False
    assert validated["performance_authority"] is False
