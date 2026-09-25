"""CPU-side validator for BENCH-001B CUDA evidence bundles.

The validator establishes protocol consistency only. Passing this validator is
not itself CUDA performance evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any


class BundleError(ValueError):
    """Raised when a BENCH-001B evidence bundle violates the frozen protocol."""


HARDWARE_TOP_KEYS = {
    "schema_version",
    "experiment_id",
    "execution_id",
    "synthetic",
    "source",
    "hardware",
    "runtime",
    "execution",
}
HARDWARE_SOURCE_KEYS = {
    "upstream_repository",
    "upstream_commit",
    "upstream_benchmark_blob",
}
HARDWARE_KEYS = {
    "gpu_name",
    "compute_capability",
    "total_vram_bytes",
    "bf16_supported",
}
RUNTIME_KEYS = {
    "python",
    "torch",
    "cuda_runtime",
    "driver",
    "flash_attn",
    "os",
}
EXECUTION_KEYS = {
    "profile",
    "command_argv",
    "bmk_results_sha256",
}

RESULT_ROW_KEYS = {
    "variant",
    "mode",
    "median_ms",
    "min_ms",
    "max_ms",
    "round_ms",
    "seq_len",
    "context_len",
    "group_size",
    "benchmark_scope",
    "offload_policy",
    "prefetch_slots",
    "offload_gpu_buffer_mib",
    "offload_pinned_mib",
    "outer_gpu_parameters",
    "gpu_parameters",
    "cpu_parameters",
    "total_parameters",
    "gpu_parameter_mib",
    "cpu_parameter_mib",
    "speedup_vs_standard",
}


def _load_json(path: Path) -> tuple[dict[str, Any], bytes]:
    raw = path.read_bytes()
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise BundleError(f"{path}: invalid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise BundleError(f"{path}: top-level JSON must be an object")
    return value, raw


def _exact_keys(name: str, obj: dict[str, Any], expected: set[str]) -> None:
    actual = set(obj)
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing or extra:
        raise BundleError(
            f"{name} keys invalid; missing={missing or []}, extra={extra or []}"
        )


def _nonempty_string(name: str, value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise BundleError(f"{name} must be a non-empty string")
    return value


def _positive_int(name: str, value: Any) -> int:
    if type(value) is not int or value <= 0:
        raise BundleError(f"{name} must be a positive integer")
    return value


def _nonnegative_int(name: str, value: Any) -> int:
    if type(value) is not int or value < 0:
        raise BundleError(f"{name} must be a non-negative integer")
    return value


def _finite_number(name: str, value: Any, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise BundleError(f"{name} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise BundleError(f"{name} must be finite")
    if positive and number <= 0:
        raise BundleError(f"{name} must be > 0")
    return number


def _version_pair(name: str, value: str) -> tuple[int, int]:
    _nonempty_string(name, value)
    parts = value.split(".")
    if len(parts) < 2:
        raise BundleError(f"{name} must contain at least major.minor")
    try:
        return int(parts[0]), int(parts[1])
    except ValueError as exc:
        raise BundleError(f"{name} must begin with numeric major.minor") from exc


def _isclose(a: float, b: float) -> bool:
    return math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12)


def _validate_hardware(
    protocol: dict[str, Any],
    manifest: dict[str, Any],
    result_raw: bytes,
    *,
    allow_synthetic: bool,
) -> None:
    _exact_keys("hardware manifest", manifest, HARDWARE_TOP_KEYS)
    if manifest["schema_version"] != 1:
        raise BundleError("hardware manifest schema_version must be 1")
    if manifest["experiment_id"] != "BENCH-001B":
        raise BundleError("hardware manifest experiment_id must be BENCH-001B")
    _nonempty_string("execution_id", manifest["execution_id"])

    if type(manifest["synthetic"]) is not bool:
        raise BundleError("synthetic must be boolean")
    if manifest["synthetic"] and not allow_synthetic:
        raise BundleError("synthetic bundles cannot enter BENCH-001B evidence")

    source = manifest["source"]
    if not isinstance(source, dict):
        raise BundleError("source must be an object")
    _exact_keys("source", source, HARDWARE_SOURCE_KEYS)
    if source != protocol["source"]:
        raise BundleError("hardware manifest source pin differs from protocol source")

    hw = manifest["hardware"]
    if not isinstance(hw, dict):
        raise BundleError("hardware must be an object")
    _exact_keys("hardware", hw, HARDWARE_KEYS)
    _nonempty_string("hardware.gpu_name", hw["gpu_name"])
    cc = hw["compute_capability"]
    if (
        not isinstance(cc, list)
        or len(cc) != 2
        or any(type(x) is not int or x < 0 for x in cc)
    ):
        raise BundleError("compute_capability must be [major, minor] integers")
    floor = protocol["hardware_floor"]["minimum_compute_capability"]
    if tuple(cc) < tuple(floor):
        raise BundleError(
            f"compute capability {cc} is below protocol floor {floor}"
        )
    _positive_int("hardware.total_vram_bytes", hw["total_vram_bytes"])
    if hw["bf16_supported"] is not True:
        raise BundleError("hardware manifest must report bf16_supported=true")

    runtime = manifest["runtime"]
    if not isinstance(runtime, dict):
        raise BundleError("runtime must be an object")
    _exact_keys("runtime", runtime, RUNTIME_KEYS)
    for key in ("python", "torch", "cuda_runtime", "driver", "flash_attn", "os"):
        _nonempty_string(f"runtime.{key}", runtime[key])
    flash = _version_pair("runtime.flash_attn", runtime["flash_attn"])
    if flash < tuple(protocol["hardware_floor"]["minimum_flash_attn_version"]):
        raise BundleError("flash-attn version is below the frozen protocol floor")

    execution = manifest["execution"]
    if not isinstance(execution, dict):
        raise BundleError("execution must be an object")
    _exact_keys("execution", execution, EXECUTION_KEYS)
    profile = _nonempty_string("execution.profile", execution["profile"])
    if profile not in protocol["profiles"]:
        raise BundleError(f"unknown BENCH-001B profile: {profile}")
    argv = execution["command_argv"]
    if not isinstance(argv, list) or not argv or any(
        not isinstance(x, str) or not x for x in argv
    ):
        raise BundleError("execution.command_argv must be a non-empty string list")
    joined = " ".join(argv)
    for required in ("profile/bmk.py", "--check-correctness", "--json"):
        if required not in joined:
            raise BundleError(f"command_argv is missing required token: {required}")

    expected_hash = hashlib.sha256(result_raw).hexdigest()
    if execution["bmk_results_sha256"] != expected_hash:
        raise BundleError("bmk_results.json SHA-256 does not match hardware manifest")


def _validate_config(protocol: dict[str, Any], config: dict[str, Any], profile: str) -> None:
    profile_spec = protocol["profiles"][profile]
    required = profile_spec["required_config"]
    expected_keys = set(required) | {"json"}
    _exact_keys("benchmark config", config, expected_keys)

    for key, expected in required.items():
        if config[key] != expected:
            raise BundleError(
                f"benchmark config {key!r} mismatch: "
                f"expected {expected!r}, got {config[key]!r}"
            )
    _nonempty_string("benchmark config json", config["json"])


def _validate_result_rows(
    protocol: dict[str, Any],
    config: dict[str, Any],
    results: Any,
    profile: str,
) -> None:
    if not isinstance(results, list):
        raise BundleError("results must be a list")

    expected_pairs = {
        tuple(x) for x in protocol["profiles"][profile]["required_result_pairs"]
    }
    if len(results) != len(expected_pairs):
        raise BundleError(
            f"expected {len(expected_pairs)} result rows, got {len(results)}"
        )

    rows: dict[tuple[str, str], dict[str, Any]] = {}
    for index, row in enumerate(results):
        if not isinstance(row, dict):
            raise BundleError(f"results[{index}] must be an object")
        _exact_keys(f"results[{index}]", row, RESULT_ROW_KEYS)

        pair = (row["variant"], row["mode"])
        if pair not in expected_pairs:
            raise BundleError(f"unexpected result pair: {pair}")
        if pair in rows:
            raise BundleError(f"duplicate result pair: {pair}")
        rows[pair] = row

        rounds = row["round_ms"]
        if not isinstance(rounds, list) or len(rounds) != config["rounds"]:
            raise BundleError(
                f"{pair}: round_ms length must equal config rounds={config['rounds']}"
            )
        samples = [
            _finite_number(f"{pair}.round_ms[{i}]", x, positive=True)
            for i, x in enumerate(rounds)
        ]
        median = _finite_number(f"{pair}.median_ms", row["median_ms"], positive=True)
        minimum = _finite_number(f"{pair}.min_ms", row["min_ms"], positive=True)
        maximum = _finite_number(f"{pair}.max_ms", row["max_ms"], positive=True)
        if not _isclose(median, statistics.median(samples)):
            raise BundleError(f"{pair}: median_ms does not match round_ms")
        if not _isclose(minimum, min(samples)):
            raise BundleError(f"{pair}: min_ms does not match round_ms")
        if not _isclose(maximum, max(samples)):
            raise BundleError(f"{pair}: max_ms does not match round_ms")

        expected_seq = config["seq_len"] if row["mode"] == "prefill" else 1
        expected_context = 0 if row["mode"] == "prefill" else config["context_len"]
        if row["seq_len"] != expected_seq:
            raise BundleError(f"{pair}: seq_len mismatch")
        if row["context_len"] != expected_context:
            raise BundleError(f"{pair}: context_len mismatch")
        if row["benchmark_scope"] != "model":
            raise BundleError(f"{pair}: benchmark_scope must be model")

        effective_group = min(config["group_size"], config["num_layers"])
        if row["variant"] == "ma_offload" and row["mode"] == "decode":
            expected_group = config["num_layers"]
            expected_policy = "bulk"
            expected_slots = 1
        elif row["variant"] == "ma_offload":
            expected_group = effective_group
            expected_policy = "pipeline"
            groups = (
                config["num_layers"] + effective_group - 1
            ) // effective_group
            expected_slots = min(config["prefetch_depth"], groups)
        else:
            expected_group = effective_group
            expected_policy = None
            expected_slots = 0

        if row["group_size"] != expected_group:
            raise BundleError(f"{pair}: group_size mismatch")
        if row["offload_policy"] != expected_policy:
            raise BundleError(f"{pair}: offload_policy mismatch")
        if row["prefetch_slots"] != expected_slots:
            raise BundleError(f"{pair}: prefetch_slots mismatch")

        for name in (
            "outer_gpu_parameters",
            "gpu_parameters",
            "cpu_parameters",
            "total_parameters",
        ):
            _nonnegative_int(f"{pair}.{name}", row[name])
        if row["total_parameters"] != row["gpu_parameters"] + row["cpu_parameters"]:
            raise BundleError(f"{pair}: total_parameters placement identity failed")
        if row["outer_gpu_parameters"] > row["gpu_parameters"]:
            raise BundleError(f"{pair}: outer_gpu_parameters exceeds GPU total")

        for name in (
            "gpu_parameter_mib",
            "cpu_parameter_mib",
            "offload_gpu_buffer_mib",
            "offload_pinned_mib",
        ):
            value = _finite_number(f"{pair}.{name}", row[name])
            if value < 0:
                raise BundleError(f"{pair}.{name} must be non-negative")

        if row["variant"] != "ma_offload":
            if row["cpu_parameters"] != 0 or row["cpu_parameter_mib"] != 0:
                raise BundleError(f"{pair}: non-offload variant has CPU model weights")
            if row["offload_gpu_buffer_mib"] != 0 or row["offload_pinned_mib"] != 0:
                raise BundleError(f"{pair}: non-offload variant has offload buffers")
        else:
            if row["cpu_parameters"] <= 0 or row["cpu_parameter_mib"] <= 0:
                raise BundleError(f"{pair}: ma_offload must report CPU table weights")
            if row["offload_gpu_buffer_mib"] <= 0 or row["offload_pinned_mib"] <= 0:
                raise BundleError(f"{pair}: ma_offload must report staging buffers")

    if set(rows) != expected_pairs:
        raise BundleError("result pair set does not match protocol")

    placement_fields = (
        "outer_gpu_parameters",
        "gpu_parameters",
        "cpu_parameters",
        "total_parameters",
        "gpu_parameter_mib",
        "cpu_parameter_mib",
    )
    for variant in ("standard", "ma_gpu", "ma_offload"):
        pre = rows[(variant, "prefill")]
        dec = rows[(variant, "decode")]
        for field in placement_fields:
            if not _isclose(float(pre[field]), float(dec[field])):
                raise BundleError(
                    f"{variant}: parameter placement differs between prefill/decode"
                )

    for mode in ("prefill", "decode"):
        standard = rows[("standard", mode)]
        std_median = float(standard["median_ms"])
        for variant in ("standard", "ma_gpu", "ma_offload"):
            row = rows[(variant, mode)]
            expected_speedup = std_median / float(row["median_ms"])
            actual = _finite_number(
                f"{variant}/{mode}.speedup_vs_standard",
                row["speedup_vs_standard"],
                positive=True,
            )
            if not _isclose(actual, expected_speedup):
                raise BundleError(
                    f"{variant}/{mode}: speedup_vs_standard is inconsistent"
                )

    for mode in ("prefill", "decode"):
        gpu = rows[("ma_gpu", mode)]
        off = rows[("ma_offload", mode)]
        if gpu["total_parameters"] != off["total_parameters"]:
            raise BundleError(
                f"{mode}: ma_gpu and ma_offload total parameter counts differ"
            )
        if gpu["gpu_parameters"] != off["gpu_parameters"] + off["cpu_parameters"]:
            raise BundleError(
                f"{mode}: MA table placement identity does not hold"
            )


def validate_bundle(
    protocol_path: Path,
    hardware_path: Path,
    results_path: Path,
    *,
    allow_synthetic: bool = False,
) -> dict[str, Any]:
    protocol, protocol_raw = _load_json(protocol_path)
    hardware, hardware_raw = _load_json(hardware_path)
    payload, result_raw = _load_json(results_path)

    if protocol.get("schema_version") != 1:
        raise BundleError("protocol schema_version must be 1")
    if protocol.get("experiment_id") != "BENCH-001B":
        raise BundleError("protocol experiment_id must be BENCH-001B")

    _validate_hardware(
        protocol,
        hardware,
        result_raw,
        allow_synthetic=allow_synthetic,
    )

    _exact_keys("bmk_results", payload, {"config", "results"})
    if not isinstance(payload["config"], dict):
        raise BundleError("bmk_results.config must be an object")

    profile = hardware["execution"]["profile"]
    _validate_config(protocol, payload["config"], profile)
    _validate_result_rows(
        protocol,
        payload["config"],
        payload["results"],
        profile,
    )

    return {
        "schema_version": 1,
        "experiment_id": "BENCH-001B-B0",
        "status": "PASS",
        "performance_authority": False,
        "synthetic": hardware["synthetic"],
        "execution_id": hardware["execution_id"],
        "protocol_sha256": hashlib.sha256(protocol_raw).hexdigest(),
        "hardware_sha256": hashlib.sha256(hardware_raw).hexdigest(),
        "results_sha256": hashlib.sha256(result_raw).hexdigest(),
        "profile": profile,
        "gpu_name": hardware["hardware"]["gpu_name"],
        "compute_capability": hardware["hardware"]["compute_capability"],
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--protocol", type=Path, required=True)
    p.add_argument("--hardware", type=Path, required=True)
    p.add_argument("--results", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    try:
        result = validate_bundle(
            args.protocol,
            args.hardware,
            args.results,
            allow_synthetic=False,
        )
    except (OSError, BundleError) as exc:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "experiment_id": "BENCH-001B-B0",
                    "status": "INVALID",
                    "performance_authority": False,
                    "error": str(exc),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        print(f"BENCH-001B-B0 INVALID: {exc}")
        return 2

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print("BENCH-001B-B0 PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
