"""Executable contract for BENCH-001A deterministic accounting."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
from pathlib import Path
from typing import Any

from memory_attention_lab.measurement.residency import (
    AccountingConfigError,
    CONFIG_KEYS,
    account_residency,
    validate_config,
)


class SpecError(ValueError):
    """Raised when the BENCH-001A reference spec is invalid."""


TOP_KEYS = {
    "schema_version",
    "experiment_id",
    "description",
    "parent_design",
    "source",
    "authority",
    "cases",
}
PARENT_KEYS = {"path", "git_blob_sha1"}
SOURCE_KEYS = {
    "upstream_repository",
    "upstream_commit",
    "upstream_benchmark_blob",
}
AUTHORITY_KEYS = {"lane", "cuda_required", "latency_authority"}
CASE_KEYS = {"id", "config", "expected"}


def _exact_keys(name: str, obj: dict[str, Any], expected: set[str]) -> None:
    actual = set(obj)
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing or extra:
        raise SpecError(
            f"{name} keys invalid; missing={missing or []}, extra={extra or []}"
        )


def git_blob_sha1(raw: bytes) -> str:
    prefix = f"blob {len(raw)}\0".encode()
    return hashlib.sha1(prefix + raw).hexdigest()


def load_spec(
    spec_path: Path,
    parent_design_path: Path,
) -> tuple[dict[str, Any], str, str]:
    raw = spec_path.read_bytes()
    spec_sha256 = hashlib.sha256(raw).hexdigest()
    try:
        spec = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise SpecError(f"invalid JSON: {exc}") from exc

    if not isinstance(spec, dict):
        raise SpecError("top-level spec must be an object")
    _exact_keys("top-level", spec, TOP_KEYS)

    for name, keys in (
        ("parent_design", PARENT_KEYS),
        ("source", SOURCE_KEYS),
        ("authority", AUTHORITY_KEYS),
    ):
        if not isinstance(spec[name], dict):
            raise SpecError(f"{name} must be an object")
        _exact_keys(name, spec[name], keys)

    if spec["schema_version"] != 1:
        raise SpecError("unsupported schema_version")
    if spec["experiment_id"] != "BENCH-001A":
        raise SpecError("experiment_id must be BENCH-001A")
    if not isinstance(spec["description"], str) or not spec["description"].strip():
        raise SpecError("description must be a non-empty string")

    if spec["parent_design"]["path"] != "specs/BENCH-001.json":
        raise SpecError("parent design path must be specs/BENCH-001.json")

    parent_raw = parent_design_path.read_bytes()
    parent_blob = git_blob_sha1(parent_raw)
    if parent_blob != spec["parent_design"]["git_blob_sha1"]:
        raise SpecError(
            "parent BENCH-001 design Git blob SHA-1 does not match the frozen reference"
        )

    authority = spec["authority"]
    if authority != {
        "lane": "deterministic_accounting",
        "cuda_required": False,
        "latency_authority": False,
    }:
        raise SpecError("authority block must remain deterministic accounting only")

    cases = spec["cases"]
    if not isinstance(cases, list) or not cases:
        raise SpecError("cases must be a non-empty list")

    seen = set()
    for index, case in enumerate(cases):
        if not isinstance(case, dict):
            raise SpecError(f"cases[{index}] must be an object")
        _exact_keys(f"cases[{index}]", case, CASE_KEYS)
        case_id = case["id"]
        if not isinstance(case_id, str) or not case_id.strip():
            raise SpecError(f"cases[{index}].id must be a non-empty string")
        if case_id in seen:
            raise SpecError(f"duplicate case id: {case_id}")
        seen.add(case_id)
        if not isinstance(case["config"], dict):
            raise SpecError(f"{case_id}.config must be an object")
        if set(case["config"]) != CONFIG_KEYS:
            missing = sorted(CONFIG_KEYS - set(case["config"]))
            extra = sorted(set(case["config"]) - CONFIG_KEYS)
            raise SpecError(
                f"{case_id}.config keys invalid; missing={missing}, extra={extra}"
            )
        try:
            validate_config(case["config"])
        except AccountingConfigError as exc:
            raise SpecError(f"{case_id}.config invalid: {exc}") from exc
        if not isinstance(case["expected"], dict):
            raise SpecError(f"{case_id}.expected must be an object")

    return spec, spec_sha256, parent_blob


def run_bench001a(
    spec: dict[str, Any],
    spec_sha256: str,
    parent_blob: str,
) -> dict[str, Any]:
    checks: dict[str, dict[str, Any]] = {}
    results: dict[str, dict[str, Any]] = {}

    for case in spec["cases"]:
        case_id = case["id"]
        actual = account_residency(case["config"])
        expected = case["expected"]
        passed = actual == expected
        checks[case_id] = {"passed": passed}
        results[case_id] = actual

    status = "PASS" if all(x["passed"] for x in checks.values()) else "FAIL"

    source_default = results.get("source_defaults_prefill")
    metrics = {
        "case_count": len(results),
        "passed_case_count": sum(1 for x in checks.values() if x["passed"]),
    }
    if source_default is not None:
        metrics["source_defaults"] = {
            "standard_wv_bytes": source_default["placement"][
                "standard_value_side"
            ]["gpu_parameter_bytes"],
            "ma_table_bytes": source_default["placement"][
                "ma_offload_value_side"
            ]["cpu_parameter_bytes"],
            "ma_offload_gpu_parameter_byte_delta_vs_standard": source_default[
                "placement"
            ]["deltas_vs_standard"]["ma_offload_gpu_parameter_bytes"],
            "pipeline_gpu_staging_bytes": source_default["staging"]["pipeline"][
                "gpu_bytes"
            ],
            "bulk_gpu_staging_bytes": source_default["staging"]["bulk"][
                "gpu_bytes"
            ],
            "h2d_payload_bytes_per_forward": source_default["transfer"][
                "h2d_payload_bytes_per_forward"
            ],
            "kv_cache_bytes": source_default["kv_cache"]["bytes"],
        }

    return {
        "manifest": {
            "schema_version": 1,
            "experiment_id": "BENCH-001A",
            "status": status,
            "spec_sha256": spec_sha256,
            "parent_design_git_blob_sha1": parent_blob,
            "source": spec["source"],
            "authority": spec["authority"],
            "provenance": {
                "python": sys.version.split()[0],
                "platform": platform.platform(),
                "git_commit": os.environ.get("GITHUB_SHA"),
            },
        },
        "metrics": metrics,
        "checks": checks,
        "observations": results,
    }


def write_evidence(out_dir: Path, result: dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in ("manifest", "metrics", "checks", "observations"):
        (out_dir / f"{name}.json").write_text(
            json.dumps(result[name], indent=2, sort_keys=True) + "\n"
        )


def _terminal_manifest(
    out_dir: Path,
    *,
    status: str,
    spec_path: Path,
    parent_path: Path,
    error: str,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "experiment_id": "BENCH-001A",
        "status": status,
        "spec_path": str(spec_path),
        "parent_design_path": str(parent_path),
        "error": error,
        "provenance": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "git_commit": os.environ.get("GITHUB_SHA"),
        },
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("spec", type=Path)
    parser.add_argument("--parent-design", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    try:
        spec, digest, parent_blob = load_spec(args.spec, args.parent_design)
    except (OSError, SpecError) as exc:
        _terminal_manifest(
            args.out,
            status="INVALID",
            spec_path=args.spec,
            parent_path=args.parent_design,
            error=str(exc),
        )
        print(f"BENCH-001A INVALID: {exc}")
        return 2

    try:
        result = run_bench001a(spec, digest, parent_blob)
        write_evidence(args.out, result)
    except Exception as exc:
        _terminal_manifest(
            args.out,
            status="ERROR",
            spec_path=args.spec,
            parent_path=args.parent_design,
            error=f"{type(exc).__name__}: {exc}",
        )
        print(f"BENCH-001A ERROR: {type(exc).__name__}: {exc}")
        return 3

    status = result["manifest"]["status"]
    print(f"BENCH-001A {status}")
    for name, check in result["checks"].items():
        print(f"  {name}: {'PASS' if check['passed'] else 'FAIL'}")
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
