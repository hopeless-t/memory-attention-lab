"""Executable contract for VAL-002."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
from pathlib import Path
from typing import Any

import numpy as np

from memory_attention_lab.attention.value_sources import (
    factorial_effects,
    factorial_value_sources,
    historical_value_embedding,
)


class SpecError(ValueError):
    """Raised when VAL-002 input is invalid."""


TOP_KEYS = {
    "schema_version",
    "experiment_id",
    "description",
    "source",
    "parameters",
    "fixture",
    "expected",
    "acceptance",
}

NESTED_KEYS = {
    "source": {
        "memory_attention_paper",
        "parent_val001_spec_sha256",
        "historical_repository",
        "historical_ref",
        "historical_valueembed_blob",
        "historical_readme_blob",
    },
    "parameters": {"dtype", "historical_lambda"},
    "fixture": {
        "projected_values",
        "content_keys",
        "ma_memory_contribution",
        "historical_token_values",
    },
    "expected": {
        "historical_value_embedding",
        "c00_projected_no_memory",
        "c01_projected_plus_memory",
        "c10_key_no_memory",
        "c11_key_plus_memory",
    },
    "acceptance": {"atol", "rtol", "min_historical_additive_separation"},
}


def _exact_keys(name: str, obj: dict[str, Any], expected: set[str]) -> None:
    actual = set(obj)
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing or extra:
        raise SpecError(
            f"{name} keys invalid; missing={missing or []}, extra={extra or []}"
        )


def _number(name: str, value: Any, *, nonnegative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SpecError(f"{name} must be a JSON number")
    number = float(value)
    if not np.isfinite(number):
        raise SpecError(f"{name} must be finite")
    if nonnegative and number < 0:
        raise SpecError(f"{name} must be >= 0")
    return number


def _array(name: str, value: Any) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise SpecError(f"{name} must be a rectangular numeric array") from exc
    if array.ndim != 3:
        raise SpecError(f"{name} must have shape [T, H, D]")
    if not np.all(np.isfinite(array)):
        raise SpecError(f"{name} contains non-finite values")
    return array


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_spec(
    spec_path: Path,
    parent_val001_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], str, str]:
    raw = spec_path.read_bytes()
    spec_sha = hashlib.sha256(raw).hexdigest()
    try:
        spec = json.loads(raw)
        parent = json.loads(parent_val001_path.read_bytes())
    except json.JSONDecodeError as exc:
        raise SpecError(f"invalid JSON: {exc}") from exc
    except OSError:
        raise

    if not isinstance(spec, dict):
        raise SpecError("top-level spec must be an object")
    _exact_keys("top-level", spec, TOP_KEYS)
    for name, keys in NESTED_KEYS.items():
        if not isinstance(spec[name], dict):
            raise SpecError(f"{name} must be an object")
        _exact_keys(name, spec[name], keys)

    if spec["schema_version"] != 1:
        raise SpecError("unsupported schema_version")
    if spec["experiment_id"] != "VAL-002":
        raise SpecError("experiment_id must be VAL-002")
    if spec["parameters"]["dtype"] != "float64":
        raise SpecError("VAL-002 frozen reference dtype must be float64")
    if not isinstance(spec["description"], str) or not spec["description"].strip():
        raise SpecError("description must be a non-empty string")
    for key, value in spec["source"].items():
        if not isinstance(value, str) or not value.strip():
            raise SpecError(f"source.{key} must be a non-empty string")

    _number("historical_lambda", spec["parameters"]["historical_lambda"])
    for key in ("atol", "rtol", "min_historical_additive_separation"):
        _number(key, spec["acceptance"][key], nonnegative=True)

    arrays = {}
    for section, keys in (
        ("fixture", NESTED_KEYS["fixture"]),
        ("expected", NESTED_KEYS["expected"]),
    ):
        for key in keys:
            arrays[f"{section}.{key}"] = _array(
                f"{section}.{key}", spec[section][key]
            )
    shapes = {array.shape for array in arrays.values()}
    if len(shapes) != 1:
        raise SpecError(
            "all VAL-002 fixture and expected arrays must share one [T,H,D] shape"
        )

    if not isinstance(parent, dict) or parent.get("experiment_id") != "VAL-001":
        raise SpecError("parent spec must be VAL-001")

    parent_sha = _sha256(parent_val001_path)
    if parent_sha != spec["source"]["parent_val001_spec_sha256"]:
        raise SpecError(
            "parent VAL-001 spec SHA-256 does not match the frozen VAL-002 source"
        )

    return spec, parent, spec_sha, parent_sha


def _check_close(
    actual: np.ndarray,
    expected: np.ndarray,
    *,
    atol: float,
    rtol: float,
) -> tuple[bool, float]:
    if actual.shape != expected.shape:
        return False, float("inf")
    error = float(np.max(np.abs(actual - expected)))
    return bool(np.allclose(actual, expected, atol=atol, rtol=rtol)), error


def run_val002(
    spec: dict[str, Any],
    parent: dict[str, Any],
    spec_sha: str,
    parent_sha: str,
) -> dict[str, Any]:
    f = spec["fixture"]
    e = spec["expected"]
    p = spec["parameters"]
    a = spec["acceptance"]

    projected = np.asarray(f["projected_values"], dtype=np.float64)
    content_keys = np.asarray(f["content_keys"], dtype=np.float64)
    memory = np.asarray(f["ma_memory_contribution"], dtype=np.float64)
    historical_tokens = np.asarray(f["historical_token_values"], dtype=np.float64)
    lam = float(p["historical_lambda"])

    historical = historical_value_embedding(projected, historical_tokens, lam)
    historical_l0 = historical_value_embedding(projected, historical_tokens, 0.0)
    historical_l1 = historical_value_embedding(projected, historical_tokens, 1.0)

    cells = factorial_value_sources(projected, content_keys, memory)
    effects = factorial_effects(cells)

    atol = float(a["atol"])
    rtol = float(a["rtol"])

    checks: dict[str, dict[str, Any]] = {}
    metrics: dict[str, float] = {}

    hist_ok, hist_err = _check_close(
        historical,
        np.asarray(e["historical_value_embedding"], dtype=np.float64),
        atol=atol,
        rtol=rtol,
    )
    checks["historical_known_answer"] = {
        "passed": hist_ok,
        "max_abs_error": hist_err,
    }
    metrics["historical_known_answer_max_abs_error"] = hist_err

    for name, actual in cells.items():
        ok, err = _check_close(
            actual,
            np.asarray(e[name], dtype=np.float64),
            atol=atol,
            rtol=rtol,
        )
        checks[f"{name}_known_answer"] = {
            "passed": ok,
            "max_abs_error": err,
        }
        metrics[f"{name}_max_abs_error"] = err

    parent_keys = np.asarray(parent["fixture"]["content_keys"], dtype=np.float64)
    parent_memory = np.asarray(parent["expected"]["normalized_memory"], dtype=np.float64)
    parent_values = np.asarray(parent["expected"]["constructed_values"], dtype=np.float64)

    for name, actual, expected in (
        ("parent_content_key_anchor", content_keys, parent_keys),
        ("parent_memory_anchor", memory, parent_memory),
        ("c11_val001_anchor", cells["c11_key_plus_memory"], parent_values),
    ):
        ok, err = _check_close(actual, expected, atol=atol, rtol=rtol)
        checks[name] = {"passed": ok, "max_abs_error": err}
        metrics[f"{name}_max_abs_error"] = err

    memory_from_projected_ok, memory_from_projected_err = _check_close(
        effects["memory_effect_from_projected"], memory, atol=atol, rtol=rtol
    )
    memory_from_key_ok, memory_from_key_err = _check_close(
        effects["memory_effect_from_key"], memory, atol=atol, rtol=rtol
    )
    source_target = content_keys - projected
    source_no_mem_ok, source_no_mem_err = _check_close(
        effects["source_effect_without_memory"], source_target, atol=atol, rtol=rtol
    )
    source_with_mem_ok, source_with_mem_err = _check_close(
        effects["source_effect_with_memory"], source_target, atol=atol, rtol=rtol
    )
    zero = np.zeros_like(projected)
    interaction_ok, interaction_err = _check_close(
        effects["additive_interaction"], zero, atol=atol, rtol=rtol
    )

    checks["memory_effect_from_projected"] = {
        "passed": memory_from_projected_ok,
        "max_abs_error": memory_from_projected_err,
    }
    checks["memory_effect_from_key"] = {
        "passed": memory_from_key_ok,
        "max_abs_error": memory_from_key_err,
    }
    checks["source_effect_without_memory"] = {
        "passed": source_no_mem_ok,
        "max_abs_error": source_no_mem_err,
    }
    checks["source_effect_with_memory"] = {
        "passed": source_with_mem_ok,
        "max_abs_error": source_with_mem_err,
    }
    checks["fixed_input_additive_interaction_zero"] = {
        "passed": interaction_ok,
        "max_abs_error": interaction_err,
    }

    for key, value in (
        ("memory_effect_from_projected_max_abs_error", memory_from_projected_err),
        ("memory_effect_from_key_max_abs_error", memory_from_key_err),
        ("source_effect_without_memory_max_abs_error", source_no_mem_err),
        ("source_effect_with_memory_max_abs_error", source_with_mem_err),
        ("fixed_input_additive_interaction_max_abs_error", interaction_err),
    ):
        metrics[key] = value

    l0_ok, l0_err = _check_close(historical_l0, projected, atol=atol, rtol=rtol)
    l1_ok, l1_err = _check_close(
        historical_l1, historical_tokens, atol=atol, rtol=rtol
    )
    checks["historical_lambda_zero_endpoint"] = {
        "passed": l0_ok,
        "max_abs_error": l0_err,
    }
    checks["historical_lambda_one_endpoint"] = {
        "passed": l1_ok,
        "max_abs_error": l1_err,
    }
    metrics["historical_lambda_zero_max_abs_error"] = l0_err
    metrics["historical_lambda_one_max_abs_error"] = l1_err

    historical_additive_delta = float(
        np.max(np.abs(historical - cells["c01_projected_plus_memory"]))
    )
    separation_ok = bool(
        historical_additive_delta
        >= float(a["min_historical_additive_separation"])
    )
    checks["historical_is_not_local_additive_control"] = {
        "passed": separation_ok,
        "max_abs_delta": historical_additive_delta,
        "minimum_required_delta": float(a["min_historical_additive_separation"]),
    }
    metrics["historical_vs_local_additive_max_abs_delta"] = (
        historical_additive_delta
    )

    status = "PASS" if all(item["passed"] for item in checks.values()) else "FAIL"

    return {
        "manifest": {
            "schema_version": 1,
            "experiment_id": "VAL-002",
            "status": status,
            "spec_sha256": spec_sha,
            "parent_val001_spec_sha256": parent_sha,
            "source": spec["source"],
            "provenance": {
                "python": sys.version.split()[0],
                "numpy": np.__version__,
                "platform": platform.platform(),
                "git_commit": os.environ.get("GITHUB_SHA"),
            },
            "authority": {
                "historical_lane": "source-fidelity control",
                "factorial_lane": "local causal control",
            },
        },
        "metrics": metrics,
        "checks": checks,
        "observations": {
            "historical_value_embedding": historical.tolist(),
            "factorial_cells": {name: value.tolist() for name, value in cells.items()},
            "factorial_effects": {
                name: value.tolist() for name, value in effects.items()
            },
        },
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
        "experiment_id": "VAL-002",
        "status": status,
        "spec_path": str(spec_path),
        "parent_val001_path": str(parent_path),
        "error": error,
        "provenance": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
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
    parser.add_argument("--parent-val001", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    try:
        spec, parent, spec_sha, parent_sha = load_spec(
            args.spec, args.parent_val001
        )
    except (OSError, SpecError) as exc:
        _terminal_manifest(
            args.out,
            status="INVALID",
            spec_path=args.spec,
            parent_path=args.parent_val001,
            error=str(exc),
        )
        print(f"VAL-002 INVALID: {exc}")
        return 2

    try:
        result = run_val002(spec, parent, spec_sha, parent_sha)
        write_evidence(args.out, result)
    except Exception as exc:
        _terminal_manifest(
            args.out,
            status="ERROR",
            spec_path=args.spec,
            parent_path=args.parent_val001,
            error=f"{type(exc).__name__}: {exc}",
        )
        print(f"VAL-002 ERROR: {type(exc).__name__}: {exc}")
        return 3

    status = result["manifest"]["status"]
    print(f"VAL-002 {status}")
    for name, check in result["checks"].items():
        print(f"  {name}: {'PASS' if check['passed'] else 'FAIL'}")
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
