"""Executable contract for VAL-001."""

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

from memory_attention_lab.attention.reference import (
    construct_values,
    fold_memory_table,
    inverse_rope_neox,
    lookup_memory,
    rope_neox,
)


class SpecError(ValueError):
    """Raised when VAL-001 input is invalid."""


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
        "paper",
        "upstream_memory_attn_blob",
        "upstream_rmsnorm_blob",
        "upstream_rotary_blob",
    },
    "parameters": {
        "num_kv_heads",
        "head_dim",
        "eps",
        "rope_base",
        "dtype",
    },
    "fixture": {
        "token_ids",
        "positions",
        "norm_weight",
        "memory_table",
        "content_keys",
    },
    "expected": {
        "normalized_memory",
        "constructed_values",
        "rotated_keys",
    },
    "acceptance": {
        "atol",
        "rtol",
        "min_post_rope_guard_delta",
    },
}


def _require_exact_keys(name: str, obj: dict[str, Any], expected: set[str]) -> None:
    actual = set(obj)
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing or extra:
        raise SpecError(
            f"{name} keys invalid; missing={missing or []}, extra={extra or []}"
        )


def _finite_scalar(
    name: str,
    value: Any,
    *,
    strictly_positive: bool = False,
    nonnegative: bool = False,
) -> float:
    if isinstance(value, bool):
        raise SpecError(f"{name} must be numeric, not boolean")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise SpecError(f"{name} must be numeric") from exc
    if not np.isfinite(number):
        raise SpecError(f"{name} must be finite")
    if strictly_positive and number <= 0:
        raise SpecError(f"{name} must be > 0")
    if nonnegative and number < 0:
        raise SpecError(f"{name} must be >= 0")
    return number


def _finite_array(
    name: str,
    value: Any,
    *,
    ndim: int,
) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise SpecError(f"{name} must be a rectangular numeric array") from exc
    if array.ndim != ndim:
        raise SpecError(f"{name} must have ndim={ndim}, got {array.ndim}")
    if not np.all(np.isfinite(array)):
        raise SpecError(f"{name} contains non-finite values")
    return array


def load_spec(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    try:
        spec = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise SpecError(f"invalid JSON: {exc}") from exc

    if not isinstance(spec, dict):
        raise SpecError("top-level spec must be an object")
    _require_exact_keys("top-level", spec, TOP_KEYS)

    for name, keys in NESTED_KEYS.items():
        if not isinstance(spec[name], dict):
            raise SpecError(f"{name} must be an object")
        _require_exact_keys(name, spec[name], keys)

    if spec["schema_version"] != 1:
        raise SpecError("unsupported schema_version")
    if spec["experiment_id"] != "VAL-001":
        raise SpecError("experiment_id must be VAL-001")
    if spec["parameters"]["dtype"] != "float64":
        raise SpecError("VAL-001 frozen reference dtype must be float64")

    p = spec["parameters"]
    if not isinstance(p["num_kv_heads"], int) or p["num_kv_heads"] <= 0:
        raise SpecError("num_kv_heads must be a positive integer")
    if not isinstance(p["head_dim"], int) or p["head_dim"] <= 0:
        raise SpecError("head_dim must be a positive integer")
    if p["head_dim"] % 2:
        raise SpecError("head_dim must be even for the frozen RoPE contract")
    _finite_scalar("eps", p["eps"], strictly_positive=True)
    _finite_scalar("rope_base", p["rope_base"], strictly_positive=True)

    a = spec["acceptance"]
    for key in ("atol", "rtol", "min_post_rope_guard_delta"):
        _finite_scalar(key, a[key], nonnegative=True)

    f = spec["fixture"]
    ids_raw = f["token_ids"]
    if (
        not isinstance(ids_raw, list)
        or len(ids_raw) == 0
        or any(type(item) is not int for item in ids_raw)
    ):
        raise SpecError("token_ids must be a non-empty list of integers")
    token_ids = np.asarray(ids_raw, dtype=np.int64)
    t = token_ids.shape[0]

    positions = _finite_array("positions", f["positions"], ndim=1)
    if positions.shape != (t,):
        raise SpecError(f"positions must have shape ({t},)")

    norm_weight = _finite_array("norm_weight", f["norm_weight"], ndim=1)
    if norm_weight.shape != (p["head_dim"],):
        raise SpecError(
            f"norm_weight must have shape ({p['head_dim']},)"
        )

    memory_table = _finite_array("memory_table", f["memory_table"], ndim=2)
    kv_dim = p["num_kv_heads"] * p["head_dim"]
    if memory_table.shape[0] == 0 or memory_table.shape[1] != kv_dim:
        raise SpecError(
            f"memory_table must have shape [vocab, {kv_dim}] with vocab > 0"
        )
    if np.any(token_ids < 0) or np.any(token_ids >= memory_table.shape[0]):
        raise SpecError("token_ids contain an out-of-range memory address")

    content_keys = _finite_array("content_keys", f["content_keys"], ndim=3)
    expected_shape = (t, p["num_kv_heads"], p["head_dim"])
    if content_keys.shape != expected_shape:
        raise SpecError(
            f"content_keys must have shape {expected_shape}, got {content_keys.shape}"
        )

    e = spec["expected"]
    for key in ("normalized_memory", "constructed_values", "rotated_keys"):
        expected = _finite_array(f"expected.{key}", e[key], ndim=3)
        if expected.shape != expected_shape:
            raise SpecError(
                f"expected.{key} must have shape {expected_shape}, "
                f"got {expected.shape}"
            )

    return spec, digest


def _max_abs_error(actual: np.ndarray, expected: np.ndarray) -> float:
    return float(np.max(np.abs(actual - expected)))


def _check_close(
    actual: np.ndarray,
    expected: np.ndarray,
    *,
    atol: float,
    rtol: float,
) -> tuple[bool, float]:
    if actual.shape != expected.shape:
        return False, float("inf")
    return bool(np.allclose(actual, expected, atol=atol, rtol=rtol)), _max_abs_error(
        actual, expected
    )


def run_val001(spec: dict[str, Any], spec_sha256: str) -> dict[str, Any]:
    p = spec["parameters"]
    f = spec["fixture"]
    e = spec["expected"]
    a = spec["acceptance"]

    token_ids = np.asarray(f["token_ids"], dtype=np.int64)
    positions = np.asarray(f["positions"], dtype=np.float64)
    norm_weight = np.asarray(f["norm_weight"], dtype=np.float64)
    memory_table = np.asarray(f["memory_table"], dtype=np.float64)
    content_keys = np.asarray(f["content_keys"], dtype=np.float64)

    values, normalized_memory = construct_values(
        content_keys,
        memory_table,
        token_ids,
        norm_weight,
        eps=float(p["eps"]),
    )

    folded_table = fold_memory_table(
        memory_table,
        norm_weight,
        num_kv_heads=int(p["num_kv_heads"]),
        head_dim=int(p["head_dim"]),
        eps=float(p["eps"]),
    )
    folded_selected = lookup_memory(
        folded_table,
        token_ids,
        num_kv_heads=int(p["num_kv_heads"]),
        head_dim=int(p["head_dim"]),
    )
    folded_values = content_keys + folded_selected

    rotated_keys = rope_neox(
        content_keys,
        positions,
        base=float(p["rope_base"]),
    )
    recovered_keys = inverse_rope_neox(
        rotated_keys,
        positions,
        base=float(p["rope_base"]),
    )

    # Deliberately wrong construction for the Red Team guard.
    post_rope_values = rotated_keys + normalized_memory
    post_rope_guard_delta = float(np.max(np.abs(post_rope_values - values)))

    expected_normalized = np.asarray(e["normalized_memory"], dtype=np.float64)
    expected_values = np.asarray(e["constructed_values"], dtype=np.float64)
    expected_rotated = np.asarray(e["rotated_keys"], dtype=np.float64)

    atol = float(a["atol"])
    rtol = float(a["rtol"])

    normalized_ok, normalized_error = _check_close(
        normalized_memory, expected_normalized, atol=atol, rtol=rtol
    )
    values_ok, values_error = _check_close(
        values, expected_values, atol=atol, rtol=rtol
    )
    rotated_ok, rotated_error = _check_close(
        rotated_keys, expected_rotated, atol=atol, rtol=rtol
    )
    folded_ok, folded_error = _check_close(
        folded_values, values, atol=atol, rtol=rtol
    )
    rope_roundtrip_ok, rope_roundtrip_error = _check_close(
        recovered_keys, content_keys, atol=atol, rtol=rtol
    )

    repeated_memory_ok = bool(
        token_ids[0] == token_ids[-1]
        and np.allclose(
            normalized_memory[0],
            normalized_memory[-1],
            atol=atol,
            rtol=rtol,
        )
        and not np.allclose(values[0], values[-1], atol=atol, rtol=rtol)
    )
    pre_rope_guard_ok = bool(
        post_rope_guard_delta >= float(a["min_post_rope_guard_delta"])
    )

    checks = {
        "known_answer_normalized_memory": {
            "passed": normalized_ok,
            "max_abs_error": normalized_error,
        },
        "known_answer_constructed_values": {
            "passed": values_ok,
            "max_abs_error": values_error,
        },
        "known_answer_rotated_keys": {
            "passed": rotated_ok,
            "max_abs_error": rotated_error,
        },
        "folded_memory_equivalence": {
            "passed": folded_ok,
            "max_abs_error": folded_error,
        },
        "rope_roundtrip": {
            "passed": rope_roundtrip_ok,
            "max_abs_error": rope_roundtrip_error,
        },
        "same_token_same_memory_different_context": {
            "passed": repeated_memory_ok,
        },
        "pre_rope_value_guard": {
            "passed": pre_rope_guard_ok,
            "wrong_post_rope_value_max_delta": post_rope_guard_delta,
            "minimum_required_delta": float(a["min_post_rope_guard_delta"]),
        },
    }

    status = "PASS" if all(item["passed"] for item in checks.values()) else "FAIL"

    return {
        "manifest": {
            "schema_version": 1,
            "experiment_id": "VAL-001",
            "status": status,
            "spec_sha256": spec_sha256,
            "source": spec["source"],
            "provenance": {
                "python": sys.version.split()[0],
                "numpy": np.__version__,
                "platform": platform.platform(),
                "git_commit": os.environ.get("GITHUB_SHA"),
            },
        },
        "metrics": {
            "normalized_memory_max_abs_error": normalized_error,
            "constructed_values_max_abs_error": values_error,
            "rotated_keys_max_abs_error": rotated_error,
            "folded_equivalence_max_abs_error": folded_error,
            "rope_roundtrip_max_abs_error": rope_roundtrip_error,
            "post_rope_guard_delta": post_rope_guard_delta,
        },
        "checks": checks,
        "observations": {
            "normalized_memory": normalized_memory.tolist(),
            "constructed_values": values.tolist(),
            "rotated_keys": rotated_keys.tolist(),
            "recovered_content_keys": recovered_keys.tolist(),
        },
    }


def write_evidence(out_dir: Path, result: dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in ("manifest", "metrics", "checks", "observations"):
        (out_dir / f"{name}.json").write_text(
            json.dumps(result[name], indent=2, sort_keys=True) + "\n"
        )


def _write_terminal_manifest(
    out_dir: Path,
    *,
    status: str,
    spec_path: Path,
    error: str,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "experiment_id": "VAL-001",
        "status": status,
        "spec_path": str(spec_path),
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
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    try:
        spec, digest = load_spec(args.spec)
    except (OSError, SpecError) as exc:
        _write_terminal_manifest(
            args.out,
            status="INVALID",
            spec_path=args.spec,
            error=str(exc),
        )
        print(f"VAL-001 INVALID: {exc}")
        return 2

    try:
        result = run_val001(spec, digest)
        write_evidence(args.out, result)
    except Exception as exc:
        _write_terminal_manifest(
            args.out,
            status="ERROR",
            spec_path=args.spec,
            error=f"{type(exc).__name__}: {exc}",
        )
        print(f"VAL-001 ERROR: {type(exc).__name__}: {exc}")
        return 3

    status = result["manifest"]["status"]
    print(f"VAL-001 {status}")
    for name, check in result["checks"].items():
        print(f"  {name}: {'PASS' if check['passed'] else 'FAIL'}")

    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
