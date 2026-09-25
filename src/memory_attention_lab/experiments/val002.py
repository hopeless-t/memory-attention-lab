"""Executable known-answer contract for VAL-002."""

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

from memory_attention_lab.attention.reference import rope_neox
from memory_attention_lab.attention.value_sources import (
    factorial_value_cells,
    historical_value_embedding,
)


class SpecError(ValueError):
    """Raised when the VAL-002 reference spec is invalid."""


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
        "design_commit",
        "memory_attention_paper",
        "historical_repository",
        "historical_path",
        "historical_repository_commit",
        "historical_file_blob",
    },
    "parameters": {
        "num_kv_heads",
        "head_dim",
        "eps",
        "rope_base",
        "historical_lambda",
        "dtype",
    },
    "fixture": {
        "token_ids",
        "positions",
        "norm_weight",
        "hidden_states",
        "wv",
        "wk",
        "memory_table",
        "historical_value_table",
    },
    "expected": {
        "memory",
        "C00",
        "C01",
        "C10",
        "C11",
        "HIST-VE",
        "rotated_contextual_wk",
    },
    "acceptance": {
        "atol",
        "rtol",
        "min_post_rope_guard_delta",
        "min_historical_vs_additive_delta",
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


def _number(
    name: str,
    value: Any,
    *,
    strictly_positive: bool = False,
    nonnegative: bool = False,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SpecError(f"{name} must be a JSON number")
    number = float(value)
    if not np.isfinite(number):
        raise SpecError(f"{name} must be finite")
    if strictly_positive and number <= 0:
        raise SpecError(f"{name} must be > 0")
    if nonnegative and number < 0:
        raise SpecError(f"{name} must be >= 0")
    return number


def _array(name: str, value: Any, *, ndim: int) -> np.ndarray:
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
    if spec["experiment_id"] != "VAL-002":
        raise SpecError("experiment_id must be VAL-002")
    if not isinstance(spec["description"], str) or not spec["description"].strip():
        raise SpecError("description must be a non-empty string")
    if spec["parameters"]["dtype"] != "float64":
        raise SpecError("VAL-002 frozen reference dtype must be float64")

    for key, value in spec["source"].items():
        if not isinstance(value, str) or not value.strip():
            raise SpecError(f"source.{key} must be a non-empty string")

    p = spec["parameters"]
    if type(p["num_kv_heads"]) is not int or p["num_kv_heads"] <= 0:
        raise SpecError("num_kv_heads must be a positive integer")
    if type(p["head_dim"]) is not int or p["head_dim"] <= 0:
        raise SpecError("head_dim must be a positive integer")
    if p["head_dim"] % 2:
        raise SpecError("head_dim must be even for the frozen RoPE guard")
    _number("eps", p["eps"], strictly_positive=True)
    _number("rope_base", p["rope_base"], strictly_positive=True)
    _number("historical_lambda", p["historical_lambda"])

    a = spec["acceptance"]
    for key in (
        "atol",
        "rtol",
        "min_post_rope_guard_delta",
        "min_historical_vs_additive_delta",
    ):
        _number(key, a[key], nonnegative=True)

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

    positions = _array("positions", f["positions"], ndim=1)
    if positions.shape != (t,):
        raise SpecError(f"positions must have shape ({t},)")

    hidden = _array("hidden_states", f["hidden_states"], ndim=2)
    if hidden.shape[0] != t:
        raise SpecError("hidden_states sequence length must match token_ids")
    d_model = hidden.shape[1]

    kv_width = p["num_kv_heads"] * p["head_dim"]

    for key in ("wv", "wk"):
        weight = _array(key, f[key], ndim=2)
        if weight.shape != (d_model, kv_width):
            raise SpecError(
                f"{key} must have shape ({d_model}, {kv_width}), got {weight.shape}"
            )

    norm_weight = _array("norm_weight", f["norm_weight"], ndim=1)
    if norm_weight.shape != (p["head_dim"],):
        raise SpecError(
            f"norm_weight must have shape ({p['head_dim']},)"
        )

    for key in ("memory_table", "historical_value_table"):
        table = _array(key, f[key], ndim=2)
        if table.shape[0] == 0 or table.shape[1] != kv_width:
            raise SpecError(
                f"{key} must have shape [vocab, {kv_width}] with vocab > 0"
            )
        if np.any(token_ids < 0) or np.any(token_ids >= table.shape[0]):
            raise SpecError(f"token_ids contain an out-of-range address for {key}")

    e = spec["expected"]
    flat_shape = (t, kv_width)
    for key in ("memory", "C00", "C01", "C10", "C11", "HIST-VE"):
        expected = _array(f"expected.{key}", e[key], ndim=2)
        if expected.shape != flat_shape:
            raise SpecError(
                f"expected.{key} must have shape {flat_shape}, got {expected.shape}"
            )

    rotated = _array(
        "expected.rotated_contextual_wk",
        e["rotated_contextual_wk"],
        ndim=3,
    )
    rotary_shape = (t, p["num_kv_heads"], p["head_dim"])
    if rotated.shape != rotary_shape:
        raise SpecError(
            "expected.rotated_contextual_wk must have shape "
            f"{rotary_shape}, got {rotated.shape}"
        )

    return spec, digest


def _max_abs_error(actual: np.ndarray, expected: np.ndarray) -> float:
    return float(np.max(np.abs(actual - expected)))


def _close(
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


def run_val002(spec: dict[str, Any], spec_sha256: str) -> dict[str, Any]:
    p = spec["parameters"]
    f = spec["fixture"]
    e = spec["expected"]
    a = spec["acceptance"]

    token_ids = np.asarray(f["token_ids"], dtype=np.int64)
    positions = np.asarray(f["positions"], dtype=np.float64)
    norm_weight = np.asarray(f["norm_weight"], dtype=np.float64)
    hidden = np.asarray(f["hidden_states"], dtype=np.float64)
    wv = np.asarray(f["wv"], dtype=np.float64)
    wk = np.asarray(f["wk"], dtype=np.float64)
    memory_table = np.asarray(f["memory_table"], dtype=np.float64)
    historical_table = np.asarray(f["historical_value_table"], dtype=np.float64)

    cells = factorial_value_cells(
        hidden,
        wv,
        wk,
        memory_table,
        token_ids,
        norm_weight,
        num_kv_heads=int(p["num_kv_heads"]),
        head_dim=int(p["head_dim"]),
        eps=float(p["eps"]),
    )
    hist = historical_value_embedding(
        hidden,
        wv,
        historical_table,
        token_ids,
        lamb=float(p["historical_lambda"]),
    )

    atol = float(a["atol"])
    rtol = float(a["rtol"])

    checks: dict[str, dict[str, Any]] = {}
    metrics: dict[str, float] = {}

    for key in ("memory", "C00", "C01", "C10", "C11"):
        expected = np.asarray(e[key], dtype=np.float64)
        ok, err = _close(cells[key], expected, atol=atol, rtol=rtol)
        checks[f"known_answer_{key}"] = {"passed": ok, "max_abs_error": err}
        metrics[f"{key}_max_abs_error"] = err

    expected_hist = np.asarray(e["HIST-VE"], dtype=np.float64)
    hist_ok, hist_err = _close(hist, expected_hist, atol=atol, rtol=rtol)
    checks["known_answer_HIST-VE"] = {
        "passed": hist_ok,
        "max_abs_error": hist_err,
    }
    metrics["HIST-VE_max_abs_error"] = hist_err

    memory_delta_wv = cells["C01"] - cells["C00"]
    memory_delta_wk = cells["C11"] - cells["C10"]
    delta_wv_ok, delta_wv_err = _close(
        memory_delta_wv, cells["memory"], atol=atol, rtol=rtol
    )
    delta_wk_ok, delta_wk_err = _close(
        memory_delta_wk, cells["memory"], atol=atol, rtol=rtol
    )
    shared_memory_ok, shared_memory_err = _close(
        memory_delta_wv, memory_delta_wk, atol=atol, rtol=rtol
    )

    checks["memory_delta_at_Wv"] = {
        "passed": delta_wv_ok,
        "max_abs_error": delta_wv_err,
    }
    checks["memory_delta_at_Wk"] = {
        "passed": delta_wk_ok,
        "max_abs_error": delta_wk_err,
    }
    checks["shared_memory_delta"] = {
        "passed": shared_memory_ok,
        "max_abs_error": shared_memory_err,
    }

    interaction = memory_delta_wk - memory_delta_wv
    interaction_max = float(np.max(np.abs(interaction)))
    interaction_ok = bool(
        np.allclose(interaction, np.zeros_like(interaction), atol=atol, rtol=rtol)
    )
    checks["factorial_interaction_zero_for_frozen_value_path"] = {
        "passed": interaction_ok,
        "max_abs_value": interaction_max,
    }
    metrics["factorial_interaction_max_abs"] = interaction_max

    source_no_memory = cells["C10"] - cells["C00"]
    source_with_memory = cells["C11"] - cells["C01"]
    source_ok, source_err = _close(
        source_no_memory, source_with_memory, atol=atol, rtol=rtol
    )
    checks["source_contrast_consistency"] = {
        "passed": source_ok,
        "max_abs_error": source_err,
    }

    kv_shape = (
        hidden.shape[0],
        int(p["num_kv_heads"]),
        int(p["head_dim"]),
    )
    content_k_heads = cells["C10"].reshape(kv_shape)
    rotated = rope_neox(
        content_k_heads,
        positions,
        base=float(p["rope_base"]),
    )
    expected_rotated = np.asarray(e["rotated_contextual_wk"], dtype=np.float64)
    rotated_ok, rotated_err = _close(
        rotated, expected_rotated, atol=atol, rtol=rtol
    )
    checks["known_answer_rotated_contextual_wk"] = {
        "passed": rotated_ok,
        "max_abs_error": rotated_err,
    }
    metrics["rotated_contextual_wk_max_abs_error"] = rotated_err

    wrong_post_rope_c11 = rotated.reshape(cells["C11"].shape) + cells["memory"]
    post_rope_delta = float(np.max(np.abs(wrong_post_rope_c11 - cells["C11"])))
    post_rope_ok = bool(
        post_rope_delta >= float(a["min_post_rope_guard_delta"])
    )
    checks["pre_rope_value_guard"] = {
        "passed": post_rope_ok,
        "wrong_post_rope_value_max_delta": post_rope_delta,
        "minimum_required_delta": float(a["min_post_rope_guard_delta"]),
    }
    metrics["post_rope_guard_delta"] = post_rope_delta

    historical_vs_additive = float(np.max(np.abs(hist - cells["C01"])))
    historical_distinct_ok = bool(
        historical_vs_additive
        >= float(a["min_historical_vs_additive_delta"])
    )
    checks["historical_lane_is_not_clean_additive_control"] = {
        "passed": historical_distinct_ok,
        "max_abs_delta": historical_vs_additive,
        "minimum_required_delta": float(a["min_historical_vs_additive_delta"]),
    }
    metrics["historical_vs_additive_max_abs_delta"] = historical_vs_additive

    hist_zero = historical_value_embedding(
        hidden, wv, historical_table, token_ids, lamb=0.0
    )
    selected_hist = historical_table[token_ids]
    hist_one = historical_value_embedding(
        hidden, wv, historical_table, token_ids, lamb=1.0
    )
    hist_zero_ok, hist_zero_err = _close(
        hist_zero, cells["C00"], atol=atol, rtol=rtol
    )
    hist_one_ok, hist_one_err = _close(
        hist_one, selected_hist, atol=atol, rtol=rtol
    )
    checks["historical_lambda_zero_recovers_Wv"] = {
        "passed": hist_zero_ok,
        "max_abs_error": hist_zero_err,
    }
    checks["historical_lambda_one_recovers_embedding"] = {
        "passed": hist_one_ok,
        "max_abs_error": hist_one_err,
    }

    wv_perturbed = wv.copy()
    wv_perturbed[0, 0] += 7.0
    perturbed_cells = factorial_value_cells(
        hidden,
        wv_perturbed,
        wk,
        memory_table,
        token_ids,
        norm_weight,
        num_kv_heads=int(p["num_kv_heads"]),
        head_dim=int(p["head_dim"]),
        eps=float(p["eps"]),
    )
    perturbed_hist = historical_value_embedding(
        hidden,
        wv_perturbed,
        historical_table,
        token_ids,
        lamb=float(p["historical_lambda"]),
    )
    c10_unchanged = bool(np.allclose(
        perturbed_cells["C10"], cells["C10"], atol=atol, rtol=rtol
    ))
    c11_unchanged = bool(np.allclose(
        perturbed_cells["C11"], cells["C11"], atol=atol, rtol=rtol
    ))
    c00_changed = not bool(np.allclose(
        perturbed_cells["C00"], cells["C00"], atol=atol, rtol=rtol
    ))
    hist_changed = not bool(np.allclose(
        perturbed_hist, hist, atol=atol, rtol=rtol
    ))
    checks["Wv_perturbation_guard"] = {
        "passed": c10_unchanged and c11_unchanged and c00_changed and hist_changed,
        "C10_unchanged": c10_unchanged,
        "C11_unchanged": c11_unchanged,
        "C00_changed": c00_changed,
        "HIST-VE_changed": hist_changed,
    }

    wk_perturbed = wk.copy()
    wk_perturbed[0, 0] += 7.0
    perturbed_wk_cells = factorial_value_cells(
        hidden,
        wv,
        wk_perturbed,
        memory_table,
        token_ids,
        norm_weight,
        num_kv_heads=int(p["num_kv_heads"]),
        head_dim=int(p["head_dim"]),
        eps=float(p["eps"]),
    )
    c00_unchanged = bool(np.allclose(
        perturbed_wk_cells["C00"], cells["C00"], atol=atol, rtol=rtol
    ))
    c01_unchanged = bool(np.allclose(
        perturbed_wk_cells["C01"], cells["C01"], atol=atol, rtol=rtol
    ))
    c10_changed = not bool(np.allclose(
        perturbed_wk_cells["C10"], cells["C10"], atol=atol, rtol=rtol
    ))
    c11_changed = not bool(np.allclose(
        perturbed_wk_cells["C11"], cells["C11"], atol=atol, rtol=rtol
    ))
    checks["Wk_perturbation_guard"] = {
        "passed": c00_unchanged and c01_unchanged and c10_changed and c11_changed,
        "C00_unchanged": c00_unchanged,
        "C01_unchanged": c01_unchanged,
        "C10_changed": c10_changed,
        "C11_changed": c11_changed,
    }

    status = "PASS" if all(item["passed"] for item in checks.values()) else "FAIL"

    return {
        "manifest": {
            "schema_version": 1,
            "experiment_id": "VAL-002",
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
        "metrics": metrics,
        "checks": checks,
        "observations": {
            "memory": cells["memory"].tolist(),
            "C00": cells["C00"].tolist(),
            "C01": cells["C01"].tolist(),
            "C10": cells["C10"].tolist(),
            "C11": cells["C11"].tolist(),
            "HIST-VE": hist.tolist(),
            "rotated_contextual_wk": rotated.tolist(),
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
    error: str,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "experiment_id": "VAL-002",
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
        _terminal_manifest(
            args.out,
            status="INVALID",
            spec_path=args.spec,
            error=str(exc),
        )
        print(f"VAL-002 INVALID: {exc}")
        return 2

    try:
        result = run_val002(spec, digest)
        write_evidence(args.out, result)
    except Exception as exc:
        _terminal_manifest(
            args.out,
            status="ERROR",
            spec_path=args.spec,
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
