"""BENCH-001B B1 capture kit for an already-approved CUDA environment.

This module never provisions cloud resources, starts paid GPU instances,
registers GitHub runners, uploads evidence, or handles billing. It accepts an
existing upstream checkout and records a source-faithful benchmark bundle.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from memory_attention_lab.experiments.bench001b_validate import (
    BundleError,
    validate_bundle,
)


class CaptureError(RuntimeError):
    """Raised when a BENCH-001B capture precondition fails."""


def _run_text(
    argv: list[str],
    *,
    cwd: Path | None = None,
    check: bool = True,
) -> str:
    proc = subprocess.run(
        argv,
        cwd=cwd,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if check and proc.returncode != 0:
        raise CaptureError(
            f"command failed ({proc.returncode}): {shlex.join(argv)}\n"
            f"stdout:\n{proc.stdout}\n"
            f"stderr:\n{proc.stderr}"
        )
    return proc.stdout.strip()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_upstream_checkout(
    upstream_dir: Path,
    *,
    expected_commit: str,
    expected_benchmark_blob: str,
) -> dict[str, str]:
    """Verify that the supplied checkout is exactly the frozen upstream source."""
    upstream_dir = upstream_dir.resolve()
    if not upstream_dir.is_dir():
        raise CaptureError(f"upstream directory does not exist: {upstream_dir}")

    head = _run_text(["git", "rev-parse", "HEAD"], cwd=upstream_dir)
    if head != expected_commit:
        raise CaptureError(
            f"upstream HEAD mismatch: expected {expected_commit}, got {head}"
        )

    dirty = _run_text(["git", "status", "--porcelain"], cwd=upstream_dir)
    if dirty:
        raise CaptureError(
            "upstream checkout must be clean; uncommitted changes were detected"
        )

    benchmark = upstream_dir / "profile" / "bmk.py"
    if not benchmark.is_file():
        raise CaptureError(f"missing pinned benchmark file: {benchmark}")

    blob = _run_text(["git", "hash-object", "profile/bmk.py"], cwd=upstream_dir)
    if blob != expected_benchmark_blob:
        raise CaptureError(
            f"profile/bmk.py blob mismatch: expected {expected_benchmark_blob}, "
            f"got {blob}"
        )

    origin = _run_text(
        ["git", "remote", "get-url", "origin"],
        cwd=upstream_dir,
        check=False,
    )
    return {
        "head": head,
        "benchmark_blob": blob,
        "origin": origin,
    }


def _append_flag(argv: list[str], flag: str, value: Any) -> None:
    if value is None:
        return
    if isinstance(value, bool):
        if value:
            argv.append(flag)
        return
    if isinstance(value, list):
        argv.append(flag)
        argv.extend(str(item) for item in value)
        return
    argv.extend([flag, str(value)])


CONFIG_FLAGS = {
    "mode": "--mode",
    "variants": "--variants",
    "decode_offload": "--decode-offload",
    "check_correctness": "--check-correctness",
    "batch_size": "--batch-size",
    "logits_to_keep": "--logits-to-keep",
    "pad_token_id": "--pad-token-id",
    "seq_len": "--seq-len",
    "context_len": "--context-len",
    "hidden_size": "--hidden-size",
    "num_heads": "--num-heads",
    "num_kv_heads": "--num-kv-heads",
    "num_layers": "--num-layers",
    "hidden_ratio": "--hidden-ratio",
    "intermediate_size": "--intermediate-size",
    "no_fuse_swiglu": "--no-fuse-swiglu",
    "vocab_size": "--vocab-size",
    "group_size": "--group-size",
    "prefetch_depth": "--prefetch-depth",
    "gpu_ma_lookup": "--gpu-ma-lookup",
    "qkv_bias": "--qkv-bias",
    "qk_norm": "--qk-norm",
    "use_gate": "--use-gate",
    "window_size": "--window-size",
    "rope_theta": "--rope-theta",
    "norm_eps": "--norm-eps",
    "cpu_threads": "--cpu-threads",
    "fold_chunk": "--fold-chunk",
    "id_pool": "--id-pool",
    "warmup": "--warmup",
    "repeats": "--repeats",
    "rounds": "--rounds",
    "timing": "--timing",
    "seed": "--seed",
    "device": "--device",
    "profile_dir": "--profile-dir",
    "kernel_trace": "--kernel-trace",
}


def build_benchmark_command(
    protocol: dict[str, Any],
    *,
    profile: str,
    python_executable: str,
    result_path: Path,
) -> list[str]:
    """Build the explicit source command from the frozen B0 protocol."""
    profiles = protocol.get("profiles")
    if not isinstance(profiles, dict) or profile not in profiles:
        raise CaptureError(f"unknown protocol profile: {profile}")
    required = profiles[profile].get("required_config")
    if not isinstance(required, dict):
        raise CaptureError("protocol profile required_config must be an object")

    missing_flags = sorted(set(required) - set(CONFIG_FLAGS))
    if missing_flags:
        raise CaptureError(
            f"capture kit has no CLI mapping for protocol keys: {missing_flags}"
        )

    argv = [python_executable, "profile/bmk.py"]
    for key, value in required.items():
        _append_flag(argv, CONFIG_FLAGS[key], value)
    argv.extend(["--json", str(result_path)])
    return argv


def _parse_version_pair(name: str, value: str) -> tuple[int, int]:
    parts = value.split(".")
    if len(parts) < 2:
        raise CaptureError(f"{name} must contain major.minor")
    try:
        return int(parts[0]), int(parts[1])
    except ValueError as exc:
        raise CaptureError(
            f"{name} must begin with numeric major.minor, got {value!r}"
        ) from exc


def probe_cuda_environment(
    protocol: dict[str, Any],
    *,
    device: str = "cuda:0",
) -> dict[str, Any]:
    """Probe an already-running environment; imports CUDA dependencies lazily."""
    try:
        import torch
        import flash_attn
    except ImportError as exc:
        raise CaptureError(
            "BENCH-001B B1 requires torch and flash-attn in the approved "
            "execution environment"
        ) from exc

    if not torch.cuda.is_available():
        raise CaptureError("CUDA is required")
    torch_device = torch.device(device)
    if torch_device.type != "cuda":
        raise CaptureError("device must be CUDA")
    torch.cuda.set_device(torch_device)

    bf16 = bool(torch.cuda.is_bf16_supported())
    if not bf16:
        raise CaptureError("BF16 support is required")

    props = torch.cuda.get_device_properties(torch_device)
    capability = list(torch.cuda.get_device_capability(torch_device))
    floor = protocol["hardware_floor"]["minimum_compute_capability"]
    if tuple(capability) < tuple(floor):
        raise CaptureError(
            f"GPU compute capability {capability} is below protocol floor {floor}"
        )

    flash_version = str(flash_attn.__version__)
    flash_pair = _parse_version_pair("flash-attn", flash_version)
    flash_floor = tuple(protocol["hardware_floor"]["minimum_flash_attn_version"])
    if flash_pair < flash_floor:
        raise CaptureError(
            f"flash-attn {flash_version} is below protocol floor "
            f"{flash_floor[0]}.{flash_floor[1]}"
        )

    driver = _run_text(
        [
            "nvidia-smi",
            "--query-gpu=driver_version",
            "--format=csv,noheader",
            "--id=0",
        ],
        check=False,
    )
    if not driver:
        driver = "unavailable"

    return {
        "hardware": {
            "gpu_name": str(torch.cuda.get_device_name(torch_device)),
            "compute_capability": capability,
            "total_vram_bytes": int(props.total_memory),
            "bf16_supported": bf16,
        },
        "runtime": {
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "cuda_runtime": str(torch.version.cuda or "unknown"),
            "driver": driver.splitlines()[0] if driver else "unavailable",
            "flash_attn": flash_version,
            "os": platform.platform(),
        },
    }


def make_hardware_manifest(
    protocol: dict[str, Any],
    *,
    execution_id: str,
    probe: dict[str, Any],
    profile: str,
    command_argv: list[str],
    result_path: Path,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "experiment_id": "BENCH-001B",
        "execution_id": execution_id,
        "synthetic": False,
        "source": dict(protocol["source"]),
        "hardware": dict(probe["hardware"]),
        "runtime": dict(probe["runtime"]),
        "execution": {
            "profile": profile,
            "command_argv": command_argv,
            "bmk_results_sha256": _sha256(result_path),
        },
    }


def _load_protocol(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise CaptureError(f"cannot load protocol {path}: {exc}") from exc
    if not isinstance(value, dict) or value.get("experiment_id") != "BENCH-001B":
        raise CaptureError("protocol must be a BENCH-001B object")
    if value.get("authority", {}).get("paid_resource_authorized") is not False:
        raise CaptureError(
            "capture kit expects the protocol to deny paid-resource authority"
        )
    return value


def _default_execution_id() -> str:
    return time.strftime("bench001b-%Y%m%dT%H%M%SZ", time.gmtime())


def main() -> int:
    p = argparse.ArgumentParser(
        description="Capture BENCH-001B evidence on an already-approved GPU."
    )
    p.add_argument("--protocol", type=Path, required=True)
    p.add_argument("--upstream-dir", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument(
        "--profile",
        default="source_default_latency",
    )
    p.add_argument("--execution-id", default=None)
    p.add_argument(
        "--probe-only",
        action="store_true",
        help="verify source/hardware and print the frozen command without timing",
    )
    args = p.parse_args()

    try:
        protocol = _load_protocol(args.protocol)
        source = protocol["source"]
        source_state = verify_upstream_checkout(
            args.upstream_dir,
            expected_commit=source["upstream_commit"],
            expected_benchmark_blob=source["upstream_benchmark_blob"],
        )

        args.out_dir.mkdir(parents=True, exist_ok=True)
        result_path = (args.out_dir / "bmk_results.json").resolve()
        hardware_path = (args.out_dir / "hardware.json").resolve()
        stdout_path = args.out_dir / "benchmark.stdout.txt"
        stderr_path = args.out_dir / "benchmark.stderr.txt"

        command = build_benchmark_command(
            protocol,
            profile=args.profile,
            python_executable=sys.executable,
            result_path=result_path,
        )

        probe = probe_cuda_environment(
            protocol,
            device=protocol["profiles"][args.profile]["required_config"]["device"],
        )

        probe_record = {
            "schema_version": 1,
            "experiment_id": "BENCH-001B-B1-PROBE",
            "performance_authority": False,
            "source_state": source_state,
            "hardware": probe["hardware"],
            "runtime": probe["runtime"],
            "profile": args.profile,
            "command_argv": command,
        }
        (args.out_dir / "probe.json").write_text(
            json.dumps(probe_record, indent=2, sort_keys=True) + "\n"
        )

        if args.probe_only:
            print(json.dumps(probe_record, indent=2, sort_keys=True))
            print("BENCH-001B probe only: no timing executed")
            return 0

        proc = subprocess.run(
            command,
            cwd=args.upstream_dir.resolve(),
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        stdout_path.write_text(proc.stdout)
        stderr_path.write_text(proc.stderr)
        if proc.returncode != 0:
            raise CaptureError(
                f"upstream benchmark failed with exit code {proc.returncode}; "
                f"see {stdout_path} and {stderr_path}"
            )
        if not result_path.is_file():
            raise CaptureError("upstream benchmark did not create bmk_results.json")

        execution_id = args.execution_id or _default_execution_id()
        manifest = make_hardware_manifest(
            protocol,
            execution_id=execution_id,
            probe=probe,
            profile=args.profile,
            command_argv=command,
            result_path=result_path,
        )
        hardware_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )

        validated = validate_bundle(
            args.protocol,
            hardware_path,
            result_path,
            allow_synthetic=False,
        )
        validation_path = args.out_dir / "validation.json"
        validation_path.write_text(
            json.dumps(validated, indent=2, sort_keys=True) + "\n"
        )

        print("BENCH-001B capture PASS")
        print(f"bundle: {args.out_dir.resolve()}")
        return 0

    except (CaptureError, BundleError, OSError) as exc:
        print(f"BENCH-001B capture ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
