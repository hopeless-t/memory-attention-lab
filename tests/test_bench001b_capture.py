from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from memory_attention_lab.experiments import bench001b_capture as capture


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "specs" / "BENCH-001B.protocol.json"


def _protocol() -> dict:
    return json.loads(PROTOCOL.read_text())


def test_build_command_is_derived_from_frozen_protocol(tmp_path: Path):
    protocol = _protocol()
    result = tmp_path / "bmk_results.json"

    argv = capture.build_benchmark_command(
        protocol,
        profile="source_default_latency",
        python_executable="/usr/bin/python3",
        result_path=result,
    )

    assert argv[:2] == ["/usr/bin/python3", "profile/bmk.py"]
    joined = " ".join(argv)
    for token in (
        "--mode both",
        "--variants standard ma_gpu ma_offload",
        "--decode-offload bulk",
        "--check-correctness",
        "--batch-size 8",
        "--seq-len 2048",
        "--context-len 2048",
        "--hidden-size 2048",
        "--num-heads 32",
        "--num-kv-heads 32",
        "--num-layers 24",
        "--intermediate-size 5632",
        "--group-size 1",
        "--prefetch-depth 4",
        "--warmup 5",
        "--repeats 30",
        "--rounds 5",
        "--timing latency",
        "--seed 1234",
        "--device cuda:0",
        "--json",
    ):
        assert token in joined

    for absent in (
        "--qkv-bias",
        "--qk-norm",
        "--use-gate",
        "--no-fuse-swiglu",
        "--kernel-trace",
        "--profile-dir",
        "--window-size",
        "--norm-eps",
        "--pad-token-id",
    ):
        assert absent not in argv

    assert argv[-2:] == ["--json", str(result)]


def test_unknown_profile_is_rejected(tmp_path: Path):
    with pytest.raises(capture.CaptureError, match="unknown protocol profile"):
        capture.build_benchmark_command(
            _protocol(),
            profile="unknown",
            python_executable=sys.executable,
            result_path=tmp_path / "result.json",
        )


def _git(cwd: Path, *args: str) -> str:
    proc = subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return proc.stdout.strip()


def _temporary_upstream(tmp_path: Path) -> tuple[Path, str, str]:
    repo = tmp_path / "upstream"
    (repo / "profile").mkdir(parents=True)
    (repo / "profile" / "bmk.py").write_text("print('fixture')\n")
    _git(repo, "init")
    _git(repo, "config", "user.email", "fixture@example.invalid")
    _git(repo, "config", "user.name", "Fixture")
    _git(repo, "add", "profile/bmk.py")
    _git(repo, "commit", "-m", "fixture")
    commit = _git(repo, "rev-parse", "HEAD")
    blob = _git(repo, "hash-object", "profile/bmk.py")
    return repo, commit, blob


def test_verify_upstream_checkout_accepts_exact_clean_source(tmp_path: Path):
    repo, commit, blob = _temporary_upstream(tmp_path)

    result = capture.verify_upstream_checkout(
        repo,
        expected_commit=commit,
        expected_benchmark_blob=blob,
    )

    assert result["head"] == commit
    assert result["benchmark_blob"] == blob


def test_verify_upstream_checkout_rejects_wrong_commit(tmp_path: Path):
    repo, _, blob = _temporary_upstream(tmp_path)

    with pytest.raises(capture.CaptureError, match="HEAD mismatch"):
        capture.verify_upstream_checkout(
            repo,
            expected_commit="0" * 40,
            expected_benchmark_blob=blob,
        )


def test_verify_upstream_checkout_rejects_dirty_tree(tmp_path: Path):
    repo, commit, blob = _temporary_upstream(tmp_path)
    (repo / "profile" / "bmk.py").write_text("print('dirty')\n")

    with pytest.raises(capture.CaptureError, match="clean"):
        capture.verify_upstream_checkout(
            repo,
            expected_commit=commit,
            expected_benchmark_blob=blob,
        )


def test_verify_upstream_checkout_rejects_wrong_blob(tmp_path: Path):
    repo, commit, _ = _temporary_upstream(tmp_path)

    with pytest.raises(capture.CaptureError, match="blob mismatch"):
        capture.verify_upstream_checkout(
            repo,
            expected_commit=commit,
            expected_benchmark_blob="0" * 40,
        )


def test_make_hardware_manifest_binds_exact_result_bytes(tmp_path: Path):
    result = tmp_path / "bmk_results.json"
    result.write_bytes(b'{"fixture":true}\n')
    probe = {
        "hardware": {
            "gpu_name": "fixture",
            "compute_capability": [8, 0],
            "total_vram_bytes": 1,
            "bf16_supported": True,
        },
        "runtime": {
            "python": "3.12",
            "torch": "2.8",
            "cuda_runtime": "12.8",
            "driver": "fixture",
            "flash_attn": "2.8",
            "os": "fixture",
        },
    }

    manifest = capture.make_hardware_manifest(
        _protocol(),
        execution_id="fixture-execution",
        probe=probe,
        profile="source_default_latency",
        command_argv=["python", "profile/bmk.py", "--json", str(result)],
        result_path=result,
    )

    assert manifest["synthetic"] is False
    assert manifest["execution"]["bmk_results_sha256"] == hashlib.sha256(
        result.read_bytes()
    ).hexdigest()


def test_default_main_path_is_probe_only_and_never_runs_benchmark(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    protocol_path = tmp_path / "protocol.json"
    protocol_path.write_text(PROTOCOL.read_text())
    upstream = tmp_path / "upstream"
    upstream.mkdir()
    out = tmp_path / "bundle"

    monkeypatch.setattr(
        capture,
        "verify_upstream_checkout",
        lambda *args, **kwargs: {
            "head": "fixture",
            "benchmark_blob": "fixture",
            "origin": "",
        },
    )
    monkeypatch.setattr(
        capture,
        "probe_cuda_environment",
        lambda *args, **kwargs: {
            "hardware": {
                "gpu_name": "fixture",
                "compute_capability": [8, 0],
                "total_vram_bytes": 1,
                "bf16_supported": True,
            },
            "runtime": {
                "python": "3.12",
                "torch": "2.8",
                "cuda_runtime": "12.8",
                "driver": "fixture",
                "flash_attn": "2.8",
                "os": "fixture",
            },
        },
    )

    def forbidden_run(*args, **kwargs):
        raise AssertionError("benchmark subprocess must not run without --execute")

    monkeypatch.setattr(capture.subprocess, "run", forbidden_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "bench001b_capture",
            "--protocol",
            str(protocol_path),
            "--upstream-dir",
            str(upstream),
            "--out-dir",
            str(out),
        ],
    )

    assert capture.main() == 0
    assert (out / "probe.json").is_file()
    assert not (out / "bmk_results.json").exists()
    assert not (out / "hardware.json").exists()


def test_protocol_refuses_paid_resource_authority(tmp_path: Path):
    payload = _protocol()
    payload["authority"]["paid_resource_authorized"] = True
    path = tmp_path / "protocol.json"
    path.write_text(json.dumps(payload))

    with pytest.raises(capture.CaptureError, match="deny paid-resource"):
        capture._load_protocol(path)
