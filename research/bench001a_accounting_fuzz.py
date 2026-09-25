#!/usr/bin/env python3
"""Randomized property sweep for BENCH-001A integer accounting."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from memory_attention_lab.measurement.residency import account_residency


def random_config(rng: random.Random) -> dict:
    heads = rng.choice([1, 2, 4, 8, 16, 32, 64, 128])
    head_dim = rng.choice([4, 8, 16, 32, 64, 128])
    hidden = heads * head_dim
    divisors = [d for d in range(1, heads + 1) if heads % d == 0]
    kv_heads = rng.choice(divisors)
    layers = rng.randint(1, 256)
    return {
        "hidden_size": hidden,
        "num_heads": heads,
        "num_kv_heads": kv_heads,
        "num_layers": layers,
        "vocab_size": rng.randint(1, 500000),
        "batch_size": rng.randint(1, 128),
        "seq_len": rng.randint(1, 8192),
        "group_size": rng.randint(1, 512),
        "prefetch_depth": rng.randint(1, 64),
        "qkv_bias": bool(rng.getrandbits(1)),
        "weight_element_bytes": rng.choice([1, 2, 4, 8]),
        "memory_element_bytes": rng.choice([1, 2, 4, 8]),
        "kv_cache_element_bytes": rng.choice([1, 2, 4, 8]),
        "kv_cache_capacity": rng.randint(0, 16384),
    }


def check(config: dict) -> None:
    r = account_residency(config)
    d = r["derived"]
    p = r["parameters"]
    placement = r["placement"]
    staging = r["staging"]
    transfer = r["transfer"]
    kv = r["kv_cache"]

    h = config["hidden_size"]
    nh = config["num_heads"]
    nkv = config["num_kv_heads"]
    layers = config["num_layers"]
    vocab = config["vocab_size"]
    batch = config["batch_size"]
    seq = config["seq_len"]
    bias = config["qkv_bias"]
    bw = config["weight_element_bytes"]
    bm = config["memory_element_bytes"]
    bkv = config["kv_cache_element_bytes"]
    cap = config["kv_cache_capacity"]

    assert d["head_dim"] * nh == h
    assert d["kv_width"] == nkv * d["head_dim"]
    assert d["effective_group"] == min(config["group_size"], layers)
    assert d["num_groups"] == (
        layers + d["effective_group"] - 1
    ) // d["effective_group"]
    assert d["pipeline_slots"] == min(
        config["prefetch_depth"], d["num_groups"]
    )

    expected_wv_per = h * d["kv_width"] + (
        d["kv_width"] if bias else 0
    )
    assert p["standard_wv_params_per_layer"] == expected_wv_per
    assert p["standard_wv_params_total"] == layers * expected_wv_per
    assert p["ma_table_elements"] == vocab * layers * d["kv_width"]

    standard = placement["standard_value_side"]
    ma_gpu = placement["ma_gpu_value_side"]
    ma_offload = placement["ma_offload_value_side"]
    delta = placement["deltas_vs_standard"]

    assert standard["gpu_parameter_bytes"] == standard["gpu_parameters"] * bw
    assert ma_gpu["gpu_parameter_bytes"] == ma_gpu["gpu_parameters"] * bm
    assert ma_offload["cpu_parameter_bytes"] == ma_offload["cpu_parameters"] * bm
    assert delta["ma_offload_gpu_parameters"] == -standard["gpu_parameters"]
    assert delta["ma_offload_cpu_parameters"] == ma_offload["cpu_parameters"]
    assert delta["ma_gpu_gpu_parameters"] == (
        ma_gpu["gpu_parameters"] - standard["gpu_parameters"]
    )
    assert delta["ma_offload_gpu_parameter_bytes"] == -standard[
        "gpu_parameter_bytes"
    ]
    assert delta["ma_offload_cpu_parameter_bytes"] == ma_offload[
        "cpu_parameter_bytes"
    ]
    assert delta["ma_gpu_gpu_parameter_bytes"] == (
        ma_gpu["gpu_parameter_bytes"] - standard["gpu_parameter_bytes"]
    )

    pipeline_elements = (
        d["pipeline_slots"]
        * batch
        * seq
        * d["effective_group"]
        * d["kv_width"]
    )
    bulk_elements = batch * seq * layers * d["kv_width"]
    assert staging["pipeline"]["allocated_elements"] == pipeline_elements
    assert staging["pipeline"]["gpu_bytes"] == pipeline_elements * bm
    assert staging["pipeline"]["pinned_cpu_bytes"] == pipeline_elements * bm
    assert staging["bulk"]["allocated_elements"] == bulk_elements
    assert staging["bulk"]["gpu_bytes"] == bulk_elements * bm
    assert transfer["h2d_payload_elements_per_forward"] == bulk_elements
    assert transfer["h2d_payload_bytes_per_forward"] == bulk_elements * bm

    expected_kv = 2 * layers * batch * cap * d["kv_width"]
    assert kv["elements"] == expected_kv
    assert kv["bytes"] == expected_kv * bkv

    if config["group_size"] >= layers:
        assert d["effective_group"] == layers
        assert d["num_groups"] == 1
        assert d["pipeline_slots"] == 1
        assert staging["pipeline"]["allocated_elements"] == bulk_elements


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--cases", type=int, default=500000)
    p.add_argument("--seed", type=int, default=20260925)
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("runs/BENCH-001A/fuzz"),
    )
    args = p.parse_args()

    rng = random.Random(args.seed)
    for _ in range(args.cases):
        check(random_config(rng))

    result = {
        "kind": "BENCH-001A_property_sweep",
        "cases": args.cases,
        "seed": args.seed,
        "status": "PASS",
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
