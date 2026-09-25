"""Deterministic BENCH-001A residency and working-set accounting."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


class AccountingConfigError(ValueError):
    """Raised when a BENCH-001A accounting configuration is invalid."""


CONFIG_KEYS = {
    "hidden_size",
    "num_heads",
    "num_kv_heads",
    "num_layers",
    "vocab_size",
    "batch_size",
    "seq_len",
    "group_size",
    "prefetch_depth",
    "qkv_bias",
    "weight_element_bytes",
    "memory_element_bytes",
    "kv_cache_element_bytes",
    "kv_cache_capacity",
}


def _positive_int(name: str, value: Any) -> int:
    if type(value) is not int or value <= 0:
        raise AccountingConfigError(f"{name} must be a positive integer")
    return value


def _nonnegative_int(name: str, value: Any) -> int:
    if type(value) is not int or value < 0:
        raise AccountingConfigError(f"{name} must be a non-negative integer")
    return value


def validate_config(config: Mapping[str, Any]) -> dict[str, Any]:
    actual = set(config)
    missing = sorted(CONFIG_KEYS - actual)
    extra = sorted(actual - CONFIG_KEYS)
    if missing or extra:
        raise AccountingConfigError(
            f"config keys invalid; missing={missing or []}, extra={extra or []}"
        )

    values = dict(config)
    for key in (
        "hidden_size",
        "num_heads",
        "num_kv_heads",
        "num_layers",
        "vocab_size",
        "batch_size",
        "seq_len",
        "group_size",
        "prefetch_depth",
        "weight_element_bytes",
        "memory_element_bytes",
        "kv_cache_element_bytes",
    ):
        values[key] = _positive_int(key, values[key])
    values["kv_cache_capacity"] = _nonnegative_int(
        "kv_cache_capacity", values["kv_cache_capacity"]
    )
    if type(values["qkv_bias"]) is not bool:
        raise AccountingConfigError("qkv_bias must be boolean")

    hidden = values["hidden_size"]
    heads = values["num_heads"]
    kv_heads = values["num_kv_heads"]
    if hidden % heads:
        raise AccountingConfigError("hidden_size must be divisible by num_heads")
    if heads % kv_heads:
        raise AccountingConfigError(
            "num_heads must be divisible by num_kv_heads"
        )

    return values


def account_residency(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return exact source-scope integer accounting for BENCH-001A.

    This function does not allocate tensors and does not estimate latency.
    Python integers are used intentionally so large sweeps do not overflow.
    """
    c = validate_config(config)

    h = c["hidden_size"]
    nh = c["num_heads"]
    nkv = c["num_kv_heads"]
    layers = c["num_layers"]
    vocab = c["vocab_size"]
    batch = c["batch_size"]
    seq = c["seq_len"]
    requested_group = c["group_size"]
    prefetch = c["prefetch_depth"]
    bias = c["qkv_bias"]
    weight_bytes = c["weight_element_bytes"]
    memory_bytes = c["memory_element_bytes"]
    kv_bytes = c["kv_cache_element_bytes"]
    capacity = c["kv_cache_capacity"]

    head_dim = h // nh
    kv_width = nkv * head_dim

    effective_group = min(requested_group, layers)
    num_groups = (layers + effective_group - 1) // effective_group
    pipeline_slots = min(prefetch, num_groups)

    wv_per_layer = h * kv_width + (kv_width if bias else 0)
    wv_total = layers * wv_per_layer
    ma_table_elements = vocab * layers * kv_width

    standard_wv_bytes = wv_total * weight_bytes
    ma_table_bytes = ma_table_elements * memory_bytes

    pipeline_elements = (
        pipeline_slots * batch * seq * effective_group * kv_width
    )
    bulk_elements = batch * seq * layers * kv_width
    kv_cache_elements = 2 * layers * batch * capacity * kv_width

    return {
        "derived": {
            "head_dim": head_dim,
            "kv_width": kv_width,
            "effective_group": effective_group,
            "num_groups": num_groups,
            "pipeline_slots": pipeline_slots,
        },
        "parameters": {
            "standard_wv_params_per_layer": wv_per_layer,
            "standard_wv_params_total": wv_total,
            "ma_table_elements": ma_table_elements,
        },
        "placement": {
            "standard_value_side": {
                "gpu_parameters": wv_total,
                "cpu_parameters": 0,
                "gpu_parameter_bytes": standard_wv_bytes,
                "cpu_parameter_bytes": 0,
            },
            "ma_gpu_value_side": {
                "gpu_parameters": ma_table_elements,
                "cpu_parameters": 0,
                "gpu_parameter_bytes": ma_table_bytes,
                "cpu_parameter_bytes": 0,
            },
            "ma_offload_value_side": {
                "gpu_parameters": 0,
                "cpu_parameters": ma_table_elements,
                "gpu_parameter_bytes": 0,
                "cpu_parameter_bytes": ma_table_bytes,
            },
            "deltas_vs_standard": {
                "ma_offload_gpu_parameters": -wv_total,
                "ma_offload_cpu_parameters": ma_table_elements,
                "ma_gpu_gpu_parameters": ma_table_elements - wv_total,
                "ma_offload_gpu_parameter_bytes": -standard_wv_bytes,
                "ma_offload_cpu_parameter_bytes": ma_table_bytes,
                "ma_gpu_gpu_parameter_bytes": ma_table_bytes - standard_wv_bytes,
            },
        },
        "staging": {
            "pipeline": {
                "allocated_elements": pipeline_elements,
                "pinned_cpu_bytes": pipeline_elements * memory_bytes,
                "gpu_bytes": pipeline_elements * memory_bytes,
            },
            "bulk": {
                "allocated_elements": bulk_elements,
                "pinned_cpu_bytes": bulk_elements * memory_bytes,
                "gpu_bytes": bulk_elements * memory_bytes,
            },
        },
        "transfer": {
            "h2d_payload_elements_per_forward": bulk_elements,
            "h2d_payload_bytes_per_forward": bulk_elements * memory_bytes,
        },
        "kv_cache": {
            "elements": kv_cache_elements,
            "bytes": kv_cache_elements * kv_bytes,
        },
    }
