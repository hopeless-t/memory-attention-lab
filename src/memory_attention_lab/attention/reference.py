"""Framework-independent numerical reference for VAL-001.

This module intentionally uses NumPy rather than importing the upstream FLA
implementation. It exists as an independent known-answer oracle for the
Memory Attention value-construction contract.
"""

from __future__ import annotations

import numpy as np


class ReferenceShapeError(ValueError):
    """Raised when a reference input violates the frozen tensor contract."""


def _require_finite(name: str, array: np.ndarray) -> None:
    if not np.all(np.isfinite(array)):
        raise ReferenceShapeError(f"{name} contains non-finite values")


def _reshape_memory_table(
    memory_table: np.ndarray,
    *,
    num_kv_heads: int,
    head_dim: int,
) -> np.ndarray:
    table = np.asarray(memory_table, dtype=np.float64)
    if table.ndim != 2:
        raise ReferenceShapeError("memory_table must have shape [vocab, kv_dim]")
    expected = num_kv_heads * head_dim
    if table.shape[1] != expected:
        raise ReferenceShapeError(
            f"memory_table kv_dim={table.shape[1]} but expected {expected}"
        )
    _require_finite("memory_table", table)
    return table.reshape(table.shape[0], num_kv_heads, head_dim)


def headwise_rmsnorm(
    x: np.ndarray,
    weight: np.ndarray,
    *,
    eps: float,
) -> np.ndarray:
    """Apply affine RMSNorm independently to the final/head dimension.

    The affine weight is shared across KV heads, matching the inspected
    upstream MemoryAttention use of RMSNorm(head_dim).
    """
    values = np.asarray(x, dtype=np.float64)
    scale = np.asarray(weight, dtype=np.float64)

    if values.ndim < 1:
        raise ReferenceShapeError("x must have at least one dimension")
    if scale.ndim != 1 or scale.shape[0] != values.shape[-1]:
        raise ReferenceShapeError(
            "norm weight must be one-dimensional with length equal to head_dim"
        )
    if eps <= 0 or not np.isfinite(eps):
        raise ReferenceShapeError("eps must be finite and > 0")

    _require_finite("x", values)
    _require_finite("weight", scale)

    rstd = 1.0 / np.sqrt(np.mean(np.square(values), axis=-1, keepdims=True) + eps)
    return values * rstd * scale


def lookup_memory(
    memory_table: np.ndarray,
    token_ids: np.ndarray,
    *,
    num_kv_heads: int,
    head_dim: int,
) -> np.ndarray:
    table = _reshape_memory_table(
        memory_table,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
    )
    ids = np.asarray(token_ids)
    if ids.ndim != 1:
        raise ReferenceShapeError("token_ids must be one-dimensional")
    if not np.issubdtype(ids.dtype, np.integer):
        raise ReferenceShapeError("token_ids must contain integers")
    if np.any(ids < 0) or np.any(ids >= table.shape[0]):
        raise ReferenceShapeError("token_ids contain an out-of-range address")
    return table[ids.astype(np.int64, copy=False)]


def fold_memory_table(
    memory_table: np.ndarray,
    norm_weight: np.ndarray,
    *,
    num_kv_heads: int,
    head_dim: int,
    eps: float,
) -> np.ndarray:
    """Precompute the inference-time normalized memory table."""
    table = _reshape_memory_table(
        memory_table,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
    )
    folded = headwise_rmsnorm(table, norm_weight, eps=eps)
    return folded.reshape(folded.shape[0], num_kv_heads * head_dim)


def construct_values(
    content_keys: np.ndarray,
    memory_table: np.ndarray,
    token_ids: np.ndarray,
    norm_weight: np.ndarray,
    *,
    eps: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Construct V = K_content + RMSNorm(M[token]).

    content_keys has shape [T, H_kv, D_h].
    Returns (values, normalized_selected_memory).
    """
    keys = np.asarray(content_keys, dtype=np.float64)
    if keys.ndim != 3:
        raise ReferenceShapeError("content_keys must have shape [T, H_kv, D_h]")
    _require_finite("content_keys", keys)

    t, num_kv_heads, head_dim = keys.shape
    ids = np.asarray(token_ids)
    if ids.shape != (t,):
        raise ReferenceShapeError(
            f"token_ids must have shape ({t},), got {ids.shape}"
        )

    selected = lookup_memory(
        memory_table,
        ids,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
    )
    normalized = headwise_rmsnorm(selected, norm_weight, eps=eps)
    return keys + normalized, normalized


def _rope_angles(
    positions: np.ndarray,
    *,
    head_dim: int,
    base: float,
) -> tuple[np.ndarray, np.ndarray]:
    if head_dim % 2 != 0:
        raise ReferenceShapeError("NeoX-style RoPE requires an even head_dim")
    if base <= 0 or not np.isfinite(base):
        raise ReferenceShapeError("RoPE base must be finite and > 0")

    pos = np.asarray(positions, dtype=np.float64)
    if pos.ndim != 1:
        raise ReferenceShapeError("positions must be one-dimensional")
    _require_finite("positions", pos)

    inv_freq = 1.0 / (
        base ** (np.arange(0, head_dim, 2, dtype=np.float64) / head_dim)
    )
    angle = pos[:, None] * inv_freq[None, :]
    cos = np.concatenate([np.cos(angle), np.cos(angle)], axis=-1)[:, None, :]
    sin = np.concatenate([np.sin(angle), np.sin(angle)], axis=-1)[:, None, :]
    return cos, sin


def rope_neox(
    x: np.ndarray,
    positions: np.ndarray,
    *,
    base: float = 10000.0,
) -> np.ndarray:
    """Apply GPT-NeoX-style non-interleaved RoPE to [T, H, D]."""
    values = np.asarray(x, dtype=np.float64)
    if values.ndim != 3:
        raise ReferenceShapeError("RoPE input must have shape [T, H, D]")
    _require_finite("RoPE input", values)

    t, _, head_dim = values.shape
    pos = np.asarray(positions)
    if pos.shape != (t,):
        raise ReferenceShapeError(f"positions must have shape ({t},)")

    cos, sin = _rope_angles(pos, head_dim=head_dim, base=base)
    half = head_dim // 2
    rotate_half = np.concatenate(
        [-values[..., half:], values[..., :half]],
        axis=-1,
    )
    return values * cos + rotate_half * sin


def inverse_rope_neox(
    x: np.ndarray,
    positions: np.ndarray,
    *,
    base: float = 10000.0,
) -> np.ndarray:
    """Invert the frozen NeoX-style RoPE convention."""
    values = np.asarray(x, dtype=np.float64)
    if values.ndim != 3:
        raise ReferenceShapeError("RoPE input must have shape [T, H, D]")
    t, _, head_dim = values.shape
    pos = np.asarray(positions)
    if pos.shape != (t,):
        raise ReferenceShapeError(f"positions must have shape ({t},)")

    cos, sin = _rope_angles(pos, head_dim=head_dim, base=base)
    half = head_dim // 2
    rotate_half = np.concatenate(
        [-values[..., half:], values[..., :half]],
        axis=-1,
    )
    return values * cos - rotate_half * sin
