"""Independent numerical value-source controls for VAL-002."""

from __future__ import annotations

import numpy as np

from memory_attention_lab.attention.reference import (
    ReferenceShapeError,
    headwise_rmsnorm,
    lookup_memory,
)


def linear_project(hidden_states: np.ndarray, weight: np.ndarray) -> np.ndarray:
    x = np.asarray(hidden_states, dtype=np.float64)
    w = np.asarray(weight, dtype=np.float64)

    if x.ndim != 2:
        raise ReferenceShapeError("hidden_states must have shape [T, d_model]")
    if w.ndim != 2:
        raise ReferenceShapeError("projection weight must be two-dimensional")
    if x.shape[1] != w.shape[0]:
        raise ReferenceShapeError(
            f"hidden width {x.shape[1]} does not match projection input {w.shape[0]}"
        )
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(w)):
        raise ReferenceShapeError("projection inputs must be finite")

    return x @ w


def normalized_token_memory(
    memory_table: np.ndarray,
    token_ids: np.ndarray,
    norm_weight: np.ndarray,
    *,
    num_kv_heads: int,
    head_dim: int,
    eps: float,
) -> np.ndarray:
    selected = lookup_memory(
        memory_table,
        token_ids,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
    )
    normalized = headwise_rmsnorm(selected, norm_weight, eps=eps)
    return normalized.reshape(normalized.shape[0], num_kv_heads * head_dim)


def factorial_value_cells(
    hidden_states: np.ndarray,
    wv: np.ndarray,
    wk: np.ndarray,
    memory_table: np.ndarray,
    token_ids: np.ndarray,
    norm_weight: np.ndarray,
    *,
    num_kv_heads: int,
    head_dim: int,
    eps: float,
) -> dict[str, np.ndarray]:
    v_proj = linear_project(hidden_states, wv)
    k_content = linear_project(hidden_states, wk)

    expected_width = num_kv_heads * head_dim
    if v_proj.shape[1] != expected_width:
        raise ReferenceShapeError(
            f"Wv output width {v_proj.shape[1]} does not match expected {expected_width}"
        )
    if k_content.shape[1] != expected_width:
        raise ReferenceShapeError(
            f"Wk output width {k_content.shape[1]} does not match expected {expected_width}"
        )

    memory = normalized_token_memory(
        memory_table,
        token_ids,
        norm_weight,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        eps=eps,
    )

    if memory.shape != v_proj.shape:
        raise ReferenceShapeError(
            f"memory shape {memory.shape} does not match projected value shape {v_proj.shape}"
        )

    return {
        "memory": memory,
        "C00": v_proj,
        "C01": v_proj + memory,
        "C10": k_content,
        "C11": k_content + memory,
    }


def historical_value_embedding(
    hidden_states: np.ndarray,
    wv: np.ndarray,
    historical_value_table: np.ndarray,
    token_ids: np.ndarray,
    *,
    lamb: float,
) -> np.ndarray:
    if isinstance(lamb, bool) or not isinstance(lamb, (int, float)):
        raise ReferenceShapeError("historical lambda must be numeric")
    lamb = float(lamb)
    if not np.isfinite(lamb):
        raise ReferenceShapeError("historical lambda must be finite")

    v_proj = linear_project(hidden_states, wv)
    table = np.asarray(historical_value_table, dtype=np.float64)
    ids = np.asarray(token_ids)

    if table.ndim != 2:
        raise ReferenceShapeError(
            "historical_value_table must have shape [vocab, value_width]"
        )
    if table.shape[1] != v_proj.shape[1]:
        raise ReferenceShapeError(
            "historical_value_table width must match Wv projection width"
        )
    if ids.ndim != 1 or ids.shape[0] != v_proj.shape[0]:
        raise ReferenceShapeError(
            "token_ids must be one-dimensional and match sequence length"
        )
    if not np.issubdtype(ids.dtype, np.integer):
        raise ReferenceShapeError("token_ids must contain integers")
    if np.any(ids < 0) or np.any(ids >= table.shape[0]):
        raise ReferenceShapeError("token_ids contain an out-of-range address")
    if not np.all(np.isfinite(table)):
        raise ReferenceShapeError("historical_value_table contains non-finite values")

    selected = table[ids.astype(np.int64, copy=False)]
    return (1.0 - lamb) * v_proj + lamb * selected
