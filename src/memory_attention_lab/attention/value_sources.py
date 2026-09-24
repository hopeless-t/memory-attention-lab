"""Independent value-source controls for VAL-002."""

from __future__ import annotations

import numpy as np


class ValueSourceError(ValueError):
    """Raised when VAL-002 array inputs violate the frozen contract."""


def _array(name: str, value: np.ndarray) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 3:
        raise ValueSourceError(f"{name} must have shape [T, H, D]")
    if not np.all(np.isfinite(array)):
        raise ValueSourceError(f"{name} contains non-finite values")
    return array


def _same_shape(**arrays: np.ndarray) -> dict[str, np.ndarray]:
    normalized = {name: _array(name, value) for name, value in arrays.items()}
    shapes = {array.shape for array in normalized.values()}
    if len(shapes) != 1:
        detail = {name: array.shape for name, array in normalized.items()}
        raise ValueSourceError(f"all value-source tensors must share shape: {detail}")
    return normalized


def historical_value_embedding(
    projected_value: np.ndarray,
    token_value_embedding: np.ndarray,
    lam: float,
) -> np.ndarray:
    """Source-faithful learned-mix form from the preserved ValueEmbed record."""
    arrays = _same_shape(
        projected_value=projected_value,
        token_value_embedding=token_value_embedding,
    )
    if isinstance(lam, bool) or not isinstance(lam, (int, float)):
        raise ValueSourceError("lambda must be a numeric scalar")
    lam64 = float(lam)
    if not np.isfinite(lam64):
        raise ValueSourceError("lambda must be finite")

    projected = arrays["projected_value"]
    token_value = arrays["token_value_embedding"]
    return (1.0 - lam64) * projected + lam64 * token_value


def factorial_value_sources(
    projected_value: np.ndarray,
    content_key: np.ndarray,
    memory_contribution: np.ndarray,
) -> dict[str, np.ndarray]:
    """Return the four-cell local causal control family.

    C01 is a local additive control, not the historical Value Embeddings method.
    """
    arrays = _same_shape(
        projected_value=projected_value,
        content_key=content_key,
        memory_contribution=memory_contribution,
    )
    p = arrays["projected_value"]
    k = arrays["content_key"]
    m = arrays["memory_contribution"]
    return {
        "c00_projected_no_memory": p,
        "c01_projected_plus_memory": p + m,
        "c10_key_no_memory": k,
        "c11_key_plus_memory": k + m,
    }


def factorial_effects(cells: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    required = {
        "c00_projected_no_memory",
        "c01_projected_plus_memory",
        "c10_key_no_memory",
        "c11_key_plus_memory",
    }
    if set(cells) != required:
        raise ValueSourceError(
            f"factorial cells must be exactly {sorted(required)}, got {sorted(cells)}"
        )
    arrays = _same_shape(**cells)
    c00 = arrays["c00_projected_no_memory"]
    c01 = arrays["c01_projected_plus_memory"]
    c10 = arrays["c10_key_no_memory"]
    c11 = arrays["c11_key_plus_memory"]
    return {
        "memory_effect_from_projected": c01 - c00,
        "memory_effect_from_key": c11 - c10,
        "source_effect_without_memory": c10 - c00,
        "source_effect_with_memory": c11 - c01,
        "additive_interaction": c11 - c10 - c01 + c00,
    }
