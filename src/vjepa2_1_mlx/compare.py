from __future__ import annotations

import numpy as np


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = a.reshape(-1).astype(np.float64)
    b = b.reshape(-1).astype(np.float64)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        raise ValueError("cosine similarity undefined for zero vector")
    return float(np.dot(a, b) / denom)


def compare_features(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    diff = reference - candidate
    return {
        "cosine_similarity": cosine_similarity(reference, candidate),
        "max_abs_error": float(np.max(np.abs(diff))),
        "mean_abs_error": float(np.mean(np.abs(diff))),
        "sum_abs_error": float(np.sum(np.abs(diff))),
    }
