"""
metrics.py
Earliness, accuracy, HM score, and stability for early classification.
"""

from __future__ import annotations

import numpy as np


def earliness(trigger_times: np.ndarray, series_length: int) -> float:
    """Mean earliness: 1 - mean(t / T). Higher = earlier decisions."""
    return float(np.mean(1.0 - trigger_times / series_length))


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Standard classification accuracy."""
    return float(np.mean(y_true == y_pred))


def harmonic_mean_score(acc: float, earl: float) -> float:
    """HM = 2*(acc*earl)/(acc+earl). Forces a model to be both accurate and early."""
    if acc + earl == 0:
        return 0.0
    return 2.0 * (acc * earl) / (acc + earl)


def stability(trigger_times_per_seed: list[np.ndarray]) -> float:
    """Mean std of trigger times across seeds. Lower = more stable."""
    if len(trigger_times_per_seed) < 2:
        return 0.0
    matrix = np.stack(trigger_times_per_seed, axis=0)  # (n_seeds, n_samples)
    return float(np.mean(np.std(matrix, axis=0)))


def evaluate(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    trigger_times: np.ndarray,
    series_length: int,
    trigger_times_per_seed: list[np.ndarray] | None = None,
) -> dict:
    """Compute all trilemma metrics for one (method, dataset) run."""
    acc  = accuracy(y_true, y_pred)
    earl = earliness(trigger_times, series_length)
    hm   = harmonic_mean_score(acc, earl)
    stab = (
        stability(trigger_times_per_seed)
        if trigger_times_per_seed is not None
        else None
    )

    return {
        "accuracy":  round(acc,  4),
        "earliness": round(earl, 4),
        "hm_score":  round(hm,   4),
        "stability": round(stab, 4) if stab is not None else None,
    }


if __name__ == "__main__":
    print("=== Metrics sanity check ===\n")

    T = 100
    n = 200
    rng = np.random.default_rng(42)

    # Simulate a model that triggers at t=30 on average
    y_true = rng.integers(0, 2, size=n)
    y_pred = y_true.copy()
    y_pred[rng.random(n) < 0.1] ^= 1      # ~10% errors
    trigger_times = rng.integers(25, 35, size=n)

    results = evaluate(y_true, y_pred, trigger_times, T)

    print(f"Accuracy  : {results['accuracy']:.4f}  (expected ~0.90)")
    print(f"Earliness : {results['earliness']:.4f}  (expected ~0.70)")
    print(f"HM score  : {results['hm_score']:.4f}")

    # Stability across 5 seeds
    seeds_trigger_times = [
        rng.integers(25, 35, size=n) for _ in range(5)
    ]
    stab = stability(seeds_trigger_times)
    print(f"Stability : {stab:.4f}  (lower = more stable)")

    print("\nAll metrics OK.")