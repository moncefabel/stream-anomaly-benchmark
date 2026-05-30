"""
evaluate.py
-----------
Unified evaluation pipeline for early time series classification.
Runs TEASER and ECONOMY-K on UCR datasets and computes
the earliness-reliability-stability trilemma metrics.
"""

from __future__ import annotations

import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from sktime.classification.early_classification import TEASER
from datasets import load_dataset, UCR_DATASETS
from metrics import evaluate, stability


# ── Format helper ─────────────────────────────────────────────────────────────

def to_3d(X: np.ndarray) -> np.ndarray:
    """
    Convert (n_samples, n_timepoints) → (n_samples, 1, n_timepoints).
    sktime TEASER expects 3D input.
    """
    return X[:, np.newaxis, :].astype(np.float64)


# ── TEASER ────────────────────────────────────────────────────────────────────

def run_teaser(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test:  np.ndarray,
    y_test:  np.ndarray,
    seeds:   list[int] = [42],
) -> dict:
    """Run TEASER and return trilemma metrics."""
    T = X_train.shape[1]
    X_tr = to_3d(X_train)
    X_te = to_3d(X_test)

    all_trigger_times = []
    last_preds = None

    for seed in seeds:
        model = TEASER(random_state=seed)
        model.fit(X_tr, y_train)

        # Recover trigger times via incremental predict
        trigger_times = np.full(len(y_test), float(T))
        classified    = np.zeros(len(y_test), dtype=bool)

        for t in range(1, T + 1):
            X_slice = X_te[:, :, :t]
            try:
                p = model.predict(X_slice)
                decided_now = (p != -1) & (~classified)
                trigger_times[decided_now] = t
                classified |= decided_now
                if classified.all():
                    break
            except Exception:
                continue

        preds = model.predict(X_te)
        all_trigger_times.append(trigger_times)
        last_preds = preds

    return evaluate(
        y_test, last_preds,
        all_trigger_times[-1], T,
        trigger_times_per_seed=all_trigger_times if len(all_trigger_times) > 1 else None,
    )


# ── ECONOMY-K (lightweight) ───────────────────────────────────────────────────

class EconomyK:
    """
    Simplified ECONOMY-K early classifier.

    Stopping rule: trigger when confidence of 1-NN prediction
    exceeds threshold. Confidence = 1 - d1/(d1+d2).

    Reference: Achenchabe et al. (2021), Machine Learning 110:1481-1517.
    """

    def __init__(self, threshold: float = 0.55, min_length: int = 3):
        self.threshold = threshold
        self.min_length = min_length
        self._train_X: np.ndarray | None = None
        self._train_y: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "EconomyK":
        self._train_X = X.astype(np.float64)
        self._train_y = y
        return self

    def _predict_one(self, x: np.ndarray) -> tuple[int, float]:
        t = len(x)
        dists = np.sqrt(np.sum(
            (x - self._train_X[:, :t]) ** 2, axis=1
        ))
        idx_sorted = np.argsort(dists)
        d1, d2 = dists[idx_sorted[0]], dists[idx_sorted[1]]
        label  = self._train_y[idx_sorted[0]]
        conf   = 1.0 - d1 / (d1 + d2 + 1e-10)
        return int(label), float(conf)

    def predict_with_triggers(
        self, X: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        n, T = X.shape
        predictions   = np.zeros(n, dtype=int)
        trigger_times = np.full(n, float(T))

        for i in range(n):
            decided = False
            for t in range(self.min_length, T + 1):
                label, conf = self._predict_one(X[i, :t])
                if conf >= self.threshold:
                    predictions[i]   = label
                    trigger_times[i] = t
                    decided = True
                    break
            if not decided:
                label, _ = self._predict_one(X[i])
                predictions[i] = label

        return predictions, trigger_times


def run_economy_k(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test:  np.ndarray,
    y_test:  np.ndarray,
    threshold: float     = 0.55,
    seeds:     list[int] = [42],
) -> dict:
    T = X_test.shape[1]
    all_trigger_times = []
    last_preds = None

    for _ in seeds:
        model = EconomyK(threshold=threshold)
        model.fit(X_train, y_train)
        preds, t_times = model.predict_with_triggers(X_test)
        all_trigger_times.append(t_times)
        last_preds = preds

    return evaluate(
        y_test, last_preds,
        all_trigger_times[-1], T,
        trigger_times_per_seed=all_trigger_times if len(all_trigger_times) > 1 else None,
    )


# ── Full benchmark ────────────────────────────────────────────────────────────

def benchmark(
    datasets: list[str] | None = None,
    seeds:    list[int]        = [42, 0, 1],
) -> pd.DataFrame:
    target = datasets or list(UCR_DATASETS.keys())
    rows   = []

    for name in target:
        print(f"\n{'='*55}")
        print(f"  Dataset: {name}")
        print(f"{'='*55}")

        try:
            X_train, y_train, X_test, y_test = load_dataset(name)
        except Exception as e:
            print(f"  Load error: {e}")
            continue

        print("  Running TEASER...", end=" ", flush=True)
        try:
            r = run_teaser(X_train, y_train, X_test, y_test, seeds=seeds)
            print(f"acc={r['accuracy']:.3f}  earl={r['earliness']:.3f}  hm={r['hm_score']:.3f}")
            rows.append({"dataset": name, "method": "TEASER", **r})
        except Exception as e:
            print(f"ERROR: {e}")

        print("  Running ECONOMY-K...", end=" ", flush=True)
        try:
            r = run_economy_k(X_train, y_train, X_test, y_test, seeds=seeds)
            print(f"acc={r['accuracy']:.3f}  earl={r['earliness']:.3f}  hm={r['hm_score']:.3f}")
            rows.append({"dataset": name, "method": "ECONOMY-K", **r})
        except Exception as e:
            print(f"ERROR: {e}")

    return pd.DataFrame(rows)


if __name__ == "__main__":
    FAST_DATASETS = ["ItalyPowerDemand", "ECG200", "SonyAIBORobotSurface1"]

    print("\n" + "="*55)
    print("  STREAM ANOMALY BENCHMARK — Pipeline validation")
    print("="*55)

    df = benchmark(datasets=FAST_DATASETS, seeds=[42, 0, 1])

    print("\n\n=== RESULTS ===\n")
    print(df.to_string(index=False))

    df.to_csv("../results/benchmark_results.csv", index=False)
    print("\nSaved → results/benchmark_results.csv")