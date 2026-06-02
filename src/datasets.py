
"""
datasets.py
UCR dataset loader using aeon. Returns 2D numpy arrays (n_samples, T).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from aeon.datasets import load_classification


# UCR datasets selected for relevance to network/infrastructure monitoring
UCR_DATASETS = {
    # Sensor / fault detection
    "FaceDetection":     {"domain": "sensor",  "classes": 2},
    "ElectricDevices":   {"domain": "sensor",  "classes": 7},
    "NonInvasiveFetalECGThorax1": {"domain": "ecg", "classes": 42},
    # ECG / physiological — safety-critical, high-frequency
    "ECG200":            {"domain": "ecg",     "classes": 2},
    "ECG5000":           {"domain": "ecg",     "classes": 5},
    # Traffic / activity
    "BasicMotions":      {"domain": "motion",  "classes": 4},
    "NATOPS":            {"domain": "motion",  "classes": 6},
    # Small / fast to test pipeline
    "ItalyPowerDemand":  {"domain": "energy",  "classes": 2},
    "SonyAIBORobotSurface1": {"domain": "robot", "classes": 2},
    "Wafer":             {"domain": "industrial", "classes": 2},
}


def load_dataset(name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load a UCR dataset by name. Returns X_train, y_train, X_test, y_test as 2D arrays."""
    if name not in UCR_DATASETS:
        raise ValueError(
            f"Dataset '{name}' not in benchmark suite.\n"
            f"Available: {list(UCR_DATASETS.keys())}"
        )

    print(f"  Loading {name}...", end=" ", flush=True)

    X_train_raw, y_train = load_classification(name, split="train")
    X_test_raw,  y_test  = load_classification(name, split="test")

    X_train = _to_2d(X_train_raw)
    X_test  = _to_2d(X_test_raw)

    # Encode labels as integers
    classes = np.unique(np.concatenate([y_train, y_test]))
    label_map = {c: i for i, c in enumerate(classes)}
    y_train = np.array([label_map[y] for y in y_train])
    y_test  = np.array([label_map[y] for y in y_test])

    print(
        f"OK  |  train={X_train.shape[0]}  test={X_test.shape[0]}"
        f"  T={X_train.shape[1]}  classes={len(classes)}"
    )
    return X_train, y_train, X_test, y_test


def _to_2d(X) -> np.ndarray:
    """
    Convert aeon panel format (n, 1, T) or DataFrame to (n, T) numpy array.
    """
    if isinstance(X, pd.DataFrame):
        # Nested DataFrame: each cell contains a pd.Series
        return np.stack(
            [np.concatenate([x.values for x in row]) for row in X.values]
        )
    arr = np.array(X)
    if arr.ndim == 3:
        # (n_samples, n_channels, n_timepoints) → take first channel
        return arr[:, 0, :]
    return arr


def dataset_summary() -> pd.DataFrame:
    """Print a summary table of all benchmark datasets."""
    rows = []
    for name, meta in UCR_DATASETS.items():
        try:
            X_train, y_train, X_test, y_test = load_dataset(name)
            rows.append({
                "dataset":  name,
                "domain":   meta["domain"],
                "n_train":  len(X_train),
                "n_test":   len(X_test),
                "T":        X_train.shape[1],
                "classes":  len(np.unique(y_train)),
            })
        except Exception as e:
            rows.append({"dataset": name, "error": str(e)})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    print("=== Dataset loading check ===\n")
    df = dataset_summary()
    print("\n", df.to_string(index=False))