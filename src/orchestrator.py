"""
orchestrator.py
---------------
HeterogeneousOrchestrator: multi-agent cooperation prototype
for early time series classification.

Three cooperation schemes:
  OR       : trigger when EITHER agent decides     → aggressive, early
  AND      : trigger only when BOTH agree          → conservative, reliable
  WEIGHTED : weighted vote by training accuracy    → adaptive compromise

Thesis connection: Orange Innovation / EURECOM (ref. 2026-51517)
"""

from __future__ import annotations
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import sys
sys.path.insert(0, ".")

from sktime.classification.early_classification import TEASER
from evaluate import EconomyK, to_3d
from datasets import load_dataset, UCR_DATASETS
from metrics import evaluate


# ── Helpers ───────────────────────────────────────────────────────────────────

def _to_int_labels(arr: np.ndarray) -> np.ndarray:
    """
    Safely convert TEASER predictions (may be strings or floats) to int.
    Returns -1 for abstain / unparseable values.
    """
    result = np.full(len(arr), -1, dtype=int)
    for i, v in enumerate(arr):
        try:
            val = int(float(str(v)))
            result[i] = val
        except (ValueError, TypeError):
            result[i] = -1
    return result


# ── Orchestrator ──────────────────────────────────────────────────────────────

class HeterogeneousOrchestrator:
    """
    Multi-agent early classifier: TEASER + ECONOMY-K.

    At each timestep t, both agents independently decide.
    The orchestrator applies a cooperation scheme.

    Schemes
    -------
    OR       : either agent triggers  → max earliness
    AND      : both must agree        → max reliability
    WEIGHTED : weighted vote          → adaptive balance
    """

    SCHEMES = ("OR", "AND", "WEIGHTED")

    def __init__(
        self,
        scheme: str = "WEIGHTED",
        economy_threshold: float = 0.55,
        random_state: int = 42,
    ):
        assert scheme in self.SCHEMES, f"scheme must be in {self.SCHEMES}"
        self.scheme            = scheme
        self.economy_threshold = economy_threshold
        self.random_state      = random_state
        self._teaser           = TEASER(random_state=random_state)
        self._economy          = EconomyK(threshold=economy_threshold)
        self._w_teaser         = 0.5
        self._w_economy        = 0.5

    def fit(self, X: np.ndarray, y: np.ndarray) -> "HeterogeneousOrchestrator":
        X3d = to_3d(X)
        self._teaser.fit(X3d, y)
        self._economy.fit(X, y)
        # Equal weights — no bias toward either agent before deployment
        self._w_teaser  = 0.5
        self._w_economy = 0.5
        print(f"    Agents fitted — scheme: {self.scheme}")
        return self

    def predict_with_triggers(
        self, X: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        n, T   = X.shape
        X3d    = to_3d(X)

        predictions   = np.zeros(n, dtype=int)
        trigger_times = np.full(n, float(T))
        classified    = np.zeros(n, dtype=bool)

        # Pre-compute all ECONOMY-K decisions
        econ_preds, econ_triggers = self._economy.predict_with_triggers(X)
        econ_preds = econ_preds.astype(int)

        for t in range(1, T + 1):
            if classified.all():
                break

            # ECONOMY-K: has this agent decided by timestep t?
            e_decided = (econ_triggers <= t) & (~classified)   # bool array

            # TEASER: get prediction for partial series
            try:
                raw = self._teaser.predict(X3d[:, :, :t])
                t_preds   = _to_int_labels(raw)                # int array
                t_decided = (t_preds != -1) & (~classified)    # bool array
            except Exception:
                t_preds   = np.full(n, -1, dtype=int)
                t_decided = np.zeros(n, dtype=bool)

            # Apply cooperation scheme
            if self.scheme == "OR":
                fire = (e_decided | t_decided)
                for i in np.where(fire)[0]:
                    # Use TEASER if it decided, else ECONOMY-K
                    predictions[i]   = int(t_preds[i]) if bool(t_decided[i]) else int(econ_preds[i])
                    trigger_times[i] = t
                    classified[i]    = True

            elif self.scheme == "AND":
                # Both must decide AND agree on the label
                both = e_decided & t_decided
                for i in np.where(both)[0]:
                    t_lbl = int(t_preds[i])
                    e_lbl = int(econ_preds[i])
                    if t_lbl == e_lbl:
                        predictions[i]   = t_lbl
                        trigger_times[i] = t
                        classified[i]    = True
                    # If they disagree: wait for next timestep

            elif self.scheme == "WEIGHTED":
                for i in range(n):
                    if bool(classified[i]):
                        continue
                    t_dec = bool(t_decided[i])
                    e_dec = bool(e_decided[i])
                    if not t_dec and not e_dec:
                        continue

                    # Accumulate weighted votes per label
                    votes: dict[int, float] = {}
                    if t_dec and int(t_preds[i]) != -1:
                        lbl = int(t_preds[i])
                        votes[lbl] = votes.get(lbl, 0.0) + self._w_teaser
                    if e_dec:
                        lbl = int(econ_preds[i])
                        votes[lbl] = votes.get(lbl, 0.0) + self._w_economy

                    if votes and max(votes.values()) >= 0.5:
                        winner = max(votes, key=lambda k: votes[k])
                        predictions[i]   = int(winner)
                        trigger_times[i] = t
                        classified[i]    = True

        # Fallback: still-unclassified → ECONOMY-K full series
        unclassified = ~classified
        if unclassified.any():
            predictions[unclassified] = econ_preds[unclassified]

        return predictions, trigger_times


# ── Run helpers ───────────────────────────────────────────────────────────────

def run_orchestrator(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test:  np.ndarray,
    y_test:  np.ndarray,
    scheme:  str        = "WEIGHTED",
    seeds:   list[int]  = [42],
) -> dict:
    T = X_test.shape[1]
    all_tt, last_preds = [], None

    for seed in seeds:
        orch = HeterogeneousOrchestrator(scheme=scheme, random_state=seed)
        orch.fit(X_train, y_train)
        preds, tt = orch.predict_with_triggers(X_test)
        all_tt.append(tt)
        last_preds = preds

    return evaluate(
        y_test, last_preds, all_tt[-1], T,
        trigger_times_per_seed=all_tt if len(all_tt) > 1 else None,
    )


def benchmark_orchestrator(
    datasets: list[str] | None = None,
    seeds:    list[int]        = [42, 0, 1],
) -> pd.DataFrame:
    from evaluate import run_teaser, run_economy_k
    target = datasets or list(UCR_DATASETS.keys())
    rows   = []

    for name in target:
        print(f"\n{'='*60}")
        print(f"  Dataset: {name}")
        print(f"{'='*60}")

        try:
            X_train, y_train, X_test, y_test = load_dataset(name)
        except Exception as e:
            print(f"  Load error: {e}"); continue

        for method, fn in [
            ("TEASER",    lambda: run_teaser(X_train, y_train, X_test, y_test, seeds=seeds)),
            ("ECONOMY-K", lambda: run_economy_k(X_train, y_train, X_test, y_test, seeds=seeds)),
        ]:
            print("  [Baseline] ECONOMY-K...", end=" ", flush=True)
            try:
                r = run_economy_k(X_train, y_train, X_test, y_test, seeds=seeds)
                print(f"acc={r['accuracy']:.3f}  earl={r['earliness']:.3f}  hm={r['hm_score']:.3f}")
                rows.append({"dataset": name, "method": "ECONOMY-K", **r})
            except Exception as e:
                print(f"ERROR: {e}")

        for scheme in ["OR", "AND", "WEIGHTED"]:
            print(f"  [Orch-{scheme}]...", end=" ", flush=True)
            try:
                r = run_orchestrator(X_train, y_train, X_test, y_test, scheme=scheme, seeds=seeds)
                print(f"acc={r['accuracy']:.3f}  earl={r['earliness']:.3f}  hm={r['hm_score']:.3f}")
                rows.append({"dataset": name, "method": f"Orch-{scheme}", **r})
            except Exception as e:
                print(f"ERROR: {e}")

    return pd.DataFrame(rows)


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    DATASETS = ["ItalyPowerDemand", "ECG200", "SonyAIBORobotSurface1",
                "ECG5000", "Wafer"]

    print("\n" + "="*60)
    print("  HETEROGENEOUS ORCHESTRATOR")
    print("  TEASER + ECONOMY-K → AND / OR / WEIGHTED")
    print("="*60)

    df = benchmark_orchestrator(datasets=DATASETS, seeds=[42, 0, 1])

    print("\n\n=== RESULTS ===\n")
    print(df.to_string(index=False))
    df.to_csv("../results/orchestrator_results.csv", index=False)
    print("\nSaved → results/orchestrator_results.csv")

    print("\n=== MEAN HM SCORE BY METHOD ===\n")
    summary = df.groupby("method")["hm_score"].mean().sort_values(ascending=False)
    print(summary.round(4).to_string())

    # Key question
    solo  = df[df["method"].isin(["TEASER","ECONOMY-K"])].groupby("method")["hm_score"].mean().max()
    orch  = df[df["method"].str.startswith("Orch")].groupby("method")["hm_score"].mean().max()
    delta = orch - solo
    print(f"\n=== Does orchestration beat solo agents? ===")
    print(f"  Best solo HM  : {solo:.4f}")
    print(f"  Best orch HM  : {orch:.4f}")
    print(f"  Delta         : {delta:+.4f}  →  {'YES' if delta > 0 else 'NO'}")