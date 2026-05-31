"""
orchestrator2_twothreshold.py
-----------------------------
Two-threshold ECONOMY-K orchestration experiment.

Replaces TEASER (which systematically abstains on small datasets)
with two ECONOMY-K agents operating at different confidence thresholds,
modeling agents with explicitly different risk tolerances.

Agent A (Aggressive)  : threshold = 0.40 -> triggers early, accepts lower confidence
Agent B (Conservative): threshold = 0.70 -> waits for high confidence, more accurate

Three cooperation schemes:
  OR       : trigger when either agent decides    -> max earliness
  AND      : trigger when both agree on the label -> max reliability
  WEIGHTED : take first-to-trigger agent          -> follows aggressive agent

Key finding: With homogeneous agent types (same algorithm, different
thresholds), no cooperation scheme outperforms individual agents on HM
score. OR collapses to EK-Aggressive. AND collapses to EK-Conservative
with lower earliness. WEIGHTED degenerates to OR.

This finding motivates the thesis research question: genuine agent
diversity (different algorithms, different reasoning paradigms) is
required for non-trivial orchestration gains.

See RESEARCH_NOTES.md — Experiment 3 for full analysis.
"""

from __future__ import annotations

import sys
import warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")

import numpy as np
import pandas as pd
from datasets import load_dataset, UCR_DATASETS
from evaluate import EconomyK
from metrics import evaluate

# ── Configuration ─────────────────────────────────────────────────────────────

THRESHOLD_A = 0.40   # aggressive agent
THRESHOLD_B = 0.70   # conservative agent
DATASETS    = ["ItalyPowerDemand", "ECG200", "SonyAIBORobotSurface1",
               "ECG5000", "Wafer"]


# ── Cooperation schemes ───────────────────────────────────────────────────────

def orchestrate(
    pA: np.ndarray, tA: np.ndarray,
    pB: np.ndarray, tB: np.ndarray,
    T:  int,
    scheme: str,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Combine two agent decisions under a cooperation scheme.

    Parameters
    ----------
    pA, tA : predictions and trigger times of Agent A (aggressive)
    pB, tB : predictions and trigger times of Agent B (conservative)
    T      : full series length
    scheme : 'OR', 'AND', or 'WEIGHTED'

    Returns
    -------
    preds, triggers : final predictions and trigger times
    """
    n = len(pA)
    preds    = np.zeros(n, dtype=int)
    triggers = np.full(n, float(T))

    for i in range(n):
        a_first = tA[i] <= tB[i]

        if scheme == "OR":
            # Trigger on first available decision
            if a_first:
                preds[i], triggers[i] = int(pA[i]), tA[i]
            else:
                preds[i], triggers[i] = int(pB[i]), tB[i]

        elif scheme == "AND":
            # Trigger only when both agree on the same label
            if int(pA[i]) == int(pB[i]):
                preds[i]    = int(pA[i])
                triggers[i] = max(tA[i], tB[i])  # wait for the slower agent
            else:
                # Disagreement: fallback to full series (conservative label)
                preds[i]    = int(pB[i])
                triggers[i] = float(T)

        elif scheme == "WEIGHTED":
            # Take first-to-trigger agent's label
            # Note: degenerates to OR in current implementation
            # (see RESEARCH_NOTES.md for discussion)
            if tA[i] < tB[i]:
                preds[i], triggers[i] = int(pA[i]), tA[i]
            elif tB[i] < tA[i]:
                preds[i], triggers[i] = int(pB[i]), tB[i]
            else:
                # Same trigger time: use consensus or conservative
                preds[i]    = int(pA[i]) if int(pA[i]) == int(pB[i]) else int(pB[i])
                triggers[i] = tA[i]

    return preds, triggers


# ── Main benchmark ────────────────────────────────────────────────────────────

def run() -> pd.DataFrame:
    rows = []

    for name in DATASETS:
        print(f"\n{'='*55}")
        print(f"  Dataset: {name}")
        print(f"{'='*55}")

        X_train, y_train, X_test, y_test = load_dataset(name)
        T = X_test.shape[1]

        # Fit both agents
        mA = EconomyK(threshold=THRESHOLD_A)
        mA.fit(X_train, y_train)
        pA, tA = mA.predict_with_triggers(X_test)

        mB = EconomyK(threshold=THRESHOLD_B)
        mB.fit(X_train, y_train)
        pB, tB = mB.predict_with_triggers(X_test)

        # Cooperation schemes
        for scheme in ["OR", "AND", "WEIGHTED"]:
            preds, triggers = orchestrate(pA, tA, pB, tB, T, scheme)
            r = evaluate(y_test, preds, triggers, T)
            print(
                f"  Orch2-{scheme:<8}  "
                f"acc={r['accuracy']:.3f}  "
                f"earl={r['earliness']:.3f}  "
                f"hm={r['hm_score']:.3f}"
            )
            rows.append({
                "dataset": name, "method": f"Orch2-{scheme}",
                "threshold_A": THRESHOLD_A, "threshold_B": THRESHOLD_B,
                **r,
            })

        # Baselines
        for th, label in [(THRESHOLD_A, "EK-Aggressive"),
                          (THRESHOLD_B, "EK-Conservative")]:
            m = EconomyK(threshold=th)
            m.fit(X_train, y_train)
            p, t = m.predict_with_triggers(X_test)
            r = evaluate(y_test, p, t, T)
            print(
                f"  {label:<17}  "
                f"acc={r['accuracy']:.3f}  "
                f"earl={r['earliness']:.3f}  "
                f"hm={r['hm_score']:.3f}"
            )
            rows.append({
                "dataset": name, "method": label,
                "threshold_A": th, "threshold_B": th,
                **r,
            })

    return pd.DataFrame(rows)


if __name__ == "__main__":
    print("\n" + "="*55)
    print("  TWO-THRESHOLD ORCHESTRATION EXPERIMENT")
    print(f"  Agent A (Aggressive)   : τ = {THRESHOLD_A}")
    print(f"  Agent B (Conservative) : τ = {THRESHOLD_B}")
    print("="*55)

    df = run()

    print("\n\n=== MEAN HM SCORE BY METHOD ===\n")
    summary = df.groupby("method")["hm_score"].mean().sort_values(ascending=False)
    print(summary.round(4).to_string())

    df.to_csv("../results/orchestrator2_results.csv", index=False)
    print("\nSaved → results/orchestrator2_results.csv")

    print("\n=== KEY FINDING ===")
    solo = df[df["method"].isin(["EK-Aggressive","EK-Conservative"])]\
             .groupby("method")["hm_score"].mean().max()
    orch = df[df["method"].str.startswith("Orch2")]\
             .groupby("method")["hm_score"].mean().max()
    print(f"  Best solo HM  : {solo:.4f}")
    print(f"  Best orch HM  : {orch:.4f}")
    print(f"  Delta         : {orch-solo:+.4f}")
    print(
        "\n  Conclusion: homogeneous agents (same algorithm, different"
        "\n  thresholds) produce no HM gain through orchestration."
        "\n  Genuine algorithmic diversity is required."
        "\n  → See RESEARCH_NOTES.md, Experiment 3."
    )