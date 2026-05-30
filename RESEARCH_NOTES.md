# Research Notes & Experimental Log

> This file documents experimental decisions, failed strategies, and pivots
> made during the development of this benchmark.
> It is intentionally kept honest — negative results are results.

---

## Session 1 — 2026-05-31

### Goal
Build a reproducible benchmark comparing TEASER and ECONOMY-K on the
earliness–reliability–stability trilemma, and prototype a heterogeneous
orchestration layer combining both agents.

---

### Experiment 1 — ECONOMY-K Confidence Threshold

**Strategy:** Use `conf = 1 - d1 / (d1 + d2)` as stopping criterion for 1-NN.

**Finding:** The metric saturates below τ = 0.50. For all datasets,
thresholds in [0.10, 0.50] produce identical results — the confidence
always exceeds 0.50 at the minimum series length (t = min_length = 3).

**Interpretation:** The 1-NN ratio confidence is a weak estimator.
It saturates because nearest-neighbour distances are rarely balanced
(d1 << d2 for most samples), giving artificially high confidence values
from the very first timestep.

**Implication for thesis:** Simple confidence metrics are insufficient
for adaptive stopping rules in production. Richer estimators — Bayesian
posteriors, ensemble-based confidence, or calibrated probability outputs
— are needed. This motivates one of the core research axes.

**Pivot:** Threshold range shifted to [0.10, 0.95] to visualise the
saturation plateau explicitly. The effective decision zone identified
as τ ∈ [0.55, 0.70].

---

### Experiment 2 — TEASER in Heterogeneous Orchestrator

**Strategy:** Combine TEASER + ECONOMY-K in an orchestrator with three
cooperation schemes (OR, AND, WEIGHTED). TEASER provides one vote,
ECONOMY-K provides another.

**Implementation:** At each timestep t, query TEASER incrementally
(`predict` on partial series `X[:, :, :t]`). TEASER returns -1 for
abstaining samples and a class label when confident.

**Finding:** TEASER returns -1 for virtually all samples across all
timesteps on all tested datasets. It only classifies at t = T (full
series). As a result:
- OR scheme → degenerates to ECONOMY-K alone (TEASER never triggers first)
- AND scheme → earliness = 0 (TEASER never agrees, so no sample is
  classified early)
- WEIGHTED scheme → degenerates to ECONOMY-K alone

**Root cause:** TEASER's internal OCSVM (One-Class SVM) stopping rule
is calibrated on the full training distribution. On small training sets
(n < 100), the OCSVM rejects almost all partial series predictions,
causing systematic abstention until the full series is observed.
TEASER is not designed for low-data regimes with short series.

**Negative result value:** This finding is scientifically meaningful.
It shows that TEASER's non-myopic stopping rule, despite its theoretical
appeal, fails to generalise to small-sample / short-series settings —
which are precisely the conditions encountered in real-time network
anomaly detection (few labelled examples, short observation windows).

---

### Experiment 3 — Two-Threshold Orchestrator (Pivot)

**Strategy:** Replace TEASER with a second ECONOMY-K agent operating
at a different threshold, creating two agents with explicitly different
risk tolerances:

- **Agent A (Aggressive):** threshold = 0.40 → decides early, accepts
  lower confidence
- **Agent B (Conservative):** threshold = 0.70 → waits for high
  confidence, more accurate

**Scientific motivation:** This directly models the multi-agent
orchestration problem in the thesis — agents with heterogeneous
stopping criteria must cooperate to produce a collective decision.
The threshold encodes the agent's "risk tolerance", analogous to
a decision-maker's cost function in ECONOMY-K's original formulation.

**Cooperation schemes applied:**
- **OR:** trigger when either agent decides → maximum earliness
- **AND:** trigger when both agree on the same label → maximum reliability
- **WEIGHTED:** vote weighted by per-agent validation accuracy

**Expected outcome:** OR ≈ Agent A (earliness dominated by aggressive
agent). AND ≈ Agent B accuracy with lower earliness. WEIGHTED between
the two — tunable operating point on the Pareto frontier.

**Status:** Running — results pending.

---

### Open Questions (to be addressed in thesis)

1. **Confidence estimation:** What is the right confidence metric for
   adaptive early classification in streaming settings with concept drift?

2. **Calibration under small samples:** TEASER's OCSVM fails on
   n < 100. How do calibrated neural early classifiers (e.g., CALIMERA,
   ShapeNet) compare in low-data regimes?

3. **Dynamic weight adaptation:** The WEIGHTED scheme uses static
   training-time weights. How should weights be updated online as the
   stream distribution shifts?

4. **Semantic cooperation medium:** Can a knowledge graph (OWL/SPARQL)
   serve as the formal cooperation contract between agents — replacing
   ad-hoc voting with structured semantic reasoning?

5. **Causal vs correlational triggers:** Can we distinguish between
   "anomaly A caused anomaly B" and "A and B share a common cause"
   using the graph topology alone, without explicit causal structure
   learning?

---

## Next Steps

- [ ] Run two-threshold orchestrator on 5 UCR datasets
- [ ] Add calibrated confidence estimation (sigmoid-transformed distances)
- [ ] Extend benchmark to 10+ datasets for robustness
- [ ] Formalise the cooperation scheme as a decision-theoretic framework
- [ ] Connect to NORIA-O ontology as semantic coordination layer