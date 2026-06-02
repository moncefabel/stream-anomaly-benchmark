# Abstract

## Benchmarking Early Time Series Classification Under the Earliness–Reliability–Stability Trilemma: Towards Heterogeneous Agent Orchestration

**Author:** Moncef Bouhabel
**Context:** Preparatory research — CIFRE PhD candidacy, Orange Innovation / EURECOM (ref. 2026-51517)

### Summary

Early classification of time series requires committing to a decision
before the full sequence is observed. In network anomaly detection and
cybersecurity incident triage, this creates a fundamental three-way
tension: decisions must be *early* (to enable rapid response), *reliable*
(to avoid costly false positives), and *stable* (to be trusted by
operators and downstream systems). We call this the
**earliness–reliability–stability trilemma**.

This work provides the first benchmark jointly evaluating two published
early classification methods — TEASER (Schäfer & Leser, 2020) and
ECONOMY-K (Achenchabe et al., 2021) — on all three trilemma dimensions
across five UCR time series datasets. We implement a unified evaluation
harness with reproducible seeds, a shared metric suite (earliness,
HM score, stability), and a threshold sensitivity analysis for ECONOMY-K.

**Main results:**

1. ECONOMY-K outperforms TEASER on 4 of 5 datasets by HM score
   (mean 0.837 vs 0.786), despite similar earliness (~0.87).
   No single method dominates across all datasets.

2. ECONOMY-K's 1-NN confidence metric saturates below τ = 0.50,
   creating a plateau in the Pareto frontier. The effective decision
   zone is τ ∈ [0.55, 0.70], motivating research on richer
   confidence estimators.

3. TEASER systematically abstains on partial series in small-sample
   settings (n < 100), rendering it ineffective as an incremental
   early classifier in low-data regimes typical of network monitoring.

4. Two-threshold orchestration (combining EK-Aggressive τ=0.40 and
   EK-Conservative τ=0.70) produces no HM gain over individual agents
   under OR, AND, or weighted-vote schemes. Genuine algorithmic
   diversity — not just parameter diversity — is required for
   non-trivial orchestration gains.

**Implications for multi-agent orchestration:**

These findings establish that (a) no static single-agent deployment
is optimal across operational contexts, and (b) homogeneous
agent ensembles cannot escape the trilemma through simple voting.
This motivates the thesis research agenda: designing self-adaptive,
heterogeneous multi-agent systems that dynamically negotiate their
cooperation scheme based on stream statistics, agent confidence signals,
and semantic contracts encoded in operational knowledge graphs.

### Keywords

Early time series classification · Anomaly detection · Multi-agent
systems · Trilemma · Pareto frontier · Confidence calibration ·
NetOps · ECONOMY-K · TEASER · UCR benchmark

### Repository

[github.com/moncefabel/stream-anomaly-benchmark](https://github.com/moncefabel/stream-anomaly-benchmark)

### References

- Schäfer, P., & Leser, U. (2020). TEASER: Early and Accurate Time
  Series Classification. *Data Mining and Knowledge Discovery*, 34,
  1598–1626.
- Achenchabe, Y., Bondu, A., Cornuéjols, A., & Dachraoui, A. (2021).
  Early Classification of Time Series. *Machine Learning*, 110,
  1481–1517.
- Mori, U., Mendiburu, A., Keogh, E., & Lozano, J. A. (2017).
  Reliable early classification of time series. *DMKD*, 31(1),
  233–263.
