# stream-anomaly-benchmark

> **Benchmarking early time series classification methods under the earliness–reliability–stability trilemma**

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status: Work in Progress](https://img.shields.io/badge/status-work%20in%20progress-orange.svg)]()

---

## Motivation

Early classification of time series asks a deceptively simple question:  
**_At what point in an incoming data stream can we make a reliable decision — and how early is early enough?_**

This question sits at the heart of network anomaly detection, predictive maintenance, and cybersecurity incident triage, where acting too late is costly but acting too early on a noisy signal is equally dangerous.

Existing methods optimize for **earliness** or **accuracy** in isolation. This benchmark explicitly targets the three-way trade-off:

| Dimension | Description |
|---|---|
| **Earliness** | How soon after stream onset is a decision triggered? |
| **Reliability** | How accurate is the decision at trigger time? |
| **Stability** | How consistent are decisions across random seeds and stream perturbations? |

No published benchmark jointly evaluates all three dimensions across methods and datasets. This repository fills that gap.

---

## Methods

| Method | Reference | Key idea |
|---|---|---|
| **TEASER** | Schäfer & Leser, 2020 | Two-tier early and accurate classifier with a non-myopic stopping rule |
| **ECONOMY-K** | Achenchabe et al., 2021 | Cost-sensitive stopping — triggers when expected future gain < current cost |

Both methods are available via [`sktime`](https://www.sktime.net/). This benchmark provides a unified evaluation harness with reproducible seeds, consistent train/test splits, and a shared metric suite.

---

## Datasets

All datasets are drawn from the [UCR Time Series Archive](https://www.cs.ucr.edu/~eamonn/time_series_data_2018/) (Dau et al., 2018).

Initial evaluation targets datasets covering the following domains, chosen for their relevance to network and infrastructure monitoring:

- **ECG / physiological signals** — high-frequency, safety-critical
- **Sensor fault detection** — industrial, class-imbalanced
- **Traffic and network traces** — variable-length, bursty

Full dataset list in [`datasets/README.md`](datasets/README.md) _(coming soon)_.

---

## Metrics

Beyond standard accuracy, this benchmark implements metrics that explicitly quantify the earliness–reliability trade-off:

```
Earliness(t)         = 1 - (t / T)           # fraction of series unseen at decision
HM_score             = 2 · (Acc · E) / (Acc + E)   # harmonic mean of accuracy and earliness
Stability(σ)         = std(trigger_time) across seeds   # decision time variance
```

These metrics are motivated by the framework of Mori et al. (2017) and extended toward stability analysis.

---

## Repository Structure

```
stream-anomaly-benchmark/
│
├── src/
│   ├── datasets.py          # UCR loader + preprocessing
│   ├── metrics.py           # Earliness, HM score, stability
│   └── evaluate.py          # Unified evaluation loop
│
├── experiments/
│   ├── teaser_baseline.py
│   └── economy_k_baseline.py
│
├── notebooks/
│   └── 01_trilemma_analysis.ipynb   # Main analysis notebook
│
├── results/
│   └── figures/
│
├── requirements.txt
└── README.md
```

---

## Roadmap

- [x] Repository structure and README
- [ ] UCR dataset loader (`sktime` + `aeon`)
- [ ] TEASER evaluation pipeline
- [ ] ECONOMY-K evaluation pipeline
- [ ] Unified metric suite (earliness, HM score, stability)
- [ ] Results tables and figures
- [ ] Analysis notebook
- [ ] Technical report (PDF)

---

## Scientific Context

This project is developed as preparatory work for a CIFRE doctoral thesis at **Orange Innovation** on the optimization of cybersecurity incident diagnostics under time and explainability constraints.

The earliness–reliability–stability trilemma maps directly onto the operational constraints of network anomaly detection:

- A **too-early** trigger fires on transient noise → false positive, wasted remediation effort
- A **too-late** trigger misses the early remediation window → incident escalates
- An **unstable** trigger produces inconsistent decisions across similar incidents → untrusted system

Understanding the Pareto frontier between these three objectives is a prerequisite for designing reliable multi-model orchestration systems for NetOps/SecOps.

---

## References

- Schäfer, P., & Leser, U. (2020). [TEASER: Early and Accurate Time Series Classification](https://doi.org/10.1007/s10618-020-00707-1). *Data Mining and Knowledge Discovery*, 34, 1598–1626.
- Achenchabe, Y., Bondu, A., Cornuéjols, A., & Dachraoui, A. (2021). [Early Classification of Time Series: Cost-Based Optimization Criterion and Algorithms](https://doi.org/10.1007/s10994-021-05974-z). *Machine Learning*, 110, 1481–1517.
- Mori, U., Mendiburu, A., Keogh, E., & Lozano, J. A. (2017). Reliable early classification of time series based on discriminating the classes over time. *Data Mining and Knowledge Discovery*, 31(1), 233–263.
- Dau, H. A., et al. (2018). [The UCR Time Series Archive](https://arxiv.org/abs/1810.07758). *IEEE/CAA Journal of Automatica Sinica*.

---

## Author

**Moncef Bouhabel** — ML Engineer, Master ML for Data Science, Université Paris Cité  
[github.com/moncefabel](https://github.com/moncefabel)
