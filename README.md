# Dynamic Correlation for Risk Assessment in Finance
This repository contains a **multivariate econometric study** that has been applied to the financial risk management of an equally-weighted (EW) portfolio.

Starting with a set of securities selected through a _machine learning_ (ML) technique and an aggregated macroeconomic-financial index, the study examines the _evolution of the pairwise correlations_ between the purpose-built index and each security over a 5-year horizon (2019-2024), and applies the final results to estimate the _Value at Risk_ (VaR) and the _Expected Shortfall_ (ES) of an EW portfolio.


## Repository Structure
The repository is organized into the following components:
- `code/`, which contains all the scripts used for data processing, clustering, econometric modelling and portfolio risk estimation;

The structure is shown below:
```
.
├── code/
│   ├── macro_index_pipeline/
│   │   ├── nasdaq_clustering_pipeline.py   # Clustering of NASDAQ-100 using KMeans
│   │   └── macro_index_builder.py          # Macro-financial index with Python/arch package
│   │
│   ├── dcc_correlation_models/
│   │   ├── main_driver.py                  # Master script for DCC/VAR pipeline
│   │   ├── compute_rolling.py              # Rolling correlation benchmark
│   │   ├── estimate_all_models.py          # DCC, GJR, TARCH, ADCC variants
│   │   ├── model_selection.py              # RMSE-based model selection
│   │   ├── compare_with_var.py             # VAR-DCC implementation + LRT
│   │   ├── generate_plots.py               # Plot generation (figures are no longer stored)
│   │   └── utils.py                        # Shared estimation helpers
│   │
│   └── tail_risk_lab/
│       ├── portfolio_allocator.py          # Portfolio P&L based on equal weights
│       ├── tail_risk_engine.py             # Dynamic VaR and ES estimation
│       └── tail_risk_breaches.py           # VaR and ES exceedance visualization
└── README.md                               # This file
```
Generated figures and supporting documents are no longer versioned. Run the provided scripts to recreate plots locally when required.


## How to Run

### Requirements
- **Python 3.11** (or later recommended)
  - Suggested stack: `pandas`, `numpy`, `scipy`, `statsmodels`, `arch`, `matplotlib`.
  - [Anaconda](https://www.anaconda.com/) (recommended for environment management)  
  - IDE: Spyder / VS Code / Jupyter (pick your preference).
  - Install dependencies via `pip install -r requirements.txt` once created, or install the packages manually.


## Main Outputs
The repository produces key outputs based on econometric modelling and dynamic risk analysis. Results are presented from two perspectives:

**1. Econometric Insights (DCC-GARCH vs. VAR-DCC-GARCH)**

The dynamic conditional correlations estimated through the DCC(1,1) and VAR(1)-DCC(1,1) models show substantial convergence over time. Removing a brief high-volatility segment from the sample:
- **_improves the overall stability of correlations_**, with smoother transitions and fewer spikes;
- **_reduces frequent regime shifts_** between positive and negative relationships;
- allows for **_more accurate modelling_** of second-order conditional moments.

Even when exogenous shocks are muted, the trends across models remain largely overlapping, but the trimmed sample displays less noisy dynamics and more consistent correlation patterns. Use `code/dcc_correlation_models/generate_plots.py` (after running the pipeline) to regenerate these correlation charts.

**2. Risk Measures: VaR and ES (Full vs. Trimmed Sample)**

The model provides a set of outputs illustrating the dynamic estimation of _daily_ VaR and ES, derived through a DCC-GARCH framework integrated with Markowitz Portfolio Theory (MPT).
Comparing the full dataset with the volatility-trimmed subset highlights that:

- **risk exposure rises** once the short volatility burst is omitted, revealing latent vulnerabilities that are otherwise masked in the aggregate sample;
- a more **_conservative risk profile_** is detected by Expected Shortfall (ES), especially during stress conditions;
- **_ES remains more informative_** than VaR for tail-risk monitoring across both samples.

| Measure   | Full Sample | Trimmed Sample |
|-----------|-------------|----------------|
| **VaR₉₅%**  | −271.34     | −338.05        |
| **VaR₉₉%**  | −378.82     | −466.59        |
| **ES₉₅%**   | −337.24     | −416.86        |
| **ES₉₉%**   | −423.27     | −530.50        |

Visualizations of loss comparisons and breach diagnostics can be recreated locally by re-running `tail_risk_breaches.py` and the associated plotting routines inside `code/tail_risk_lab/`.

## Limits

Despite the robustness of the implemented methodology, several limitations must be acknowledged:

- _Variable Scope:_ the inclusion/exclusion of additional macro-financial variables could significantly alter correlation dynamics and risk estimations.

- _Computational Burden:_ DCC-GARCH models are computationally intensive, especially in high-dimensional settings or during extended sample periods.

- _Normality Assumption:_ the model relies on the assumption of multivariate normality, which may not fully capture the fat tails and asymmetries observed in financial returns.

- _Simplified Portfolio Construction:_ the portfolio assumes equal weights and ignores transaction costs, potentially oversimplifying real-world investment conditions.
