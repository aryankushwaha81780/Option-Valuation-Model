# Option Valuation Model
## Authors
- [Aryan Kushwaha](https://github.com/aryankushwaha81780) and
 [Divyansh Bansal](https://github.com/bansaldiv)
## Overview

This repository, **option-valuation-model**, delivers an end-to-end **option pricing, risk analytics, and simulation framework** aligned with institutional trading and risk-management workflows. The model integrates analytical valuation, hedging intelligence, scenario stress testing, and Monte Carlo simulation to provide a forward-looking and decision-oriented view of option behavior.

The implementation is intentionally modular, transparent, and reproducible, making it suitable for academic evaluation, portfolio demonstrations, and quantitative finance prototyping.

---

## Model Explanation

The model is built around the **Black–Scholes–Merton (BSM)** framework and is executed in three strategic layers:

### 1. Market Data & Calibration

* Pulls one year of historical equity price data using Yahoo Finance
* Computes log returns from closing prices
* Estimates **annualized historical volatility (HV₍₂₅₂₎)**
* Anchors valuation to the latest observed spot price

### 2. Analytical Valuation & Risk Decomposition

* Prices European call and put options using closed-form BSM equations
* Computes key option Greeks:

  * **Delta** – directional exposure
  * **Gamma** – convexity and rebalancing risk
  * **Vega** – sensitivity to volatility changes
  * **Theta** – daily time decay
* Extracts **implied volatility** using Newton–Raphson iteration (when market price is supplied)

### 3. Strategy, Stress Testing & Simulation

* Generates **delta-neutral hedging recommendations** at the contract level
* Produces a **spot–volatility sensitivity matrix** for scenario analysis
* Simulates forward stock prices using **Geometric Brownian Motion (GBM)**
* Reprices the option dynamically until expiry, illustrating time decay and intrinsic value convergence

This architecture mirrors how professional trading desks transition from pricing to risk control and forward scenario planning.

---

## Algorithm Design

The algorithmic flow implemented in code is summarized in the following pseudocode flowchart:

👉 **Algorithm Design Diagram:**
[View Pseudocode Flowchart](Assets/Pseudocode_Flowchart.png)

---

## Prerequisites

### Required Python Libraries

The model requires the following dependencies:

* `numpy` – numerical computation
* `pandas` – time-series and tabular analysis
* `yfinance` – market data ingestion
* `scipy` – statistical distributions and math utilities
* `matplotlib` – data visualization
* `seaborn` – visualization styling
* `datetime` – date arithmetic

Install all dependencies using:

```bash
pip install numpy pandas yfinance scipy matplotlib seaborn
```

---

## Output

The model generates **both numerical and visual outputs**, all of which are reproducible from a single execution.

### Console Output (Analytics & Risk)
👉 [View Console Output](Assets/Console_Output.png)


### Visual Outputs
  1. Historical Market Context: Displays one year of real market data with the strike price overlaid.
  2. Simulated Future Stock Path: Shows the Monte Carlo–simulated stock trajectory until option expiry.
  3. Option Valuation & Time Decay: Illustrates BSM option value convergence toward intrinsic value as expiry approaches.

👉 [View Graph Output](Assets/Graph_Output.png)

---

## Disclaimer

This project is intended **solely for educational and analytical purposes**. It does not constitute financial advice. Market assumptions such as constant volatility and lognormal price dynamics may not hold under real-world stress conditions.

---
