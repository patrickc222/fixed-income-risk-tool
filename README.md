# Fixed Income Pricing & Risk Engine

A Python-based analytical engine designed to price fixed-income securities and calculate key interest rate risk sensitivities ("Greeks") utilized on institutional sales and trading desks.

## Overview

This repository contains a quantitative finance tool designed to model bond valuations and analyze risk metrics. It computes:
* **Yield to Maturity (YTM):** Implied return solved via numerical optimization (Newton-Raphson).
* **Macaulay & Modified Duration:** Measures of a bond’s price sensitivity to interest rate movements.
* **DV01 (Dollar Value of a Basis Point):** The exact dollar change in bond value for a 1 basis point (0.01%) parallel shift in the yield curve.
* **Convexity:** The second-order derivative of price with respect to yield, measuring the curvature of risk.

---

## Installation & Setup

Ensure you have Python installed, then set up the required dependencies:

```bash
pip install numpy scipy
