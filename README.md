# Tail Dependence and Joint Extreme Events in Financial Markets: A Copula-Based Analysis

A rigorous empirical study of tail dependence structures in multi-asset portfolios
using copula methods, with emphasis on the limitations of the Gaussian copula and
the implications of model choice for portfolio risk management.

**Author**: Robbie | Duke University, M.S. in Statistical Science
**Course**: [Course name/number]
**Status**: In progress

---

## Motivation

The Gaussian copula — once called *"the formula that killed Wall Street"* for its
role in the 2008 financial crisis — implies **zero asymptotic tail dependence**,
meaning it structurally cannot capture the phenomenon that asset returns tend to
crash together. This project quantifies the cost of this assumption and compares
alternative copula families on a realistic multi-asset portfolio.

## Research Questions

1. How much does tail dependence estimation differ across copula families?
2. How does joint tail behavior change between normal and stress regimes?
3. What is the impact of copula choice on VaR, Expected Shortfall, and
   perceived diversification benefits?

## Data

- **Assets**: SPY (US equity), EFA (intl developed equity), TLT (long Treasuries),
  GLD (gold)
- **Sample**: 2005-01-01 to 2024-12-31 (daily)
- **Source**: Yahoo Finance via `yfinance`
- **Crisis regimes**: GFC (2008-09), Euro crisis (2011), COVID (2020), Rate shock (2022)

## Methodology

1. **Marginal modeling**: ARMA(p,q)-GARCH(1,1) with skewed-t innovations;
   PIT transformation validated via Ljung-Box, ARCH-LM, and K-S tests.
2. **Copula estimation**: IFM (Inference for Margins) for Gaussian, Student-t,
   Clayton, Gumbel, Frank, and rotated Archimedean copulas.
3. **Model selection**: AIC/BIC + Cramér-von Mises GoF with parametric bootstrap.
4. **Tail dependence**: Parametric λ_L, λ_U from fitted copulas vs. non-parametric
   Schmidt-Stadtmüller estimator.
5. **Risk measures**: Monte Carlo VaR/ES (1%, 5%) with Kupiec and Christoffersen
   backtests; Acerbi-Székely ES test.

## Repository Structure

```
src/              Core modules (marginals, copulas, tail dep, risk)
notebooks/        Analysis pipeline (01_EDA → 05_risk_analysis)
scripts/          One-shot entry points (download, full pipeline)
results/          Figures and tables
report/           Final report and presentation
```

## Reproducibility

```bash
git clone https://github.com/YOUR_USERNAME/copula-tail-dependence.git
cd copula-tail-dependence
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/download_data.py
python scripts/run_full_pipeline.py
```

## Key Findings

*[To be populated as results come in]*

## References

- Li, D. X. (2000). On default correlation: A copula function approach.
- Patton, A. J. (2006). Modelling asymmetric exchange rate dependence.
- Joe, H. (2014). *Dependence Modeling with Copulas*.
- McNeil, Frey & Embrechts (2015). *Quantitative Risk Management*.

## License

MIT
