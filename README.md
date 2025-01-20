# Statistical Analysis of the NASDAQ Stock Exchange Index

## Overview
This repository contains the code and analysis of the statistical properties of the NASDAQ stock exchange index, utilizing daily and weekly data from February 5, 1971, to November 15, 2022. The paper explores fundamental characteristics of financial time series data, including volatility modeling, stationarity, heavy tails, and Value at Risk (VaR). 
This work is especially relevant for researchers, traders, and financial risk managers who aim to better understand the behavior of stock market indices and assess potential risks.


## Stylized Facts
- **Log-Returns Distribution:** Returns are centered around zero but exhibit extreme values (heavy tails) compared to a normal distribution.
- **Serial Correlation:** Low autocorrelation in returns but a positive correlation in squared returns, indicating volatility clustering.

## Value at Risk (VaR)
Different methods are used to compute VaR at 5% and 1% levels for daily and weekly data:

- The **Normal Distribution method** often underestimates risk, especially at the 1% VaR level, because it fails to account for heavy tails.
- The **Historical Simulation method** performs best at the 99% VaR level, but struggles during extreme market conditions due to its reliance on past data.
- The **EWMA method** adapts to changing volatility, providing robust VaR estimates in periods of volatility clustering.
- The **GARCH-GJR model** captures volatility clustering and asymmetries in market returns, offering better performance during periods of high volatility than other methods.


## Methods

### Statistical Tests
- **Unit Root Tests:** Using the Augmented Dickey-Fuller (ADF) test, the study shows that log-prices have a unit root (non-stationary), while log-returns do not (stationary).
- **Ljung-Box Test:** Significant cumulative serial correlation is observed in returns and squared returns at specific lags, suggesting deviations from weak stationarity.
- **Tail Index Estimation:** Using the Hill estimator and rank-size regression, the analysis shows that returns have finite second moments but infinite third moments.

### Volatility Modeling
- **CUSUM Test:** Indicates time-varying variance, motivating the use of GARCH-type models.
- **GARCH Models:** A GJR(1,1) model is fitted for daily data, and a GARCH(1,1) model for weekly data, with model selection based on Bayesian Information Criteria (BIC).
- **Misspecification Analysis:** Standardized residuals reveal non-normality and serial correlation.

### Bootstrap for ARCH(1) Model
Bootstrap methods are used to estimate the sampling distribution of a statistic, especially when traditional methods might not be applicable or robust. In our analysis of the ARCH(1) model, we use the Fixed-Volatility Bootstrap because it preserves the heteroskedasticity inherent in financial data and accounts for the time-dependence of returns. Unlike other bootstrapping methods, the Fixed-Volatility variant maintains volatility across bootstrap samples, which is essential for testing volatility parameters
