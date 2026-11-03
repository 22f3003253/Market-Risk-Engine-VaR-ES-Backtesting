# Market-Risk-Engine-VaR-ES-Backtesting
A Python risk engine for a 12-position multi-asset book (equities, FX, gold,
US Treasuries). It computes 1-day 99% VaR and Expected Shortfall, back-tests
them, and produces stress tests and a daily risk report.

## What it does
- **Portfolio:** SPY, QQQ, AAPL, MSFT, INDA, EURUSD, USDINR, GLD, TLT and
  5Y/10Y/30Y Treasuries, $1M each, daily data 2019-2024 (yfinance)
- **VaR / ES (99%, 1-day, 250-day rolling window):** historical simulation
  and variance-covariance
- **Bonds:** PV01 from a par-bond price bump; P&L = -PV01 x yield change (bp)
- **Back-testing:** exceedance counts, Basel traffic-light zones, Kupiec POF test
- **Stress tests:** +100bp rates, equities -10%, FX -5%, COVID (Mar 2020)
  and 2022 rate-shock windows
- **Report:** daily HTML summary with VaR, ES, limit utilisation, PV01 and
  back-test results

## Results
| Method | Days | Exceedances | Expected | Kupiec (5%) |
|---|---|---|---|---|
| Historical | 1,313 | 21 | 13.1 | Reject (p=0.045) |
| Variance-Covariance | 1,313 | 33 | 13.1 | Reject (p<0.001) |

![Exceedances](output/exceedances.png)

**Findings**
- Variance-covariance under-covers heavily: the normality assumption
  underestimates fat-tailed losses.
- Historical VaR was too low going into March 2020, then stayed high for
  about a year as the crash remained in the 250-day window.
- Exceedances cluster in the 2020 and 2022 stress periods.
- Over the last 250 days both methods are in the Basel yellow zone and
  Kupiec does not reject.

## Assumptions and limitations
- Bonds are modelled as par bonds from Yahoo yield indices, with constant maturity
- Equal $1M notionals; VaR limit is illustrative
- Normality assumption in variance-covariance
- Possible extensions: EWMA volatility, Monte Carlo VaR, option greeks

## Run
pip install -r requirements.txt
python risk_engine.py
