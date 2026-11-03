"""
Market Risk Engine: VaR, ES & Backtesting
-----------------------------------------
12-position book (5 equity/ETF, 2 FX, 2 other ETFs, 3 US Treasuries).
- 1-day 99% VaR and ES: historical simulation + variance-covariance
- Backtest: exceedances, Basel traffic light, Kupiec POF test
- Bond PV01, stress scenarios, daily HTML report, exceedance chart

Run:   python risk_engine.py            (real data: yfinance + FRED)
Test:  python risk_engine.py --demo     (synthetic data, no internet)
Install: pip install yfinance pandas numpy scipy matplotlib
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import norm, binom, chi2
from scipy.special import xlogy

# ------------------------------------------------------------------ config
START, END = "2019-01-01", "2024-12-31"   # 6 years incl. March 2020 and 2022
WINDOW = 250                              # rolling VaR window (trading days)
CONF = 0.99
P = 1 - CONF
NOTIONAL = 1_000_000                      # per position, USD
VAR_LIMIT=None

EQUITIES = ["SPY", "QQQ", "AAPL", "MSFT", "INDA"]
FX = ["EURUSD=X", "USDINR=X"]             # long EUR, long USD (vs INR)
OTHER = ["GLD", "TLT"]
PRICE_ASSETS = EQUITIES + FX + OTHER      # 9 price-based positions
FRED_BONDS = {"^FVX": 5, "^TNX": 10, "^TYX": 30}   # Yahoo Treasury yield indices

OUT = Path("output")
OUT.mkdir(exist_ok=True)


# ------------------------------------------------------------------ data
def load_real():
    import yfinance as yf
    tickers = PRICE_ASSETS + list(FRED_BONDS)
    px = yf.download(tickers, start=START, end=END,
                     auto_adjust=True, progress=False)["Close"]
    px.index = px.index.tz_localize(None)
    yl = px[list(FRED_BONDS)].copy()      # yields in percent
    px = px[PRICE_ASSETS]
    if yl.median().median() > 20:         # guard: some feeds quote yields x10
        yl = yl / 10
    return px, yl


# ------------------------------------------------------------------ bonds
def bond_price(y, T, c, freq=2):
    """Price per 100 face of a bond with annual coupon rate c, yield y."""
    n = int(T * freq)
    t = np.arange(1, n + 1)
    cf = np.full(n, c / freq * 100.0)
    cf[-1] += 100.0
    return (cf / (1 + y / freq) ** t).sum()


def bond_pv01(y_pct, T, notional):
    """$ price change for a 1bp rise in yield. Par bond (coupon = yield)."""
    y = y_pct / 100
    p0 = bond_price(y, T, y)
    p1 = bond_price(y + 0.0001, T, y)
    return (p0 - p1) / p0 * notional


# ------------------------------------------------------------------ P&L
def build_pnl(px, yl):
    data = pd.concat([px, yl], axis=1).ffill().dropna()
    pnl = data[PRICE_ASSETS].pct_change() * NOTIONAL        # $ P&L, equal notionals
    dy_bp = data[list(FRED_BONDS)].diff() * 100              # yield change in bp
    pv01_latest = {}
    for s, T in FRED_BONDS.items():
        pv01_t = data[s].shift(1).apply(lambda y: bond_pv01(y, T, NOTIONAL))
        pnl[s] = -pv01_t * dy_bp[s]                           # P&L = -PV01 x dy(bp)
        pv01_latest[s] = bond_pv01(data[s].iloc[-1], T, NOTIONAL)
    pnl = pnl.dropna()
    assert pnl.shape[1] == 12, "book should have 12 positions"
    return pnl, pv01_latest


# ------------------------------------------------------------------ VaR / ES
def rolling_var_es(port, window=WINDOW, p=P):
    """VaR/ES for day t use only the previous `window` days (shift(1) = no look-ahead)."""
    r = port.rolling(window)
    z = norm.ppf(p)
    mu, sd = r.mean().shift(1), r.std().shift(1)

    def es(x):
        q = np.quantile(x, p)
        return -x[x <= q].mean()

    out = pd.DataFrame({
        "hist_var": -r.quantile(p).shift(1),
        "hist_es": r.apply(es, raw=True).shift(1),
        # for $ P&L, portfolio sigma = sqrt(1' Cov 1) = std of summed P&L
        "vc_var": -(mu + z * sd),
        "vc_es": -mu + sd * norm.pdf(z) / p,
    })
    return out.dropna()


# ------------------------------------------------------------------ backtest
def kupiec_pof(n, T, p=P):
    x = n / T
    ll0 = xlogy(T - n, 1 - p) + xlogy(n, p)
    ll1 = xlogy(T - n, 1 - x) + xlogy(n, x)
    lr = -2 * (ll0 - ll1)
    return lr, chi2.sf(lr, 1)


def traffic_light(n, T, p=P):
    c = binom.cdf(n, T, p)            # T=250: green 0-4, yellow 5-9, red 10+
    return "Green" if c < 0.95 else ("Yellow" if c < 0.9999 else "Red")


def backtest(port, var, label):
    rows = []
    for name, sl in [("full sample", slice(None)), (f"last {WINDOW} days", slice(-WINDOW, None))]:
        v = var.iloc[sl]
        a = port.loc[v.index]
        exc = a < -v
        n, T = int(exc.sum()), len(v)
        lr, pval = kupiec_pof(n, T)
        rows.append({"method": label, "period": name, "days": T, "exceedances": n,
                     "expected": round(P * T, 1), "basel_zone": traffic_light(n, T),
                     "kupiec_LR": round(lr, 2), "kupiec_p": round(pval, 3),
                     "kupiec_result": "Reject" if pval < 0.05 else "Accept"})
    return rows


# ------------------------------------------------------------------ stress
def stress(pnl, port, pv01):
    hyp = {
        "Rates +100bp parallel": -sum(pv01.values()) * 100,
        "Equities -10% (5 positions)": -0.10 * NOTIONAL * len(EQUITIES),
        "EURUSD & USDINR -5%": -0.05 * NOTIONAL * len(FX),
    }
    rows = [{"scenario": k, "pnl_usd": round(v)} for k, v in hyp.items()]
    hist = {"COVID crash (9-23 Mar 2020)": ("2020-03-09", "2020-03-23"),
            "2022 rate shock (Jan-Oct 2022)": ("2022-01-03", "2022-10-14")}
    for k, (a, b) in hist.items():
        s = port.loc[a:b]
        if len(s):
            rows.append({"scenario": k + f" | worst day {s.min():,.0f}", "pnl_usd": round(s.sum())})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ outputs
def plot(port, var, path):
    v = var.loc[port.index.intersection(var.index)]
    a = port.loc[v.index]
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(a.index, a, lw=0.6, color="grey", label="Daily P&L")
    ax.plot(v.index, -v["hist_var"], color="tab:red", lw=1, label="-Hist VaR 99%")
    ax.plot(v.index, -v["vc_var"], color="tab:blue", lw=1, label="-VC VaR 99%")
    exc = a < -v["hist_var"]
    ax.scatter(a.index[exc], a[exc], color="black", s=14, zorder=3, label="Hist VaR exceedances")
    ax.set_title("Portfolio P&L vs 1-day 99% VaR"); ax.set_ylabel("USD"); ax.legend()
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def daily_report(var, port, bt, st, pv01, path):
    d = var.index[-1]
    last = var.iloc[-1]
    util = last["hist_var"] / VAR_LIMIT
    flag = "BREACH" if util > 1 else ("WARNING (>80%)" if util > 0.8 else "OK")
    summ = pd.DataFrame({
        "Hist": [last["hist_var"], last["hist_es"]],
        "Var-Cov": [last["vc_var"], last["vc_es"]]}, index=["VaR 99% (1d)", "ES 99% (1d)"]).round(0)
    pv = pd.Series(pv01).round(0).rename("PV01 (USD/bp)").to_frame()
    html = f"""<html><body style="font-family:sans-serif">
<h2>Daily Market Risk Report - {d:%d %b %Y}</h2>
<p>Day P&L: {port.loc[d]:,.0f} USD | Hist VaR limit: {VAR_LIMIT:,} USD |
Utilisation: {util:.0%} | Status: <b>{flag}</b> |
VaR exceedance today: {'YES' if port.loc[d] < -last['hist_var'] else 'No'}</p>
<h3>VaR / ES</h3>{summ.to_html()}
<h3>Bond PV01</h3>{pv.to_html()}
<h3>Backtest</h3>{bt.to_html(index=False)}
<h3>Stress scenarios</h3>{st.to_html(index=False)}
<img src="exceedances.png" width="900">
</body></html>"""
    Path(path).write_text(html)


def main():
    px, yl = load_real()
    pnl, pv01 = build_pnl(px, yl)
    port = pnl.sum(axis=1)
    var = rolling_var_es(port)
    global VAR_LIMIT
    VAR_LIMIT = round(1.25 * var["hist_var"].mean(), -3)   # illustrative: 1.25x average VaR

    bt = pd.DataFrame(backtest(port, var["hist_var"], "Historical") +
                      backtest(port, var["vc_var"], "Variance-Covariance"))
    st = stress(pnl, port, pv01)

    print(f"Positions: {pnl.shape[1]} | Days of P&L: {len(port)} | "
          f"Backtest days: {len(var)} | {port.index[0]:%Y-%m-%d} to {port.index[-1]:%Y-%m-%d}\n")
    print(bt.to_string(index=False), "\n")
    print(st.to_string(index=False), "\n")
    print("Latest PV01 (USD/bp):", {k: round(v) for k, v in pv01.items()})

    plot(port, var, OUT / "exceedances.png")
    daily_report(var, port, bt, st, pv01, OUT / "daily_report.html")
    bt.to_csv(OUT / "backtest_results.csv", index=False)
    print(f"\nSaved report, chart and CSV to ./{OUT}/")


if __name__ == "__main__":
    main()