#!/usr/bin/env python3
"""
StressTest AI for Indian Stocks (NSE)
-------------------------------------
Optimized multi‑agent prototype with portfolio support, market‑factor stress, and
vectorized Monte Carlo — tailored for MachineHack × Groq.

What's new vs v1
- ✅ Portfolio mode (multiple tickers + weights; equal weights by default)
- ✅ NIFTY‑conditioned stress using a 1‑factor model (beta to market + residuals)
- ✅ (Stub) SentimentAgent scales market shock (fear/optimism)
- ✅ Charts: PnL histogram + sample portfolio price paths
- ✅ Vectorized simulation (matrix bootstrap) for low latency
- ✅ Simple in‑memory Yahoo cache to avoid repeat downloads
- ✅ MCP‑friendly tool wrappers; optional Groq LLM bullet summaries

Examples
--------
python stresstest_ai.py --ticker RELIANCE --scenario covid_2020 --days 10
python stresstest_ai.py --tickers RELIANCE,TCS,INFY --weights 0.34,0.33,0.33 --scenario gfc_2008 --days 15 --sims 8000
python stresstest_ai.py --tickers HDFCBANK,ICICIBANK --scenario oil_2022 --days 7 --json

Notes
-----
- NSE symbols are auto‑suffixed with ".NS" for Yahoo Finance.
- Requires: yfinance, numpy, pandas, matplotlib. Optional: groq
- Outputs two PNGs per run (histogram + sample paths) into ./outputs/
"""
from __future__ import annotations
import os
import sys
import json
import math
import argparse
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Optional Groq client (pip install groq)
try:
    from groq import Groq
except Exception:
    Groq = None  # type: ignore

# Market data (pip install yfinance)
try:
    import yfinance as yf
except Exception:
    yf = None  # type: ignore

# -----------------------------
# Config / Constants
# -----------------------------
CRISIS_WINDOWS = {
    "covid_2020": ("2020-02-15", "2020-04-15"),
    "gfc_2008": ("2008-09-01", "2008-12-01"),
    "oil_2022": ("2022-02-15", "2022-06-30"),
}

MARKET_TICKER = "^NSEI"  # NIFTY 50 index on Yahoo
PNG_DIR = os.environ.get("STRESSTEST_PNG_DIR", "./outputs")
os.makedirs(PNG_DIR, exist_ok=True)
RNG = np.random.default_rng(int(os.environ.get("STRESSTEST_SEED", "42")))

# -----------------------------
# Helpers
# -----------------------------

def _nse_symbol(t: str) -> str:
    t = t.upper().strip()
    return t if t.endswith(".NS") else t + ".NS"


class _LRUCache:
    """Very small in‑mem cache for Yahoo calls within a single run."""
    def __init__(self):
        self.h: Dict[Tuple[str, str], pd.DataFrame] = {}

    def get(self, key: Tuple[str, str]) -> Optional[pd.DataFrame]:
        return self.h.get(key)

    def put(self, key: Tuple[str, str], df: pd.DataFrame) -> None:
        self.h[key] = df


CACHE = _LRUCache()

# -----------------------------
# MCP‑style tool wrappers
# -----------------------------
class YahooFinanceTool:
    def __init__(self):
        if yf is None:
            raise RuntimeError("yfinance not installed. Run: pip install yfinance")

    def get_full_history(self, ticker: str, period: str = "5y") -> pd.DataFrame:
        key = (f"hist:{_nse_symbol(ticker)}", period)
        cached = CACHE.get(key)
        if cached is not None:
            return cached.copy()
        df = yf.download(_nse_symbol(ticker), period=period, progress=False)
        if df is None or df.empty:
            raise ValueError(f"No data for {ticker} period={period}")
        df = df.dropna()
        CACHE.put(key, df)
        return df

    def get_range(self, ticker: str, start: str, end: str) -> pd.DataFrame:
        key = (f"range:{_nse_symbol(ticker)}", f"{start}:{end}")
        cached = CACHE.get(key)
        if cached is not None:
            return cached.copy()
        df = yf.download(_nse_symbol(ticker), start=start, end=end, progress=False)
        if df is None or df.empty:
            raise ValueError(f"No data for {ticker} {start}→{end}")
        df = df.dropna()
        CACHE.put(key, df)
        return df

    def last_close(self, ticker: str) -> float:
        df = yf.Ticker(_nse_symbol(ticker)).history(period="1d")
        if df is None or df.empty:
            raise ValueError(f"No last price for {ticker}")
        return float(df["Close"].iloc[-1])


class GroqLLMTool:
    def __init__(self, model: str = "llama-3.1-70b-versatile"):
        key = os.environ.get("GROQ_API_KEY")
        self.model = model
        self.client = Groq(api_key=key) if key and Groq else None

    def summarize(self, prompt: str) -> str:
        if not self.client:
            return "(Groq unavailable) " + prompt[:160]
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=260,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            return f"(Groq error: {e})"


# -----------------------------
# Agents
# -----------------------------
@dataclass
class SentimentAgent:
    """Stub for MCP news/Twitter; returns score in [-1, 1]."""
    def get_sentiment_score(self, tickers: List[str]) -> float:
        return 0.0  # neutral for now


@dataclass
class DataAgent:
    tool: YahooFinanceTool

    def baseline_prices(self, tickers: List[str]) -> pd.DataFrame:
        dfs = []
        for t in tickers:
            df = self.tool.get_full_history(t, period="5y")[['Adj Close']].rename(columns={'Adj Close': t.upper()})
            dfs.append(df)
        out = pd.concat(dfs, axis=1).dropna(how='all')
        return out

    def crisis_returns(self, tickers: List[str], scenario: str) -> pd.DataFrame:
        start, end = CRISIS_WINDOWS[scenario]
        dfs = []
        for t in tickers:
            df = self.tool.get_range(t, start, end)[['Adj Close']].rename(columns={'Adj Close': t.upper()})
            dfs.append(df)
        out = pd.concat(dfs, axis=1).dropna(how='all')
        rets = out.pct_change().dropna(how='any')
        return rets

    def market_crisis_returns(self, scenario: str) -> pd.Series:
        start, end = CRISIS_WINDOWS[scenario]
        nifty = self.tool.get_range(MARKET_TICKER, start, end)["Adj Close"].pct_change().dropna()
        return nifty


@dataclass
class StressAgent:
    groq: Optional[GroqLLMTool] = None

    def _fit_betas(self, asset_rets: pd.DataFrame, mkt_rets: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
        """OLS betas per asset for the crisis window; returns (alpha, beta)."""
        X = mkt_rets.values
        alphas, betas = [], []
        for col in asset_rets.columns:
            y = asset_rets[col].reindex_like(mkt_rets).dropna()
            x = mkt_rets.reindex_like(y).values
            yv = y.values
            if len(yv) < 5:
                alphas.append(0.0); betas.append(1.0)
                continue
            # OLS: beta = cov/var
            var_x = np.var(x, ddof=1)
            cov_xy = np.cov(x, yv, ddof=1)[0,1] if var_x > 0 else 0.0
            beta = cov_xy / var_x if var_x > 0 else 1.0
            alpha = float(np.mean(yv - beta * x))
            alphas.append(alpha); betas.append(beta)
        return np.array(alphas, dtype=float), np.array(betas, dtype=float)

    def _residual_pool(self, asset_rets: pd.DataFrame, mkt_rets: pd.Series, alphas: np.ndarray, betas: np.ndarray) -> np.ndarray:
        resids = []
        X = mkt_rets.values
        for i, col in enumerate(asset_rets.columns):
            y = asset_rets[col].reindex_like(mkt_rets).values
            r = y - (alphas[i] + betas[i] * X)
            r = r[~np.isnan(r)]
            if len(r) == 0:
                r = np.array([0.0])
            resids.append(r)
        # ragged → store as list; sampling handles per asset separately
        return resids  # type: ignore

    def simulate_portfolio(
        self,
        scenario: str,
        asset_rets: pd.DataFrame,
        mkt_rets: pd.Series,
        prices_now: np.ndarray,
        weights: np.ndarray,
        sentiment_score: float,
        horizon_days: int = 10,
        sims: int = 5000,
    ) -> Dict[str, any]:
        n_assets = len(weights)
        alphas, betas = self._fit_betas(asset_rets, mkt_rets)
        resid_pool = self._residual_pool(asset_rets, mkt_rets, alphas, betas)

        # Bootstrap market returns for (sims × days)
        mkt_vals = mkt_rets.values
        idx_m = RNG.integers(0, len(mkt_vals), size=(sims, horizon_days))
        M = mkt_vals[idx_m]

        # Sentiment scaling: shift market by small multiple of its std
        if sentiment_score != 0.0:
            M = M + sentiment_score * (np.std(mkt_vals) * 0.2)

        # Build asset returns: r_i = alpha_i + beta_i*M + eps_i  (eps_i bootstrapped per asset)
        A = np.zeros((sims, horizon_days, n_assets), dtype=float)
        for i in range(n_assets):
            eps_pool = resid_pool[i]
            idx_e = RNG.integers(0, len(eps_pool), size=(sims, horizon_days))
            E = eps_pool[idx_e]
            A[..., i] = alphas[i] + betas[i] * M + E

        # Price paths & portfolio values (vectorized)
        base = float(np.sum(prices_now * weights))
        paths = prices_now * np.cumprod(1.0 + A, axis=1)  # sims × days × assets
        port_vals = np.sum(paths * weights[None, None, :], axis=2)  # sims × days
        final_vals = port_vals[:, -1]
        pnl = (final_vals - base) / base

        # Risk stats
        var95 = float(np.percentile(pnl, 5))
        cvar95 = float(pnl[pnl <= np.percentile(pnl, 5)].mean())
        exp = float(np.mean(pnl))

        res = {
            "scenario": scenario,
            "horizon_days": horizon_days,
            "sims": sims,
            "base_value": base,
            "exp_return": exp,
            "var95": var95,
            "cvar95": cvar95,
            "exp_price": float(base * (1 + exp)),
            "var95_price": float(base * (1 + var95)),
            "cvar95_price": float(base * (1 + cvar95)),
            "distribution_sample": pnl[:1000].tolist(),
        }

        if self.groq:
            prompt = (
                f"Stress results ({scenario}):\n"
                f"Horizon {horizon_days}d, sims {sims}. Base ₹{base:,.0f}.\n"
                f"Exp {exp:.2%}, VaR95 {var95:.2%}, CVaR95 {cvar95:.2%}.\n"
                f"Explain in 3 concise bullets for a retail investor."
            )
            res["llm_summary"] = self.groq.summarize(prompt)
        return res, port_vals

    # ---------- Plots ----------
    @staticmethod
    def plot_hist(pnl: np.ndarray, outfile: str) -> str:
        plt.figure(figsize=(8,5))
        plt.hist(pnl, bins=40, alpha=0.85)
        plt.title("Portfolio PnL Distribution")
        plt.xlabel("Return (fraction)")
        plt.ylabel("Frequency")
        plt.tight_layout(); plt.savefig(outfile, dpi=160); plt.close()
        return outfile

    @staticmethod
    def plot_paths(port_vals: np.ndarray, outfile: str, n_paths: int = 30) -> str:
        plt.figure(figsize=(8,5))
        base0 = port_vals[0,0]
        idx = RNG.integers(0, port_vals.shape[0], size=min(n_paths, port_vals.shape[0]))
        for i in idx:
            plt.plot(port_vals[i], alpha=0.5, linewidth=1)
        plt.title("Sample Portfolio Price Paths (stress sim)")
        plt.xlabel("Day")
        plt.ylabel("Portfolio Value (₹)")
        plt.tight_layout(); plt.savefig(outfile, dpi=160); plt.close()
        return outfile


# -----------------------------
# Orchestrator
# -----------------------------
@dataclass
class Orchestrator:
    data_agent: DataAgent
    stress_agent: StressAgent
    sentiment_agent: SentimentAgent

    def run(self, tickers: List[str], weights: Optional[List[float]], scenario: str, days: int, sims: int) -> Dict[str, any]:
        tickers = [t.upper().strip() for t in tickers]
        if weights is None:
            weights = [1.0/len(tickers)] * len(tickers)
        if len(weights) != len(tickers):
            raise ValueError("weights length must match tickers length")
        W = np.array(weights, dtype=float)
        W = W / W.sum()

        # Data
        base_hist = self.data_agent.baseline_prices(tickers)
        prices_now = base_hist[tickers].iloc[-1].values.astype(float)
        asset_crisis = self.data_agent.crisis_returns(tickers, scenario)
        mkt_crisis = self.data_agent.market_crisis_returns(scenario)

        # Sentiment (stub)
        s_score = self.sentiment_agent.get_sentiment_score(tickers)

        # Simulate
        res, port_vals = self.stress_agent.simulate_portfolio(
            scenario=scenario,
            asset_rets=asset_crisis,
            mkt_rets=mkt_crisis,
            prices_now=prices_now,
            weights=W,
            sentiment_score=s_score,
            horizon_days=days,
            sims=sims,
        )

        # Plots
        base_val = float(np.sum(prices_now * W))
        pnl_all = (port_vals[:, -1] - base_val) / base_val
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        tag = f"{scenario}_{'_'.join(tickers)}_{ts}"
        hist_png = os.path.join(PNG_DIR, f"pnl_hist_{tag}.png")
        paths_png = os.path.join(PNG_DIR, f"paths_{tag}.png")
        self.stress_agent.plot_hist(pnl_all, hist_png)
        self.stress_agent.plot_paths(port_vals, paths_png)

        res.update({
            "tickers": tickers,
            "weights": W.tolist(),
            "price_now": prices_now.tolist(),
            "pnl_hist_png": hist_png,
            "paths_png": paths_png,
        })
        return res


# -----------------------------
# CLI
# -----------------------------

def parse_args():
    p = argparse.ArgumentParser(description="StressTest AI: NSE stress under crisis scenarios")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--ticker", help="Single ticker (e.g., RELIANCE)")
    g.add_argument("--tickers", help="Comma‑separated tickers (e.g., RELIANCE,TCS,INFY)")
    p.add_argument("--weights", help="Comma‑separated weights (same order as tickers)")
    p.add_argument("--scenario", required=True, choices=list(CRISIS_WINDOWS.keys()))
    p.add_argument("--days", type=int, default=10)
    p.add_argument("--sims", type=int, default=5000)
    p.add_argument("--json", action="store_true", help="Print JSON result")
    return p.parse_args()


def main():
    args = parse_args()
    tickers = [args.ticker] if args.ticker else args.tickers.split(",")
    weights = None
    if args.weights:
        weights = [float(x) for x in args.weights.split(",")]

    try:
        data_agent = DataAgent(YahooFinanceTool())
        stress_agent = StressAgent(GroqLLMTool())
        sentiment_agent = SentimentAgent()
        orch = Orchestrator(data_agent, stress_agent, sentiment_agent)
        result = orch.run(tickers, weights, args.scenario, args.days, args.sims)
        if args.json:
            print(json.dumps(result, indent=2))
        else:
            print("\n=== StressTest AI (Portfolio) ===")
            print(f"Tickers   : {', '.join(result['tickers'])}")
            print(f"Weights   : {result['weights']}")
            print(f"Scenario  : {result['scenario']}")
            print(f"Horizon   : {result['horizon_days']} days | Sims: {result['sims']}")
            print("--- Risk ---")
            print(f"Exp Ret   : {result['exp_return']:.2%}  -> Exp Value: ₹{result['exp_price']:,.2f}")
            print(f"VaR 95%   : {result['var95']:.2%}  -> VaR Value: ₹{result['var95_price']:,.2f}")
            print(f"CVaR 95%  : {result['cvar95']:.2%} -> CVaR Value: ₹{result['cvar95_price']:,.2f}")
            if 'llm_summary' in result:
                print("--- Groq Summary ---")
                print(result['llm_summary'])
            print(f"Histogram : {result['pnl_hist_png']}")
            print(f"Paths     : {result['paths_png']}")
            print("===============================\n")
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
