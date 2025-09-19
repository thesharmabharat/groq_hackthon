#!/usr/bin/env python3
"""
StressTest AI for Indian Stocks (NSE)
-------------------------------------
Multi-agent prototype that stress-tests NSE stocks against historical crisis
regimes (e.g., COVID-19, GFC 2008) and custom shocks. Meets hackathon asks:

- Agents: DataAgent (market data) + StressAgent (scenario simulation)
- Real-time: queries live prices via Yahoo Finance if online
- MCP-ready: tool wrappers are structured so they can be exposed via MCP
- Multi-modal: generates charts (PNG) + text summary; easy to add TTS
- Groq: optional LLM summaries / decisions if GROQ_API_KEY is set

Usage examples
--------------
python stresstest_ai.py --ticker RELIANCE --scenario covid_2020 --days 10
python stresstest_ai.py --ticker TCS --scenario gfc_2008 --sims 5000 --days 15
python stresstest_ai.py --ticker HDFCBANK --scenario custom --shock_pct -0.18

Notes
-----
- NSE tickers use ".NS" on Yahoo (e.g., RELIANCE.NS). The script adds it.
- Requires: yfinance, numpy, pandas, matplotlib. Optional: groq
- Outputs PNG chart and JSON-like printed metrics (VaR, CVaR, drawdowns).

"""
from __future__ import annotations
import os
import sys
import json
import math
import time
import argparse
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Optional Groq client (pip install groq)
try:
    from groq import Groq
except Exception:  # pragma: no cover
    Groq = None  # type: ignore

# Optional: yfinance (pip install yfinance)
try:
    import yfinance as yf
except Exception as e:  # pragma: no cover
    yf = None  # type: ignore

# -----------------------------
# Utility & Config
# -----------------------------

CRISIS_WINDOWS = {
    # Approx crisis windows; you can tune these further
    "covid_2020": ("2020-02-15", "2020-04-15"),
    "gfc_2008": ("2008-09-01", "2008-12-01"),
    # Example: oil shock / inflation 2022
    "oil_2022": ("2022-02-15", "2022-06-30"),
}

PNG_DIR = os.environ.get("STRESSTEST_PNG_DIR", "./outputs")
os.makedirs(PNG_DIR, exist_ok=True)

SEED = int(os.environ.get("STRESSTEST_SEED", "42"))
np.random.seed(SEED)


def _nse_symbol(ticker: str) -> str:
    t = ticker.upper().strip()
    if not t.endswith(".NS"):
        t += ".NS"
    return t


# -----------------------------
# MCP-style tool wrappers (thin)
# -----------------------------

class YahooFinanceTool:
    """Thin wrapper that can be exposed as an MCP Tool.
    Provides: get_history(ticker, start, end), get_last_price(ticker)
    """

    def __init__(self):
        if yf is None:
            raise RuntimeError(
                "yfinance not available. Install via `pip install yfinance`."
            )

    def get_history(self, ticker: str, start: str, end: str) -> pd.DataFrame:
        sym = _nse_symbol(ticker)
        df = yf.download(sym, start=start, end=end, progress=False)
        if df is None or df.empty:
            raise ValueError(f"No data for {ticker} between {start} and {end}")
        return df

    def get_full_history(self, ticker: str, period: str = "max") -> pd.DataFrame:
        sym = _nse_symbol(ticker)
        df = yf.download(sym, period=period, progress=False)
        if df is None or df.empty:
            raise ValueError(f"No data for {ticker} period={period}")
        return df

    def get_last_price(self, ticker: str) -> float:
        sym = _nse_symbol(ticker)
        info = yf.Ticker(sym).history(period="1d")
        if info is None or info.empty:
            raise ValueError(f"No last price for {ticker}")
        return float(info["Close"].iloc[-1])


class GroqLLMTool:
    """Optional Groq LLM helper for fast summaries and small decisions."""

    def __init__(self, model: str = "llama-3.1-70b-versatile"):
        self.api_key = os.environ.get("GROQ_API_KEY")
        self.model = model
        self.client = None
        if self.api_key and Groq is not None:
            self.client = Groq(api_key=self.api_key)

    def summarize(self, prompt: str) -> str:
        if not self.client:
            return "(Groq unavailable) " + prompt[:200]
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=300,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:  # pragma: no cover
            return f"(Groq error: {e})\n" + prompt[:200]


# -----------------------------
# Agents
# -----------------------------

@dataclass
class DataAgent:
    data_tool: YahooFinanceTool

    def fetch_baseline(self, ticker: str) -> pd.DataFrame:
        """Fetch broad history for feature calc (5y by default)."""
        df = self.data_tool.get_full_history(ticker, period="5y")
        df = df.dropna()
        return df

    def crisis_sample(self, ticker: str, scenario: str) -> pd.DataFrame:
        if scenario not in CRISIS_WINDOWS:
            raise ValueError(f"Unknown scenario '{scenario}'. Choose from {list(CRISIS_WINDOWS)}")
        start, end = CRISIS_WINDOWS[scenario]
        df = self.data_tool.get_history(ticker, start, end).dropna()
        return df

    @staticmethod
    def daily_returns(prices: pd.Series) -> pd.Series:
        return prices.pct_change().dropna()

    def compute_features(self, df: pd.DataFrame) -> Dict[str, float]:
        ret = self.daily_returns(df["Adj Close"]) if "Adj Close" in df.columns else self.daily_returns(df["Close"]) 
        vol = float(ret.std() * math.sqrt(252))
        avg_ret = float(ret.mean() * 252)
        downside = float(ret[ret < 0].mean() * 252) if (ret < 0).any() else 0.0
        return {"ann_vol": vol, "ann_ret": avg_ret, "downside_ret": downside}


@dataclass
class StressAgent:
    groq: Optional[GroqLLMTool] = None

    def simulate(
        self,
        current_price: float,
        crisis_returns: pd.Series,
        horizon_days: int = 10,
        sims: int = 5000,
    ) -> Dict[str, any]:
        """Bootstrapped Monte Carlo from crisis-period daily return distribution."""
        if len(crisis_returns) < 5:
            raise ValueError("Insufficient crisis returns to simulate.")

        returns = crisis_returns.values
        paths_end = np.empty(sims)
        drawdowns = np.empty(sims)
        for i in range(sims):
            # sample with replacement
            sampled = np.random.choice(returns, size=horizon_days, replace=True)
            path = current_price * np.cumprod(1 + sampled)
            paths_end[i] = path[-1]
            peak = np.maximum.accumulate(path)
            dd = np.min(path / peak - 1.0)
            drawdowns[i] = dd

        pnl = (paths_end - current_price) / current_price
        var95 = float(np.percentile(pnl, 5))
        cvar95 = float(pnl[pnl <= np.percentile(pnl, 5)].mean())
        exp = float(np.mean(pnl))

        res = {
            "horizon_days": horizon_days,
            "sims": sims,
            "price_now": float(current_price),
            "exp_return": exp,
            "var95": var95,
            "cvar95": cvar95,
            "exp_price": float(current_price * (1 + exp)),
            "var95_price": float(current_price * (1 + var95)),
            "cvar95_price": float(current_price * (1 + cvar95)),
            "avg_max_drawdown": float(np.mean(drawdowns)),
            "distribution_sample": pnl[:100].tolist(),  # preview only
        }

        # Optional LLM summary
        if self.groq:
            prompt = (
                f"Summarize in 3 bullet points for a retail investor:\n"
                f"- Current price: {current_price:.2f}\n"
                f"- Horizon: {horizon_days} days, Simulations: {sims}\n"
                f"- Expected return: {exp:.2%}\n"
                f"- 95% VaR return: {var95:.2%}\n"
                f"- 95% CVaR return: {cvar95:.2%}\n"
                f"- Avg max drawdown: {res['avg_max_drawdown']:.2%}\n"
            )
            res["llm_summary"] = self.groq.summarize(prompt)
        return res

    @staticmethod
    def plot_simulation(
        ticker: str,
        scenario: str,
        current_price: float,
        results: Dict[str, any],
        outfile: Optional[str] = None,
    ) -> str:
        if outfile is None:
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            outfile = os.path.join(PNG_DIR, f"stress_{ticker}_{scenario}_{stamp}.png")
        # Histogram of end PnL
        pnl = np.array(results["distribution_sample"])  # sample only for speed in PNG
        plt.figure(figsize=(8, 5))
        plt.hist(pnl, bins=30, alpha=0.8)
        plt.title(f"PnL Distribution @ {ticker} | {scenario} | {results['horizon_days']}d")
        plt.xlabel("Return (fraction)")
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.savefig(outfile, dpi=160)
        plt.close()
        return outfile


# -----------------------------
# Orchestrator
# -----------------------------

@dataclass
class Orchestrator:
    data_agent: DataAgent
    stress_agent: StressAgent

    def run(
        self,
        ticker: str,
        scenario: str,
        days: int = 10,
        sims: int = 3000,
        shock_pct: Optional[float] = None,
    ) -> Dict[str, any]:
        # 1) Fetch baseline & current price
        base = self.data_agent.fetch_baseline(ticker)
        price_col = "Adj Close" if "Adj Close" in base.columns else "Close"
        current_price = float(base[price_col].iloc[-1])

        # 2) Crisis returns
        if scenario == "custom":
            if shock_pct is None:
                raise ValueError("custom scenario requires --shock_pct (e.g., -0.15)")
            # Create a pseudo-crisis distribution around the shock (single-day) + noise
            loc = shock_pct / days
            crisis_returns = pd.Series(np.random.normal(loc=loc, scale=abs(loc) * 0.6, size=60))
        else:
            crisis_df = self.data_agent.crisis_sample(ticker, scenario)
            price = crisis_df["Adj Close"] if "Adj Close" in crisis_df.columns else crisis_df["Close"]
            crisis_returns = self.data_agent.daily_returns(price)

        # 3) Compute features (optional for reporting)
        feats = self.data_agent.compute_features(base)

        # 4) Simulate
        sim = self.stress_agent.simulate(
            current_price=current_price,
            crisis_returns=crisis_returns,
            horizon_days=days,
            sims=sims,
        )

        # 5) Plot
        png = self.stress_agent.plot_simulation(ticker, scenario, current_price, sim)

        # 6) Package
        out = {
            "ticker": ticker.upper(),
            "scenario": scenario,
            "features": feats,
            "simulation": sim,
            "chart_png": png,
        }
        return out


# -----------------------------
# CLI
# -----------------------------

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="StressTest AI: simulate NSE stock under crisis scenarios"
    )
    p.add_argument("--ticker", required=True, help="NSE symbol without .NS suffix (e.g., RELIANCE)")
    p.add_argument(
        "--scenario",
        required=True,
        choices=list(CRISIS_WINDOWS.keys()) + ["custom"],
        help="Stress scenario (covid_2020, gfc_2008, oil_2022, or custom)",
    )
    p.add_argument("--days", type=int, default=10, help="Simulation horizon in trading days")
    p.add_argument("--sims", type=int, default=3000, help="Number of Monte Carlo simulations")
    p.add_argument("--shock_pct", type=float, default=None, help="Custom scenario: total shock pct (e.g., -0.2)")
    p.add_argument("--json", action="store_true", help="Print JSON output")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    try:
        data_tool = YahooFinanceTool()
        groq_tool = GroqLLMTool()
        data_agent = DataAgent(data_tool)
        stress_agent = StressAgent(groq=groq_tool)
        orch = Orchestrator(data_agent, stress_agent)
        result = orch.run(
            ticker=args.ticker,
            scenario=args.scenario,
            days=args.days,
            sims=args.sims,
            shock_pct=args.shock_pct,
        )
        if args.json:
            print(json.dumps(result, indent=2, default=str))
        else:
            sim = result["simulation"]
            feats = result["features"]
            print("\n=== StressTest AI Result ===")
            print(f"Ticker        : {result['ticker']}")
            print(f"Scenario      : {result['scenario']}")
            print(f"Horizon (d)   : {sim['horizon_days']}")
            print(f"Simulations   : {sim['sims']}")
            print("--- Features (5y) ---")
            print(f"Annual Vol    : {feats['ann_vol']:.2%}")
            print(f"Annual Return : {feats['ann_ret']:.2%}")
            print(f"Downside Ret  : {feats['downside_ret']:.2%}")
            print("--- Scenario Stats ---")
            print(f"Price Now     : {sim['price_now']:.2f}")
            print(f"Expected Ret  : {sim['exp_return']:.2%}  -> Exp Price: {sim['exp_price']:.2f}")
            print(f"VaR 95% Ret   : {sim['var95']:.2%}  -> VaR Price: {sim['var95_price']:.2f}")
            print(f"CVaR 95% Ret  : {sim['cvar95']:.2%} -> CVaR Price: {sim['cvar95_price']:.2f}")
            print(f"Avg Max DD    : {sim['avg_max_drawdown']:.2%}")
            if sim.get("llm_summary"):
                print("--- LLM Summary (Groq) ---")
                print(sim["llm_summary"])
            print(f"Chart PNG     : {result['chart_png']}")
            print("===========================\n")
        return 0
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
