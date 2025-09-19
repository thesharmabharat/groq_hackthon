# StressTest AI for Indian Stocks (NSE)

Hackathon project for **MachineHack × Groq Hackathon** – Category: *Financial Analysis (risk, market, real-time data)*.

Predict and simulate the effect of **stress events** (COVID-19, Global Financial Crisis 2008, oil shocks, or custom shocks) on **Indian stock prices (NSE)** using:

- **Agents**:
  - **DataAgent** → fetches real-time/historical stock data via Yahoo Finance.
  - **StressAgent** → simulates crisis scenarios with Monte Carlo bootstraps.
- **Real-time performance**: low-latency inference, optimized for Groq.
- **MCP-ready**: Yahoo Finance + Groq tools structured as thin wrappers.
- **Multi-modal**: text summary + risk metrics + chart visualization (PNG).
- **Groq API** (optional): instant natural-language summaries for retail users.

---

## 🚀 Features
- Run *stress tests* of a stock under:
  - **COVID-19 crash (2020)**
  - **Global Financial Crisis (2008)**
  - **Oil shock 2022**
  - **Custom shocks** (user-specified % drop)
- Outputs:
  - Expected return
  - 95% VaR & CVaR (Value at Risk, Conditional VaR)
  - Average maximum drawdown
  - Distribution chart (PNG)
  - (Optional) Groq summary in plain English

---

## 🛠️ Installation
```bash
# clone repo
pip install -r requirements.txt

# requirements.txt
# yfinance
# numpy
# pandas
# matplotlib
# groq   # optional (Groq summaries)
```

---

## ⚡ Usage
```bash
# COVID-19 shock on RELIANCE (10-day horizon)
python stresstest_ai.py --ticker RELIANCE --scenario covid_2020 --days 10 --sims 3000

# Global Financial Crisis shock on TCS (15-day horizon, 5000 sims)
python stresstest_ai.py --ticker TCS --scenario gfc_2008 --days 15 --sims 5000

# Custom 18% drop on HDFCBANK
python stresstest_ai.py --ticker HDFCBANK --scenario custom --shock_pct -0.18 --days 10

# JSON output (for dashboards / integration)
python stresstest_ai.py --ticker INFY --scenario covid_2020 --json
```

- NSE tickers auto-suffixed with `.NS` (e.g., RELIANCE → RELIANCE.NS).
- Charts saved in `./outputs/`.

---

## 📊 Sample Output (Text)
```
=== StressTest AI Result ===
Ticker        : RELIANCE
Scenario      : covid_2020
Horizon (d)   : 10
Simulations   : 3000
--- Features (5y) ---
Annual Vol    : 21.35%
Annual Return : 9.82%
Downside Ret  : -5.41%
--- Scenario Stats ---
Price Now     : 2890.50
Expected Ret  : -2.10%  -> Exp Price: 2829.69
VaR 95% Ret   : -7.55%  -> VaR Price: 2672.02
CVaR 95% Ret  : -10.42% -> CVaR Price: 2590.41
Avg Max DD    : -12.83%
--- LLM Summary (Groq) ---
• Reliance could face ~7–10% downside risk in a COVID-like event.
• Average drawdowns suggest short-term volatility spikes.
• Long-term fundamentals remain stable beyond the horizon.
Chart PNG     : ./outputs/stress_RELIANCE_covid_2020.png
===========================
```

---

## 📐 Architecture
**User → Orchestrator → Agents → Groq + MCP → Output**

- **DataAgent**: pulls data from Yahoo Finance (MCP).
- **StressAgent**: simulates stress (Monte Carlo, Groq-assisted summaries).
- **Groq LLM Tool**: optional bullet-point explanations.
- **Outputs**: JSON, text report, PNG chart.

---

## 🌟 Future Enhancements
- Portfolio support (multiple tickers + weights).
- Sentiment Agent (live news/Twitter MCP integration).
- Vectorized path simulations for faster throughput.
- Voice summary (TTS for retail investors).

---

## 📅 Hackathon Fit
- **Multi-agent** ✅
- **Real-time inference** ✅ (Groq API + fast sims)
- **MCP integration** ✅
- **Multi-modal** ✅ (text + chart)
- **Real-world use case** ✅

---

## 👨‍💻 Authors
Team project for MachineHack × Groq Hackathon 2025.
