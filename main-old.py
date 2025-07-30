import time
import pandas as pd
from tabulate import tabulate
from dotenv import load_dotenv

from agents.strategy_gen import StrategyGenAgent
from agents.backtest_agent import BacktestAgent
from agents.explain_agent import ExplainabilityAgent
from mcp.schema import StrategyMCP
from metrics import StrategyRunMetrics  # 🧪 Metrics integration

load_dotenv()

PROMPTS = [
    "Design a mean-reversion strategy for AAPL using RSI and Bollinger Bands.",
    "Create a momentum trading strategy using MACD and moving averages on TSLA.",
    "Propose a breakout trading strategy using ATR and Donchian Channels for the ETF IWM, which tracks small-cap U.S. stocks"
]

def run_single_pipeline(prompt: str, idx: int, verbose: bool = True) -> dict:
    metrics = StrategyRunMetrics(prompt)  # 🧪 Initialize metrics

    print(f"\n🧠 [Strategy {idx}] Prompt: {prompt}")
    try:
        # Strategy generation
        strat_agent = StrategyGenAgent()
        mcp: StrategyMCP = strat_agent.generate_strategy(prompt=prompt)
        metrics.mark("strategy_gen")
        if verbose: print("✅ Strategy generated.")

        # Backtest
        bt_agent = BacktestAgent(mcp)
        results = bt_agent.simulate()
        metrics.mark("backtest")
        if verbose: print("📊 Backtest complete.")

        # Explanation
        ex_agent = ExplainabilityAgent()
        explanation = ex_agent.explain(mcp, results)
        metrics.mark("explain")
        if verbose: print("📝 Explanation generated.")

        # Complete metrics tracking
        metrics.complete(
            sharpe=results.get("sharpe_ratio", 0),
            win_rate=results.get("win_rate", 0),
            avg_return=results.get("average_return", 0)
        )

        metrics.print_summary()

        return {
            "Prompt": prompt,
            "Strategy Name": mcp.strategy_name,
            "Symbol": results.get("symbol", "N/A"),
            "Sharpe Ratio": round(results.get("sharpe_ratio", 0), 3),
            "Sortino Ratio": round(results.get("sortino_ratio", 0), 3),
            "Avg Return": round(results.get("average_return", 0), 3),
            "CAGR (%)": round(results.get("cagr", 0), 2),
            "Max Drawdown (%)": round(results.get("max_drawdown", 0), 2),
            "Hit Ratio (%)": round(results.get("hit_ratio", 0), 2),
            "Win Rate (%)": round(results.get("win_rate", 0), 2),
            "Total Trades": results.get("total_trades", 0),
            "Equity Curve": results.get("equity_curve", [])[:10],
            "Explanation": explanation,
            "Runtime (s)": round(time.time() - metrics.start_time, 2)
        }

    except Exception as e:
        metrics.fail(e)
        metrics.print_summary()
        print(f"❌ Error running strategy {idx}: {e}")
        return None


def main():
    all_results = []
    start = time.time()

    for idx, prompt in enumerate(PROMPTS):
        result = run_single_pipeline(prompt, idx)
        if result:
            all_results.append(result)

    if not all_results:
        print("\n⚠️ No successful strategies to display.")
        return

    # Leaderboard
    df = pd.DataFrame(all_results)
    df_sorted = df.sort_values(by="Sharpe Ratio", ascending=False)

    print("\n🏆 Strategy Leaderboard:\n")
    print(tabulate(
        df_sorted.drop(columns=["Prompt", "Explanation", "Equity Curve"]),
        headers="keys",
        tablefmt="fancy_grid",
        showindex=False
    ))

    # Explanations
    print("\n🧠 Strategy Explanations:\n")
    for _, row in df_sorted.iterrows():
        print(f"🔹 {row['Strategy Name']} ({row['Symbol']}):")
        print(row["Explanation"].strip() + "\n")

    # Equity Curve Preview
    print("📈 Equity Curve Previews:")
    for _, row in df_sorted.iterrows():
        curve = row["Equity Curve"]
        preview = f"{curve[:5]}..." if curve else "No data"
        print(f"  ▸ {row['Strategy Name']} ({row['Symbol']}): {preview}")

    print(f"\n⏱️ Total runtime: {round(time.time() - start, 2)}s")

if __name__ == "__main__":
    main()
