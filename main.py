import time
import pandas as pd
from tabulate import tabulate

from agents.strategy_gen import StrategyGenAgent
from agents.backtest_agent import BacktestAgent
from agents.explain_agent import ExplainabilityAgent
from mcp.schema import StrategyMCP
from dotenv import load_dotenv
load_dotenv()

PROMPTS = [
    "Design a mean-reversion strategy for AAPL using RSI and Bollinger Bands.",
    "Create a momentum trading strategy using MACD and moving averages on TSLA.",
    "Propose a breakout trading strategy using ATR and Donchian Channels for the ETF IWM, which tracks small-cap U.S. stocks"
]

def run_single_pipeline(prompt: str, idx: int) -> dict:
    print(f"\n🧠 [Strategy {idx}] Prompt: {prompt}")
    try:
        strat_agent = StrategyGenAgent()
        mcp: StrategyMCP = strat_agent.generate_strategy(prompt=prompt)  
        print("✅ Strategy generated.")

        bt_agent = BacktestAgent(mcp)
        print("🚀 Running backtest...")
        results = bt_agent.simulate()
        print("📊 Backtest complete.")

        ex_agent = ExplainabilityAgent()
        print("🔍 Generating explanation...")
        explanation = ex_agent.explain(mcp, results)
        print("📝 Explanation generated.")

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
            "Equity Curve": results.get("equity_curve", []),
            "Explanation": explanation
        }

    except Exception as e:
        print(f"❌ Error running strategy {idx}: {e}")
        return None

def main():
    all_results = []
    start_time = time.time()

    for idx, prompt in enumerate(PROMPTS):
        result = run_single_pipeline(prompt, idx)
        if result:
            all_results.append(result)

    if all_results:
        df = pd.DataFrame(all_results)
        df_sorted = df.sort_values(by="Sharpe Ratio", ascending=False)

        print("\n🏆 Strategy Leaderboard:\n")
        print(tabulate(
            df_sorted.drop(columns=["Prompt", "Explanation", "Equity Curve"]),
            headers="keys",
            tablefmt="fancy_grid",
            showindex=False
        ))

        print("\n🧠 Strategy Explanations:\n")
        for _, row in df_sorted.iterrows():
            print(f"🔹 {row['Strategy Name']} ({row['Symbol']}):\n{row['Explanation']}\n")

        # Optionally, print equity curves for each strategy
        for _, row in df_sorted.iterrows():
            print(f"📈 Equity Curve for {row['Strategy Name']} ({row['Symbol']}):")
            print(row["Equity Curve"][:5], "...")  # Show first 5 points as a preview

    else:
        print("\n⚠️ No successful strategies to display.")

    print(f"\n⏱️ Total runtime: {round(time.time() - start_time, 2)}s")

if __name__ == "__main__":
    main()
