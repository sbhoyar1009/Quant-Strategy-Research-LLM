import os
from dotenv import load_dotenv
import google.generativeai as genai
from mcp.schema import StrategyMCP

load_dotenv()

class ExplainabilityAgent:
    def __init__(self):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise EnvironmentError("GEMINI_API_KEY not found in environment variables.")
        
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel("gemini-1.5-flash")

    def explain(
        self,
        strategy: StrategyMCP,
        backtest_results: dict,
        tone: str = "simple",  # "simple", "technical", "quant"
        output_format: str = "markdown"  # "markdown", "latex", "plain"
    ) -> str:
        summary = self.sanitize_results(backtest_results)
        risks = self.detect_risks(backtest_results)
        overfit_flag = self.detect_overfit(backtest_results)

        tone_instructions = {
            "simple": "Explain in clear, everyday language suitable for a non-expert.",
            "technical": "Use technical finance language suitable for an experienced trader.",
            "quant": "Use quantitative finance language and reference statistical concepts."
        }
        format_instructions = {
            "markdown": "Format your answer in Markdown.",
            "latex": "Format your answer in LaTeX (enclose equations in $$).",
            "plain": "Respond in plain text."
        }

        prompt = (
            f"You are a financial trading assistant. {tone_instructions.get(tone, tone_instructions['simple'])} "
            f"{format_instructions.get(output_format, format_instructions['markdown'])}\n\n"
            f"Analyze and explain the results of the following backtest.\n\n"
            f"📌 **Strategy Name:** {strategy.strategy_name}\n"
            f"📘 **Description:** {getattr(strategy, 'description', 'N/A')}\n"
            f"📊 **Assets:** {strategy.symbol}\n"
            f"⏱️ **Timeframe:** {strategy.timeframe}\n"
            f"🔼 **Entry Rules:** {getattr(strategy, 'entry_rules', 'N/A')}\n"
            f"🔽 **Exit Rules:** {getattr(strategy, 'exit_rules', 'N/A')}\n\n"
            f"### Backtest Summary\n"
            f"- **Average Return:** {summary['avg_return']}\n"
            f"- **Sharpe Ratio:** {summary['sharpe_ratio']}\n"
            f"- **Sortino Ratio:** {summary.get('sortino_ratio', 'N/A')}\n"
            f"- **Max Drawdown:** {summary.get('max_drawdown', 'N/A')}\n"
            f"- **CAGR:** {summary.get('cagr', 'N/A')}\n"
            f"- **Win Rate:** {summary['win_rate']}\n"
            f"- **Total Trades:** {summary['trades']}\n\n"
            f"### Risks & Failure Modes\n"
            f"{risks}\n\n"
            f"{overfit_flag}\n"
            "💬 Provide an explanation of the strategy performance, highlighting strengths, weaknesses, and any signs of overfitting or risk."
        )

        response = self.model.generate_content(prompt)
        return response.text.strip()

    def sanitize_results(self, results):
        sanitized = {
            "avg_return": round(float(results.get("average_return", 0.0)), 3),
            "sharpe_ratio": round(float(results.get("sharpe_ratio", 0.0)), 3),
            "sortino_ratio": round(float(results.get("sortino_ratio", 0.0)), 3) if "sortino_ratio" in results else "N/A",
            "max_drawdown": f"{round(results.get('max_drawdown', 0.0), 2)}%" if "max_drawdown" in results else "N/A",
            "cagr": f"{round(results.get('cagr', 0.0), 2)}%" if "cagr" in results else "N/A",
            "win_rate": f"{round(results.get('win_rate', 0.0), 2)}%" if "win_rate" in results else "N/A",
            "trades": len(results.get("trades", [])) if isinstance(results.get("trades"), list) else "N/A"
        }
        return sanitized

    def detect_risks(self, results):
        risks = []
        if results.get("max_drawdown", 0) < -20:
            risks.append("- High drawdown may indicate significant risk of large losses.")
        if results.get("total_trades", 0) < 10:
            risks.append("- Very few trades: results may not be statistically robust.")
        if results.get("cagr", 0) < 0:
            risks.append("- Negative CAGR: strategy lost money over the test period.")
        if results.get("sharpe_ratio", 0) < 1:
            risks.append("- Low Sharpe Ratio: risk-adjusted returns are weak.")
        if not risks:
            risks.append("- No major risks detected, but always validate with out-of-sample data.")
        return "\n".join(risks)

    def detect_overfit(self, results):
        sharpe = results.get("sharpe_ratio", 0)
        trades = results.get("total_trades", 0)
        if sharpe > 2 and trades < 10:
            return "⚠️ **Potential Overfitting:** High Sharpe Ratio with very few trades may indicate overfitting."
        return ""
