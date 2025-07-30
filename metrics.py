import time
from tabulate import tabulate


class StrategyRunMetrics:
    def __init__(self, prompt: str):
        self.prompt = prompt
        self.start_time = time.time()
        self.times = {}
        self.results = {}
        self.success = False
        self.error = None

    def mark(self, stage: str):
        self.times[stage] = time.time()

    def complete(self, sharpe: float, win_rate: float, avg_return: float):
        self.results = {
            "Sharpe Ratio": round(sharpe, 3),
            "Win Rate (%)": round(win_rate, 2),
            "Avg Return": round(avg_return, 3)
        }
        self.success = True

    def fail(self, error: Exception):
        self.error = str(error)
        self.success = False

    def print_summary(self):
        print("\n🧪 Performance Summary:")
        data = [
            ["Prompt", self.prompt],
            ["Status", "✅ Success" if self.success else "❌ Failure"],
            ["Total Runtime (s)", round(time.time() - self.start_time, 2)]
        ]

        if self.success:
            data += [
                ["Time to Strategy Gen (s)", round(self.times.get("strategy_gen", 0) - self.start_time, 2)],
                ["Time to Backtest (s)", round(self.times.get("backtest", 0) - self.times.get("strategy_gen", 0), 2)],
                ["Time to Explain (s)", round(self.times.get("explain", 0) - self.times.get("backtest", 0), 2)],
                ["Sharpe Ratio", self.results["Sharpe Ratio"]],
                ["Win Rate (%)", self.results["Win Rate (%)"]],
                ["Avg Return", self.results["Avg Return"]],
            ]
        else:
            data.append(["Error", self.error])

        print(tabulate(data, headers=["Metric", "Value"], tablefmt="fancy_grid"))
