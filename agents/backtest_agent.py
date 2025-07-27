import yfinance as yf
import pandas as pd
from datetime import datetime
from typing import List, Tuple, Dict, Union
import numpy as np

class BacktestAgent:
    def __init__(self, strategy_mcp, initial_cash: float = 100000, position_size: float = 1.0, slippage: float = 0.001, transaction_cost: float = 0.0005):
        self.symbol = strategy_mcp.symbol
        self.start_date = "2020-01-01"
        self.end_date = datetime.now().strftime("%Y-%m-%d")
        self.initial_cash = initial_cash
        self.position_size = position_size  # Fraction of cash to use per trade (1.0 = all-in)
        self.slippage = slippage            # Slippage as a fraction of price (e.g., 0.001 = 0.1%)
        self.transaction_cost = transaction_cost  # Per trade as a fraction of notional

    def fetch_data(self) -> pd.DataFrame:
        print(f"📈 Downloading data for {self.symbol}")
        df = yf.download(self.symbol, start=self.start_date, end=self.end_date, interval="1d")

        if df.empty:
            raise ValueError(f"No data found for {self.symbol}")

        if isinstance(df.columns, pd.MultiIndex):
            df = df.xs(self.symbol, axis=1, level=1)

        df.dropna(inplace=True)

        df["RSI"] = self.compute_rsi(df["Close"])
        df["MA"] = df["Close"].rolling(window=20).mean()
        std = df["Close"].rolling(window=20).std()
        df["UpperBB"] = df["MA"] + 2 * std
        df["LowerBB"] = df["MA"] - 2 * std

        return df.dropna()

    @staticmethod
    def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
        delta = series.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    def simulate(self) -> Dict:
        print("🚀 Running backtest simulation...")
        df = self.fetch_data()

        cash = self.initial_cash
        equity = self.initial_cash
        shares = 0
        holding = False
        entry_price = 0
        trades: List[Tuple[str, str, float, int, float]] = []  # (date, side, price, shares, equity)
        equity_curve: List[Tuple[str, float]] = []

        for idx, row in df.iterrows():
            date = idx.strftime('%Y-%m-%d')
            rsi = float(row["RSI"])
            close = float(row["Close"])
            lower_bb = float(row["LowerBB"])
            ma = float(row["MA"])

            # Track equity daily
            if holding:
                equity = shares * close + cash
            else:
                equity = cash
            equity_curve.append((date, equity))

            # Buy signal
            if not holding and rsi < 30 and close < lower_bb:
                # Position sizing
                alloc_cash = cash * self.position_size
                buy_price = close * (1 + self.slippage)
                n_shares = int(alloc_cash // buy_price)
                if n_shares > 0:
                    cost = n_shares * buy_price
                    fee = cost * self.transaction_cost
                    cash -= (cost + fee)
                    shares += n_shares
                    holding = True
                    entry_price = buy_price
                    trades.append((date, "BUY", buy_price, n_shares, equity))

            # Sell signal
            elif holding and (rsi > 70 or close > ma):
                sell_price = close * (1 - self.slippage)
                proceeds = shares * sell_price
                fee = proceeds * self.transaction_cost
                cash += (proceeds - fee)
                trades.append((date, "SELL", sell_price, shares, equity))
                shares = 0
                holding = False
                entry_price = 0

        # Final equity update if still holding
        if holding and shares > 0:
            final_price = df.iloc[-1]["Close"] * (1 - self.slippage)
            proceeds = shares * final_price
            fee = proceeds * self.transaction_cost
            cash += (proceeds - fee)
            trades.append((df.index[-1].strftime('%Y-%m-%d'), "SELL", final_price, shares, cash))
            shares = 0
            holding = False

        # Final equity curve update
        equity = cash
        equity_curve.append((df.index[-1].strftime('%Y-%m-%d'), equity))

        # Calculate trade returns
        trade_returns = []
        for i in range(0, len(trades) - 1, 2):
            buy = trades[i]
            sell = trades[i + 1] if i + 1 < len(trades) else None
            if sell and buy[1] == "BUY" and sell[1] == "SELL":
                ret = (sell[2] - buy[2]) / buy[2]
                trade_returns.append(ret)

        # Metrics
        profits = [r * self.position_size * self.initial_cash for r in trade_returns]
        num_wins = sum(1 for r in trade_returns if r > 0)
        hit_ratio = round((num_wins / len(trade_returns)) * 100, 2) if trade_returns else 0.0

        # Max Drawdown
        eq_vals = [e[1] for e in equity_curve]
        roll_max = np.maximum.accumulate(eq_vals)
        drawdowns = (eq_vals - roll_max) / roll_max
        max_drawdown = round(drawdowns.min() * 100, 2) if len(drawdowns) > 0 else 0.0

        # CAGR
        if len(equity_curve) > 1:
            start_val = equity_curve[0][1]
            end_val = equity_curve[-1][1]
            days = (pd.to_datetime(equity_curve[-1][0]) - pd.to_datetime(equity_curve[0][0])).days
            years = days / 365.25
            cagr = round(((end_val / start_val) ** (1 / years) - 1) * 100, 2) if years > 0 else 0.0
        else:
            cagr = 0.0

        # Sharpe & Sortino
        returns = np.diff(eq_vals) / eq_vals[:-1] if len(eq_vals) > 1 else np.array([0])
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        sharpe_ratio = round(mean_return / std_return * np.sqrt(252), 4) if std_return > 0 else 0.0
        downside_returns = returns[returns < 0]
        std_downside = np.std(downside_returns) if len(downside_returns) > 0 else 0.0
        sortino_ratio = round(mean_return / std_downside * np.sqrt(252), 4) if std_downside > 0 else 0.0

        avg_return = round(np.mean(profits), 4) if profits else 0.0

        return {
            "symbol": self.symbol,
            "total_trades": len(trade_returns),
            "average_return": avg_return,
            "sharpe_ratio": sharpe_ratio,
            "sortino_ratio": sortino_ratio,
            "max_drawdown": max_drawdown,
            "cagr": cagr,
            "hit_ratio": hit_ratio,
            "win_rate": hit_ratio,  # alias
            "equity_curve": equity_curve,
            "trades": trades
        }
