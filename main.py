from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from agents.strategy_gen import StrategyGenAgent
from agents.backtest_agent import BacktestAgent
from agents.explain_agent import ExplainabilityAgent
from mcp.schema import StrategyMCP
from metrics import StrategyRunMetrics
from models.run_store import run_store

app = FastAPI()


class PromptRequest(BaseModel):
    prompt: str


@app.post("/run-strategy")
def run_strategy(req: PromptRequest):
    metrics = StrategyRunMetrics(prompt=req.prompt)
    try:
        strat_agent = StrategyGenAgent()
        metrics.mark("strategy_gen")
        mcp: StrategyMCP = strat_agent.generate_strategy(req.prompt)

        bt_agent = BacktestAgent(mcp)
        metrics.mark("backtest")
        results = bt_agent.simulate()

        ex_agent = ExplainabilityAgent()
        metrics.mark("explain")
        explanation = ex_agent.explain(mcp, results)

        metrics.complete(
            sharpe=results.get("sharpe_ratio", 0),
            win_rate=results.get("win_rate", 0),
            avg_return=results.get("average_return", 0)
        )

        run_id = run_store.add({
            "prompt": req.prompt,
            "strategy": mcp.dict(),
            "results": results,
            "explanation": explanation,
            "metrics": metrics.results
        })

        return {
            "id": run_id,
            "strategy": mcp,
            "results": results,
            "explanation": explanation,
            "metrics": metrics.results
        }

    except Exception as e:
        metrics.fail(e)
        metrics.print_summary()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/strategies")
def list_strategies():
    return run_store.all()


@app.get("/strategy/{id}")
def get_strategy(id: int):
    item = run_store.get(id)
    if item is None:
        raise HTTPException(status_code=404, detail="Strategy not found")
    return item
