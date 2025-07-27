from typing import List, Dict, Union
from pydantic import BaseModel

class Indicator(BaseModel):
    name: str
    parameters: Dict[str, Union[str, float, int]]
    description: str

class CriteriaBlock(BaseModel):
    conditions: List[str]
    description: str

class EntryCriteria(BaseModel):
    long_entry: CriteriaBlock
    short_entry: CriteriaBlock
    considerations: str

class StopLoss(BaseModel):
    long_stop_loss: str
    short_stop_loss: str
    description: str

class TakeProfit(BaseModel):
    long_take_profit: str
    short_take_profit: str
    description: str

class TrailingStop(BaseModel):
    long_exit: str
    short_exit: str
    description: str

class ExitCriteria(BaseModel):
    stop_loss: StopLoss
    take_profit: TakeProfit
    trailing_stop_alternative: TrailingStop

class StrategyMCP(BaseModel):
    strategy_name: str
    symbol: str
    indicators: List[Indicator]
    entry_criteria: EntryCriteria
    exit_criteria: ExitCriteria
    timeframe: str
