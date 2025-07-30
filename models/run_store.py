class StrategyRunStore:
    def __init__(self):
        self._runs = []
        self._counter = 0

    def add(self, data: dict) -> int:
        self._counter += 1
        data["id"] = self._counter
        self._runs.append(data)
        return self._counter

    def all(self):
        return self._runs

    def get(self, run_id: int):
        return next((r for r in self._runs if r["id"] == run_id), None)


run_store = StrategyRunStore()
