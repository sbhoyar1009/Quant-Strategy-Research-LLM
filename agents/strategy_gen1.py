import json
import re
from transformers import pipeline
from langchain_huggingface import HuggingFacePipeline
from pydantic import ValidationError
from mcp.schema import StrategyMCP


class StrategyGenAgent:
    def __init__(self):
        pipe = pipeline(
            "text2text-generation",
            model="google/flan-t5-small",
            device="mps",
            # return_full_text=False,
            # max_new_tokens=512
        )
        self.llm = HuggingFacePipeline(pipeline=pipe)

    def _build_prompt(self, user_prompt: str) -> str:
        return f"""You are a quantitative trading expert.
Respond ONLY with a valid JSON object using this structure:

{{
  "strategy_name": "Momentum MA Strategy",
  "description": "Buy when 50-day MA crosses above 200-day MA",
  "assets": ["AAPL"],
  "timeframe": "1d",
  "entry_rules": "50-day MA crosses above 200-day MA",
  "exit_rules": "50-day MA crosses below 200-day MA"
}}

Now generate the strategy for this prompt:
\"\"\"{user_prompt}\"\"\"
"""

    def _extract_json(self, s: str) -> str:
      # Step 1: Fix malformed colon and add "description" key
        s_fixed = re.sub(r'("Strategy_name"\s*:\s*"[^"]+)"\s*:\s*"', r'\1", "description": "', s, count=1)

        # Step 2: Wrap in braces to make it a valid JSON object
        s_fixed = '{' + s_fixed + '}'

        try:
            # Step 3: Load JSON to dict
            data = json.loads(s_fixed)

            # Step 4: Lowercase keys and string values
            lowercased_data = {
                k.lower(): v.lower() if isinstance(v, str) else v
                for k, v in data.items()
            }

            # Step 5: Return as a formatted JSON string
            return json.dumps(lowercased_data, indent=2)

        except json.JSONDecodeError as e:
            print("❌ Failed to parse JSON:", e)
            return "{}"

    def generate_strategy(self, prompt: str) -> StrategyMCP:
        full_prompt = self._build_prompt(prompt)
        print("📤 Prompting model...")
        print("with this prompt", prompt)

        raw = self.llm.invoke(full_prompt)
        raw_text = raw[0]['generated_text'] if isinstance(raw, list) else raw
        print("📥 Raw LLM Output:", repr(raw_text))

        try:
            cleaned = self._extract_json(raw_text)
            parsed = json.loads(cleaned)
        except (json.JSONDecodeError, ValueError) as e:
            raise ValueError(f"❌ Model output is not valid JSON: {e}")

        try:
            return StrategyMCP(
                strategy_name=parsed["strategy_name"],
                description=parsed["description"],
                assets=parsed["assets"],
                timeframe=parsed["timeframe"],
                entry_rules=parsed["entry_rules"],
                exit_rules=parsed["exit_rules"]
            )
        except ValidationError as ve:
            raise ValueError(f"❌ Invalid StrategyMCP fields: {ve}")
