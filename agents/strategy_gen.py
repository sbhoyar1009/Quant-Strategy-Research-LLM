import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
from mcp.schema import StrategyMCP
from google.generativeai import configure, GenerativeModel
from dotenv import load_dotenv

load_dotenv()

class StrategyGenAgent:
    def __init__(self):
        configure(api_key=os.getenv("GEMINI_API_KEY"))
        self.model = GenerativeModel("gemini-2.5-flash")
        self.system_prompt = {
            "strategy_name": "string",
            "symbol": "string (e.g., AAPL)",
            "indicators": [
                {
                    "name": "string",
                    "parameters": { "key": "value" },
                    "description": "string"
                }
            ],
            "entry_criteria": {
                "long_entry": {
                    "conditions": [ "string", "string" ],
                    "description": "string"
                },
                "short_entry": {
                    "conditions": [ "string", "string" ],
                    "description": "string"
                },
                "considerations": "string"
            },
            "exit_criteria": {
                "stop_loss": {
                    "long_stop_loss": "string",
                    "short_stop_loss": "string",
                    "description": "string"
                },
                "take_profit": {
                    "long_take_profit": "string",
                    "short_take_profit": "string",
                    "description": "string"
                },
                "trailing_stop_alternative": {
                    "long_exit": "string",
                    "short_exit": "string",
                    "description": "string"
                }
            },
            "timeframe": "string (e.g., 1d, Daily)"
        }

    def generate_strategy(self, prompt: str, max_retries: int = 2) -> StrategyMCP:
        # Stage 1: Generate a text-based plan
        plan_prompt = (
            f"Given the following user request, write a detailed trading strategy plan in plain English. "
            f"Do not use JSON or code blocks. Be as clear and structured as possible.\n\n"
            f"User request: {prompt}"
        )
        print("📤 Prompting Gemini for plan...")
        plan_response = self.model.generate_content(plan_prompt)
        plan_text = plan_response.text.strip()

        # Stage 2: Generate structured JSON from the plan
        json_prompt = (
            f"Given this trading strategy plan:\n\n"
            f"{plan_text}\n\n"
            f"Convert it into a JSON object matching this schema:\n{json.dumps(self.system_prompt, indent=2)}\n"
            f"Respond ONLY with a JSON code block."
        )

        locked_symbol = None  # <-- lock the symbol after first parse

        for attempt in range(max_retries + 1):
            print(f"📤 Prompting Gemini for structured JSON (attempt {attempt+1})...")
            response = self.model.generate_content(json_prompt)
            raw_output = response.text.strip()

            try:
                json_str = self.extract_json_from_code_block(raw_output)
                parsed = json.loads(json_str)
                # Lock the symbol on first parse, always enforce after
                if locked_symbol is None:
                    locked_symbol = parsed.get("symbol", "AAPL")
                parsed["symbol"] = locked_symbol
                parsed["assets"] = [locked_symbol]
            except Exception as e:
                print(f"❌ Model output is not valid JSON: {e}")
                if attempt == max_retries:
                    raise ValueError("Failed to get valid JSON after retries.")
                # Retry with a correction prompt
                json_prompt = (
                    f"The previous output was not valid JSON or did not match the schema. "
                    f"Please try again. Here is the schema:\n{json.dumps(self.system_prompt, indent=2)}\n"
                    f"Respond ONLY with a JSON code block."
                )
                continue

            try:
                return StrategyMCP(**parsed)
            except Exception as e:
                print(f"❌ Failed to validate strategy: {e}")
                if attempt == max_retries:
                    raise ValueError("Failed to validate strategy after retries.")
                # Retry with a correction prompt
                missing_fields = self.get_missing_fields(parsed)
                json_prompt = (
                    f"The previous output was missing required fields: {missing_fields}. "
                    f"Please correct and output valid JSON matching this schema:\n{json.dumps(self.system_prompt, indent=2)}\n"
                    f"Respond ONLY with a JSON code block."
                )
        raise ValueError("Failed to generate a valid strategy after retries.")

    @staticmethod
    def extract_json_from_code_block(text: str) -> str:
        start = text.find("```json")
        end = text.find("```", start + 1)
        if start == -1 or end == -1:
            raise ValueError("❌ JSON code block not found.")
        return text[start + 7:end].strip()

    @staticmethod
    def get_missing_fields(parsed: dict) -> list:
        # Example: check for top-level fields in the schema
        required = [
            "strategy_name", "symbol", "indicators", "entry_criteria",
            "exit_criteria", "timeframe"
        ]
        return [field for field in required if field not in parsed]

