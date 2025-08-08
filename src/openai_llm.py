# src/openai_llm.py
import os, time, random
from typing import Dict, Optional
from openai import OpenAI
try:
    from openai import APIError, RateLimitError  # SDK >= 1.x
except Exception:
    APIError = Exception
    RateLimitError = Exception

ISSUES_SCHEMA: Dict = {
    "name": "issues_schema",
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "issues": {
                "type": "array",
                "maxItems": 5,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "title": {"type": "string"},
                        "description": {"type": "string"},
                        "frequency_estimate": {"type": "string", "enum": ["low","medium","high"]},
                        "example_comments": {"type": "array", "items": {"type": "string"}, "maxItems": 3}
                    },
                    "required": ["title","description","frequency_estimate","example_comments"]
                }
            }
        },
        "required": ["issues"]
    }
}

_SYSTEM = (
    "You extract recurring issues/dissatisfactions from social comments for one Instagram post. "
    "Group similar complaints; ignore spam and @user tags. "
    "Your output MUST be valid JSON and should match the provided JSON schema."
)

_client: Optional[OpenAI] = None
def _get_client(api_key: Optional[str] = None) -> OpenAI:
    global _client
    if api_key:  # explicit key wins
        return OpenAI(api_key=api_key)
    if _client is None:
        key = os.getenv("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("Set OPENAI_API_KEY or pass api_key to openai_llm_call().")
        _client = OpenAI(api_key=key)
    return _client

def openai_llm_call(prompt: str, model: str = "gpt-4o-mini", api_key: Optional[str] = None) -> str:
    """
    Calls Chat Completions with JSON output.
    - First tries structured JSON schema (newer SDKs).
    - Falls back to generic JSON mode if schema isn't supported.
    Returns a JSON string.
    """
    client = _get_client(api_key)

    for attempt in range(4):
        try:
            # Try strict JSON schema (supported in recent SDKs)
            try:
                resp = client.chat.completions.create(
                    model=model,
                    temperature=0.2,
                    response_format={"type": "json_schema", "json_schema": ISSUES_SCHEMA},
                    messages=[
                        {"role": "system", "content": _SYSTEM},
                        {"role": "user", "content": prompt},
                    ],
                )
                return resp.choices[0].message.content  # JSON string
            except TypeError:
                # Older SDK: fall back to generic JSON mode + embed schema text
                resp = client.chat.completions.create(
                    model=model,
                    temperature=0.2,
                    response_format={"type": "json_object"},
                    messages=[
                        {"role": "system", "content": _SYSTEM + " Use this schema strictly."},
                        {"role": "system", "content": str(ISSUES_SCHEMA["schema"])},
                        {"role": "user", "content": prompt},
                    ],
                )
                return resp.choices[0].message.content  # JSON string
        except (RateLimitError, APIError):
            if attempt == 3:
                raise
            time.sleep(1.2 * (2 ** attempt) + random.random() * 0.4)
        except Exception:
            if attempt == 3:
                raise
            time.sleep(1.0 + random.random() * 0.5)
