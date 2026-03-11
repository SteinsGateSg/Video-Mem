from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

try:
    from openai import OpenAI
except Exception:  # pragma: no cover - optional dependency
    OpenAI = None  # type: ignore

from .utils import extract_json_object


class LLMAdapterError(RuntimeError):
    pass


@dataclass
class OpenAICompatibleLLMAdapter:
    api_base: str
    api_key: str
    model: str
    temperature: float = 0.1

    def _client(self) -> Any:
        if OpenAI is None:
            raise LLMAdapterError(
                "openai package is not available. Install it or run the script in an environment with OpenAI support."
            )
        if not self.api_base or not self.api_key:
            raise LLMAdapterError(
                "Missing API configuration. Provide both --api_base and --api_key, or use --dry_run to inspect the grounded prompt."
            )
        return OpenAI(base_url=self.api_base, api_key=self.api_key)

    def answer(self, prompts: Dict[str, str], evidence: Dict[str, Any]) -> Dict[str, Any]:
        if not evidence.get("events") and not evidence.get("entities"):
            return {
                "answer": "Insufficient evidence to answer the question.",
                "sufficient_evidence": False,
                "used_event_ids": [],
                "used_entity_ids": [],
                "short_rationale": "No event or entity evidence was retrieved above threshold.",
            }

        client = self._client()
        response = client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            messages=[
                {"role": "system", "content": prompts["system_prompt"]},
                {"role": "user", "content": prompts["user_prompt"]},
            ],
        )
        content = response.choices[0].message.content or ""
        parsed = self._parse_response(content)
        return parsed

    def _parse_response(self, content: str) -> Dict[str, Any]:
        try:
            payload = extract_json_object(content)
        except Exception:
            payload = {
                "answer": content.strip(),
                "sufficient_evidence": False,
                "used_event_ids": [],
                "used_entity_ids": [],
                "short_rationale": "Model did not return valid JSON; raw text was preserved.",
            }

        payload.setdefault("answer", "")
        payload["sufficient_evidence"] = bool(payload.get("sufficient_evidence", False))
        payload["used_event_ids"] = [str(item) for item in payload.get("used_event_ids", [])]
        payload["used_entity_ids"] = [str(item) for item in payload.get("used_entity_ids", [])]
        payload.setdefault("short_rationale", "")
        return payload
