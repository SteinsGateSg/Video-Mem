"""
LLM 适配器模块

本模块封装了与 OpenAI 兼容 API 的交互逻辑，使 STG 系统能将结构化证据发送给任意
支持 OpenAI chat completions 接口的大语言模型（如 GPT-4、Qwen、DeepSeek 等）。

核心类：
    OpenAICompatibleLLMAdapter:
        - 初始化参数：api_base（API 端点）、api_key、model（模型名）、temperature
        - answer(prompts, evidence): 将 grounded prompt 发送给 LLM，解析并返回结构化答案
        - 如果 evidence 中既没有事件也没有实体，直接返回 insufficient evidence 结果
        - 自动解析 LLM 返回的 JSON 响应；若解析失败则保留原始文本

异常：
    LLMAdapterError: API 配置缺失或 openai 包未安装时抛出

注意：openai 包是可选依赖，未安装时系统仍可正常构建和检索，只是无法调用 LLM。
"""

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
