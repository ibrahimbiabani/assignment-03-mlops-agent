"""LLM access for the orchestrator.

The orchestrator only sees LLMTurn / ToolCall, so it does not depend on Gemini types
and tests can substitute a scripted LLM. GeminiLLM uses manual function calling
(automatic function calling is disabled) so every tool call passes through the
orchestrator's validation, permission and logging layer.
"""

import os
from dataclasses import dataclass, field
from typing import Any, Protocol

from agent.registry import ToolSpec


@dataclass
class ToolCall:
    name: str
    args: dict[str, Any]
    id: str | None = None


@dataclass
class LLMTurn:
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)


class LLMError(Exception):
    """The language model could not produce a turn (API error, timeout, empty reply)."""


class LLMSession(Protocol):
    def send_user(self, message: str) -> LLMTurn: ...

    def send_tool_results(
        self, results: list[tuple[ToolCall, dict[str, Any]]], allow_tools: bool = True
    ) -> LLMTurn: ...


class LLMClient(Protocol):
    model: str

    def start(self, system_prompt: str, tools: list[ToolSpec]) -> LLMSession: ...


class GeminiLLM:
    """Google Gemini via the google-genai SDK. Reads GEMINI_API_KEY / GEMINI_MODEL."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        thinking_level: str | None = None,
        timeout_s: float | None = None,
        client: Any = None,
    ) -> None:
        from google import genai
        from google.genai import types

        self._types = types
        self.model = model or os.environ.get("GEMINI_MODEL", "gemini-3.5-flash")
        self.thinking_level = thinking_level or os.environ.get("GEMINI_THINKING_LEVEL", "low")
        if client is None:
            key = api_key or os.environ.get("GEMINI_API_KEY")
            if not key:
                raise LLMError("GEMINI_API_KEY is not set")
            timeout_ms = int(1000 * (timeout_s or float(os.environ.get("GEMINI_TIMEOUT_S", "90"))))
            client = genai.Client(
                api_key=key,
                http_options=types.HttpOptions(
                    timeout=timeout_ms,
                    retry_options=types.HttpRetryOptions(
                        attempts=3, http_status_codes=[429, 500, 503]
                    ),
                ),
            )
        self._client = client

    def start(self, system_prompt: str, tools: list[ToolSpec]) -> "GeminiSession":
        t = self._types
        declarations = [
            t.FunctionDeclaration(
                name=s.name, description=s.description, parameters_json_schema=s.parameters_schema()
            )
            for s in tools
        ]
        return GeminiSession(self, system_prompt, [t.Tool(function_declarations=declarations)])


class GeminiSession:
    def __init__(self, llm: GeminiLLM, system_prompt: str, tools: list[Any]) -> None:
        self._llm = llm
        self._system = system_prompt
        self._tools = tools
        self.contents: list[Any] = []

    def _config(self, allow_tools: bool) -> Any:
        t = self._llm._types
        mode = t.FunctionCallingConfigMode.AUTO if allow_tools else t.FunctionCallingConfigMode.NONE
        return t.GenerateContentConfig(
            system_instruction=self._system,
            tools=self._tools,
            tool_config=t.ToolConfig(function_calling_config=t.FunctionCallingConfig(mode=mode)),
            automatic_function_calling=t.AutomaticFunctionCallingConfig(disable=True),
            thinking_config=t.ThinkingConfig(thinking_level=self._llm.thinking_level),
        )

    def _generate(self, allow_tools: bool) -> LLMTurn:
        try:
            response = self._llm._client.models.generate_content(
                model=self._llm.model, contents=self.contents, config=self._config(allow_tools)
            )
        except Exception as exc:  # noqa: BLE001 - SDK raises APIError, httpx timeouts, ...
            raise LLMError(
                f"Gemini request failed: {type(exc).__name__}: {str(exc)[:300]}"
            ) from exc
        candidates = response.candidates or []
        content = candidates[0].content if candidates else None
        if content is None or not content.parts:
            reason = candidates[0].finish_reason if candidates else "no candidates"
            raise LLMError(f"Gemini returned an empty response (finish_reason={reason})")
        # Keep the model's content unchanged: it carries thought signatures Gemini needs back.
        self.contents.append(content)
        calls = [
            ToolCall(name=c.name, args=dict(c.args or {}), id=c.id)
            for c in response.function_calls or []
        ]
        text = "".join(p.text for p in content.parts if p.text and not p.thought)
        return LLMTurn(text=text, tool_calls=calls)

    def send_user(self, message: str) -> LLMTurn:
        t = self._llm._types
        self.contents.append(t.Content(role="user", parts=[t.Part(text=message)]))
        return self._generate(allow_tools=True)

    def send_tool_results(
        self, results: list[tuple[ToolCall, dict[str, Any]]], allow_tools: bool = True
    ) -> LLMTurn:
        t = self._llm._types
        parts = [
            t.Part(
                function_response=t.FunctionResponse(id=call.id, name=call.name, response=payload)
            )
            for call, payload in results
        ]
        self.contents.append(t.Content(role="user", parts=parts))
        return self._generate(allow_tools=allow_tools)
