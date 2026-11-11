"""LLM access for the orchestrator.

The orchestrator only sees LLMTurn / ToolCall, so it does not depend on Gemini types
and tests can substitute a scripted LLM. GeminiLLM uses manual function calling
(automatic function calling is disabled) so every tool call passes through the
orchestrator's validation, permission and logging layer.

Each Gemini request goes through agent.gemini_pool.ModelPool: the first available model in
the free-tier fallback chain is used, requests are paced and never run concurrently, and a
model that fails with a quota/availability error is put on cooldown while the same request
moves to the next model. The conversation history is sent unchanged to whichever model
serves the next turn (Gemini manages thought-signature compatibility across models).
"""

import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

from agent.gemini_pool import FatalLLMError, ModelPool, ModelUnavailable, classify_error
from agent.registry import ToolSpec
from agent.tools.base import request_id_var

log = logging.getLogger("mlops_agent.llm")


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
    """Google Gemini via the google-genai SDK, across a free-tier model fallback chain.

    Configuration: GEMINI_API_KEY, GEMINI_MODELS (comma-separated, best first; GEMINI_MODEL
    is accepted for a single model), GEMINI_THINKING_LEVEL, GEMINI_TIMEOUT_S, plus the pacing
    settings read by ModelPool.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        thinking_level: str | None = None,
        timeout_s: float | None = None,
        client: Any = None,
        pool: ModelPool | None = None,
    ) -> None:
        from google import genai
        from google.genai import types

        self._types = types
        self.pool = pool or ModelPool(models=[model] if model else None)
        self.model = " > ".join(self.pool.models)  # the chain, for logs and responses
        self.thinking_level = thinking_level or os.environ.get("GEMINI_THINKING_LEVEL", "low")
        self.no_thinking: set[str] = set()  # models that rejected thinking_config
        if client is None:
            key = api_key or os.environ.get("GEMINI_API_KEY")
            if not key:
                raise LLMError("GEMINI_API_KEY is not set")
            timeout_ms = int(1000 * (timeout_s or float(os.environ.get("GEMINI_TIMEOUT_S", "60"))))
            client = genai.Client(
                api_key=key,
                http_options=types.HttpOptions(
                    timeout=timeout_ms,
                    # No SDK retries: retrying a 429 only spends more of the free quota.
                    # Failures are handled by moving to the next model instead.
                    retry_options=types.HttpRetryOptions(attempts=1),
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
        self.models_used: list[str] = []
        self.fallbacks: list[str] = []  # "model: reason" for every model that was skipped

    def _config(self, allow_tools: bool, model: str) -> Any:
        t = self._llm._types
        mode = t.FunctionCallingConfigMode.AUTO if allow_tools else t.FunctionCallingConfigMode.NONE
        thinking = (
            None
            if model in self._llm.no_thinking
            else t.ThinkingConfig(thinking_level=self._llm.thinking_level)
        )
        return t.GenerateContentConfig(
            system_instruction=self._system,
            tools=self._tools,
            tool_config=t.ToolConfig(function_calling_config=t.FunctionCallingConfig(mode=mode)),
            automatic_function_calling=t.AutomaticFunctionCallingConfig(disable=True),
            thinking_config=thinking,
        )

    def _log(self, model: str, status: str, start: float, reason: str | None = None) -> None:
        log.info(
            json.dumps(
                {
                    "event": "llm_call",
                    "request_id": request_id_var.get(),
                    "model": model,
                    "status": status,
                    "duration_ms": round((time.perf_counter() - start) * 1000, 1),
                    "reason": reason,
                }
            )
        )

    def _call_with_fallback(self, allow_tools: bool) -> Any:
        pool, tried = self._llm.pool, set()
        while True:
            with pool.in_flight:  # never more than one Gemini request at a time
                try:
                    model = pool.acquire(exclude=tried)
                except ModelUnavailable as exc:
                    raise LLMError(f"No Gemini model is available right now: {exc.reason}") from exc
                start = time.perf_counter()
                try:
                    response = self._llm._client.models.generate_content(
                        model=model, contents=self.contents, config=self._config(allow_tools, model)
                    )
                except Exception as exc:  # noqa: BLE001 - classified below
                    message = str(getattr(exc, "message", None) or exc).lower()
                    if (
                        getattr(exc, "code", None) == 400
                        and "thinking" in message
                        and model not in self._llm.no_thinking
                    ):
                        self._llm.no_thinking.add(model)  # retry this model without thinking
                        self._log(model, "retry_without_thinking", start, message[:120])
                        continue
                    err = classify_error(model, exc, pool.clock())
                    if isinstance(err, FatalLLMError):
                        self._log(model, "fatal", start, str(err))
                        raise LLMError(str(err)) from exc
                    self._log(model, "unavailable", start, err.reason)
                    pool.cooldown(model, err.cooldown_s, err.reason)
                    self.fallbacks.append(f"{model}: {err.reason}")
                    tried.add(model)
                    continue
            self._log(model, "ok", start)
            if model not in self.models_used:
                self.models_used.append(model)
            return response

    def _generate(self, allow_tools: bool) -> LLMTurn:
        response = self._call_with_fallback(allow_tools)
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
