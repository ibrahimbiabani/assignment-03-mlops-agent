"""GeminiLLM adapter tests with a fake SDK client returning real google-genai types."""

import pytest
from google.genai import types

from agent.llm import GeminiLLM, LLMError, ToolCall
from agent.registry import default_registry


class FakeModels:
    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []

    def generate_content(self, *, model, contents, config):
        self.requests.append({"model": model, "contents": list(contents), "config": config})
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def response(*parts):
    return types.GenerateContentResponse(
        candidates=[types.Candidate(content=types.Content(role="model", parts=list(parts)))]
    )


def make_llm(*responses):
    client = type("Client", (), {})()
    client.models = FakeModels(responses)
    return GeminiLLM(model="gemini-test", thinking_level="low", client=client), client.models


def test_function_calls_round_trip_with_ids_and_thought_signature():
    call_part = types.Part(
        function_call=types.FunctionCall(
            id="call-1", name="mlflow_find_best_run", args={"metric": "f1"}
        ),
        thought_signature=b"sig",
    )
    llm, models = make_llm(response(call_part), response(types.Part(text="Run r1 is best.")))
    session = llm.start("system prompt", list(default_registry().values()))

    turn = session.send_user("best model?")
    assert turn.tool_calls == [
        ToolCall(name="mlflow_find_best_run", args={"metric": "f1"}, id="call-1")
    ]

    final = session.send_tool_results([(turn.tool_calls[0], {"status": "success", "data": {}})])
    assert final.text == "Run r1 is best." and final.tool_calls == []

    second = models.requests[1]
    model_turn, tool_turn = second["contents"][1], second["contents"][2]
    assert model_turn.parts[0].thought_signature == b"sig"  # model content kept unchanged
    fr = tool_turn.parts[0].function_response
    assert (fr.id, fr.name, fr.response["status"]) == ("call-1", "mlflow_find_best_run", "success")


def test_config_disables_automatic_calls_and_declares_all_tools():
    llm, models = make_llm(response(types.Part(text="hi")))
    llm.start("sys", list(default_registry().values())).send_user("hello")

    config = models.requests[0]["config"]
    assert config.automatic_function_calling.disable is True
    assert config.system_instruction == "sys"
    names = {d.name for d in config.tools[0].function_declarations}
    assert "k8s_restart_deployment" in names and len(names) == len(default_registry())
    assert config.tool_config.function_calling_config.mode == types.FunctionCallingConfigMode.AUTO


def test_final_turn_can_disable_tools():
    llm, models = make_llm(response(types.Part(text="a")), response(types.Part(text="b")))
    session = llm.start("sys", [])
    session.send_user("q")
    session.send_tool_results([], allow_tools=False)
    mode = models.requests[1]["config"].tool_config.function_calling_config.mode
    assert mode == types.FunctionCallingConfigMode.NONE


def test_thought_parts_are_not_returned_as_answer_text():
    llm, _ = make_llm(
        response(types.Part(text="thinking...", thought=True), types.Part(text="Answer"))
    )
    assert llm.start("sys", []).send_user("q").text == "Answer"


@pytest.mark.parametrize(
    "failure",
    [
        RuntimeError("503 UNAVAILABLE"),
        types.GenerateContentResponse(candidates=[]),
    ],
)
def test_api_errors_and_empty_responses_raise_llm_error(failure):
    llm, _ = make_llm(failure)
    with pytest.raises(LLMError):
        llm.start("sys", []).send_user("q")


def test_missing_api_key_is_reported(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(LLMError, match="GEMINI_API_KEY"):
        GeminiLLM()
