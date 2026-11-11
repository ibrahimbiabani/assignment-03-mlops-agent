"""Free-tier fallback chain and request pacing (no network: fake clock and fake SDK client)."""

import threading
import time
from datetime import datetime

import pytest
from google.genai import errors, types

from agent.gemini_pool import (
    LONG_COOLDOWN_S,
    PACIFIC,
    FatalLLMError,
    ModelPool,
    ModelUnavailable,
    classify_error,
)
from agent.llm import GeminiLLM, LLMError


def quota_error(quota_id, value="20", delay=None, code=429):
    """Same body shape as the real 429 responses observed from the Gemini API."""
    details = [
        {
            "@type": "type.googleapis.com/google.rpc.QuotaFailure",
            "violations": [{"quotaId": quota_id, "quotaValue": value}],
        }
    ]
    if delay:
        details.append({"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": delay})
    body = {
        "error": {
            "code": code,
            "message": "quota",
            "status": "RESOURCE_EXHAUSTED",
            "details": details,
        }
    }
    return errors.APIError(code, body)


DAILY = "GenerateRequestsPerDayPerProjectPerModel-FreeTier"
MINUTE = "GenerateRequestsPerMinutePerProjectPerModel-FreeTier"
NOW = datetime(2026, 11, 11, 22, 0, tzinfo=PACIFIC).timestamp()  # 10 PM Pacific


# ---------- error classification ----------


def test_daily_quota_uses_server_retry_delay():
    err = classify_error("m", quota_error(DAILY, "20", "3400s"), NOW)
    assert isinstance(err, ModelUnavailable)
    assert err.cooldown_s == 3400
    assert "daily free-tier quota (20/day)" in err.reason


def test_daily_quota_without_delay_waits_until_midnight_pacific():
    err = classify_error("m", quota_error(DAILY), NOW)
    assert err.cooldown_s == 2 * 3600  # 10 PM -> midnight Pacific


def test_model_without_free_tier():
    err = classify_error("gemini-4-argon", quota_error(DAILY, None, "3385s"), NOW)
    assert err.cooldown_s == LONG_COOLDOWN_S
    assert "no free-tier quota" in err.reason


def test_per_minute_limit_uses_short_delay():
    assert classify_error("m", quota_error(MINUTE, "10", "17s"), NOW).cooldown_s == 17
    assert classify_error("m", quota_error(MINUTE, "10"), NOW).cooldown_s == 60


@pytest.mark.parametrize(
    "exc, kind, cooldown",
    [
        (
            errors.APIError(404, {"error": {"message": "not found"}}),
            ModelUnavailable,
            LONG_COOLDOWN_S,
        ),
        (errors.APIError(503, {"error": {"message": "overloaded"}}), ModelUnavailable, 60),
        (errors.APIError(504, {"error": {"message": "deadline"}}), ModelUnavailable, 60),
        (TimeoutError("read timed out"), ModelUnavailable, 60),
        (errors.APIError(401, {"error": {"message": "bad key"}}), FatalLLMError, None),
        (errors.APIError(400, {"error": {"message": "invalid schema"}}), FatalLLMError, None),
    ],
)
def test_other_errors(exc, kind, cooldown):
    err = classify_error("m", exc, NOW)
    assert isinstance(err, kind)
    if cooldown is not None:
        assert err.cooldown_s == cooldown


# ---------- pool: selection, persistence, pacing ----------


class Clock:
    def __init__(self, t=NOW):
        self.t = t
        self.slept = []

    def __call__(self):
        return self.t

    def sleep(self, s):
        self.slept.append(round(s, 3))
        self.t += s


def make_pool(tmp_path, models=("a", "b", "c"), **kw):
    clock = kw.pop("clock", None) or Clock()
    pool = ModelPool(list(models), tmp_path / "state.json", clock=clock, sleep=clock.sleep, **kw)
    return pool, clock


def test_chain_order_and_cooldown_survive_a_new_process(tmp_path):
    pool, clock = make_pool(tmp_path, min_interval_s=0)
    assert pool.acquire(set()) == "a"
    pool.cooldown("a", 3400, "daily quota exhausted")

    fresh = ModelPool(
        ["a", "b", "c"], tmp_path / "state.json", clock=clock, sleep=clock.sleep, min_interval_s=0
    )  # e.g. the next CLI run
    assert fresh.acquire(set()) == "b"
    status = {s["model"]: s for s in fresh.status()}
    assert status["a"]["available"] is False and status["a"]["reason"] == "daily quota exhausted"

    clock.t += 3401  # quota reset: the preferred model is used again
    assert fresh.acquire(set()) == "a"


def test_requests_are_spaced_by_min_interval(tmp_path):
    pool, clock = make_pool(tmp_path, min_interval_s=2)
    pool.acquire(set())
    clock.t += 0.5
    pool.acquire(set())
    assert clock.slept == [1.5]


def test_per_model_rpm_window(tmp_path):
    pool, clock = make_pool(tmp_path, models=("a",), rpm=2, min_interval_s=0, max_wait_s=60)
    pool.acquire(set())
    clock.t += 10
    pool.acquire(set())
    clock.t += 10
    pool.acquire(set())  # third request in the same minute waits for the window
    assert clock.slept == [40]


def test_long_rpm_wait_falls_back_instead_of_blocking(tmp_path):
    pool, clock = make_pool(tmp_path, models=("a", "b"), rpm=1, min_interval_s=0, max_wait_s=5)
    assert pool.acquire(set()) == "a"
    clock.t += 1
    assert pool.acquire(set()) == "b"  # 'a' would need a 59 s wait
    assert clock.slept == []


def test_ready_fallback_is_used_instead_of_waiting_for_a_short_cooldown(tmp_path):
    pool, clock = make_pool(tmp_path, models=("a", "b"), min_interval_s=0, max_wait_s=30)
    pool.cooldown("a", 17, "server error 503")
    assert pool.acquire(set()) == "b"
    assert clock.slept == []

    clock.t += 18  # cooldown over: back to the preferred model
    assert pool.acquire(set()) == "a"


def test_waits_for_the_soonest_model_when_none_is_ready(tmp_path):
    pool, clock = make_pool(tmp_path, models=("a", "b"), min_interval_s=0, max_wait_s=30)
    pool.cooldown("a", 20, "rate limited")
    pool.cooldown("b", 5, "rate limited")
    assert pool.acquire(set()) == "b"
    assert clock.slept == [5]


def test_all_models_unavailable_explains_why(tmp_path):
    pool, _ = make_pool(tmp_path, models=("a", "b"), min_interval_s=0, max_wait_s=30)
    pool.cooldown("a", 3400, "daily free-tier quota (20/day) exhausted")
    pool.cooldown("b", 86400, "no free-tier quota for this model")
    with pytest.raises(ModelUnavailable) as info:
        pool.acquire(set())
    assert "a: daily free-tier quota (20/day) exhausted (available in 3400s)" in info.value.reason
    assert "b: no free-tier quota" in info.value.reason


def test_corrupt_state_file_is_ignored(tmp_path):
    (tmp_path / "state.json").write_text("{not json")
    pool, _ = make_pool(tmp_path, min_interval_s=0)
    assert pool.acquire(set()) == "a"


def test_models_from_env(monkeypatch, tmp_path):
    monkeypatch.setenv("GEMINI_MODELS", "x, y ,z")
    assert ModelPool(state_file=tmp_path / "s.json").models == ["x", "y", "z"]


# ---------- fallback inside a conversation ----------


def ok(text="done", signature=None):
    part = types.Part(text=text, thought_signature=signature)
    return types.GenerateContentResponse(
        candidates=[types.Candidate(content=types.Content(role="model", parts=[part]))]
    )


class FakeModels:
    """Responds per model: a list of responses/exceptions consumed in order."""

    def __init__(self, script):
        self.script = {m: list(v) for m, v in script.items()}
        self.calls = []
        self.active = 0
        self.max_active = 0
        self.lock = threading.Lock()

    def generate_content(self, *, model, contents, config):
        with self.lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        self.calls.append({"model": model, "contents": list(contents), "config": config})
        try:
            time.sleep(0.01)
            item = self.script[model].pop(0)
            if isinstance(item, Exception):
                raise item
            return item
        finally:
            with self.lock:
                self.active -= 1


def gemini(tmp_path, script, models=("gemini-3.5-flash", "gemini-3.5-flash-lite")):
    client = type("Client", (), {})()
    client.models = FakeModels(script)
    pool = ModelPool(list(models), tmp_path / "state.json", min_interval_s=0)
    return GeminiLLM(client=client, pool=pool), client.models


def test_daily_quota_falls_back_to_next_model_and_remembers(tmp_path):
    llm, models = gemini(
        tmp_path,
        {
            "gemini-3.5-flash": [quota_error(DAILY, "20", "3400s")],
            "gemini-3.5-flash-lite": [ok("first"), ok("second")],
        },
    )
    session = llm.start("sys", [])

    assert session.send_user("q1").text == "first"
    assert session.models_used == ["gemini-3.5-flash-lite"]
    assert session.fallbacks == ["gemini-3.5-flash: daily free-tier quota (20/day) exhausted"]

    session.send_user("q2")  # exhausted model is skipped without spending a request
    assert [c["model"] for c in models.calls] == [
        "gemini-3.5-flash",
        "gemini-3.5-flash-lite",
        "gemini-3.5-flash-lite",
    ]


def test_switching_models_keeps_history_unchanged(tmp_path):
    llm, models = gemini(
        tmp_path,
        {
            "gemini-3.5-flash": [
                ok("from flash", signature=b"flash-sig"),
                errors.APIError(503, {}),
            ],
            "gemini-3.5-flash-lite": [ok("from lite")],
        },
    )
    session = llm.start("sys", [])
    session.send_user("q1")
    session.send_user("q2")

    lite_request = models.calls[-1]
    assert lite_request["model"] == "gemini-3.5-flash-lite"
    previous_model_turn = lite_request["contents"][1]
    assert previous_model_turn.parts[0].thought_signature == b"flash-sig"


def test_fatal_error_does_not_try_other_models(tmp_path):
    llm, models = gemini(
        tmp_path, {"gemini-3.5-flash": [errors.APIError(401, {"error": {"message": "bad key"}})]}
    )
    with pytest.raises(LLMError, match="401"):
        llm.start("sys", []).send_user("q")
    assert len(models.calls) == 1


def test_every_model_exhausted_raises_clear_error(tmp_path):
    llm, _ = gemini(
        tmp_path,
        {
            "gemini-3.5-flash": [quota_error(DAILY, "20", "3400s")],
            "gemini-3.5-flash-lite": [quota_error(DAILY, "1000", "3400s")],
        },
    )
    with pytest.raises(LLMError, match="No Gemini model is available") as info:
        llm.start("sys", []).send_user("q")
    assert "gemini-3.5-flash-lite: daily free-tier quota (1000/day)" in str(info.value)


def test_model_rejecting_thinking_config_is_retried_without_it(tmp_path):
    bad = errors.APIError(400, {"error": {"message": "thinking_level is not supported"}})
    llm, models = gemini(tmp_path, {"gemini-3.5-flash": [bad, ok("fine")]})

    assert llm.start("sys", []).send_user("q").text == "fine"
    assert models.calls[0]["config"].thinking_config is not None
    assert models.calls[1]["config"].thinking_config is None


def test_concurrent_requests_never_overlap(tmp_path):
    llm, models = gemini(tmp_path, {"gemini-3.5-flash": [ok() for _ in range(6)]})
    threads = [
        threading.Thread(target=lambda: llm.start("sys", []).send_user("q")) for _ in range(6)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(models.calls) == 6
    assert models.max_active == 1
