"""Free-tier friendly access to several Gemini models: fallback chain + request pacing.

Why: the free tier allows few requests per model (e.g. 20/day for gemini-3.5-flash, reset at
midnight Pacific) and rejects bursts with 429. Limits are per project and per model and are
not published, so this module learns them from the API's own error responses.

* Fallback chain (GEMINI_MODELS, best first). Each request uses the first model that is not
  cooling down. Errors put a model on cooldown:
    429 per-day quota       -> until the reset the server reports (retryDelay), else midnight PT
    429 no free-tier access -> 24 h
    429 per-minute limit    -> retryDelay (default 60 s)
    404 / 403 model         -> 24 h (retired, not served, or no access)
    5xx / timeout / network -> 60 s
  Cooldowns are saved to GEMINI_STATE_FILE, so separate CLI runs do not re-spend requests
  discovering that a model is exhausted.
* Pacing: one Gemini request in flight per process, at least GEMINI_MIN_INTERVAL_S between
  requests, and at most GEMINI_RPM requests per model in any 60 s window (also persisted).
* No automatic SDK retries: a retried 429 is just another rejected request.
"""

import json
import logging
import os
import re
import threading
import time
from collections.abc import Callable
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

log = logging.getLogger("mlops_agent.gemini")

DEFAULT_MODELS = "gemini-3.5-flash,gemini-3.6-flash,gemini-3.5-flash-lite,gemini-3.1-flash-lite"
PACIFIC = ZoneInfo("America/Los_Angeles")
LONG_COOLDOWN_S = 24 * 3600
SHORT_COOLDOWN_S = 60


class ModelUnavailable(Exception):
    """A model cannot serve requests right now; try the next one."""

    def __init__(self, model: str, cooldown_s: float, reason: str) -> None:
        super().__init__(f"{model}: {reason}")
        self.model, self.cooldown_s, self.reason = model, cooldown_s, reason


class FatalLLMError(Exception):
    """An error no other model can fix (bad API key, malformed request)."""


def _seconds_until_pacific_midnight(now: float) -> float:
    local = datetime.fromtimestamp(now, PACIFIC)
    midnight = (local + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    return max(60.0, (midnight - local).total_seconds())


def _retry_delay(details: dict[str, Any]) -> float | None:
    for d in details.get("error", {}).get("details", []):
        if m := re.fullmatch(r"([\d.]+)s", str(d.get("retryDelay", ""))):
            return float(m.group(1))
    return None


def _violations(details: dict[str, Any]) -> list[dict[str, Any]]:
    return [v for d in details.get("error", {}).get("details", []) for v in d.get("violations", [])]


def classify_error(model: str, exc: Exception, now: float) -> Exception:
    """Map an SDK/HTTP exception to ModelUnavailable (try another model) or FatalLLMError."""
    code = getattr(exc, "code", None)
    details = getattr(exc, "details", None)
    details = details if isinstance(details, dict) else {}
    message = str(getattr(exc, "message", None) or exc)[:200]

    if code == 429:
        violations = _violations(details)
        delay = _retry_delay(details)
        free = [v for v in violations if "FreeTier" in str(v.get("quotaId", ""))]
        if free and all(str(v.get("quotaValue")) in ("None", "0") for v in free):
            return ModelUnavailable(model, LONG_COOLDOWN_S, "no free-tier quota for this model")
        if any("PerDay" in str(v.get("quotaId", "")) for v in violations):
            limit = next(
                (v.get("quotaValue") for v in violations if "PerDay" in v.get("quotaId", "")), "?"
            )
            wait = delay or _seconds_until_pacific_midnight(now)
            return ModelUnavailable(model, wait, f"daily free-tier quota ({limit}/day) exhausted")
        return ModelUnavailable(model, delay or SHORT_COOLDOWN_S, "rate limited (per-minute quota)")
    if code in (404, 403):
        return ModelUnavailable(model, LONG_COOLDOWN_S, f"model not available ({code}): {message}")
    if code == 401:
        return FatalLLMError(f"Gemini rejected the API key (401): {message}")
    if code == 400:
        return FatalLLMError(f"Gemini rejected the request (400): {message}")
    if isinstance(code, int) and code >= 500:
        return ModelUnavailable(model, SHORT_COOLDOWN_S, f"server error {code}: {message}")
    # Timeouts, connection resets and other transport errors.
    return ModelUnavailable(model, SHORT_COOLDOWN_S, f"{type(exc).__name__}: {message}")


class ModelPool:
    """Thread-safe model selection, cooldowns and request pacing, persisted to a JSON file."""

    def __init__(
        self,
        models: list[str] | None = None,
        state_file: str | Path | None = None,
        rpm: int | None = None,
        min_interval_s: float | None = None,
        max_wait_s: float | None = None,
        clock: Callable[[], float] = time.time,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        env = os.environ.get
        raw = models or env("GEMINI_MODELS", "") or env("GEMINI_MODEL", "") or DEFAULT_MODELS
        self.models = (
            raw if isinstance(raw, list) else [m.strip() for m in raw.split(",") if m.strip()]
        )
        default_state = Path.home() / ".cache" / "mlops-agent" / "gemini_state.json"
        self.state_file = Path(state_file or env("GEMINI_STATE_FILE", "") or default_state)
        self.rpm = rpm or int(env("GEMINI_RPM", "8"))
        self.min_interval = (
            min_interval_s
            if min_interval_s is not None
            else float(env("GEMINI_MIN_INTERVAL_S", "2"))
        )
        self.max_wait = (
            max_wait_s if max_wait_s is not None else float(env("GEMINI_MAX_WAIT_S", "30"))
        )
        self.clock, self.sleep = clock, sleep
        self._lock = threading.Lock()  # protects state
        self.in_flight = threading.Lock()  # one Gemini request at a time per process

    # ---------- persisted state ----------

    def _load(self) -> dict[str, Any]:
        try:
            return json.loads(self.state_file.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}

    def _save(self, state: dict[str, Any]) -> None:
        try:
            self.state_file.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.state_file.with_suffix(".tmp")
            tmp.write_text(json.dumps(state, indent=1), encoding="utf-8")
            tmp.replace(self.state_file)
        except OSError as exc:  # state is an optimisation; never fail a request over it
            log.warning("could not save Gemini state to %s: %s", self.state_file, exc)

    def cooldown(self, model: str, seconds: float, reason: str) -> None:
        with self._lock:
            state = self._load()
            entry = state.setdefault(model, {})
            entry["cooldown_until"] = self.clock() + seconds
            entry["reason"] = reason
            self._save(state)
        log.info(
            json.dumps(
                {
                    "event": "llm_model_cooldown",
                    "model": model,
                    "seconds": round(seconds),
                    "reason": reason,
                }
            )
        )

    def status(self) -> list[dict[str, Any]]:
        """Each model in chain order with its availability (for /health and errors)."""
        state, now = self._load(), self.clock()
        out = []
        for m in self.models:
            entry = state.get(m, {})
            until = entry.get("cooldown_until", 0)
            out.append(
                {
                    "model": m,
                    "available": until <= now,
                    "available_in_s": max(0, round(until - now)),
                    "reason": entry.get("reason") if until > now else None,
                }
            )
        return out

    # ---------- selection + pacing ----------

    def _wait_for(self, model: str, state: dict[str, Any], now: float) -> float:
        """Seconds until `model` may be called (cooldown, per-model RPM window, spacing)."""
        entry = state.get(model, {})
        wait = entry.get("cooldown_until", 0) - now
        recent = [t for t in entry.get("recent", []) if now - t < 60]
        if len(recent) >= self.rpm:
            wait = max(wait, recent[-self.rpm] + 60 - now)
        last_any = max((t for e in state.values() for t in e.get("recent", [])), default=0)
        wait = max(wait, last_any + self.min_interval - now)
        return max(0.0, wait)

    def acquire(self, exclude: set[str]) -> str:
        """Pick the best model that can be used within max_wait, wait for it, record the call.

        Must be called while holding `in_flight`. Raises ModelUnavailable listing why nothing
        is usable if every model is excluded or cooling down beyond max_wait.
        """
        with self._lock:
            state, now = self._load(), self.clock()
            waits = {m: self._wait_for(m, state, now) for m in self.models if m not in exclude}
        usable = [m for m, w in waits.items() if w <= self.max_wait]
        if not usable:
            summary = "; ".join(
                f"{s['model']}: {s['reason'] or 'tried this request'}"
                + (f" (available in {s['available_in_s']}s)" if s["available_in_s"] else "")
                for s in self.status()
            )
            raise ModelUnavailable("all", min(waits.values(), default=0), summary)
        # Soonest usable model; chain order breaks ties. The global spacing applies to every
        # model equally, so a ready fallback beats waiting out a preferred model's cooldown.
        model = min(usable, key=lambda m: (round(waits[m], 1), self.models.index(m)))
        if waits[model] > 0:
            self.sleep(waits[model])
        with self._lock:
            state = self._load()
            entry = state.setdefault(model, {})
            now = self.clock()
            entry["recent"] = [t for t in entry.get("recent", []) if now - t < 60] + [now]
            self._save(state)
        return model
