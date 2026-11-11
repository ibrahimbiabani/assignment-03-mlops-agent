import os

import pytest

# Keep MLflow's console hints out of test output.
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")


@pytest.fixture(autouse=True)
def _isolated_gemini_state(tmp_path, monkeypatch):
    """Never read/write the developer's real Gemini cooldown state, and never pace in tests."""
    monkeypatch.setenv("GEMINI_STATE_FILE", str(tmp_path / "gemini_state.json"))
    monkeypatch.setenv("GEMINI_MIN_INTERVAL_S", "0")
    for var in ("GEMINI_MODELS", "GEMINI_MODEL", "GEMINI_RPM", "GEMINI_MAX_WAIT_S"):
        monkeypatch.delenv(var, raising=False)
