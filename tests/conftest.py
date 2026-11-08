import os

# Keep MLflow's console hints out of test output.
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
