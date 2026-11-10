# Prediction API image (deployed to Kubernetes by kubernetes/deployment.yaml).
#   docker build -t prediction-api:1.0.0 .
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MLFLOW_DISABLE_AGENT_HINT=1 \
    HOME=/tmp

WORKDIR /app
COPY api/requirements.txt api/requirements.txt
RUN pip install -r api/requirements.txt

COPY api/ api/

# Fixed non-root UID so the Pod's runAsNonRoot / runAsUser settings can be enforced.
RUN useradd --uid 10001 --no-create-home --shell /usr/sbin/nologin appuser
USER 10001

EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
