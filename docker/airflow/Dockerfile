# Airflow 3 with the training dependencies the DAG needs.
# scikit-learn and pandas already ship in the base image; MLflow is added here.
FROM apache/airflow:3.4.0-python3.12

ARG AIRFLOW_VERSION=3.4.0
ARG PYTHON_VERSION=3.12

USER airflow
# Holds the SQLite metadata DB and auth passwords file (a named volume in docker-compose).
# Creating it here as the airflow user gives a fresh volume the right ownership.
RUN mkdir -p /opt/airflow/state

# Installing apache-airflow==<version> under the official constraints file keeps pip
# from upgrading packages Airflow depends on (FastAPI, SQLAlchemy, ...).
RUN pip install --no-cache-dir \
      "apache-airflow==${AIRFLOW_VERSION}" \
      "mlflow-skinny==3.18.0" \
      "skops<1" \
      --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-${AIRFLOW_VERSION}/constraints-${PYTHON_VERSION}.txt"
