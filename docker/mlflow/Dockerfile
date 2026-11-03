# В официальном образе MLflow нет драйвера PostgreSQL и boto3 (для артефактов в S3).
FROM ghcr.io/mlflow/mlflow:v3.17.2

RUN pip install --no-cache-dir psycopg2-binary==2.9.13 boto3==1.43.123
