FROM pytorch/pytorch:2.6.0-cuda12.6-cudnn9-runtime

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .
COPY inference.py .

RUN mkdir -p /app/model

ENV FLASK_APP=app.py
ENV FLASK_HOST=0.0.0.0
ENV FLASK_PORT=5000
ENV MODEL_PATH=/app/model
ENV PYTHONUNBUFFERED=1
# MLflow Model Registry (DagsHub) — à surcharger via docker-compose ou -e
ENV MLFLOW_TRACKING_URI=https://dagshub.com/DrEPL/ml-translation-wo-fr.mlflow
ENV MLFLOW_MODEL_URI=models:/nllb-fr-wo-translation/latest
# Jeton d'authentification DagsHub — obligatoire pour accéder au registry
# À injecter via --env DAGSHUB_USER_TOKEN=<votre-token> ou docker-compose secrets
ENV DAGSHUB_USER_TOKEN=

HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1

EXPOSE 5000

CMD ["gunicorn", \
    "--bind", "0.0.0.0:5000", \
    "--workers", "2", \
    "--threads", "2", \
    "--worker-class", "gthread", \
    "--timeout", "120", \
    "--access-logfile", "-", \
    "--error-logfile", "-", \
    "app:app"]
