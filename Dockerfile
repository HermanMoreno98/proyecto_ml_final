# ============================================================
# Dockerfile — Pipeline ML + Dashboard Streamlit
# ============================================================
# Imagen base: Python slim (sin Airflow; éste se maneja
# con la imagen oficial en docker-compose.yml)
# ============================================================

FROM python:3.9-slim

LABEL maintainer="proyecto_ml_final"
LABEL description="Pipeline ML E2E CU Venta: preprocessing, training HPO, monitoring PSI, postprocessing TLV, Streamlit dashboard"

# Evitar prompts interactivos durante el build
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Instalar dependencias del sistema (librerías C para XGBoost / pandas)
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copiar primero solo el archivo de dependencias para aprovechar la caché
COPY pyproject.toml ./

# Instalar pip actualizado y las dependencias del proyecto
RUN pip install --upgrade pip && \
    pip install --no-cache-dir \
        "pandas>=2.0.0" \
        "numpy>=1.26.0" \
        "scikit-learn>=1.4.0" \
        "xgboost>=2.0.0" \
        "optuna>=3.6.0" \
        "mlflow>=2.13.0" \
        "boto3>=1.34.0" \
        "s3fs>=2024.1.0" \
        "matplotlib>=3.8.0" \
        "streamlit>=1.35.0" \
        "pyyaml>=6.0"

# Copiar el código fuente
COPY src/ ./src/
COPY main.py ./
COPY dashboard.py ./

# Crear carpetas de datos (los CSVs se montan como volumen)
RUN mkdir -p data/raw data/processed data/postprocessed data/monitoring data/replica

# Puerto del dashboard Streamlit
EXPOSE 8501

# Por defecto arranca el dashboard; para el pipeline usar:
#   docker run ... python main.py --mode train --hpo --n-trials 30
CMD ["streamlit", "run", "dashboard.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true"]
