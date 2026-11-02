# EventOracle / Project Apex – Streamlit dashboard image
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    ENABLE_METRICS=true \
    METRICS_PORT=9100

# libgomp is required by LightGBM; curl for the health check
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt \
    && python -c "import nltk; nltk.download('vader_lexicon', download_dir='/usr/local/share/nltk_data', quiet=True)"

COPY . .

# Run as non-root
RUN useradd --create-home --uid 1000 apex && chown -R apex:apex /app
USER apex

EXPOSE 8501 9100

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Secrets (MONGO_URI, GEMINI_API_KEY) are injected at runtime – never baked in.
CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
