# FM Knowledge Assistant API. The image holds code only; the index is built at first start from the official URLs
# in sources.json (the documents' terms do not allow redistributing them or anything derived from their text).
FROM python:3.11-slim
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONPATH=/app/src HF_HOME=/app/.cache
COPY requirements.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements.txt
COPY src ./src
COPY sources.json ./
EXPOSE 8000
HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
CMD ["sh", "-c", "if [ ! -f data/processed/fmka.duckdb ]; then python -m fmka.cli fetch || exit $?; python -m fmka.cli build || exit $?; fi; exec uvicorn fmka.api:app --host 0.0.0.0 --port 8000"]
