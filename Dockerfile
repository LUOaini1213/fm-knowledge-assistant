# FM Knowledge Assistant API. The image holds code only; the index is built at first start from the official URLs
# in sources.json (the documents' terms do not allow redistributing them or anything derived from their text).
# Base image pinned by digest (tag kept for reading); refresh it when the security job reports fixable CVEs.
FROM python:3.11-slim@sha256:e88e9763f943ec1834f992a4b51e0f24500486803e8bc534e5767af9ea65f6ce
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONPATH=/app/src HF_HOME=/app/.cache
COPY requirements.txt .
# The base image's own pip, setuptools and wheel carry fixable CVEs (wheel, and jaraco.context vendored in setuptools)
RUN pip install --no-cache-dir --upgrade pip setuptools wheel \
 && pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements.txt
COPY src ./src
COPY sources.json ./
# Run as an unprivileged user; it owns only the folders the first start writes (data/ for the index, .cache for the model)
RUN useradd --create-home --uid 10001 app && mkdir -p data .cache && chown app:app data .cache
USER app
EXPOSE 8000
HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
CMD ["sh", "-c", "if [ ! -f data/processed/fmka.duckdb ]; then python -m fmka.cli fetch || exit $?; python -m fmka.cli build || exit $?; fi; exec uvicorn fmka.api:app --host 0.0.0.0 --port 8000 --no-server-header"]
