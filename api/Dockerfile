FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8080

WORKDIR /app

# Dependencies before source: source changes are frequent, dependency changes
# are not, so this layer stays cached across most rebuilds.
COPY pyproject.toml ./
RUN pip install --no-cache-dir \
      "fastapi>=0.115" \
      "uvicorn[standard]>=0.32" \
      "httpx>=0.28" \
      "pydantic>=2.9" \
      "pydantic-settings>=2.6" \
      "prometheus-fastapi-instrumentator>=7.0"

COPY app ./app

# Non-root. uid 1000 matches the convention used across these services.
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8080

# Shell form so ${PORT} expands at runtime.
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8080}"]
