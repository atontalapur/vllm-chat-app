FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8080

WORKDIR /app

# Dependencies come from pyproject.toml + uv.lock, never from a list repeated
# here. A hand-copied list is a second source of truth that drifts silently:
# `uv add psycopg` updated the project and the tests passed, while the image
# kept building without it and only failed at runtime on the box.
#
# --frozen: fail if the lock is stale rather than silently resolving something
# different from what CI tested. --no-dev: no pytest or mypy in the runtime
# image. --no-install-project: the app is COPYed below, not pip-installed, so
# a source change does not invalidate this layer.
COPY --from=ghcr.io/astral-sh/uv:0.9.26 /uv /usr/local/bin/uv
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

# uv builds a venv; put it first so `uvicorn` and `python` resolve to it.
ENV PATH="/app/.venv/bin:$PATH"

COPY app ./app

# Non-root. uid 1000 matches the convention used across these services.
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8080

# Shell form so ${PORT} expands at runtime.
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8080}"]
