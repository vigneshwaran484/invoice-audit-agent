FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy \
    AUDIT_CHECKPOINT_DB=/app/state/checkpoints.sqlite \
    OLLAMA_BASE_URL=http://host.docker.internal:11434

COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --extra agent --no-install-project

COPY src ./src
COPY data ./data
RUN uv sync --frozen --no-dev --extra agent && uv run python data/generate.py

VOLUME /app/state
EXPOSE 8000
CMD ["uv", "run", "--no-dev", "uvicorn", "audit_agent.api:app", "--host", "0.0.0.0", "--port", "8000"]
