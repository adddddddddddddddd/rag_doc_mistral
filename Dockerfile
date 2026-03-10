FROM ghcr.io/astral-sh/uv:latest
WORKDIR /app
COPY . .
RUN uv sync
CMD ["uv", "run", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "5000"]