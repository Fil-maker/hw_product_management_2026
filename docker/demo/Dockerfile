# Образ демо-фронтенда (Gradio): окружение ставится из uv.lock, PyTorch в него не попадает.
FROM ghcr.io/astral-sh/uv:0.13.6 AS uv

FROM python:3.12.12-slim-bookworm

COPY --from=uv /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH=/opt/venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    # Без этого Gradio слушает только 127.0.0.1 и с хоста интерфейс не открывается.
    GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_SERVER_PORT=7860 \
    GRADIO_ANALYTICS_ENABLED=False

WORKDIR /app

# Сначала только зависимости — этот слой кешируется, пока не меняется uv.lock.
COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project --extra demo

COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --extra demo

COPY demo/examples ./demo/examples

EXPOSE 7860

CMD ["cifar-demo"]
