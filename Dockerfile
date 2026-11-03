# Образ для обучения: окружение ставится из uv.lock.
# TORCH_VARIANT=cpu (по умолчанию) или gpu — какая сборка PyTorch попадёт в образ.
FROM ghcr.io/astral-sh/uv:0.13.6 AS uv

FROM python:3.12.12-slim-bookworm

COPY --from=uv /uv /uvx /bin/

ARG TORCH_VARIANT=cpu
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH=/opt/venv/bin:$PATH \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Сначала только зависимости — этот слой кешируется, пока не меняется uv.lock.
COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project --extra "$TORCH_VARIANT"

COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --extra "$TORCH_VARIANT"

COPY configs ./configs

CMD ["cifar-train", "--help"]
