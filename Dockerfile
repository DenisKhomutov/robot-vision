FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1
ENV UV_LINK_MODE=copy

# системные библиотеки для opencv (нужен ultralytics)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libxcb1 libsm6 libxext6 libxrender1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.11.21 /uv /uvx /bin/

WORKDIR /app

COPY pyproject.toml uv.lock ./

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-dev

COPY module2_localization/ ./module2_localization/
COPY module1_traffic_light/ ./module1_traffic_light/
COPY module3_segmentation/ ./module3_segmentation/
COPY app/ ./app/

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev
