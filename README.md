
```bash
# Проверка форматирования
uv run --group dev ruff format --check .

# Проверка стиля и ошибок
uv run --group dev ruff check .

# Автоисправление
uv run --group dev ruff check --fix .

# Проверка типов
uv run --group dev mypy --config-file pyproject.toml module1_traffic_light/ module2_localization/ tools/
```

## Запуск Docker

```bash
docker compose up --build

docker compose -f docker-compose.jetson.yml up --build

curl.exe -X POST "http://192.168.40.203:8091/api/capture-and-send/" -H "X-Capture-Token: e3e7af7028979acf0fbccfc01174c91ea32aab173eaf5c1b1f271a7c146e3fa2" -H "Content-Type: application/json" -d '{\"request_id\": \"test-1\"}'

uv run uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## Основные команды Docker

```bash
# запустить все сервисы (qdrant + api) с пересборкой
docker compose up --build

# в фоне (терминал свободен)
docker compose up --build -d

# конкретный compose-файл (Jetson)
docker compose -f docker-compose.jetson.yml up --build -d

# логи сервиса в реальном времени
docker compose logs -f localization_api

# статус контейнеров (запущены, порты)
docker compose ps

# остановить и удалить контейнеры (тома и образы остаются)
docker compose down

# остановить и удалить + тома (снесёт базу qdrant и кэш модели)
docker compose down -v

# перезапустить без пересборки
docker compose restart

# зайти внутрь работающего контейнера (отладка)
docker compose exec localization_api bash

# список образов / контейнеров / томов
docker images
docker ps -a
docker volume ls

# сколько места занято
docker system df

# нагрузка контейнеров в реальном времени (CPU/память)
docker stats
```

## Очистка кэша Docker

```bash
# кэш сборки (build cache, cache mounts)
docker builder prune -f

# остановленные контейнеры, неиспользуемые сети, висячие образы
docker system prune -f

# то же + все неиспользуемые образы
docker system prune -a -f

# ОСТОРОЖНО: + тома (снесёт qdrant_storage и кэш модели)
docker system prune -a --volumes -f
```

