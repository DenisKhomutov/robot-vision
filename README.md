# Robot Vision

Микросервис компьютерного зрения для робота-доставщика. Основные модули:

- **Модуль 1 — распознавание светофоров** (`module1_traffic_light`)
- **Модуль 2 — локализация** (`module2_localization`) — DINOv2 + Qdrant + XFeat
- **Модуль 3 — сегментация дорожного покрытия** (`module3_segmentation`)

Эксперименты: `experiment/` (офисный BEV, оценка поворота), `3D_VPR_exp/` (локализация по 3D-карте маршрута).

## Установка

```bash
cp .env.example .env      # заполнить значения
./tools/sync.sh           # одна команда: сама определит GPU и поставит нужный torch
```

`sync.sh` смотрит `nvidia-smi` и выбирает сборку torch. Если надо задать явно:

```bash
uv sync --extra cpu       # машина без GPU
uv sync --extra cu126     # машина с NVIDIA GPU
```

Автоматически по железу uv выбирать не умеет — маркеры в `pyproject.toml` статические
(платформа, версия Python), «есть ли видеокарта» среди них нет. Поэтому torch вынесен
в два конфликтующих extra, а `sync.sh` — обёртка, которая подставляет нужный.

Проверить, что встало правильно:

```bash
.venv/bin/python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

Версия должна оканчиваться на `+cu126` при наличии GPU и на `+cpu` при отсутствии.
Несовпадение = torch считает на процессоре, молча. `sync.sh` эту сверку делает сам
и ругается, если сборка не совпала с железом.

### Ловушка: голый `uv run` ломает torch

```bash
uv run python script.py            # ПЛОХО: пересинхронизирует и подменит torch
uv run --no-sync python script.py  # хорошо (или UV_NO_SYNC=1)
.venv/bin/python script.py         # хорошо
```

`uv run` перед запуском синхронизирует окружение. Раз torch вынесен в extras, в базовых
зависимостях его нет — но его требуют `ultralytics` (`torch>=1.8.0`) и `kornia`
(`torch>=2.0.0`). Не видя extra, uv тянет torch транзитивно **с обычного PyPI**, затирая
выбранную сборку. Молча. На ноутбуке это особенно обидно: приедет CUDA-вариант
с ~2.5 ГБ пакетов `nvidia-*` на машину без видеокарты.

Окружение меняется только через `./tools/sync.sh`. Всё остальное — с `--no-sync`.

**Jetson идёт мимо этого.** У него `Dockerfile.jetson` на образе `dustynv/pytorch` (ARM64,
L4T), где torch уже собран под общую память Jetson. Обычные колёса с download.pytorch.org
там не работают. `uv sync` в том образе не запускается — пакеты ставятся через `pip3`,
поэтому список зависимостей в `Dockerfile.jetson` надо править руками при изменении
зависимостей проекта.

Веса моделей положить в `./weights`: `best_det.pt`, `best_cls.pt`, `best_seg.pt`.

## 3D-карта маршрута (3D_VPR_exp)

Нужны `colmap` и `glomap` в системе — собираются из исходников, см. `3D_VPR_exp/README.md`.
Проверить: `colmap -h | head -2` (должно быть `with CUDA`, не `without`), `glomap -h`.

```bash
# нарезка + SIFT + sequential_matcher + GLOMAP
uv run --no-sync python 3D_VPR_exp/scripts/build_map.py --video july --fps 6 --overlap 25 --out july6
```

| флаг | смысл |
|---|---|
| `--video` | `july` или `june` (пути зашиты в `VIDEOS` в скрипте) |
| `--fps` | частота нарезки. 3 fps = шаг ~0.5 м при ходьбе; на поворотах мало, отсюда 6 |
| `--overlap` | сколько следующих кадров сопоставлять с каждым. Растит время нелинейно |
| `--out` | папка результата. **Задавать всегда**, иначе затрёт предыдущую карту |
| `--extract-only` | только нарезка, без SfM — проверить, что видео читается |
| `--skip-extract` | пропустить нарезку, гнать SfM по готовым кадрам |

Результат: `3D_VPR_exp/maps/<out>/sparse/0` + `database.db`, кадры в `3D_VPR_exp/data/<out>`.

Смотреть:

```bash
colmap gui --import_path 3D_VPR_exp/maps/july6/sparse/0 \
           --database_path 3D_VPR_exp/maps/july6/database.db \
           --image_path 3D_VPR_exp/data/july6
```

**На что смотреть в выводе:** сколько кадров зарегистрировано из общего числа и средняя
ошибка репроекции. Ориентир от прогона `july` (3 fps, overlap 10, ~14 мин): 443/486 = 91%,
0.587 px, одна модель. Не сошёлся хвост 443-485 — там разворот на пандусе, при 3 fps
соседние кадры после поворота перестают перекрываться.

## Линтеры и типы

```bash
# Проверка форматирования
uv run --group dev ruff format --check .

# Проверка стиля и ошибок
uv run --group dev ruff check .

# Автоисправление
uv run --group dev ruff check --fix .

# Проверка типов
uv run --group dev mypy --config-file pyproject.toml \
    app/ module1_traffic_light/ module2_localization/ module3_segmentation/ tools/
```

## Запуск Docker

```bash
docker compose up --build

docker compose -f docker-compose.jetson.yml up --build

# токен и URL берутся из .env (CAMERA_URL, CAMERA_TOKEN) — не хардкодить их здесь
source .env && curl -X POST "$CAMERA_URL" \
  -H "X-Capture-Token: $CAMERA_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"request_id": "test-1"}'

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

