# Robot Vision

Компьютерное зрение для робота-доставщика. Три модуля:

- **Модуль 1 — светофоры** (`module1_traffic_light`): детектор + классификатор.
- **Модуль 2 — локализация** (`module2_localization`): ALIKED + 3D-карта маршрута,
  выдаёт команды рулю. Единственный, у которого сейчас есть боевой демон.
- **Модуль 3 — сегментация покрытия** (`module3_segmentation`).

Ни FastAPI, ни Qdrant, ни WebRTC не используются: кадры берутся локально из сокета
камеры (GStreamer), команды уходят в NATS.

## Как это работает

```
камера -> fan-out (tee -> shmsink) -> демон локализации -> NATS -> мозг робота
                                                              \-> веб-админка
```

Демон читает кадр, ищет позу в 3D-карте маршрута (ALIKED + PnP через OpenCV), считает команду
и публикует её в топик `robot.vision.localization`. Топик слушают контроллер
робота и веб-админка — NATS раздаёт копию каждому.

### Формат сообщений

Едем:
```json
{"move_type": "straight", "deg": 1.58, "offset": -0.0017, "dist_to_route": 0.0043,
 "offset_m": -0.002, "dist_to_route_m": 0.006, "node": 263, "inliers": 378,
 "ts": 1784804951.819}
```

Потерялись (позы нет — ехать нельзя):
```json
{"move_type": "lost", "node": 301, "reason": "inliers: 5", "ts": 1784804967.926}
```

Приехали (латч, до перезапуска демона не сбрасывается):
```json
{"move_type": "stop", "node": 309, "ts": 1784804969.4}
```

| поле | смысл |
|---|---|
| `move_type` | `left` / `right` / `straight` / `stop` / `lost` |
| `deg` | азимут на цель; минус — влево, плюс — вправо |
| `offset`, `offset_m` | боковое смещение от эталона со знаком |
| `dist_to_route`, `_m` | расстояние до маршрута, всегда положительное |
| `node` | номер узла эталона — где мы на маршруте |
| `inliers` | сколько точек подтвердили позу |
| `ts` | время публикации; **по нему потребитель ведёт сторожевой таймер** |

Молчание в топике — это не «продолжай ехать». NATS доставляет максимум один раз и
без буфера, поэтому контроллер обязан тормозить, если сообщений нет дольше таймаута:
только так покрываются падение демона, обрыв сети и зависание Jetson.

## Установка (машина разработки)

```bash
./tools/sync.sh            # сам определит GPU и поставит нужную сборку torch
```

Явно, если нужно:

```bash
uv sync --extra cpu        # без GPU
uv sync --extra cu126      # с NVIDIA GPU
```

Проверка, что torch встал правильно (иначе будет молча считать на процессоре):

```bash
uv run --no-sync python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

Веса ALIKED и LightGlue лежат в `module2_localization/weights/` и подключаются
автоматически ([core/model_weights.py](module2_localization/core/model_weights.py)) — сеть при запуске не нужна.

### Ловушка: `uv run` без `--no-sync`

```bash
uv run python script.py            # ПЛОХО: пересинхронизирует и подменит torch
uv run --no-sync python script.py  # хорошо
.venv/bin/python script.py         # хорошо
```

Torch вынесен в конфликтующие extra (`cpu` / `cu126`), потому что uv не умеет выбирать
сборку по наличию GPU — маркеры в `pyproject.toml` статические. Не видя extra, uv тянет
torch транзитивно с обычного PyPI и затирает выбранную сборку. Молча.

## Запуск

Брокер (на той машине, где работает демон):

```bash
docker run -d --name nats -p 4222:4222 --restart unless-stopped nats:latest
```

Демон — кадры из сокета камеры (боевой режим):

```bash
uv run --no-sync python -m module2_localization.service --shm
```

Демон — по видеофайлу (отладка, без камеры и без NATS):

```bash
uv run --no-sync python -m module2_localization.service \
    --video module2_localization/map_for_office/rec3/camera-2.mkv \
    --source-step 20 --no-nats
```

Веб-админка:

```bash
python -m module2_localization.admin \
  --bind 0.0.0.0 --port 8080 \
  --nats-url nats://127.0.0.1:4222
```

После запуска интерфейс доступен на порту `8080` машины, где запущена админка.

## Инструменты

Нужны `colmap` и `glomap` в системе (собираются из исходников, CUDA-сборка).
Проверить: `colmap -h | head -2`, `glomap -h`.

```bash
# карта из одной камеры
uv run --no-sync python module2_localization/tools/build_map.py --images <папка> --tag map_new

# длинный последовательный маршрут: единая инкрементальная карта, без словаря
uv run --no-sync python -m module2_localization.tools.build_map \
    --images frames_ns1_rear --tag map_ns1_rear \
    --det-threshold 0.04 --pairs sequential --pair-offsets 1,3,6,10,16,25 \
    --init-image-ids 1 11

# карта из 3-камерного рига (кадры синхронны, имена {время}_c{N}.jpg)
uv run --no-sync python module2_localization/tools/build_map_rig.py \
    --videos rec3/camera-1.mkv rec3/camera-2.mkv rec3/camera-3.mkv --tag map_rig3

# экспорт карты для рантайма (после сборки карты — обязательно)
uv run --no-sync python module2_localization/tools/export_map.py --map map_rig3

# метрический масштаб карты из одометрии
uv run --no-sync python module2_localization/tools/scale_from_odometry.py \
    --map map_rig3 --log rec3/encoder-log.jsonl

# картинка карты и видео-самотест (карта + кадр + позиция + команда)
uv run --no-sync python module2_localization/tools/draw_map.py --map map_rig3
uv run --no-sync python module2_localization/tools/make_video.py <видео> \
    --map map_rig3 --route-cam _c2 --step 10 --kpts 8192 --q-threshold 0.05

# посмотреть карту в COLMAP
colmap gui --import_path module2_localization/maps/map_rig3/sparse/0
```

Демон и видеопрогон считают команду одним кодом ([core/command_filter.py](module2_localization/core/command_filter.py)),
поэтому видео показывает ровно то, что уйдёт роботу.

## Деплой на Jetson (Orin Nano 8 ГБ, JetPack 6 / Ubuntu 22.04)

Кадры приходят из ветки `tee` fan-out-сервиса камеры. Разрешение нашей ветки —
**1280×720**, как у кадров, из которых собрана карта: при совпадении используется
откалиброванная камера карты, иначе фокус угадывается и точность падает.

Сокет и размеры — в [config.py](module2_localization/config.py) (`CAM_SHM_SOCKET`, `CAM_WIDTH`,
`CAM_HEIGHT`). Они обязаны совпадать с caps, которые подаются в `shmsink`: через
разделяемую память едут голые байты без описания формата, и расхождение даст мусор
без единой ошибки.

Проверить источник до запуска демона:

```bash
gst-inspect-1.0 shmsrc
gst-launch-1.0 shmsrc socket-path=/tmp/cam_raw ! \
  video/x-raw,format=I420,width=1280,height=720,framerate=30/1 ! \
  videoconvert ! fakesink -v
```

### Вариант 1: контейнер

```bash
docker compose -f docker-compose.jetson.yml build
docker compose -f docker-compose.jetson.yml up -d
docker compose -f docker-compose.jetson.yml logs -f localization
docker compose -f docker-compose.jetson.yml down
```

`network_mode: host` — чтобы демон и веб-админка видели локальный NATS
снаружи. `ipc: host` и монтирование `/tmp` — чтобы `shmsrc` добрался до разделяемой
памяти fan-out. Без них сокет откроется, а кадры не придут.

### Вариант 2: systemd, без контейнера

Torch и OpenCV на Jetson берутся из JetPack, `uv sync` здесь не применяется —
недостающие пакеты ставятся `pip3` поверх системных. Компилировать нечего:
`pycolmap` в рантайме не нужен, карта читается из `runtime.npz`.

```bash
rsync -a --exclude .venv --exclude .git <ноутбук>:~/projects/robot-vision/ ~/projects/robot-vision/
cd ~/projects/robot-vision
pip3 install "nats-py>=2.6.0" kornia loguru
pip3 install --no-deps "lightglue @ git+https://github.com/cvg/LightGlue.git"
sudo cp tools/systemd/*.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now robot-vision-localization
journalctl -u robot-vision-localization -f
```

Юнит зависит от `camera-fanout.service`: демон стартует после того, как появились кадры.

### Что везти на робота

Рантайму нужны только `config.py`, `service.py`, `nats_client.py`, `core/`, `services/`,
`weights/aliked-n16.pth` и карта — файлы `runtime.npz`, `aliked_bank.npz`, `scale.json`,
около 110 МБ. Папка `sparse/0`, `database.db` и `pairs.txt` нужны только при сборке карты
и на робота не едут. Из пакетов на роботе: numpy, opencv, torch (из JetPack),
lightglue, kornia, nats-py, loguru — ничего компилировать не требуется.

## Настройки

Всё в [module2_localization/config.py](module2_localization/config.py). Значимое:

| параметр | смысл |
|---|---|
| `DEFAULT_MAP`, `ROUTE_CAM` | какая карта и по какой камере рига строится эталон |
| `ROUTE_NODES` | обрезка маршрута: хвост карты у стены разваливается |
| `MIN_INLIERS` | ниже — позе не верим, уходим в `lost` |
| `MAX_NODES_PER_SEC`, `MIN_NODE_JUMP` | отбраковка скачков вперёд; назад не ограничено |
| `MAX_REJECTS` | столько отказов подряд — верим кадру (защита от вечного `lost`) |
| `STOP_CONFIRM`, `STOP_MIN_INLIERS` | подтверждение конца маршрута перед латчем |
| `LOOKAHEAD_NODES`, `DEADZONE_DEG` | упреждение и мёртвая зона азимута |
| `NATS_HOST`, `NATS_URL`, `NATS_TOPIC` | параметры подключения к NATS |

## Линтеры и типы

```bash
uv run --group dev ruff check .
uv run --group dev ruff check --fix .
uv run --group dev mypy --config-file pyproject.toml \
    module1_traffic_light/ module2_localization/ module3_segmentation/
```

## Ограничения, о которых надо помнить

**Зрению нужна фактура.** Голая стена, стекло, темнота — признаков нет, позы нет.
Это физика метода, а не баг: на конце офисного маршрута локализация разваливается,
поэтому маршрут обрезан до `ROUTE_NODES`. На улице фактуры больше, но иммунитета нет.
Лечится только вторым датчиком — счислением по одометрии.

**Оценка точности завышена.** 93% узнавания получены на той же записи, из которой
построена карта. Честная цифра появится только на втором проезде другой записью.

**Латч `stop` не сбрасывается.** Повторный проезд требует перезапуска демона.

PYTORCH_NO_CUDA_MEMORY_CACHING=1 python3 -m module2_localization.service --shm

OFFICE
adapt = 3.0
min = 10
nodes = 30

STREET
adapt = 30
min = 4
nodes = 10
