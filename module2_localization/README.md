# Module 2: runtime visual localization and navigation

`module2_localization` — runtime-модуль робота. Он загружает готовую карту, локализует кадры камеры, ведёт робота по маршруту и формирует управляющие команды.

Сборка и экспорт новых карт в module2 запрещены. Новый конвейер сборки находится в [`module4_map_builder`](../module4_map_builder/README.md). Старый COLMAP/GLOMAP и MSLD стек удалён из ветки `10`; его история сохранена в ветке `9` и проекте `robot-vision-legacy`.

## Структура

- `core/` — локализатор, runtime-map loader, геометрия маршрута и фильтрация команд;
- `runtime/` — цикл навигации, источники кадров, NATS, диагностика и управление;
- `services/` — загрузка карт и orchestration локализаторов;
- `maps/` — готовые runtime-карты;
- `tools/` — только запуск, проверка и визуальная диагностика runtime;
- `docs/` — эксплуатационная документация робота.

## Запуск

```bash
python -m module2_localization.service
python -m module2_localization.admin
```

Запуск по видео без NATS:

```bash
python -m module2_localization.service \
  --video <video.mkv> \
  --source-step 2 \
  --no-nats
```

## Документация

- [SYSTEM_START_QUICK.md](docs/SYSTEM_START_QUICK.md) — запуск на Jetson;
- [NATS_COMMAND_REFERENCE.md](docs/NATS_COMMAND_REFERENCE.md) — NATS-контракт;
- [RUNTIME_SHARDS_AND_WEB_ADMIN.md](docs/RUNTIME_SHARDS_AND_WEB_ADMIN.md) — карты, recovery и веб-интерфейс;
- [NAVIGATION_LOGS.md](docs/NAVIGATION_LOGS.md) — журнал навигации;
- [STREET_NAVIGATION_TUNING.md](docs/STREET_NAVIGATION_TUNING.md) — настройка движения;
- [REVERSE_STEERING_MODEL.md](docs/REVERSE_STEERING_MODEL.md) — задний ход.

## Runtime-инструменты

```bash
python -m module2_localization.tools.run_localization
python -m module2_localization.tools.verify_runtime_maps
python -m module2_localization.tools.bench_locate --help
python -m module2_localization.tools.preview_sharded_video --help
```

Module2 не должен зависеть от DPVO, COLMAP/GLOMAP или инструментов сборки карты. Для деплоя ему нужны только runtime-зависимости и готовая карта.
