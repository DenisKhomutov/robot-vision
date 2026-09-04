# Модуль визуальной локализации и навигации

Рабочий runtime запускается через совместимые точки входа:

```bash
python -m module2_localization.service
python -m module2_localization.admin
```

Реализация процесса навигации находится в `runtime`, математическое ядро — в
`core`, управление цепочкой карт — в `services`, а оффлайн-команды подготовки и
проверки карт — в `tools`.

## Документация

- [SYSTEM_START_QUICK.md](docs/SYSTEM_START_QUICK.md) — краткий запуск системы на Jetson.
- [NATS_COMMAND_REFERENCE.md](docs/NATS_COMMAND_REFERENCE.md) — команды, состояния и NATS-контракт.
- [RUNTIME_SHARDS_AND_WEB_ADMIN.md](docs/RUNTIME_SHARDS_AND_WEB_ADMIN.md) — маршруты, шарды, recovery и веб-админка.
- [NAVIGATION_LOGS.md](docs/NAVIGATION_LOGS.md) — журнал навигации и его разбор.
- [STREET_NAVIGATION_TUNING.md](docs/STREET_NAVIGATION_TUNING.md) — настройка упреждения и рулевого управления.
- [REVERSE_STEERING_MODEL.md](docs/REVERSE_STEERING_MODEL.md) — подтверждённая защита и правильная модель заднего хода.
- [STREET_MAP_BUILDING_HANDOFF.md](docs/STREET_MAP_BUILDING_HANDOFF.md) — полный конвейер сборки уличной карты.
- [MSLD_LANDMARK_DISTILLATION.md](docs/MSLD_LANDMARK_DISTILLATION.md) — экспериментальная дистилляция стабильных ориентиров.

Текущая рабочая схема использует локализацию по ALIKED, runtime-карты с банком
3D-ориентиров и последовательные шарды. MSLD остаётся экспериментом и не заменяет
рабочие карты до отдельного полного сравнения.

TODO List:
1. Analyze the current pipeline through the module 2 code review (algorithms, code, maths. threory).