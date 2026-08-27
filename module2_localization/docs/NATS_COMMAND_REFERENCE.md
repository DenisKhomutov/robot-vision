# Команды и состояния системы локализации

## NATS-топики

| Топик | Направление | Назначение |
|---|---|---|
| `robot.vision.localization` | демон → контроллер | Команды движения и состояние навигации |
| `robot.vision.control` | админка/оператор → демон | Управление маршрутом и режимами |
| `gateway.robot.speed` | контроллер → демон | Текущая скорость PWM для адаптивного упреждения |
| `robot.vision.traffic_light` | зарезервирован | Топик светофора из конфигурации |

Все payload передаются как JSON в UTF-8.

## Команды движения

### Движение прямо

```json
{
  "move_type": "straight",
  "deg": 1.25,
  "node": 42,
  "target_node": 47,
  "inliers": 63,
  "pos": [1.2345, 6.789],
  "head": [0.998, 0.063],
  "ts": 1787660000.123,
  "cam": "front",
  "map": "2-1/front_shard/02_of_25",
  "global_node": 69,
  "global_target_node": 74,
  "mode": "front",
  "route": "2-1",
  "direction_enabled": true,
  "direction": "forward",
  "traffic_enabled": true,
  "traffic_loading": false,
  "traffic_in_zone": false,
  "traffic_state": "WAIT_RED"
}
```

### Поворот направо

```json
{
  "move_type": "right",
  "deg": 18.4,
  "node": 15,
  "target_node": 20,
  "inliers": 57,
  "direction": "backward"
}
```

Положительный `deg` означает `right`.

### Поворот налево

```json
{
  "move_type": "left",
  "deg": -12.7,
  "node": 310,
  "target_node": 315,
  "inliers": 48,
  "direction": "forward"
}
```

Отрицательный `deg` означает `left`.

### Локализация потеряна

```json
{
  "move_type": "lost",
  "node": 310,
  "reason": "inliers: 8",
  "ts": 1787660000.123,
  "cam": "front",
  "map": "2-1/front_shard/05_of_25",
  "global_node": 415,
  "mode": "front",
  "route": "2-1"
}
```

Контроллер не должен продолжать движение по последнему углу после `lost`.

## Все варианты STOP

### Маршрут завершён

После `STOP_CONFIRM` подтверждённых конечных фиксов:

```json
{
  "move_type": "stop",
  "reason": "route_complete",
  "node": 36,
  "global_node": 1632,
  "ts": 1787660000.123,
  "cam": "front",
  "map": "2-1/front_shard/25_of_25",
  "mode": "front",
  "route": "2-1",
  "direction_enabled": true,
  "direction": "backward",
  "traffic_enabled": true,
  "traffic_loading": false,
  "traffic_in_zone": false,
  "traffic_state": "GO"
}
```

Это окончательная остановка маршрута. Для нового движения оператор должен выбрать/сбросить маршрут и нажать старт.

### Маршрут не выбран

```json
{
  "move_type": "stop",
  "reason": "route_not_selected",
  "route": null,
  "requested_route": null,
  "map": null,
  "mode": "idle",
  "paused": true
}
```

### Маршрут загружается

```json
{
  "move_type": "stop",
  "reason": "route_loading",
  "route": null,
  "requested_route": "2-1",
  "map": null,
  "mode": "idle",
  "paused": true
}
```

### Маршрут загружен и ожидает START

```json
{
  "move_type": "stop",
  "reason": "route_loaded",
  "route_loaded": true,
  "route": "2-1",
  "map": "2-1/front_shard/01_of_25",
  "mode": "front",
  "cam": "front",
  "paused": true
}
```

Загрузка маршрута не запускает движение автоматически.

### Оператор поставил систему на паузу

```json
{
  "move_type": "stop",
  "node": 420,
  "paused": true,
  "cam": "front",
  "map": "2-1/front_shard/06_of_25",
  "global_node": 473,
  "mode": "front",
  "route": "2-1"
}
```

### Остановка перед светофором

```json
{
  "move_type": "stop",
  "deg": 0.0,
  "node": 34,
  "global_node": 714,
  "traffic_enabled": true,
  "traffic_loading": false,
  "traffic_in_zone": true,
  "traffic_state": "WAIT_RED"
}
```

У этой остановки нет `reason=route_complete`. После последовательности `RED → GREEN` состояние становится `GO`, движение разрешается, а детекция до конца текущего проезда больше не влияет на команды.

### Нет кадра или карты камеры

Возможные причины:

```json
{"move_type":"stop","reason":"нет кадра передней камеры","cam":"front"}
```

```json
{"move_type":"stop","reason":"нет карты передней камеры","cam":"front"}
```

```json
{"move_type":"stop","reason":"нет карты задней камеры","cam":"rear"}
```

### Карты камер принадлежат разным маршрутам

```json
{
  "move_type": "stop",
  "reason": "front/rear maps belong to different routes",
  "map_mismatch": true,
  "cam": "front"
}
```

## Возможные причины LOST

| `reason` | Значение |
|---|---|
| `inliers: N` | Недостаточно подтверждённых соответствий |
| `скачок: N > M` | Недопустимый скачок номера ноды |
| `мало пар: N` | Недостаточно 2D–3D соответствий для PnP |
| `PnP не сошёлся` | Решение позы не найдено |
| `resume: релокализация по полной карте` | После START ещё не завершена релокализация |
| `шард не выбран — нажмите «СБРОС ШАРДА»` | Активный шард не определён |

Текст диагностических причин не следует использовать как основной машинный протокол. Контроллер должен прежде всего проверять `move_type`.

## Направление движения

При включённом расчёте направления добавляются:

```json
{
  "direction_enabled": true,
  "direction": "forward"
}
```

или:

```json
{
  "direction_enabled": true,
  "direction": "backward"
}
```

Для маршрута `2-1` задний ход действует в глобальных диапазонах `0–26` и `1596–1632`. В этих диапазонах защитная логика заменяет ошибочный `left` на `right`, сохраняя модуль угла:

```json
{
  "move_type": "right",
  "deg": 7.01,
  "direction": "backward"
}
```

## Состояния светофора

| Состояние | Значение |
|---|---|
| `WAIT_RED` | Ожидание первого красного; начальный зелёный не разрешает движение |
| `WAIT_GREEN` | Красный получен, ожидается зелёный |
| `GO` | Зелёный после красного получен; разрешение защёлкнуто |

Связанные поля:

```json
{
  "traffic_enabled": true,
  "traffic_loading": false,
  "traffic_in_zone": true,
  "traffic_state": "WAIT_GREEN"
}
```

## Управляющие команды

Команды публикуются в `robot.vision.control`.

### Пауза

```json
{"cmd":"pause"}
```

### Старт/продолжение

```json
{"cmd":"resume"}
```

### Сброс состояния пилота и светофора

```json
{"cmd":"reset"}
```

### Повторная релокализация и выбор шарда

```json
{"cmd":"reset_shard"}
```

### Сброс автомата светофора в WAIT_RED

```json
{"cmd":"reset_traffic"}
```

### Выбор маршрута

```json
{"cmd":"set_route","route":"1-2"}
```

```json
{"cmd":"set_route","route":"2-1"}
```

```json
{"cmd":"set_route","route":"3-1"}
```

```json
{"cmd":"set_route","route":"2"}
```

```json
{"cmd":"set_route","route":"office"}
```

Смена маршрута разрешена только на паузе. После загрузки публикуется `reason=route_loaded`; для начала движения нужна отдельная команда `resume`.

### Выгрузить маршрут и освободить память

```json
{"cmd":"clear_route"}
```

Команда разрешена только на паузе. После неё демон публикует `reason=route_not_selected`.

### Выбор камеры

```json
{"cmd":"set_mode","mode":"front"}
```

```json
{"cmd":"set_mode","mode":"rear"}
```

```json
{"cmd":"set_mode","mode":"dual"}
```

Смена режима разрешена только на паузе и только при наличии соответствующей карты маршрута.

### Включить или выключить светофор

```json
{"cmd":"set_traffic","enabled":true}
```

```json
{"cmd":"set_traffic","enabled":false}
```

### Включить или выключить зоны направления

```json
{"cmd":"set_direction","enabled":true}
```

```json
{"cmd":"set_direction","enabled":false}
```

### Стартовый и конечный манёвры маршрута 2-1

Переключение разрешено только на паузе после выгрузки маршрута.

```json
{"cmd":"set_terminal_maneuvers","enabled":true}
```

Стандартный режим использует шарды 01–25, включая начальный и конечный задний ход.

```json
{"cmd":"set_terminal_maneuvers","enabled":false}
```

Режим без манёвров использует только шарды 02–24. До узла 27 публикуется `stop` с `reason=outside_route_segment`. На узле 1595 остановка защёлкивается с `reason=route_complete`; шард 25 не загружается. Поле `terminal_maneuvers` присутствует в каждой команде.

## Сообщение скорости

Контроллер публикует в `gateway.robot.speed`:

```json
{"speed_pwm":85}
```

Упреждение вычисляется так:

```text
base_nodes = clamp(speed_pwm / 10, 4, 20)
effective_nodes = max(base_nodes - 30 × abs(offset), 4)
```

Если сообщение отсутствует или в нём нет `speed_pwm`, используется постоянное упреждение `LOOKAHEAD_NODES=5`.

## HTTP API админки

Статус:

```bash
curl http://127.0.0.1:8080/api/status
```

Список маршрутов:

```bash
curl http://127.0.0.1:8080/api/routes
```

Отправка команды:

```bash
curl -X POST http://127.0.0.1:8080/api/control \
  -H 'Content-Type: application/json' \
  -d '{"cmd":"pause"}'
```

```bash
curl -X POST http://127.0.0.1:8080/api/control \
  -H 'Content-Type: application/json' \
  -d '{"cmd":"set_route","route":"2-1"}'
```

```bash
curl -X POST http://127.0.0.1:8080/api/control \
  -H 'Content-Type: application/json' \
  -d '{"cmd":"resume"}'
```

## Приоритет обработки на контроллере

Рекомендуемый порядок:

1. `move_type=stop` — немедленная остановка.
2. `move_type=lost` — безопасная остановка, не использовать предыдущий угол.
3. `direction=backward` — включить заднее направление движения.
4. `move_type=left/right/straight` — применить руление.
5. `deg` использовать только для `left`, `right` и `straight`; у штатного конечного `stop` поле `deg` отсутствует.
