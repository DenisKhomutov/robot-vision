# Пайплайн сборки уличной карты

Этот документ описывает полный проверенный путь: от исходных видео до карты, которую загружает
runtime-локализатор. Основная схема проверена на задней камере маршрута 3.

## Итоговая структура

После завершения остаются две карты:

```text
module2_localization/maps/ROUTE_reconstruction/   # полная исходная SfM-модель и база
module2_localization/maps/ROUTE_runtime/          # очищенная модель и файлы навигации
```

Исходную реконструкцию нельзя заменять очищенной runtime-копией: она нужна для повторной проверки,
окрашивания и изменения правил фильтрации.

## 1. Подготовка видео

Для построения использовать сжатые видео `1280x720`, `30 FPS`. Для одного маршрута, разделённого
на два файла, порядок должен быть строго следующим:

```text
camera-rear-1-720.mkv
camera-rear-2-720.mkv
```

Конец первого файла должен соответствовать началу второго. Нельзя перемешивать камеры и маршруты.

## 2. Нарезка каждого видео

Уличные видео нарезаются равномерно через 8 исходных кадров. Адаптивное учащение на поворотах не
используется: yaw создаёт большой optical flow, но не создаёт полезную трансляционную базу.

```bash
uv run --no-sync python -m module2_localization.tools.extract_adaptive \
  module2_localization/data/street_video/camera-rear-1-720.mkv \
  --out frames_route12_rear_part1_gap8_nomask \
  --fixed-gap 8 --min-move 2

uv run --no-sync python -m module2_localization.tools.extract_adaptive \
  module2_localization/data/street_video/camera-rear-2-720.mkv \
  --out frames_route12_rear_part2_gap8_nomask \
  --fixed-gap 8 --min-move 2
```

Параметр `--min-move 2` исключает стоянки. Маски людей и машин в проверенной схеме не применяются:
они ухудшали связность уличной реконструкции.

После нарезки проверить в обоих `extraction.json`:

```json
{
  "fixed_gap": 8,
  "min_gap": 8,
  "max_gap": 8,
  "min_move_px_1280": 2.0
}
```

## 3. Объединение частей маршрута

Кадры складываются в один каталог с префиксами частей, чтобы лексикографическая сортировка
полностью совпадала с порядком движения:

```text
frames_route12_rear_gap8_nomask/
├── p1_000000.jpg
├── p1_000008.jpg
├── ...
├── p2_000000.jpg
├── p2_000008.jpg
└── ...
```

`p1_` всегда идёт раньше `p2_`. Дублирующий стыковой кадр можно оставить: COLMAP сопоставит его с
соседями. Если он относится к стоянке, нарезчик обычно исключит последующие неподвижные кадры.

В итоговом каталоге сохранить общий `extraction.json` с источниками, параметрами нарезки и числом
кадров каждой части.

## 4. Проверка датасета

Перед тяжёлым запуском проверить:

- все изображения имеют размер `1280x720`;
- имена уникальны и отсортированы по времени маршрута;
- отсутствуют маски и кадры другой камеры;
- переход `p1 -> p2` визуально непрерывен;
- нет длинных серий одинаковых стоячих кадров;
- фактическое число последовательных пар при `overlap=25` примерно равно `25 * N`.

## 5. Построение полной COLMAP-карты

ALIKED и LightGlue обязательно запускать с доступной CUDA. Mapper — COLMAP incremental, не GLOMAP.

```bash
uv run --no-sync python -m module2_localization.tools.build_map \
  --images frames_route12_rear_gap8_nomask \
  --tag route12_rear_colmap_gap8 \
  --mapper colmap \
  --pairs sequential \
  --overlap 25 \
  --det-threshold 0.04 \
  --camera-model simple-radial \
  --refine-intrinsics \
  --colors
```

Проверенные настройки mapper в `build_map.py`:

```text
multiple_models                 0
tri_ignore_two_view_tracks      0
tri_min_angle                   1.5
filter_min_tri_angle            1.5
local_ba_min_tri_angle          6
ba_global_images_freq           100
ba_global_points_freq           50000
ba_global_max_num_iterations    100
ba_global_max_refinements       10
ba_refine_focal_length          1
ba_refine_principal_point       0
ba_refine_extra_params          1
```

Двухкадровые точки разрешаются во время реконструкции. Их удаление выполняется только после
успешной глобальной BA.

## 6. Контроль реконструкции

```bash
colmap model_analyzer \
  --path module2_localization/maps/route12_rear_colmap_gap8/sparse/0
```

До экспорта проверить:

- зарегистрировано желательно не меньше 90–95% движущихся кадров;
- траектория непрерывна и следует реальному маршруту;
- нет уменьшения или увеличения людей, заборов и зданий к концу;
- дорога и стены не выгибаются после поворотов;
- высота камеры визуально соответствует 0,87 м;
- средняя ошибка репроекции ориентировочно находится в диапазоне 1–1,5 px;
- уточнённые интринсики физически правдоподобны.

Карту нельзя принимать только по числу зарегистрированных кадров. Решающее условие — единый
масштаб и корректная геометрия на всём маршруте.

## 7. Окрашенная GUI-модель

Если карта строилась с `--colors`, цвета уже присутствуют в `sparse/0`. Для отдельной GUI-копии с
абсолютными путями к кадрам используется очиститель на следующем шаге. Благодаря абсолютным путям
COLMAP GUI показывает исходное изображение при выборе позы.

## 8. Очистка коротких треков

Создать отдельную модель, не изменяя исходную реконструкцию:

```bash
uv run --no-sync python -m module2_localization.tools.clean_colmap_model \
  --input module2_localization/maps/route12_rear_colmap_gap8/sparse/0 \
  --output module2_localization/maps/route12_rear_colmap_gap8/viewer_model_clean \
  --min-track-length 3 \
  --min-retained-ratio 0.70 \
  --absolute-image-root module2_localization/data/frames_route12_rear_gap8_nomask
```

Удаляются только 3D-точки с длиной трека меньше 3. Если остаётся меньше 70% точек, скрипт
останавливается и не принимает агрессивную очистку.

## 9. Создание runtime-карты

Создать отдельный каталог и поместить очищенную модель в стандартный путь `sparse/0`:

```text
module2_localization/maps/route12_rear_gap8_final/
└── sparse/
    └── 0/
        ├── cameras.bin
        ├── images.bin
        ├── points3D.bin
        └── rigs.bin / frames.bin, если они присутствуют
```

Затем экспортировать геометрию:

```bash
uv run --no-sync python -m module2_localization.tools.export_map \
  --map route12_rear_gap8_final
```

И построить ALIKED descriptor bank на GPU:

```bash
uv run --no-sync python -m module2_localization.tools.build_descriptor_bank \
  --map route12_rear_gap8_final \
  --images frames_route12_rear_gap8_nomask \
  --kpts 4096 \
  --det-threshold 0.04
```

Финальная runtime-карта должна содержать:

```text
route12_rear_gap8_final/
├── sparse/0/
├── runtime.npz
└── aliked_bank.npz
```

`runtime.npz` и `aliked_bank.npz` обязаны быть построены из одной и той же очищенной модели. Нельзя
смешивать файлы от полной и очищенной реконструкций: индексы 3D-точек будут различаться.

## 10. Проверка runtime-карты

До запуска робота прогнать исходное видео через готовую карту:

```bash
uv run --no-sync python -m module2_localization.tools.make_video \
  module2_localization/data/street_video/camera-rear-1-720.mkv \
  --map route12_rear_gap8_final \
  --back \
  --step 3 \
  --fps 10 \
  --out module2_localization/out/route12_rear_part1_localized.mp4
```

Проверить непрерывность узлов, число инлайеров, отсутствие ложных скачков и адекватность команд.
Параметры руления описаны отдельно в `STREET_NAVIGATION_PARAMETERS.md`.
