# Module 4: DPVO + ALIKED/LightGlue map builder

Экспериментальный модуль для быстрой офлайн-сборки карт без COLMAP/GLOMAP-реконструкции и без изменения рабочего runtime из `module2_localization`.

## Архитектура

```text
video + camera calibration
  -> DPVO
     -> полная траектория всех обработанных кадров
     -> фиксированные позы DPVO-keyframes
  -> исходные изображения ровно этих keyframes
  -> ALIKED keypoints + descriptors
  -> LightGlue matching между keyframes
  -> multi-view tracks
  -> triangulation при фиксированных позах DPVO
  -> runtime descriptor/landmark bank
  -> COLMAP text model только для просмотра
```

Распределение ответственности строгое:

- DPVO определяет положения и ориентации камер. После DPVO позы не оптимизируются и не меняются.
- ALIKED определяет 2D keypoints и descriptors.
- LightGlue связывает ALIKED-точки между кадрами.
- 3D landmarks триангулируются по LightGlue-трекам при фиксированных DPVO-позах.
- COLMAP не участвует в сборке или оптимизации. Он используется только как GUI viewer.

## Полная траектория и keyframe-позы

`trajectory_tum.txt` содержит позу каждого обработанного DPVO кадра. Например, текущий эксперимент со stride 8 содержит 320 поз.

Гибридная карта содержит только DPVO-keyframes из `dpvo_state.pt`. В текущем эксперименте их 128. Они являются подмножеством полной траектории, их координаты и ориентации совпадают с DPVO. Поэтому отображение keyframes визуально менее плавное, чем график всех 320 поз, но сами позы не сдвинуты.

Не следует добавлять интерполированные позы в sparse map как зарегистрированные изображения: на них нет построенных ALIKED/LightGlue landmark tracks. Полную плавную линию нужно отображать отдельно из `trajectory_tum.txt`.

## Главный инвариант сопоставления кадров

DPVO `video_stream` при `stride=N` сначала считывает N исходных кадров и передаёт последний из них с timestamp 0. Следовательно:

```text
source_frame_1_based = skip + (dpvo_timestamp + 1) * stride
```

Пример для stride 8:

```text
dpvo timestamp 23 -> source video frame 192
```

Нельзя извлекать `frame_000024.jpg` как исходный кадр 24. Такая ошибка связывает правильную DPVO-позу с чужим изображением; LightGlue продолжает находить совпадения, но триангуляция создаёт редкую и разорванную геометрию. `extract_dpvo_keyframes.py` повторяет чтение кадров DPVO и устраняет эту ошибку.

## Одна команда сборки

Из корня репозитория, в активированном `.venv`:

```bash
python -m module4_map_builder.tools.build_hybrid_map \
  --name route_test \
  --video 'module2_localization/data/new_street_video/1-2_short_720/camera-front(5)-720.mkv' \
  --calib module4_map_builder/configs/front_720_from_runtime.txt \
  --stride 8
```

Повторная сборка того же имени требует явного `--force`. Команда последовательно:

1. запускает DPVO и сохраняет полный state;
2. извлекает изображения именно DPVO-keyframes;
3. извлекает ALIKED features;
4. строит LightGlue matches для сдвигов `1,2,4,8`;
5. формирует multi-view tracks и триангулирует landmarks;
6. сохраняет исходный гибридный bank;
7. экспортирует совместимые с module2 `runtime_map/runtime.npz` и `runtime_map/aliked_bank.npz`;
8. создаёт обычную и чёрную COLMAP viewer-модели;
9. запускает `colmap model_analyzer`, если COLMAP установлен.

Настраиваемые параметры:

```text
--pair-strides 1,2,4,8
--max-reproj-error 5.0
--min-parallax-deg 0.2
--skip 0
```

## Результаты

```text
module4_map_builder/out/dpvo/<name>/
├── dpvo_state.pt
├── dpvo_state_summary.json
├── trajectory_tum.txt
├── images/
├── dpvo_aliked_tracks_bank.npz
├── dpvo_aliked_tracks_bank.json
├── runtime_map/
│   ├── runtime.npz
│   ├── aliked_bank.npz
│   └── runtime_map.json
├── colmap_view/
│   ├── images/
│   └── sparse_text/{cameras,images,points3D}.txt
└── colmap_view_black/
    ├── images/
    └── sparse_text/{cameras,images,points3D}.txt
```

`dpvo_aliked_tracks_bank.npz` содержит:

- `xyz`: триангулированные 3D landmarks;
- `desc`: агрегированные ALIKED descriptors;
- `node`: keyframe/node для landmark;
- `route`: центры DPVO-keyframes;
- `track_len`, `reproj`, `parallax`: метрики качества;
- `image_names`, `tstamps`, `K`, `dist`;
- `obs_offsets`, `obs_frames`, `obs_keypoint_indices`, `obs_xy`: реальные ALIKED/LightGlue observations для COLMAP tracks и диагностики.

## Открытие в COLMAP GUI

Для обычной карты:

```text
File -> Import model
Model: module4_map_builder/out/dpvo/<name>/colmap_view/sparse_text
Images: module4_map_builder/out/dpvo/<name>/colmap_view/images
```

Для чёрных точек используются аналогичные пути в `colmap_view_black`.

После пересборки необходимо закрыть старую модель и импортировать её снова: COLMAP GUI не перечитывает изменённые text-файлы автоматически.

В `images.txt` записаны world-to-camera quaternion/translation в соглашении COLMAP. Они вычисляются из camera-to-world DPVO pose. Экспортёр проверяет обратное восстановление центра каждой камеры и прекращает работу, если ошибка превышает `1e-9`.

Каждая строка `points3D.txt` содержит настоящий LightGlue track `IMAGE_ID POINT2D_IDX`, а соответствующая вторая строка изображения содержит настоящий ALIKED `X Y POINT3D_ID`. Синтетические observations использовать нельзя.

## Проверка качества

```bash
colmap model_analyzer \
  --path module4_map_builder/out/dpvo/<name>/colmap_view/sparse_text
```

Проверять нужно как минимум:

- количество registered images равно количеству DPVO-keyframes;
- points и observations ненулевые;
- mean track length больше 2;
- reprojection error находится в пределах выбранного фильтра;
- выбор 3D-точки в GUI показывает её track;
- выбор камеры открывает соответствующее исходное изображение;
- центры камер повторяют keyframe-подмножество DPVO-траектории.

Эталон текущей исправленной сборки:

```text
DPVO processed poses: 320
DPVO keyframes / registered images: 128
ALIKED/LightGlue landmarks: 18151
observations: 95399
mean track length: 5.255854
mean COLMAP reprojection error: 1.136922 px
max DPVO-to-COLMAP camera-center conversion error: 1.26e-15
```

## Что было ошибочным и больше не должно повторяться

- Использовать последовательные исходные кадры вместо кадров с учётом DPVO stride/timestamp.
- Подменять ALIKED tracks синтетическими проекциями только ради отображения точек в GUI.
- Экспортировать identity orientation или несогласованные quaternion и translation.
- Считать COLMAP viewer-модель источником runtime-карты.
- Смешивать все позы полной траектории с keyframe-позами, имеющими feature observations.

## Локализация

Следующий этап использует этот банк без перестроения карты:

```text
query frame
  -> ALIKED
  -> LightGlue / descriptor matching с map landmarks
  -> 2D-3D correspondences
  -> PnP RANSAC
  -> inliers >= 40
  -> позиция относительно DPVO route + управляющая команда
```

Результат прогона должен сохраняться как видео с исходным кадром, эталонной DPVO-линией, оценённой позицией, node, числом matches/inliers, расстоянием до линии и итоговой командой. COLMAP для runtime-локализации не нужен.
