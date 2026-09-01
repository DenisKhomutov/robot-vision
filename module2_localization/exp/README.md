# MSLD — многосессионная дистилляция ориентиров

Эксперимент проверяет, сохраняет ли отбор устойчивых 3D-точек качество
локализации при сокращении банка. Боевые карты не изменяются.

Быстрый тест солнечного участка маршрута `1-2`:

```bash
python -m module2_localization.exp.msld_experiment
```

По умолчанию используются:

- `1-2/front_shard/03_of_25`;
- обычный участок первого видео, соответствующий этому шарду;
- первая половина `data/test_neg.mkv` для оценки стабильности;
- вторая половина `test_neg` для контроля.

Сравниваются:

- полный исходный банк;
- стабильностный отбор 75%, 50% и 25% точек;
- случайный отбор тех же 75%, 50% и 25% точек.

Результаты сохраняются в:

```text
module2_localization/exp/artifacts/msld/<дата_время>/
├── maps/
├── diagnostics/
├── landmark_statistics.npz
├── summary.csv
└── summary.json
```

Каждый `stable_*` дополнительно содержит готовую модель COLMAP GUI:

```text
maps/stable_XX/gui/images/
maps/stable_XX/gui/sparse/0/
maps/stable_XX/gui/gui.json
```

Повторный ручной экспорт:

```bash
python -m module2_localization.exp.export_msld_gui \
  module2_localization/exp/artifacts/msld/<запуск>/maps/stable_25
```

Главные метрики: `recall15`, `recall20`, `inliers_p10`, `inliers_median`,
`longest_lost15`, `frame_ms_median` и `frame_ms_p90`.

Первый подтверждающий запуск сохранён в `20260901_142131`: стабильностный банк
на 25% точек дал `recall20=1.0`, минимум 43 inliers и медиану 65.41 мс;
случайный банк того же размера дал `recall20=0.6`, минимум 8 inliers и серию
из четырёх кадров ниже 15 inliers.
