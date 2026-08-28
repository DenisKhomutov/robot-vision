# Эксперимент единой компактной карты

Эксперимент не изменяет `module2_localization/maps` и не подключён к runtime-демону.
Все производные файлы по умолчанию сохраняются в:

```text
module2_localization/exp/artifacts/<маршрут>/<карта>/
```

Для `2-1/front_full`:

```text
module2_localization/exp/artifacts/2-1/front_full/
├── compact_bank.npz
├── metadata.json
├── compact_flat.faiss
└── compact_ivf.faiss
```

Каталог `artifacts` исключён из Git.

## Дополнение банка новым освещением

Скрипт добавляет только геометрически подтверждённые дескрипторы к уже
существующим 3D-точкам. Рабочие карты и шарды не перезаписываются.

```bash
uv run --no-sync python -m module2_localization.exp.augment_descriptor_bank \
  module2_localization/data/test_neg.mkv \
  --map 1-2/front_full \
  --frame-step 20 --max-frames 12
```

Результат сохраняется в
`module2_localization/exp/artifacts/1-2/front_full/sunny_bank/`.

## 1. Компактный банк

Без FAISS, два medoid на 3D-точку:

```bash
cd ~/projects/robot-vision
uv run --no-sync python -m module2_localization.exp.build_compact_bank \
  --map 2-1/front_full \
  --prototypes 2
```

Однопрототипный вариант нужно сохранять в другой `--output`, чтобы не затереть
двухпрототипный результат.

## 2. Точный контрольный прогон

```bash
uv run --no-sync python -m module2_localization.exp.test_compact_fullmap_video \
  module2_localization/data/front_2-1/camera-front-1-720.mkv \
  --map 2-1/front_full \
  --bank compact \
  --backend torch-exact \
  --step 8 \
  --out module2_localization/out/compact_21_exact
```

Результат содержит `result.mp4`, `diagnostics.jsonl` и `summary.json`.

Для контрольного полного исходного банка используется `--bank original`. Он
может потребовать значительно больше GPU-памяти и времени.

## 3. FAISS

FAISS не добавлен в зависимости проекта: на x86 и Jetson требуются разные
совместимые сборки. После установки FAISS в используемое Python-окружение:

```bash
python -m module2_localization.exp.build_faiss_index \
  --map 2-1/front_full \
  --mode both \
  --nlist 1024
```

Прогон IVF:

```bash
python -m module2_localization.exp.test_compact_fullmap_video \
  module2_localization/data/front_2-1/camera-front-1-720.mkv \
  --map 2-1/front_full \
  --backend faiss-ivf \
  --nprobe 32 \
  --step 8 \
  --out module2_localization/out/compact_21_ivf32
```

Сначала сравниваются исходный банк и `torch-exact`, затем `faiss-flat`, после
этого IVF с `nprobe=64`, `32` и `16`. Рабочий демон до принятия результатов не
изменяется.
