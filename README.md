# hw-product-management-2026

Учебный проект: инфраструктура (ДЗ №1), воспроизводимый конвейер данных CIFAR-10 поверх S3 (ДЗ №2) и эксперименты с трекингом в MLflow (ДЗ №3).

## Установка и запуск

### Требования

- Linux / macOS (на Windows — WSL 2)
- [uv](https://docs.astral.sh/uv/) — Python 3.12 он поставит сам
- `make`
- Docker с Compose v2 (`docker compose`) — для S3, MLflow и обучения; в WSL нужна включённая WSL integration в Docker Desktop
- для обучения на GPU (необязательно) — видеокарта NVIDIA и NVIDIA Container Toolkit

### Поднять окружение с нуля

```bash
make setup
```

Команда создаёт `.venv` из `uv.lock` (PyTorch — CPU-сборка), ставит git-хук pre-commit и, если `.env` ещё нет, копирует его из `.env.example`.

### Команды

| Команда       | Что делает                                                                   |
|---------------|------------------------------------------------------------------------------|
| `make setup`  | поднять окружение с нуля                                                     |
| `make check`  | все проверки: ruff (стиль), `ruff format --check`, mypy (типы); ничего не меняет |
| `make lint`   | автоисправление линтером                                                     |
| `make format` | автоформатирование                                                           |
| `make type`   | только mypy                                                                  |
| `make s3-up`  | поднять S3 (RustFS) и создать бакет                                          |
| `make ingest` | загрузить сырой архив CIFAR-10 в S3                                          |
| `make data`   | собрать обработанный датасет и выложить его в S3 с версией                    |
| `make s3-down`| остановить хранилище (данные сохраняются в volume)                           |
| `make infra-up` | собрать образ для обучения, поднять S3, PostgreSQL и MLflow                |
| `make baseline` | baseline-запуск (линейная модель) в контейнере                             |
| `make train CONFIG=...` | один запуск по конфигу в контейнере                                |
| `make experiments CONFIG=...` | серия запусков по конфигу сетки, каждый — в своём контейнере  |
| `make infra-down` | остановить всё (данные, запуски и модели сохраняются в volumes)          |

На каждом коммите pre-commit запускает ruff и mypy из окружения проекта, поэтому их версии совпадают с `uv.lock`.

## Данные

### Хранилище

S3-совместимое хранилище [RustFS](https://rustfs.com) поднимается в Docker Compose (`docker-compose.yml`):

- `s3` — RustFS `1.0.3`, данные лежат в именованном volume `s3-data` и переживают `docker compose down`;
- `s3-init` — одноразовый контейнер на `amazon/aws-cli`: дожидается healthcheck хранилища и создаёт бакет `S3_BUCKET`, если его ещё нет.

Логин, пароль, бакет и порты задаются в `.env` (пример — `.env.example`; реальный `.env` в git не попадает).

Где посмотреть содержимое:

- веб-консоль RustFS: <http://localhost:9001>, логин и пароль — `S3_ACCESS_KEY` / `S3_SECRET_KEY` из `.env`;
- AWS CLI: `aws --endpoint-url http://localhost:9000 s3 ls s3://datasets/ --recursive | head`
  (ключи — в `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY`).

### Порядок запуска

```bash
make setup      # окружение и .env
make s3-up      # хранилище + бакет
make ingest     # архив с сайта CIFAR -> s3://datasets/raw/cifar-10-python.tar.gz
make data       # конвейер -> s3://datasets/processed/cifar10/v1/
make s3-down    # остановить хранилище
```

`make ingest` идемпотентен: если архив уже лежит в S3, он ничего не скачивает и не загружает. Существующий объект в `raw/` не перезаписывается никогда: перед загрузкой есть проверка, а сама загрузка условная (`If-None-Match: *`). Архив кладётся в S3, только если его MD5 совпадает с опубликованным на сайте CIFAR.

Сервер CIFAR бывает очень медленным. Если официальный архив уже скачан, можно загрузить его из файла — MD5 проверяется так же:

```bash
make ingest ARCHIVE=~/Downloads/cifar-10-python.tar.gz
```

Новая версия датасета с другими параметрами собирается без правки кода:

```bash
make data VERSION=v2 SEED=7 VAL_SIZE=10000
```

Версии неизменяемы. Если версия уже есть в S3, `make data` сразу завершается с ошибкой и ничего не пересобирает.

### Шаги конвейера

`make data` последовательно вызывает отдельные шаги. Каждый из них — модуль в `src/hw_product_management_2026/data/` со своей CLI-командой (`--help` показывает параметры), их можно запускать по одному:

| Шаг | Команда | Что делает |
|-----|---------|------------|
| 0 | `cifar-upload --version v1 --check-only` | проверить, что версия свободна |
| 1 | `cifar-download` | `raw/cifar-10-python.tar.gz` из S3 -> `data/raw/`, проверка MD5 |
| 2 | `cifar-convert` | pickle -> PNG в `data/processed/cifar10/images/` + разметка train/test в `data/interim/cifar10/`; pickle открывается только после сверки MD5 с официальным |
| 3 | `cifar-split --val-size 5000 --seed 42` | стратифицированное разбиение train -> train/val, test без изменений |
| 4 | `cifar-stats` | mean/std по каналам RGB только по train -> `stats.json` |
| 5 | `cifar-upload --version v1` | выгрузка в S3 (параллельно), `manifest.json` — последним |

Запуск отдельного шага: `uv run cifar-split --val-size 2000 --seed 1`.

Работа с S3 — `s3.py`, конвертация — `convert.py`, разбиение — `split.py`, статистики — `stats.py`, выгрузка — `upload.py`.

### Что лежит в S3

```
s3://datasets/
├── raw/
│   └── cifar-10-python.tar.gz      # официальный архив как есть
└── processed/cifar10/v1/
    ├── images/                     # 60 000 PNG: train_00000.png … train_49999.png, test_00000.png … test_09999.png
    ├── annotations/
    │   ├── train.json              # 45 000
    │   ├── val.json                # 5 000
    │   └── test.json               # 10 000
    ├── stats.json                  # mean/std по каналам (train, пиксели в [0, 1])
    └── manifest.json               # число картинок по выборкам, файлы с размером и sha256, параметры, git-коммит
```

Формат разметки:

```json
{
  "classes": ["airplane", "automobile", "..."],
  "images": [
    {"image_id": "train_00000", "file_name": "images/train_00000.png",
     "label": 6, "label_name": "frog", "sha256": "<sha256 байтов PNG>"}
  ]
}
```

### Карточка датасета

| | |
|---|---|
| **Источник** | [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html), A. Krizhevsky, 2009; версия для Python, `cifar-10-python.tar.gz`, MD5 `c58f30108f718f92721af3b95e74349a` |
| **Размер** | 60 000 цветных изображений 32×32 RGB: 50 000 в исходном train, 10 000 в test |
| **Классы (10)** | airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck — по 6 000 на класс, классы не пересекаются |
| **Разбиение (v1)** | train 45 000 / val 5 000 / test 10 000. Val выделен из исходного train стратифицированно (по 500 на класс), seed 42. Test — исходный, без изменений. Разбиение задаётся файлами разметки; картинки val сохраняют id `train_…` |
| **Нормализация** | `stats.json`: mean и std по каналам R, G, B, посчитаны только по train (45 000), пиксели в [0, 1] |

Известные проблемы:

- **Низкое разрешение.** 32×32: мелкие детали теряются, часть картинок трудно распознать даже человеку.
- **Шум в разметке.** В test около 0,5% ошибочных меток (Northcutt et al., *Pervasive Label Errors in Test Sets*, 2021).
- **Почти-дубликаты между train и test.** Около 3% картинок test имеют почти точные копии в train (Barz & Denzler, *Do we train on test data?*, 2020). Это немного завышает оценку на test.
- **Похожие классы.** automobile и truck, cat и dog путаются чаще остальных.
- **Сбалансированность.** Классы идеально сбалансированы, в реальных данных так бывает редко.

## Эксперименты

### Инфраструктура

`docker-compose.yml` из ДЗ №2 дополнен трекером экспериментов:

- `postgres` — PostgreSQL `17.6`, метаданные запусков MLflow; данные в volume `postgres-data`, порт наружу не публикуется;
- `mlflow` — сервер MLflow `3.17.2` (образ `docker/mlflow/Dockerfile`: официальный `ghcr.io/mlflow/mlflow:v3.17.2` плюс `psycopg2-binary` и `boto3`), интерфейс — <http://localhost:5000>, есть `healthcheck`. Артефакты (модели, графики) идут через сервер (`--serve-artifacts`) в отдельный бакет `mlflow` того же RustFS, поэтому контейнерам с обучением ключи от S3 для MLflow не нужны;
- `s3-init` теперь создаёт оба бакета: с данными (`S3_BUCKET`) и для артефактов (`MLFLOW_BUCKET`);
- `trainer` — образ для обучения (`Dockerfile`: `python:3.12.12-slim-bookworm`, окружение из `uv.lock` через `uv sync --frozen`). У сервиса профиль `train`, поэтому `docker compose up` его не запускает: каждый запуск обучения выполняется командой `docker compose run --rm trainer ...` в отдельном контейнере в той же docker-сети, что S3 и MLflow. Папка `configs/` монтируется в контейнер, так что конфиги можно менять без пересборки образа. Скачанный датасет и кеш весов torchvision лежат в volumes `train-data` и `torch-cache`.

Пароли и ключи берутся из `.env` (пример — `.env.example`).

### Запуск

```bash
make setup                                       # окружение и .env
make infra-up                                    # образ для обучения + S3 + PostgreSQL + MLflow
make ingest && make data                         # данные из ДЗ №2, если их ещё нет в S3 (v1)
make baseline                                    # baseline: линейная модель
make experiments CONFIG=configs/grid.yaml        # сетка: 12 запусков
make experiments CONFIG=configs/grid_final.yaml  # выбранная конфигурация на полном train, 20 эпох
make infra-down                                  # остановить всё
```

На CPU основная сетка занимает несколько часов. Для быстрой проверки есть сокращённая сетка: 6 запусков по 3 эпохи на 5000 картинках train, это несколько минут.

```bash
make experiments CONFIG=configs/grid_quick.yaml
```

С видеокартой NVIDIA образ собирается с CUDA-сборкой PyTorch (CUDA 13.2, нужен драйвер с поддержкой CUDA 13), а контейнеру отдаётся GPU (`docker-compose.gpu.yml`). Проверено на RTX 3070 Laptop: 20 эпох на полном train занимают около полутора минут вместо ~13 на CPU. Образ весит ~9.4 ГБ против ~1.9 ГБ у CPU-образа.

```bash
make infra-up GPU=1
make experiments GPU=1 CONFIG=configs/grid.yaml
```

Чтобы провести другой эксперимент, достаточно поменять конфиг, код трогать не нужно. Например, поменяйте `train.lr` в `configs/simple_cnn.yaml` и выполните `make train`, или добавьте значения в `grid:` конфига сетки.

### Конфиги

Конфиг одного запуска (`configs/simple_cnn.yaml`, `configs/baseline.yaml`) содержит всё для запуска: имя эксперимента и запуска, seed, версию данных в S3 (`data.s3_path`), модель (`model.type`: `linear`, `simple_cnn` или `torchvision.<имя>`, например `torchvision.resnet18`) и параметры обучения. Отдельные значения можно переопределить из командной строки: `cifar-train --config configs/simple_cnn.yaml --set train.lr=0.003`.

Конфиг сетки (`configs/grid.yaml`) — это базовый конфиг, шаблон имени запуска и перебираемые параметры:

```yaml
base: configs/simple_cnn.yaml
run_name: "cnn-lr{train.lr}-bs{train.batch_size}-aug{data.augment}"
grid:
  train.lr: [0.0003, 0.001, 0.003]
  train.batch_size: [64, 256]
  data.augment: [false, true]
```

`cifar-grid` (`src/hw_product_management_2026/training/grid.py`) строит все комбинации через `itertools.product`, до старта проверяет все получившиеся конфиги и запускает каждую комбинацию в своём контейнере. `--dry-run` только печатает команды, `--local` запускает обучение без docker.

### Что делает один запуск

`cifar-train` (`src/hw_product_management_2026/training/train.py`):

1. скачивает версию данных из S3 в `data/datasets/<s3_path>/`, если её там ещё нет (`manifest.json` скачивается последним и сверяются sha256 файлов). При первом чтении картинки сверяются с sha256 из разметки и кешируются в `arrays.npz`;
2. нормализует изображения по `stats.json` той же версии;
3. обучает модель на train (кросс-энтропия). После каждой эпохи считает loss, accuracy и macro-F1 на **val**;
4. хранит чекпоинт эпохи с минимальным val loss, а не последней;
5. сохраняет лучший чекпоинт в `model.pt`, загружает его обратно в новую модель и только тогда один раз измеряет **test**. Test ни на какой выбор не влияет;
6. логирует всё в MLflow и выкладывает модель в S3.

Что записывается в MLflow для каждого запуска:

| Что | Где |
|-----|-----|
| конфиг целиком | параметры (`train.lr`, `model.params.channels`, …) и артефакт `model/config.yaml` |
| версия данных, seed, git-коммит | теги `data_version`, `seed`, `git_commit` (`mlflow.source.git.commit`), `git_dirty` |
| метрики на val по эпохам | `val_loss`, `val_accuracy`, `val_macro_f1` (а также `train_loss`, `train_accuracy`, `lr`, `epoch_time_sec`) |
| время обучения | `train_time_sec` |
| эпоха лучшего чекпоинта | `best_epoch`, метрики `best_val_*` |
| качество лучшей модели на test | `test_loss`, `test_accuracy`, `test_macro_f1` |
| матрица ошибок на val (лучшая эпоха) | `val_confusion_matrix.png` и `val_confusion_matrix.json` |
| лучшая модель | `model/model.pt`, `model/metrics.json` |

Лучшая модель вместе с конфигом и метриками выкладывается в бакет с данными:

```
s3://datasets/models/<experiment_name>/<run_name>/
    model.pt        # state_dict + описание модели, классы, нормализация
    config.yaml     # итоговый конфиг запуска
    metrics.json    # val и test лучшей эпохи, история по эпохам, матрица ошибок
```

Контейнер не видит `.git`, поэтому коммит передаётся ему из `make` (переменные `GIT_COMMIT` и `GIT_DIRTY`).

### Результаты

Все запуски сделаны на версии данных `processed/cifar10/v1` (train 45 000, val 5 000, test 10 000) с seed 42 и чистым деревом git (коммиты `e957b55` и `e9a707b`, код обучения в них одинаковый). Метрики val взяты для лучшей эпохи (минимальный val loss), test измерен один раз для модели этой эпохи.

| Эксперимент | Запуск | Train | Эпох (лучшая) | val loss | val acc | val macro-F1 | test acc | test macro-F1 |
|---|---|---|---|---|---|---|---|---|
| `cifar10-baseline` | `linear-lr0.001-bs256-seed42` (линейный слой) | 45 000 | 20 (8) | 1.798 | 0.375 | 0.371 | 0.377 | 0.372 |
| `cifar10-simple-cnn` | `cnn-lr0.001-bs64-augfalse` — лучший в сетке | 20 000 | 10 (10) | **0.685** | **0.755** | **0.753** | 0.753 | 0.752 |
| `cifar10-simple-cnn` | `cnn-lr0.003-bs64-augfalse` | 20 000 | 10 (10) | 0.719 | 0.744 | 0.743 | 0.742 | 0.740 |
| `cifar10-simple-cnn` | `cnn-lr0.001-bs256-augfalse` | 20 000 | 10 (10) | 0.742 | 0.736 | 0.736 | 0.732 | 0.732 |
| `cifar10-simple-cnn` | `cnn-lr0.0003-bs64-augfalse` | 20 000 | 10 (10) | 0.758 | 0.729 | 0.727 | 0.728 | 0.726 |
| `cifar10-simple-cnn` | `cnn-lr0.001-bs64-augtrue` | 20 000 | 10 (10) | 0.764 | 0.726 | 0.724 | 0.724 | 0.723 |
| `cifar10-simple-cnn` | худший: `cnn-lr0.0003-bs256-augtrue` | 20 000 | 10 (10) | 0.978 | 0.655 | 0.653 | 0.650 | 0.649 |
| `cifar10-simple-cnn-final` | **`final-cnn-lr0.001-bs64-augfalse-ep20` — выбранная** | 45 000 | 20 (18) | **0.510** | **0.830** | **0.829** | **0.817** | **0.817** |
| `cifar10-simple-cnn-final` | `final-cnn-lr0.001-bs64-augtrue-ep20` | 45 000 | 20 (20) | 0.514 | 0.824 | 0.822 | 0.811 | 0.810 |

Сетка (`configs/grid.yaml`, 12 запусков на CPU, 20 000 картинок train, 10 эпох): `train.lr` ∈ {0.0003, 0.001, 0.003} × `train.batch_size` ∈ {64, 256} × `data.augment` ∈ {false, true}. Остальные параметры — из `configs/simple_cnn.yaml`: 3 блока conv (32-64-128), BatchNorm, dropout 0.3, Adam, weight decay 1e-4, cosine-расписание. Финальные запуски (`configs/grid_final.yaml`) выполнены на GPU.

**Вывод.** Выбрана конфигурация `lr=0.001, batch_size=64` без аугментации: в сетке у неё минимальный val loss (0.685 против 0.719 у следующей). На полном train за 20 эпох она даёт val loss 0.510 и val accuracy 83.0%, а это вдвое лучше baseline (37.5%). Вариант с аугментацией на коротком обучении заметно проигрывал, а на 20 эпохах почти догнал (val loss 0.514) и ещё не вышел на плато: лучшая эпоха у него последняя, train loss намного выше. Значит, при более длинном обучении он, скорее всего, выиграет, но по имеющимся val-метрикам выбран вариант без аугментации. Test в выборе не участвовал. Для выбранной модели test accuracy 81.7% и macro-F1 0.817, что близко к val, то есть на val модель не переобучена.

Модель выбранной конфигурации: `s3://datasets/models/cifar10-simple-cnn-final/final-cnn-lr0.001-bs64-augfalse-ep20/`.
