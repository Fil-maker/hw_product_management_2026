# hw-product-management-2026

Учебный проект: инфраструктура (ДЗ №1) и воспроизводимый конвейер данных CIFAR-10 поверх S3 (ДЗ №2).

## Установка и запуск

### Требования

- Linux / macOS (на Windows — WSL 2)
- [uv](https://docs.astral.sh/uv/) — Python 3.12 он поставит сам
- `make`
- Docker с Compose v2 (`docker compose`) — для хранилища S3; в WSL нужна включённая WSL integration в Docker Desktop

### Поднять окружение с нуля

```bash
make setup
```

Команда создаёт `.venv` из `uv.lock`, ставит git-хук pre-commit и, если `.env` ещё нет, копирует его из `.env.example`.

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
