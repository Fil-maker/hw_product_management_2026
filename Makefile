.PHONY: setup check lint format type s3-up s3-down ingest data \
	data-check-version data-download data-convert data-split data-stats data-upload \
	export serve-up verify load-test serve-down

# Параметры конвейера данных; переопределяются из командной строки:
#   make data VERSION=v2 SEED=7 VAL_SIZE=10000
VERSION ?= v1
SEED ?= 42
VAL_SIZE ?= 5000

# Обучение на NVIDIA GPU: `make infra-up GPU=1`, `make experiments GPU=1`.
# Образ собирается с CUDA-сборкой PyTorch, контейнеру отдаётся видеокарта.
GPU ?= 0
ifeq ($(GPU),1)
export TORCH_VARIANT := gpu
export COMPOSE_FILE := docker-compose.yml:docker-compose.gpu.yml
export COMPOSE_PATH_SEPARATOR := :
endif

# Коммит, на котором выполняются запуски: в контейнере нет .git, передаём его снаружи.
export GIT_COMMIT := $(shell git rev-parse HEAD 2>/dev/null)
export GIT_DIRTY := $(shell [ -n "$$(git status --porcelain --untracked-files=no 2>/dev/null)" ] && echo true || echo false)

# Остановка затрагивает и сервисы с профилями, иначе docker-сеть останется занятой.
COMPOSE_ALL := docker compose --profile train --profile serve

# Какой запуск развёрнут в сервисе; другой конфиг: make export SERVING_CONFIG=...
SERVING_CONFIG ?= configs/serving.yaml

setup: .env
	uv sync --extra cpu --extra export --extra serve
	uv run pre-commit install

# Локальный .env из примера — только если его ещё нет (существующий не трогаем).
.env:
	cp .env.example .env

check:
	uv run ruff check .
	uv run ruff format --check .
	uv run mypy src/

lint:
	uv run ruff check . --fix

format:
	uv run ruff format .

type:
	uv run mypy src/

# --- S3 (RustFS) ---------------------------------------------------------------

s3-up: .env
	docker compose up -d --wait
	docker compose wait s3-init

s3-down:
	$(COMPOSE_ALL) down

# --- Данные --------------------------------------------------------------------

# Для медленной сети можно передать уже скачанный официальный архив (сверяется MD5):
#   make ingest ARCHIVE=~/Downloads/cifar-10-python.tar.gz
ingest:
	uv run cifar-ingest $(if $(ARCHIVE),--from-file $(ARCHIVE))

# Весь конвейер: сначала проверяем, что версия свободна, чтобы не делать работу зря.
data: data-check-version data-download data-convert data-split data-stats data-upload

data-check-version:
	uv run cifar-upload --version $(VERSION) --check-only

data-download:
	uv run cifar-download

data-convert:
	uv run cifar-convert

data-split:
	uv run cifar-split --val-size $(VAL_SIZE) --seed $(SEED)

data-stats:
	uv run cifar-stats

data-upload:
	uv run cifar-upload --version $(VERSION)

# --- Эксперименты (MLflow + обучение в контейнерах) ----------------------------

# Собрать образ для обучения и поднять S3, PostgreSQL и MLflow (http://localhost:5000).
infra-up: .env
	docker compose --profile train build
	docker compose up -d --wait
	docker compose wait s3-init

infra-down:
	$(COMPOSE_ALL) down

# Один запуск в отдельном контейнере: make train CONFIG=configs/simple_cnn.yaml
train: CONFIG ?= configs/simple_cnn.yaml
train:
	docker compose run --rm trainer cifar-train --config $(CONFIG)

baseline:
	docker compose run --rm trainer cifar-train --config configs/baseline.yaml

# Серия по конфигу сетки, каждый запуск — в своём контейнере:
#   make experiments CONFIG=configs/grid.yaml
experiments: CONFIG ?= configs/grid.yaml
experiments:
	uv run cifar-grid --config $(CONFIG)

# --- Сервинг (Triton + HTTP-обёртка) -------------------------------------------

# Скачать модель из S3, экспортировать в ONNX и собрать model repository.
export: .env
	docker compose up -d --wait s3
	uv run cifar-export --config $(SERVING_CONFIG)

# Поднять Triton, обёртку и S3 (он нужен для make verify). Ждёт, пока все станут healthy.
serve-up: .env
	@test -d model_repository && test -f serving_assets/stats.json \
		|| { echo "model repository is missing: run make export first"; exit 1; }
	docker compose --profile serve up -d --build --wait s3 triton api

# Прогнать весь test через обёртку и сравнить метрики с обучением.
# Допуски: make verify VERIFY_ARGS="--accuracy-tolerance 0.001"
verify:
	uv run cifar-verify --config $(SERVING_CONFIG) $(VERIFY_ARGS)

# Нагрузочный тест: 1, 2, 5, 10 workers, перезаписывает reports/load_test.md.
# Параметры: make load-test LOAD_ARGS="--workers 1 5 --requests 2000"
load-test:
	uv run cifar-load-test --config $(SERVING_CONFIG) $(LOAD_ARGS)

serve-down:
	$(COMPOSE_ALL) down
