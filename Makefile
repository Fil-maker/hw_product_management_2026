.PHONY: setup check lint format type s3-up s3-down ingest data \
	data-check-version data-download data-convert data-split data-stats data-upload

# Параметры конвейера данных; переопределяются из командной строки:
#   make data VERSION=v2 SEED=7 VAL_SIZE=10000
VERSION ?= v1
SEED ?= 42
VAL_SIZE ?= 5000

setup: .env
	uv sync
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
	docker compose down

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
