.PHONY: setup check lint format type

setup:
	uv sync
	uv run pre-commit install

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
