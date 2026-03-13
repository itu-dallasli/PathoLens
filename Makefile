.PHONY: help install install-dev test lint format check pipeline

help:
	@echo "PathoLens Developer Commands:"
	@echo "  make install      - Install production dependencies"
	@echo "  make install-dev  - Install development dependencies + pre-commit hooks"
	@echo "  make test         - Run the test suite (extremely fast, entirely synthetic data)"
	@echo "  make test-cov     - Run tests with coverage report"
	@echo "  make lint         - Run ruff linter to find errors"
	@echo "  make format       - Run ruff formatter to fix code style"
	@echo "  make check        - Run format, lint, and tests (CI pipeline equivalent)"

install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"
	pre-commit install

test:
	pytest tests/ -v

test-cov:
	pytest tests/ -v --cov=patholens --cov-report=term-missing

lint:
	ruff check src tests

format:
	ruff format src tests
	ruff check --fix src tests

check: format lint test
