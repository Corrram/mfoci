.PHONY: setup clean install test lint format docs build all update upgrade

# Default Python interpreter
PYTHON = python
# Virtual environment path
VENV = .venv
# Path to uv
UV = uv
# Path to the main package
PACKAGE = mfoci

# Create and set up virtual environment
setup:
	$(UV) venv $(VENV)
	$(UV) pip install --requirement pyproject.toml

# Install the package in development mode
install:
	$(UV) pip install -e .

# Install dev dependencies
dev-install:
	$(UV) pip install --requirement pyproject.toml --dev

# Clean build artifacts and cache files
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf $(PACKAGE).egg-info/
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf docs/_build/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

# Run tests
test:
	$(PYTHON) -m pytest

# Run tests with coverage
coverage:
	$(PYTHON) -m pytest --cov=$(PACKAGE) --cov-report=html

# Format code with black
format:
	$(PYTHON) -m black .

# Build documentation
docs:
	cd docs && $(PYTHON) -m sphinx.cmd.build -b html . _build/html

# Build package distribution
build:
	$(UV) build

# Install all dependencies (including dev)
all: setup dev-install

# Update dependencies
update:
	$(UV) sync --extra dev

upgrade:
	$(UV) sync --active --extra dev
	$(UV) export --format requirements-txt --extra dev --no-hashes --output-file requirements.txt > $(if $(filter $(OS),Windows_NT),NUL,/dev/null) 2>&1