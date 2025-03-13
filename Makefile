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
	python -c "import shutil; [shutil.rmtree(p, ignore_errors=True) for p in ['build', 'dist', '$(PACKAGE).egg-info', '.pytest_cache', 'htmlcov', 'docs/_build', '.coverage']]"
	python -c "import pathlib, shutil; [shutil.rmtree(p, ignore_errors=True) for p in pathlib.Path('.').rglob('__pycache__')]"
	python -c "import pathlib; [p.unlink() for p in pathlib.Path('.').rglob('*.pyc')]"

# Run tests
test:
	$(PYTHON) -m pytest -n 4

# Run tests with coverage
coverage:
	$(PYTHON) -m pytest --cov=$(PACKAGE) --cov-report=html

# Format code with black
format:
	$(UV) run ruff check --fix .
	$(UV) run ruff format .

# Build documentation
docs:
	cd docs && $(PYTHON) -m sphinx.cmd.build -b html . _build/html

# Build package distribution
build:
	$(UV) build

# Install all dependencies (including dev)
all: setup dev-install

publish: clean build
	twine upload dist/*

# Update dependencies
update:
	$(UV) sync --extra dev

upgrade:
	$(UV) sync --active --extra dev
	$(UV) export --format requirements-txt --extra dev --no-hashes --output-file requirements.txt > $(if $(filter $(OS),Windows_NT),NUL,/dev/null) 2>&1