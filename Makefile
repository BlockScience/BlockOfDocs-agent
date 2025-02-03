.PHONY: install run clean wipe-db test lint help

# Variables
PYTHON = python3
PIP = pip3
FLASK_PORT = 3000

help:
	@echo "Available commands:"
	@echo "  make install    - Install dependencies"
	@echo "  make run       - Run the Slack bot"
	@echo "  make clean     - Remove Python cache files"
	@echo "  make wipe-db   - Wipe Neo4j database"
	@echo "  make test      - Run tests"
	@echo "  make lint      - Run linter"

install:
	$(PIP) install -r requirements.txt

run:
	$(PYTHON) src/main.py

clean:
	find . -type d -name "__pycache__" -exec rm -r {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.pyd" -delete
	find . -type f -name ".DS_Store" -delete
	find . -type d -name "*.egg-info" -exec rm -r {} +
	find . -type f -name "*.egg" -delete

wipe-db:
	$(PYTHON) wipe_db.py

test:
	pytest tests/

lint:
	flake8 src/
	black src/ --check

# Default target
.DEFAULT_GOAL := help
