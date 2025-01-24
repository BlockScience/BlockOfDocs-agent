.PHONY: setup install test run clean clear format lint

setup: 
	uv venv && source .venv/bin/activate

install:
	uv pip install -r requirements.txt

test:
	pytest tests/

run:
	streamlit run src/app.py --autoreload

clean:
	rm -rf .venv/

clear:
	rm -rf data/processed/* data/vector_store/*

format:
	black src/ tests/

lint:
	ruff src/ tests/

update-deps:
	uv pip compile requirements.txt --upgrade

