.PHONY: install test train app docker-build docker-run clean help

PYTHON := python
PIP := pip
APP_FILE := app/streamlit_app.py
IMAGE_NAME := bank-marketing-ml
PORT := 8501

help:  ## Show available targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install:  ## Install production dependencies (for running the app)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

install-dev:  ## Install all dependencies including dev tools (for training)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements-dev.txt

test:  ## Run all pytest tests
	$(PYTHON) -m pytest tests/ -v --tb=short

train:  ## Run the full training pipeline
	$(PYTHON) scripts/train.py

app:  ## Launch the Streamlit application locally
	streamlit run $(APP_FILE) --server.port $(PORT)

docker-build:  ## Build the Docker image
	docker build -t $(IMAGE_NAME) .

docker-run:  ## Run the Docker container
	docker run -p $(PORT):$(PORT) $(IMAGE_NAME)

generate-reports:  ## Generate EDA figures and markdown reports
	$(PYTHON) scripts/generate_reports.py

batch-predict:  ## Run batch prediction (usage: make batch-predict INPUT=path/to/input.csv OUTPUT=path/to/output.csv)
	$(PYTHON) scripts/predict_batch.py --input $(INPUT) --output $(OUTPUT)

clean:  ## Remove generated artefacts
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache htmlcov .coverage
