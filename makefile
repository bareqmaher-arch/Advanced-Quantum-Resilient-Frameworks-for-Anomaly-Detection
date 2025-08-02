# Q-ZAP Framework Makefile
# ========================
# Automation for development, testing, and deployment

# Project configuration
PROJECT_NAME := qzap
VERSION := $(shell python setup.py --version 2>/dev/null || echo "1.0.0")
DOCKER_IMAGE := $(PROJECT_NAME):$(VERSION)
DOCKER_REGISTRY := ghcr.io/bareqmaher-arch
PYTHON_VERSION := 3.11
VENV_NAME := venv

# Colors for output
RED := \033[0;31m
GREEN := \033[0;32m
YELLOW := \033[0;33m
BLUE := \033[0;34m
NC := \033[0m # No Color

# Default target
.DEFAULT_GOAL := help

# Helper function to print colored output
define print_color
	@echo "$(2)$(1)$(NC)"
endef

.PHONY: help
help: ## Show this help message
	@echo "$(GREEN)Q-ZAP Framework - Available Commands$(NC)"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "$(YELLOW)%-20s$(NC) %s\n", $$1, $$2}'
	@echo ""
	@echo "$(BLUE)Environment Variables:$(NC)"
	@echo "  PYTHON_VERSION=$(PYTHON_VERSION)"
	@echo "  DOCKER_IMAGE=$(DOCKER_IMAGE)"
	@echo "  DOCKER_REGISTRY=$(DOCKER_REGISTRY)"

# =============================================================================
# DEVELOPMENT ENVIRONMENT
# =============================================================================

.PHONY: install
install: ## Install the package in development mode
	$(call print_color,"Setting up development environment...",$(GREEN))
	python -m pip install --upgrade pip
	pip install -e ".[dev,all]"
	$(call print_color,"Development environment ready!",$(GREEN))

.PHONY: install-prod
install-prod: ## Install production dependencies only
	$(call print_color,"Installing production dependencies...",$(GREEN))
	python -m pip install --upgrade pip
	pip install -e "."
	$(call print_color,"Production installation complete!",$(GREEN))

.PHONY: venv
venv: ## Create virtual environment
	$(call print_color,"Creating virtual environment...",$(GREEN))
	python$(PYTHON_VERSION) -m venv $(VENV_NAME)
	$(VENV_NAME)/bin/pip install --upgrade pip
	@echo "$(GREEN)Virtual environment created. Activate with:$(NC)"
	@echo "  source $(VENV_NAME)/bin/activate"

.PHONY: requirements
requirements: ## Generate requirements.txt from setup.py
	$(call print_color,"Generating requirements.txt...",$(GREEN))
	pip-compile setup.py --output-file requirements.txt
	pip-compile setup.py --extra dev --output-file requirements-dev.txt

# =============================================================================
# CODE QUALITY AND TESTING
# =============================================================================

.PHONY: format
format: ## Format code with black and isort
	$(call print_color,"Formatting code...",$(GREEN))
	black src/ tests/ setup.py
	isort src/ tests/ setup.py
	$(call print_color,"Code formatting complete!",$(GREEN))

.PHONY: lint
lint: ## Run linting with flake8 and mypy
	$(call print_color,"Running linters...",$(YELLOW))
	flake8 src/ tests/
	mypy src/
	$(call print_color,"Linting complete!",$(GREEN))

.PHONY: security
security: ## Run security checks with bandit and safety
	$(call print_color,"Running security checks...",$(YELLOW))
	bandit -r src/ -f json -o security-report.json || true
	safety check --json --output security-deps.json || true
	$(call print_color,"Security checks complete!",$(GREEN))

.PHONY: test
test: ## Run all tests
	$(call print_color,"Running tests...",$(YELLOW))
	pytest tests/ -v --cov=src --cov-report=html --cov-report=xml
	$(call print_color,"Tests complete!",$(GREEN))

.PHONY: test-unit
test-unit: ## Run unit tests only
	$(call print_color,"Running unit tests...",$(YELLOW))
	pytest tests/unit/ -v

.PHONY: test-integration
test-integration: ## Run integration tests only
	$(call print_color,"Running integration tests...",$(YELLOW))
	pytest tests/integration/ -v

.PHONY: test-e2e
test-e2e: ## Run end-to-end tests
	$(call print_color,"Running end-to-end tests...",$(YELLOW))
	pytest tests/e2e/ -v

.PHONY: benchmark
benchmark: ## Run performance benchmarks
	$(call print_color,"Running benchmarks...",$(YELLOW))
	python -m pytest benchmarks/ -v --benchmark-only

.PHONY: check
check: format lint security test ## Run all quality checks
	$(call print_color,"All quality checks passed!",$(GREEN))

# =============================================================================
# DOCKER OPERATIONS
# =============================================================================

.PHONY: docker-build
docker-build: ## Build Docker image
	$(call print_color,"Building Docker image...",$(GREEN))
	docker build -t $(DOCKER_IMAGE) -f deployment/docker/Dockerfile .
	docker tag $(DOCKER_IMAGE) $(PROJECT_NAME):latest
	$(call print_color,"Docker image built: $(DOCKER_IMAGE)",$(GREEN))

.PHONY: docker-build-no-cache
docker-build-no-cache: ## Build Docker image without cache
	$(call print_color,"Building Docker image (no cache)...",$(GREEN))
	docker build --no-cache -t $(DOCKER_IMAGE) -f deployment/docker/Dockerfile .
	docker tag $(DOCKER_IMAGE) $(PROJECT_NAME):latest

.PHONY: docker-run
docker-run: ## Run Docker container
	$(call print_color,"Starting Docker container...",$(GREEN))
	docker run --rm -it \
		-p 8080:8080 \
		-p 8443:8443 \
		-p 9090:9090 \
		-v $(PWD)/data:/app/data \
		-v $(PWD)/config:/app/config \
		$(DOCKER_IMAGE)

.PHONY: docker-push
docker-push: ## Push Docker image to registry
	$(call print_color,"Pushing Docker image to registry...",$(GREEN))
	docker tag $(DOCKER_IMAGE) $(DOCKER_REGISTRY)/$(DOCKER_IMAGE)
	docker push $(DOCKER_REGISTRY)/$(DOCKER_IMAGE)
	docker push $(DOCKER_REGISTRY)/$(PROJECT_NAME):latest

.PHONY: docker-pull
docker-pull: ## Pull Docker image from registry
	$(call print_color,"Pulling Docker image from registry...",$(GREEN))
	docker pull $(DOCKER_REGISTRY)/$(DOCKER_IMAGE)
	docker tag $(DOCKER_REGISTRY)/$(DOCKER_IMAGE) $(DOCKER_IMAGE)

# =============================================================================
# DOCKER COMPOSE OPERATIONS
# =============================================================================

.PHONY: up
up: ## Start all services with docker-compose
	$(call print_color,"Starting Q-ZAP services...",$(GREEN))
	docker-compose up -d
	$(call print_color,"Services started! Access UI at http://localhost:8080",$(GREEN))

.PHONY: down
down: ## Stop all services
	$(call print_color,"Stopping Q-ZAP services...",$(YELLOW))
	docker-compose down

.PHONY: logs
logs: ## Show logs from all services
	docker-compose logs -f

.PHONY: logs-core
logs-core: ## Show logs from core service only
	docker-compose logs -f qzap-core

.PHONY: restart
restart: down up ## Restart all services

.PHONY: ps
ps: ## Show running services
	docker-compose ps

.PHONY: build-compose
build-compose: ## Build docker-compose services
	$(call print_color,"Building docker-compose services...",$(GREEN))
	docker-compose build

# =============================================================================
# DATA AND MODELS
# =============================================================================

.PHONY: download-data
download-data: ## Download CIC-IDS2017 dataset
	$(call print_color,"Downloading CIC-IDS2017 dataset...",$(GREEN))
	mkdir -p data/datasets
	python scripts/download_dataset.py --dataset cic-ids2017 --output data/datasets/

.PHONY: preprocess-data
preprocess-data: ## Preprocess datasets for training
	$(call print_color,"Preprocessing data...",$(GREEN))
	python scripts/preprocess_data.py --input data/datasets/ --output data/processed/

.PHONY: train-hae
train-hae: ## Train the Hybrid Autoencoder model
	$(call print_color,"Training HAE model...",$(GREEN))
	qzap-train --config config/training.yaml --output models/hae_model

.PHONY: train-distributed
train-distributed: ## Train with distributed/federated learning
	$(call print_color,"Starting federated learning...",$(GREEN))
	docker-compose --profile federated-learning up -d
	$(call print_color,"Federated learning coordinator started on port 8765",$(GREEN))

.PHONY: benchmark-models
benchmark-models: ## Benchmark model performance
	$(call print_color,"Benchmarking models...",$(GREEN))
	python experiments/benchmarks/anomaly_detection.py --models hae,classical_ae,isolation_forest

# =============================================================================
# DEPLOYMENT
# =============================================================================

.PHONY: deploy-local
deploy-local: docker-build up ## Deploy locally with Docker Compose
	$(call print_color,"Q-ZAP deployed locally!",$(GREEN))
	@echo "Access points:"
	@echo "  - Main UI: http://localhost:8080"
	@echo "  - Grafana: http://localhost:3000 (admin/admin_password)"
	@echo "  - Kibana: http://localhost:5601"
	@echo "  - Prometheus: http://localhost:9091"

.PHONY: deploy-k8s
deploy-k8s: ## Deploy to Kubernetes
	$(call print_color,"Deploying to Kubernetes...",$(GREEN))
	kubectl apply -f deployment/kubernetes/
	$(call print_color,"Kubernetes deployment complete!",$(GREEN))

.PHONY: deploy-aws
deploy-aws: ## Deploy to AWS EKS
	$(call print_color,"Deploying to AWS EKS...",$(GREEN))
	terraform -chdir=deployment/terraform init
	terraform -chdir=deployment/terraform plan
	terraform -chdir=deployment/terraform apply -auto-approve

.PHONY: undeploy-k8s
undeploy-k8s: ## Remove Kubernetes deployment
	$(call print_color,"Removing Kubernetes deployment...",$(YELLOW))
	kubectl delete -f deployment/kubernetes/

# =============================================================================
# MONITORING AND MAINTENANCE
# =============================================================================

.PHONY: monitor
monitor: ## Start monitoring services
	$(call print_color,"Starting monitoring stack...",$(GREEN))
	docker-compose --profile monitoring up -d prometheus grafana kibana
	$(call print_color,"Monitoring services started!",$(GREEN))

.PHONY: backup
backup: ## Backup data and models
	$(call print_color,"Creating backup...",$(GREEN))
	mkdir -p backups/$(shell date +%Y%m%d_%H%M%S)
	tar -czf backups/$(shell date +%Y%m%d_%H%M%S)/qzap_backup.tar.gz \
		data/ models/ config/ --exclude=data/temp

.PHONY: restore
restore: ## Restore from backup (specify BACKUP_FILE)
	$(call print_color,"Restoring from backup...",$(GREEN))
	@if [ -z "$(BACKUP_FILE)" ]; then \
		echo "$(RED)Please specify BACKUP_FILE=path/to/backup.tar.gz$(NC)"; \
		exit 1; \
	fi
	tar -xzf $(BACKUP_FILE)

.PHONY: health-check
health-check: ## Check health of all services
	$(call print_color,"Checking service health...",$(GREEN))
	@docker-compose ps | grep "Up" && echo "$(GREEN)All services healthy$(NC)" || echo "$(RED)Some services down$(NC)"

# =============================================================================
# DOCUMENTATION
# =============================================================================

.PHONY: docs
docs: ## Build documentation
	$(call print_color,"Building documentation...",$(GREEN))
	sphinx-build -b html docs/ docs/_build/html
	$(call print_color,"Documentation built in docs/_build/html/",$(GREEN))

.PHONY: docs-serve
docs-serve: docs ## Serve documentation locally
	$(call print_color,"Serving documentation at http://localhost:8000",$(GREEN))
	cd docs/_build/html && python -m http.server 8000

.PHONY: docs-api
docs-api: ## Generate API documentation
	$(call print_color,"Generating API documentation...",$(GREEN))
	sphinx-apidoc -o docs/api src/qzap --force

# =============================================================================
# CLEANUP
# =============================================================================

.PHONY: clean
clean: ## Clean up build artifacts
	$(call print_color,"Cleaning up...",$(YELLOW))
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	rm -rf build/ dist/ .coverage htmlcov/ .pytest_cache/ .mypy_cache/
	$(call print_color,"Cleanup complete!",$(GREEN))

.PHONY: clean-docker
clean-docker: ## Clean up Docker images and containers
	$(call print_color,"Cleaning Docker resources...",$(YELLOW))
	docker-compose down -v --remove-orphans
	docker system prune -f
	$(call print_color,"Docker cleanup complete!",$(GREEN))

.PHONY: clean-all
clean-all: clean clean-docker ## Clean everything
	$(call print_color,"Full cleanup complete!",$(GREEN))

# =============================================================================
# DEVELOPMENT HELPERS
# =============================================================================

.PHONY: shell
shell: ## Open interactive shell in container
	docker-compose exec qzap-core /bin/bash

.PHONY: jupyter
jupyter: ## Start Jupyter notebook server
	$(call print_color,"Starting Jupyter server...",$(GREEN))
	jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root
	$(call print_color,"Jupyter available at http://localhost:8888",$(GREEN))

.PHONY: generate-keys
generate-keys: ## Generate PQC keys
	$(call print_color,"Generating PQC keys...",$(GREEN))
	qzap crypto generate-keys --algorithms ML-KEM-768,ML-DSA-65 --output keys/

.PHONY: setup-dev
setup-dev: venv install generate-keys download-data ## Complete development setup
	$(call print_color,"Development environment setup complete!",$(GREEN))

# =============================================================================
# CI/CD HELPERS
# =============================================================================

.PHONY: ci-test
ci-test: ## Run CI tests
	$(call print_color,"Running CI tests...",$(GREEN))
	python -m pytest tests/ --junitxml=test-results.xml --cov=src --cov-report=xml

.PHONY: ci-build
ci-build: ## Build for CI
	$(call print_color,"Building for CI...",$(GREEN))
	docker build -t $(DOCKER_IMAGE) .

.PHONY: ci-security
ci-security: ## Security scan for CI
	$(call print_color,"Running security scan...",$(GREEN))
	bandit -r src/ -f json -o bandit-report.json
	safety check --json --output safety-report.json

# =============================================================================
# VERSION MANAGEMENT
# =============================================================================

.PHONY: version
version: ## Show current version
	@echo "$(GREEN)Current version: $(VERSION)$(NC)"

.PHONY: tag
tag: ## Create git tag for current version
	$(call print_color,"Creating tag v$(VERSION)...",$(GREEN))
	git tag -a v$(VERSION) -m "Release version $(VERSION)"
	git push origin v$(VERSION)

# =============================================================================
# EXAMPLES AND DEMOS
# =============================================================================

.PHONY: demo
demo: ## Run demonstration
	$(call print_color,"Running Q-ZAP demonstration...",$(GREEN))
	python examples/demo.py

.PHONY: example-training
example-training: ## Run training example
	$(call print_color,"Running training example...",$(GREEN))
	python examples/train_hae_example.py

.PHONY: example-detection
example-detection: ## Run anomaly detection example
	$(call print_color,"Running detection example...",$(GREEN))
	python examples/anomaly_detection_example.py

# Ensure some targets run in order
.PHONY: all
all: check docker-build deploy-local ## Run complete build and deploy pipeline