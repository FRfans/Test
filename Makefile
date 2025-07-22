install:
	pip install --upgrade pip &&\
	pip install -r requirements.txt

format:
	black *.py App/*.py

train:
	python train.py

eval:
	echo "## Model Metrics" > report.md
	cat ./Results/metrics.txt >> report.md

	echo '\n## Data Exploration Plot' >> report.md
	echo '![Data Exploration](./Results/data_exploration.png)' >> report.md

	echo '\n## Model Evaluation Plot' >> report.md
	echo '![Model Evaluation](./Results/model_evaluation.png)' >> report.md

	@echo "Report generated: report.md"

update-branch:
	git config user.name "$(USER_NAME)"
	git config user.email "$(USER_EMAIL)"
	git add Results/ Model/ report.md
	git commit -m "Update: training and evaluation results" || echo "Nothing to commit"
	git push --force origin HEAD:update || echo "Nothing to push"

deploy:
	@echo "Deploying to Hugging Face Spaces..."
	@echo "Checking HF token..."
	@if [ -z "$(HF_TOKEN)" ]; then \
		echo "❌ Error: HF token is empty!"; \
		echo "Please check that HF_TOKEN secret is set in GitHub repository"; \
		exit 1; \
	else \
		echo "✅ HF token is present"; \
	fi
	pip install huggingface_hub[cli]
	huggingface-cli login --token "$(HF_TOKEN)"
	@echo "🚀 Uploading main app file..."
	huggingface-cli upload firmnnm/Tugas1MLOps ./App/app.py app.py --repo-type=space --commit-message="Deploy personality classifier app"
	@echo "📁 Uploading model files..."
	huggingface-cli upload firmnnm/Tugas1MLOps ./Model/personality_classifier.skops Model/personality_classifier.skops --repo-type=space --commit-message="Upload personality classifier model"
	huggingface-cli upload firmnnm/Tugas1MLOps ./Model/label_encoder.skops Model/label_encoder.skops --repo-type=space --commit-message="Upload label encoder"
	huggingface-cli upload firmnnm/Tugas1MLOps ./Model/feature_names.skops Model/feature_names.skops --repo-type=space --commit-message="Upload feature names"
	@echo "📋 Uploading requirements..."
	huggingface-cli upload firmnnm/Tugas1MLOps ./requirements.txt requirements.txt --repo-type=space --commit-message="Upload requirements"
	@echo "📄 Uploading README..."
	huggingface-cli upload firmnnm/Tugas1MLOps ./README.md README.md --repo-type=space --commit-message="Upload README for Spaces"
	@echo "✅ Deployment to Hugging Face Spaces completed!"

run:
	@echo "🚀 Starting Personality Classifier app..."
	python App/app.py

# ========================
# Docker Commands
# ========================

docker-build:
	@echo "🐳 Building Docker image..."
	docker build -t personality-classifier:latest .

docker-run:
	@echo "🚀 Running Docker container..."
	docker run -p 7860:7860 --name personality-app personality-classifier:latest

docker-run-detached:
	@echo "🚀 Running Docker container in background..."
	docker run -d -p 7860:7860 --name personality-app personality-classifier:latest

docker-stop:
	@echo "⏹️ Stopping Docker container..."
	docker stop personality-app || echo "Container not running"
	docker rm personality-app || echo "Container not found"

docker-compose-up:
	@echo "🐳 Starting services with Docker Compose..."
	docker-compose up -d

docker-compose-down:
	@echo "⏹️ Stopping Docker Compose services..."
	docker-compose down

docker-train:
	@echo "🎯 Running training in Docker..."
	docker-compose --profile training up mlops-training

docker-clean:
	@echo "🧹 Cleaning Docker resources..."
	docker system prune -f

# ========================
# MLOps Commands
# ========================

drift-detect:
	@echo "🔍 Running data drift detection..."
	python data_drift.py

monitoring:
	@echo "📊 Starting MLOps monitoring..."
	python monitoring.py

mlflow-server:
	@echo "🚀 Starting MLflow tracking server..."
	mlflow server --host 0.0.0.0 --port 5000

full-pipeline:
	@echo "🚀 Running complete MLOps pipeline..."
	python train.py
	python data_drift.py
	python monitoring.py

test-mlops:
	@echo "🧪 Running MLOps test suite..."
	python test_mlops.py

# ========================
# Monitoring Infrastructure
# ========================

start-monitoring-stack:
	@echo "🚀 Starting complete monitoring stack..."
	docker-compose up -d mlflow prometheus grafana

stop-monitoring-stack:
	@echo "⏹️ Stopping monitoring stack..."
	docker-compose down

monitoring-logs:
	@echo "📋 Showing monitoring logs..."
	docker-compose logs -f mlflow prometheus grafana

# ========================
# MLOps Simplified Makefile
# ========================

# Basic setup
install:
	pip install --upgrade pip &&\
	pip install -r requirements.txt

format:
	black *.py App/*.py

format-check:
	black --check --diff *.py App/*.py

# Core MLOps workflows
train:
	python train.py

eval:
	echo "## Model Metrics" > report.md
	cat ./Results/metrics.txt >> report.md
	echo '\n## Data Exploration Plot' >> report.md
	echo '![Data Exploration](./Results/data_exploration.png)' >> report.md
	echo '\n## Model Evaluation Plot' >> report.md
	echo '![Model Evaluation](./Results/model_evaluation.png)' >> report.md
	@echo "Report generated: report.md"

# Deployment
deploy:
	@echo "Deploying to Hugging Face Spaces..."
	@if [ -z "$(HF_TOKEN)" ]; then \
		echo "❌ Error: HF token is empty!"; \
		exit 1; \
	else \
		echo "✅ HF token is present"; \
	fi
	pip install huggingface_hub[cli]
	huggingface-cli login --token "$(HF_TOKEN)"
	huggingface-cli upload firmnnm/Tugas1MLOps ./App/app.py app.py --repo-type=space --commit-message="Deploy personality classifier app"
	huggingface-cli upload firmnnm/Tugas1MLOps ./Model/personality_classifier.skops Model/personality_classifier.skops --repo-type=space --commit-message="Upload personality classifier model"
	huggingface-cli upload firmnnm/Tugas1MLOps ./Model/label_encoder.skops Model/label_encoder.skops --repo-type=space --commit-message="Upload label encoder"
	huggingface-cli upload firmnnm/Tugas1MLOps ./Model/feature_names.skops Model/feature_names.skops --repo-type=space --commit-message="Upload feature names"
	huggingface-cli upload firmnnm/Tugas1MLOps ./requirements.txt requirements.txt --repo-type=space --commit-message="Upload requirements"
	huggingface-cli upload firmnnm/Tugas1MLOps ./README.md README.md --repo-type=space --commit-message="Upload README for Spaces"
	@echo "✅ Deployment to Hugging Face Spaces completed!"

# CI/CD Support Commands
ci-setup:
	pip install --upgrade pip
	pip install -r requirements.txt
	pip install pytest black flake8 pylint bandit safety

ci-full:
	make format-check
	make lint
	make security-scan
	make train
	make validate-artifacts
	make health-check

lint:
	flake8 *.py App/*.py --count --select=E9,F63,F7,F82 --show-source --statistics
	pylint *.py App/*.py --exit-zero --score=yes

security-scan:
	bandit -r *.py App/*.py --format json --output bandit-report.json || true
	safety check --json --output safety-report.json || true
	bandit -r *.py App/*.py
	safety check

validate-artifacts:
	python -c "\
	import os; import skops.io as sio; \
	files = ['Model/personality_classifier.skops', 'Model/label_encoder.skops', 'Model/feature_names.skops']; \
	[print(f'✅ {f} validated') if os.path.exists(f) and sio.load(f, trusted=sio.get_untrusted_types(f)) else exit(1) for f in files]"

# Data and Drift Management
generate-synthetic:
	python -c "\
	from data_synthetic_generator import adaptiveDriftGenerator; import os; \
	generator = adaptiveDriftGenerator('Data/personality_datasert.csv') if os.path.exists('Data/personality_datasert.csv') else exit(1); \
	data = generator.generate_drift_data(n_samples=1000); \
	data.to_csv('Data/synthetic_ctgan_data.csv', index=False); \
	print(f'✅ Generated: {data.shape}')"

drift-detect:
	python -c "\
	from data_drift import DataDriftDetector; import os; \
	detector = DataDriftDetector(); \
	results = detector.detect_drift('Data/synthetic_ctgan_data.csv') if os.path.exists('Data/synthetic_ctgan_data.csv') else exit(1); \
	[print(f'{k}: {v}') for k,v in results.items()]"

# Application and Monitoring
run:
	python App/app.py

monitoring:
	python monitoring.py

retrain-api:
	python -m uvicorn retrain_api:app --host 0.0.0.0 --port 8001

# Docker Operations
docker-build:
	docker build -t mlops-app:latest .

docker-run:
	docker run -p 7860:7860 --name mlops-app mlops-app:latest

docker-compose-up:
	docker-compose up -d

docker-compose-down:
	docker-compose down

start-monitoring-stack:
	docker-compose up -d mlflow prometheus grafana

# Performance and Health
benchmark:
	python -c "\
	import time, numpy as np, skops.io as sio, os; \
	model = sio.load('Model/personality_classifier.skops', trusted=sio.get_untrusted_types('Model/personality_classifier.skops')) if os.path.exists('Model/personality_classifier.skops') else exit(1); \
	[print(f'Size {s}: {s/(time.time()-(t:=time.time())or(model.predict(np.random.rand(s,5)))and time.time()):.2f} pred/sec') for s in [1,10,100,1000] for t in [time.time()]]"

health-check:
	python -c "\
	import os, pandas as pd; \
	print('🔍 Health Check:'); \
	print('✅ Data exists' if os.path.exists('Data/personality_datasert.csv') else '❌ Data missing'); \
	[print(f'✅ {f} exists' if os.path.exists(f) else f'❌ {f} missing') for f in ['Model/personality_classifier.skops', 'Results/metrics.txt']]"

# Git operations
update-branch:
	git config user.name "$(USER_NAME)"
	git config user.email "$(USER_EMAIL)"
	git add Results/ Model/ report.md
	git commit -m "Update: training and evaluation results" || echo "Nothing to commit"
	git push --force origin HEAD:update || echo "Nothing to push"

# Cleanup
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	rm -f *.log *_report.json *_report.txt

# Help
help:
	@echo "📚 MLOps Makefile Commands:"
	@echo ""
	@echo "🏗️ Setup & Dependencies:"
	@echo "  install              - Install Python dependencies"
	@echo "  ci-setup             - Setup CI environment"
	@echo ""
	@echo "🎯 ML Training & Evaluation:"
	@echo "  train                - Train the model"
	@echo "  eval                 - Evaluate model and generate report"
	@echo "  validate-artifacts   - Validate model artifacts"
	@echo ""
	@echo "🧪 Code Quality & Testing:"
	@echo "  format               - Format code with Black"
	@echo "  format-check         - Check code formatting"
	@echo "  lint                 - Run code linting"
	@echo "  security-scan        - Run security scans"
	@echo "  ci-full              - Run complete CI pipeline locally"
	@echo ""
	@echo "🔄 Data & Drift:"
	@echo "  generate-synthetic   - Generate synthetic data"
	@echo "  drift-detect         - Run data drift detection"
	@echo ""
	@echo "🚀 Application & Deployment:"
	@echo "  run                  - Run app locally"
	@echo "  deploy               - Deploy to Hugging Face"
	@echo "  monitoring           - Start MLOps monitoring"
	@echo "  retrain-api          - Start retrain API"
	@echo ""
	@echo "🐳 Docker:"
	@echo "  docker-build         - Build Docker image"
	@echo "  docker-run           - Run Docker container"
	@echo "  docker-compose-up    - Start all services"
	@echo "  start-monitoring-stack - Start monitoring infrastructure"
	@echo ""
	@echo "📊 Performance & Health:"
	@echo "  benchmark            - Run performance benchmark"
	@echo "  health-check         - Run system health check"
	@echo ""
	@echo "🧹 Maintenance:"
	@echo "  clean                - Clean temporary files"
	@echo "  update-branch        - Update git branch with results"

.PHONY: install format train eval deploy run monitoring ci-setup ci-full lint security-scan \
		validate-artifacts generate-synthetic drift-detect benchmark health-check \
		docker-build docker-compose-up clean help
