#!/bin/bash

# MLOps Setup Script
# Sets up MLflow tracking server and DVC for the project

set -e

echo "🚀 Setting up MLOps environment..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Python is installed
if ! command -v python &> /dev/null; then
    print_error "Python is not installed. Please install Python 3.8 or higher."
    exit 1
fi

# Check Python version
python_version=$(python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
print_status "Using Python $python_version"

# Install requirements
print_status "Installing Python dependencies..."
pip install -r requirements.txt

# Initialize DVC if not already initialized
if [ ! -d ".dvc" ]; then
    print_status "Initializing DVC..."
    dvc init --no-scm
else
    print_status "DVC already initialized"
fi

# Set up DVC remote (use local storage for development)
print_status "Setting up DVC remote storage..."
dvc remote add -d local-storage ./dvc-storage -f
mkdir -p ./dvc-storage

# Initialize MLflow directory
print_status "Setting up MLflow..."
mkdir -p ./mlruns
mkdir -p ./mlflow-artifacts

# Create MLflow startup script
cat > start_mlflow.sh << 'EOF'
#!/bin/bash
echo "🔧 Starting MLflow tracking server..."
mlflow server \
    --backend-store-uri ./mlruns \
    --default-artifact-root ./mlflow-artifacts \
    --host 0.0.0.0 \
    --port 5000 \
    --serve-artifacts
EOF

chmod +x start_mlflow.sh

# Create MLflow startup script for Windows
cat > start_mlflow.bat << 'EOF'
@echo off
echo Starting MLflow tracking server...
mlflow server ^
    --backend-store-uri ./mlruns ^
    --default-artifact-root ./mlflow-artifacts ^
    --host 0.0.0.0 ^
    --port 5000 ^
    --serve-artifacts
EOF

# Create environment file
if [ ! -f ".env" ]; then
    print_status "Creating environment configuration..."
    cat > .env << 'EOF'
# MLflow Configuration
MLFLOW_TRACKING_URI=http://localhost:5000
MLFLOW_EXPERIMENT_NAME=personality-classification

# DVC Configuration
DVC_REMOTE_URL=./dvc-storage

# AWS Configuration (for production)
# AWS_ACCESS_KEY_ID=your-access-key
# AWS_SECRET_ACCESS_KEY=your-secret-key
# AWS_DEFAULT_REGION=us-east-1

# Model Configuration
MODEL_NAME=personality-classifier
MODEL_STAGE=staging
EOF
else
    print_warning "Environment file already exists"
fi

# Set up pre-commit hooks (optional)
if command -v pre-commit &> /dev/null; then
    print_status "Setting up pre-commit hooks..."
    cat > .pre-commit-config.yaml << 'EOF'
repos:
  - repo: https://github.com/psf/black
    rev: 23.12.1
    hooks:
      - id: black
        language_version: python3

  - repo: https://github.com/pycqa/flake8
    rev: 7.0.0
    hooks:
      - id: flake8
        args: [--max-line-length=88, --extend-ignore=E203,W503]

  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.5.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files
EOF
    pre-commit install
fi

# Create directories
print_status "Creating project directories..."
mkdir -p Data
mkdir -p Model
mkdir -p Results
mkdir -p scripts
mkdir -p notebooks
mkdir -p configs

# Add data to DVC tracking
if [ -f "Data/personality_datasert.csv" ]; then
    print_status "Adding original data to DVC tracking..."
    dvc add Data/personality_datasert.csv
fi

if [ -f "Data/synthetic_ctgan_data.csv" ]; then
    print_status "Adding synthetic data to DVC tracking..."
    dvc add Data/synthetic_ctgan_data.csv
fi

# Create helpful scripts
print_status "Creating utility scripts..."

# Train script
cat > train.sh << 'EOF'
#!/bin/bash
echo "🚀 Starting training pipeline..."
python train_with_mlflow.py
EOF
chmod +x train.sh

# Evaluation script
cat > evaluate.sh << 'EOF'
#!/bin/bash
echo "📊 Starting model evaluation..."
python evaluate_model.py
EOF
chmod +x evaluate.sh

# Data drift check script
cat > check_drift.sh << 'EOF'
#!/bin/bash
echo "🔍 Checking for data drift..."
python scripts/check_data_drift.py
EOF
chmod +x check_drift.sh

print_status "MLOps setup completed successfully! 🎉"
echo ""
print_status "Next steps:"
echo "  1. Start MLflow server: ./start_mlflow.sh (Linux/Mac) or start_mlflow.bat (Windows)"
echo "  2. Access MLflow UI at: http://localhost:5000"
echo "  3. Run training: ./train.sh or python train_with_mlflow.py"
echo "  4. Run evaluation: ./evaluate.sh or python evaluate_model.py"
echo "  5. Check data drift: ./check_drift.sh or python scripts/check_data_drift.py"
echo ""
print_status "GitHub Actions workflow is configured in .github/workflows/ml-pipeline.yml"
print_warning "Don't forget to set up your secrets in GitHub repository settings!"
