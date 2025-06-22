#!/bin/bash

# ML Pipeline Setup Script
# This script sets up the entire ML pipeline project

set -e  # Exit on any error

echo "🚀 Setting up ML Pipeline Project..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Python is installed
check_python() {
    print_status "Checking Python installation..."
    if command -v python3 &> /dev/null; then
        PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
        print_success "Python $PYTHON_VERSION found"
    else
        print_error "Python 3 is not installed. Please install Python 3.8+ first."
        exit 1
    fi
}

# Check if pip is installed
check_pip() {
    print_status "Checking pip installation..."
    if command -v pip3 &> /dev/null; then
        print_success "pip3 found"
    elif command -v pip &> /dev/null; then
        print_success "pip found"
    else
        print_error "pip is not installed. Please install pip first."
        exit 1
    fi
}

# Create virtual environment
create_venv() {
    print_status "Creating virtual environment..."
    if [ ! -d "venv" ]; then
        python3 -m venv venv
        print_success "Virtual environment created"
    else
        print_warning "Virtual environment already exists"
    fi
}

# Activate virtual environment
activate_venv() {
    print_status "Activating virtual environment..."
    source venv/bin/activate
    print_success "Virtual environment activated"
}

# Install dependencies
install_dependencies() {
    print_status "Installing Python dependencies..."
    pip install --upgrade pip
    pip install -r requirements.txt
    print_success "Dependencies installed"
}

# Create necessary directories
create_directories() {
    print_status "Creating project directories..."
    mkdir -p models
    mkdir -p logs
    mkdir -p data
    mkdir -p .github/workflows
    print_success "Directories created"
}

# Set up git repository (if not already initialized)
setup_git() {
    if [ ! -d ".git" ]; then
        print_status "Initializing git repository..."
        git init
        git add .
        git commit -m "Initial commit: ML Pipeline setup"
        print_success "Git repository initialized"
    else
        print_warning "Git repository already exists"
    fi
}

# Train initial model
train_model() {
    print_status "Training initial model..."
    python train.py
    print_success "Model trained successfully"
}

# Run tests
run_tests() {
    print_status "Running tests..."
    python -m pytest test_pipeline.py -v
    python -m pytest test_api.py -v
    print_success "All tests passed"
}

# Check Docker installation
check_docker() {
    print_status "Checking Docker installation..."
    if command -v docker &> /dev/null; then
        print_success "Docker found"
        return 0
    else
        print_warning "Docker not found. Docker setup will be skipped."
        return 1
    fi
}

# Build Docker image
build_docker() {
    if check_docker; then
        print_status "Building Docker image..."
        docker build -t ml-pipeline:latest .
        print_success "Docker image built successfully"
    fi
}

# Start services with Docker Compose
start_services() {
    if command -v docker-compose &> /dev/null; then
        print_status "Starting services with Docker Compose..."
        docker-compose up -d
        print_success "Services started. API available at http://localhost:8000"
    else
        print_warning "docker-compose not found. Starting API manually..."
        nohup uvicorn main:app --host 0.0.0.0 --port 8000 > logs/api.log 2>&1 &
        sleep 5
        print_success "API started manually. Check logs/api.log for details"
    fi
}

# Test API endpoints
test_api() {
    print_status "Testing API endpoints..."
    sleep 5  # Wait for API to start
    
    # Test health endpoint
    if curl -f -s http://localhost:8000/health > /dev/null; then
        print_success "Health endpoint is working"
    else
        print_error "Health endpoint failed"
        return 1
    fi
    
    # Test prediction endpoint
    RESPONSE=$(curl -s -X POST "http://localhost:8000/predict" \
        -H "Content-Type: application/json" \
        -d '{"features": [0.1, -0.5, 1.2, 0.8, -1.1, 0.3, -0.7, 1.5, 0.2, -0.9]}')
    
    if echo "$RESPONSE" | grep -q "prediction"; then
        print_success "Prediction endpoint is working"
    else
        print_error "Prediction endpoint failed"
        return 1
    fi
}

# Main setup function
main() {
    echo "================================================================"
    echo "🤖 ML Pipeline End-to-End Setup"
    echo "================================================================"
    
    # Basic checks
    check_python
    check_pip
    
    # Setup virtual environment
    create_venv
    activate_venv
    
    # Install dependencies and setup
    install_dependencies
    create_directories
    
    # Train model and run tests
    train_model
    run_tests
    
    # Docker setup (optional)
    if [ "$1" = "--docker" ]; then
        build_docker
        start_services
        test_api
    else
        print_status "Starting API server..."
        nohup uvicorn main:app --host 0.0.0.0 --port 8000 > logs/api.log 2>&1 &
        sleep 5
        test_api
    fi
    
    # Git setup
    setup_git
    
    echo ""
    echo "================================================================"
    print_success "🎉 ML Pipeline setup completed successfully!"
    echo "================================================================"
    echo ""
    echo "📚 Next steps:"
    echo "  • API Documentation: http://localhost:8000/docs"
    echo "  • Health Check: http://localhost:8000/health"
    echo "  • View logs: tail -f logs/api.log"
    echo "  • Run tests: python -m pytest -v"
    echo "  • Stop API: pkill -f uvicorn (or docker-compose down)"
    echo ""
    echo "🐳 Docker commands:"
    echo "  • Build: docker build -t ml-pipeline ."
    echo "  • Run: docker-compose up -d"
    echo "  • Stop: docker-compose down"
    echo ""
    echo "🔄 CI/CD:"
    echo "  • Push to GitHub to trigger automated pipeline"
    echo "  • GitHub Actions will run tests and deploy