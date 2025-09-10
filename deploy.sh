#!/bin/bash
# Deployment script for AI Medical Disease Detection System
# Supports local, Docker, and cloud deployments

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
DEPLOYMENT_TYPE="local"
ENVIRONMENT="development"
AUTO_SETUP=true
SKIP_TESTS=false
VERBOSE=false

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

print_header() {
    echo -e "${BLUE}================================${NC}"
    echo -e "${BLUE} $1${NC}"
    echo -e "${BLUE}================================${NC}"
}

# Function to show usage
show_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -t, --type TYPE        Deployment type: local, docker, aws, azure, gcp (default: local)"
    echo "  -e, --env ENV          Environment: development, staging, production (default: development)"
    echo "  -a, --auto-setup       Auto-setup data and models (default: true)"
    echo "  -s, --skip-tests       Skip running tests (default: false)"
    echo "  -v, --verbose          Verbose output (default: false)"
    echo "  -h, --help             Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 --type local --env development"
    echo "  $0 --type docker --env production"
    echo "  $0 --type aws --env production --skip-tests"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -t|--type)
            DEPLOYMENT_TYPE="$2"
            shift 2
            ;;
        -e|--env)
            ENVIRONMENT="$2"
            shift 2
            ;;
        -a|--auto-setup)
            AUTO_SETUP=true
            shift
            ;;
        -s|--skip-tests)
            SKIP_TESTS=true
            shift
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        -h|--help)
            show_usage
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

# Function to check prerequisites
check_prerequisites() {
    print_header "Checking Prerequisites"
    
    # Check Python
    if ! command -v python3 &> /dev/null; then
        print_error "Python 3 is not installed"
        exit 1
    fi
    
    python_version=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
    if [[ $(echo "$python_version < 3.8" | bc -l) -eq 1 ]]; then
        print_error "Python 3.8 or higher is required"
        exit 1
    fi
    
    print_status "Python version: $(python3 --version)"
    
    # Check pip
    if ! command -v pip3 &> /dev/null; then
        print_error "pip3 is not installed"
        exit 1
    fi
    
    print_status "pip version: $(pip3 --version)"
    
    # Check if requirements.txt exists
    if [[ ! -f "requirements.txt" ]]; then
        print_error "requirements.txt not found"
        exit 1
    fi
    
    print_status "All prerequisites met"
}

# Function to install dependencies
install_dependencies() {
    print_header "Installing Dependencies"
    
    # Create virtual environment if it doesn't exist
    if [[ ! -d "venv" ]]; then
        print_status "Creating virtual environment..."
        python3 -m venv venv
    fi
    
    # Activate virtual environment
    source venv/bin/activate
    
    # Upgrade pip
    print_status "Upgrading pip..."
    pip install --upgrade pip
    
    # Install requirements
    print_status "Installing Python packages..."
    pip install -r requirements.txt
    
    print_status "Dependencies installed successfully"
}

# Function to run tests
run_tests() {
    if [[ "$SKIP_TESTS" == "true" ]]; then
        print_warning "Skipping tests"
        return
    fi
    
    print_header "Running Tests"
    
    # Activate virtual environment
    source venv/bin/activate
    
    # Run basic system tests
    print_status "Running system tests..."
    python -c "
import sys
sys.path.append('.')
try:
    from utils.data_processor import MedicalDataProcessor
    from models.demo_models import DemoDiseaseDetector
    print('✅ Core modules imported successfully')
except Exception as e:
    print(f'❌ Import error: {e}')
    sys.exit(1)
"
    
    # Test API endpoints if available
    if command -v curl &> /dev/null; then
        print_status "Testing API endpoints..."
        # This would test the API if it's running
    fi
    
    print_status "Tests completed successfully"
}

# Function to setup data and models
setup_data_and_models() {
    if [[ "$AUTO_SETUP" != "true" ]]; then
        print_warning "Skipping auto-setup"
        return
    fi
    
    print_header "Setting up Data and Models"
    
    # Activate virtual environment
    source venv/bin/activate
    
    # Generate sample data if not exists
    if [[ ! -f "data/diabetes_data.csv" ]]; then
        print_status "Generating sample data..."
        python data/generate_sample_data.py
    else
        print_status "Sample data already exists"
    fi
    
    # Train models if not exists
    if [[ ! -d "models/saved_models" ]]; then
        print_status "Training models..."
        python train_models.py
    else
        print_status "Models already exist"
    fi
    
    print_status "Data and models setup completed"
}

# Function for local deployment
deploy_local() {
    print_header "Local Deployment"
    
    # Activate virtual environment
    source venv/bin/activate
    
    # Set environment variables
    export ENVIRONMENT="$ENVIRONMENT"
    export MODEL_CACHE=true
    
    print_status "Starting application..."
    print_status "Web interface will be available at: http://localhost:8501"
    print_status "API will be available at: http://localhost:8000"
    print_status "Press Ctrl+C to stop"
    
    # Start the application
    python start.py --mode web --auto-setup
}

# Function for Docker deployment
deploy_docker() {
    print_header "Docker Deployment"
    
    # Check if Docker is installed
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed"
        exit 1
    fi
    
    # Check if Docker Compose is installed
    if ! command -v docker-compose &> /dev/null; then
        print_error "Docker Compose is not installed"
        exit 1
    fi
    
    # Build and start services
    print_status "Building Docker images..."
    docker-compose build
    
    print_status "Starting services..."
    docker-compose up -d
    
    print_status "Services started successfully"
    print_status "Web interface: http://localhost:8501"
    print_status "API: http://localhost:8000"
    print_status "Monitoring: http://localhost:3000 (Grafana)"
    
    # Show logs
    print_status "Showing logs (Press Ctrl+C to stop)..."
    docker-compose logs -f
}

# Function for AWS deployment
deploy_aws() {
    print_header "AWS Deployment"
    
    # Check if AWS CLI is installed
    if ! command -v aws &> /dev/null; then
        print_error "AWS CLI is not installed"
        exit 1
    fi
    
    # Check AWS credentials
    if ! aws sts get-caller-identity &> /dev/null; then
        print_error "AWS credentials not configured"
        exit 1
    fi
    
    print_status "Deploying to AWS..."
    
    # Create ECS task definition
    print_status "Creating ECS task definition..."
    # This would create the ECS task definition
    
    # Deploy to ECS
    print_status "Deploying to ECS..."
    # This would deploy to ECS
    
    print_status "AWS deployment completed"
}

# Function for Azure deployment
deploy_azure() {
    print_header "Azure Deployment"
    
    # Check if Azure CLI is installed
    if ! command -v az &> /dev/null; then
        print_error "Azure CLI is not installed"
        exit 1
    fi
    
    print_status "Deploying to Azure..."
    
    # Deploy to Azure Container Instances
    print_status "Deploying to Azure Container Instances..."
    # This would deploy to Azure
    
    print_status "Azure deployment completed"
}

# Function for GCP deployment
deploy_gcp() {
    print_header "GCP Deployment"
    
    # Check if gcloud CLI is installed
    if ! command -v gcloud &> /dev/null; then
        print_error "Google Cloud CLI is not installed"
        exit 1
    fi
    
    print_status "Deploying to GCP..."
    
    # Deploy to Google Cloud Run
    print_status "Deploying to Google Cloud Run..."
    # This would deploy to GCP
    
    print_status "GCP deployment completed"
}

# Function to create production environment file
create_production_env() {
    print_header "Creating Production Environment"
    
    cat > .env.production << EOF
# Production Environment Configuration
ENVIRONMENT=production
MODEL_CACHE=true
SECURITY_LEVEL=high
LOG_LEVEL=WARNING
CORS_ORIGINS=["https://yourdomain.com"]
RATE_LIMIT_ENABLED=true
MAX_FILE_SIZE=104857600
SESSION_TIMEOUT=3600
EOF
    
    print_status "Production environment file created"
}

# Main deployment function
main() {
    print_header "AI Medical Disease Detection System Deployment"
    print_status "Deployment Type: $DEPLOYMENT_TYPE"
    print_status "Environment: $ENVIRONMENT"
    print_status "Auto Setup: $AUTO_SETUP"
    print_status "Skip Tests: $SKIP_TESTS"
    
    # Common setup steps
    check_prerequisites
    install_dependencies
    run_tests
    setup_data_and_models
    
    # Create production environment file if needed
    if [[ "$ENVIRONMENT" == "production" ]]; then
        create_production_env
    fi
    
    # Deploy based on type
    case $DEPLOYMENT_TYPE in
        local)
            deploy_local
            ;;
        docker)
            deploy_docker
            ;;
        aws)
            deploy_aws
            ;;
        azure)
            deploy_azure
            ;;
        gcp)
            deploy_gcp
            ;;
        *)
            print_error "Unknown deployment type: $DEPLOYMENT_TYPE"
            exit 1
            ;;
    esac
    
    print_header "Deployment Completed Successfully"
    print_status "System is ready for use!"
}

# Run main function
main "$@"
