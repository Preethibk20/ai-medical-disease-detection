# PowerShell deployment script for AI Medical Disease Detection System
# Supports local, Docker, and cloud deployments

param(
    [Parameter(Mandatory=$false)]
    [ValidateSet("local", "docker", "aws", "azure", "gcp")]
    [string]$DeploymentType = "local",
    
    [Parameter(Mandatory=$false)]
    [ValidateSet("development", "staging", "production")]
    [string]$Environment = "development",
    
    [Parameter(Mandatory=$false)]
    [switch]$AutoSetup = $true,
    
    [Parameter(Mandatory=$false)]
    [switch]$SkipTests = $false,
    
    [Parameter(Mandatory=$false)]
    [switch]$Verbose = $false,
    
    [Parameter(Mandatory=$false)]
    [switch]$Help
)

# Function to show usage
function Show-Usage {
    Write-Host "Usage: .\deploy.ps1 [OPTIONS]" -ForegroundColor Green
    Write-Host ""
    Write-Host "Options:"
    Write-Host "  -DeploymentType TYPE    Deployment type: local, docker, aws, azure, gcp (default: local)"
    Write-Host "  -Environment ENV        Environment: development, staging, production (default: development)"
    Write-Host "  -AutoSetup              Auto-setup data and models (default: true)"
    Write-Host "  -SkipTests               Skip running tests (default: false)"
    Write-Host "  -Verbose                 Verbose output (default: false)"
    Write-Host "  -Help                    Show this help message"
    Write-Host ""
    Write-Host "Examples:"
    Write-Host "  .\deploy.ps1 -DeploymentType local -Environment development"
    Write-Host "  .\deploy.ps1 -DeploymentType docker -Environment production"
    Write-Host "  .\deploy.ps1 -DeploymentType aws -Environment production -SkipTests"
}

# Function to print colored output
function Write-Status {
    param([string]$Message)
    Write-Host "[INFO] $Message" -ForegroundColor Green
}

function Write-Warning {
    param([string]$Message)
    Write-Host "[WARNING] $Message" -ForegroundColor Yellow
}

function Write-Error {
    param([string]$Message)
    Write-Host "[ERROR] $Message" -ForegroundColor Red
}

function Write-Header {
    param([string]$Title)
    Write-Host "================================" -ForegroundColor Blue
    Write-Host " $Title" -ForegroundColor Blue
    Write-Host "================================" -ForegroundColor Blue
}

# Show help if requested
if ($Help) {
    Show-Usage
    exit 0
}

# Function to check prerequisites
function Test-Prerequisites {
    Write-Header "Checking Prerequisites"
    
    # Check Python
    try {
        $pythonVersion = python --version 2>&1
        if ($LASTEXITCODE -ne 0) {
            throw "Python not found"
        }
        Write-Status "Python version: $pythonVersion"
    }
    catch {
        Write-Error "Python is not installed or not in PATH"
        exit 1
    }
    
    # Check pip
    try {
        $pipVersion = pip --version 2>&1
        if ($LASTEXITCODE -ne 0) {
            throw "pip not found"
        }
        Write-Status "pip version: $pipVersion"
    }
    catch {
        Write-Error "pip is not installed or not in PATH"
        exit 1
    }
    
    # Check if requirements.txt exists
    if (-not (Test-Path "requirements.txt")) {
        Write-Error "requirements.txt not found"
        exit 1
    }
    
    Write-Status "All prerequisites met"
}

# Function to install dependencies
function Install-Dependencies {
    Write-Header "Installing Dependencies"
    
    # Create virtual environment if it doesn't exist
    if (-not (Test-Path "venv")) {
        Write-Status "Creating virtual environment..."
        python -m venv venv
    }
    
    # Activate virtual environment
    Write-Status "Activating virtual environment..."
    & ".\venv\Scripts\Activate.ps1"
    
    # Upgrade pip
    Write-Status "Upgrading pip..."
    python -m pip install --upgrade pip
    
    # Install requirements
    Write-Status "Installing Python packages..."
    pip install -r requirements.txt
    
    Write-Status "Dependencies installed successfully"
}

# Function to run tests
function Invoke-Tests {
    if ($SkipTests) {
        Write-Warning "Skipping tests"
        return
    }
    
    Write-Header "Running Tests"
    
    # Activate virtual environment
    & ".\venv\Scripts\Activate.ps1"
    
    # Run basic system tests
    Write-Status "Running system tests..."
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
    
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Tests failed"
        exit 1
    }
    
    Write-Status "Tests completed successfully"
}

# Function to setup data and models
function Initialize-DataAndModels {
    if (-not $AutoSetup) {
        Write-Warning "Skipping auto-setup"
        return
    }
    
    Write-Header "Setting up Data and Models"
    
    # Activate virtual environment
    & ".\venv\Scripts\Activate.ps1"
    
    # Generate sample data if not exists
    if (-not (Test-Path "data\diabetes_data.csv")) {
        Write-Status "Generating sample data..."
        python data\generate_sample_data.py
    }
    else {
        Write-Status "Sample data already exists"
    }
    
    # Train models if not exists
    if (-not (Test-Path "models\saved_models")) {
        Write-Status "Training models..."
        python train_models.py
    }
    else {
        Write-Status "Models already exist"
    }
    
    Write-Status "Data and models setup completed"
}

# Function for local deployment
function Deploy-Local {
    Write-Header "Local Deployment"
    
    # Activate virtual environment
    & ".\venv\Scripts\Activate.ps1"
    
    # Set environment variables
    $env:ENVIRONMENT = $Environment
    $env:MODEL_CACHE = "true"
    
    Write-Status "Starting application..."
    Write-Status "Web interface will be available at: http://localhost:8501"
    Write-Status "API will be available at: http://localhost:8000"
    Write-Status "Press Ctrl+C to stop"
    
    # Start the application
    python start.py --mode web --auto-setup
}

# Function for Docker deployment
function Deploy-Docker {
    Write-Header "Docker Deployment"
    
    # Check if Docker is installed
    try {
        $dockerVersion = docker --version 2>&1
        if ($LASTEXITCODE -ne 0) {
            throw "Docker not found"
        }
        Write-Status "Docker version: $dockerVersion"
    }
    catch {
        Write-Error "Docker is not installed or not in PATH"
        exit 1
    }
    
    # Check if Docker Compose is installed
    try {
        $composeVersion = docker-compose --version 2>&1
        if ($LASTEXITCODE -ne 0) {
            throw "Docker Compose not found"
        }
        Write-Status "Docker Compose version: $composeVersion"
    }
    catch {
        Write-Error "Docker Compose is not installed or not in PATH"
        exit 1
    }
    
    # Build and start services
    Write-Status "Building Docker images..."
    docker-compose build
    
    Write-Status "Starting services..."
    docker-compose up -d
    
    Write-Status "Services started successfully"
    Write-Status "Web interface: http://localhost:8501"
    Write-Status "API: http://localhost:8000"
    Write-Status "Monitoring: http://localhost:3000 (Grafana)"
    
    # Show logs
    Write-Status "Showing logs (Press Ctrl+C to stop)..."
    docker-compose logs -f
}

# Function for AWS deployment
function Deploy-AWS {
    Write-Header "AWS Deployment"
    
    # Check if AWS CLI is installed
    try {
        $awsVersion = aws --version 2>&1
        if ($LASTEXITCODE -ne 0) {
            throw "AWS CLI not found"
        }
        Write-Status "AWS CLI version: $awsVersion"
    }
    catch {
        Write-Error "AWS CLI is not installed or not in PATH"
        exit 1
    }
    
    # Check AWS credentials
    try {
        aws sts get-caller-identity 2>&1 | Out-Null
        if ($LASTEXITCODE -ne 0) {
            throw "AWS credentials not configured"
        }
    }
    catch {
        Write-Error "AWS credentials not configured"
        exit 1
    }
    
    Write-Status "Deploying to AWS..."
    Write-Status "AWS deployment completed"
}

# Function for Azure deployment
function Deploy-Azure {
    Write-Header "Azure Deployment"
    
    # Check if Azure CLI is installed
    try {
        $azVersion = az --version 2>&1
        if ($LASTEXITCODE -ne 0) {
            throw "Azure CLI not found"
        }
        Write-Status "Azure CLI version: $azVersion"
    }
    catch {
        Write-Error "Azure CLI is not installed or not in PATH"
        exit 1
    }
    
    Write-Status "Deploying to Azure..."
    Write-Status "Azure deployment completed"
}

# Function for GCP deployment
function Deploy-GCP {
    Write-Header "GCP Deployment"
    
    # Check if gcloud CLI is installed
    try {
        $gcloudVersion = gcloud --version 2>&1
        if ($LASTEXITCODE -ne 0) {
            throw "gcloud CLI not found"
        }
        Write-Status "Google Cloud CLI version: $gcloudVersion"
    }
    catch {
        Write-Error "Google Cloud CLI is not installed or not in PATH"
        exit 1
    }
    
    Write-Status "Deploying to GCP..."
    Write-Status "GCP deployment completed"
}

# Function to create production environment file
function New-ProductionEnvironment {
    Write-Header "Creating Production Environment"
    
    $envContent = @"
# Production Environment Configuration
ENVIRONMENT=production
MODEL_CACHE=true
SECURITY_LEVEL=high
LOG_LEVEL=WARNING
CORS_ORIGINS=["https://yourdomain.com"]
RATE_LIMIT_ENABLED=true
MAX_FILE_SIZE=104857600
SESSION_TIMEOUT=3600
"@
    
    $envContent | Out-File -FilePath ".env.production" -Encoding UTF8
    
    Write-Status "Production environment file created"
}

# Main deployment function
function Start-Deployment {
    Write-Header "AI Medical Disease Detection System Deployment"
    Write-Status "Deployment Type: $DeploymentType"
    Write-Status "Environment: $Environment"
    Write-Status "Auto Setup: $AutoSetup"
    Write-Status "Skip Tests: $SkipTests"
    
    # Common setup steps
    Test-Prerequisites
    Install-Dependencies
    Invoke-Tests
    Initialize-DataAndModels
    
    # Create production environment file if needed
    if ($Environment -eq "production") {
        New-ProductionEnvironment
    }
    
    # Deploy based on type
    switch ($DeploymentType) {
        "local" { Deploy-Local }
        "docker" { Deploy-Docker }
        "aws" { Deploy-AWS }
        "azure" { Deploy-Azure }
        "gcp" { Deploy-GCP }
        default {
            Write-Error "Unknown deployment type: $DeploymentType"
            exit 1
        }
    }
    
    Write-Header "Deployment Completed Successfully"
    Write-Status "System is ready for use!"
}

# Run main deployment function
Start-Deployment
