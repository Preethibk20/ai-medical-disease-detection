# AI Medical Disease Detection System - Deployment Guide

## Overview

This guide provides comprehensive instructions for deploying the AI Medical Disease Detection System across different environments and platforms. The system supports multiple deployment options including local development, Docker containers, and cloud platforms (AWS, Azure, GCP).

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Quick Start](#quick-start)
3. [Local Deployment](#local-deployment)
4. [Docker Deployment](#docker-deployment)
5. [Cloud Deployment](#cloud-deployment)
6. [Environment Configuration](#environment-configuration)
7. [Monitoring and Logging](#monitoring-and-logging)
8. [Security Considerations](#security-considerations)
9. [Troubleshooting](#troubleshooting)
10. [Maintenance](#maintenance)

## Prerequisites

### System Requirements

- **Operating System**: Windows 10/11, macOS 10.15+, or Linux (Ubuntu 20.04+)
- **Python**: 3.8 or higher
- **Memory**: Minimum 4GB RAM (8GB+ recommended for production)
- **Storage**: Minimum 10GB free space (50GB+ recommended for production)
- **CPU**: 2+ cores (4+ cores recommended for production)

### Software Dependencies

- **Python 3.8+** with pip
- **Docker** and **Docker Compose** (for containerized deployment)
- **Git** (for version control)
- **Cloud CLI tools** (for cloud deployment):
  - AWS CLI
  - Azure CLI
  - Google Cloud CLI

### Cloud Platform Accounts (Optional)

- **AWS Account** with appropriate permissions
- **Azure Subscription** with contributor access
- **Google Cloud Platform** project with billing enabled

## Quick Start

### Option 1: Automated Deployment Script

```bash
# Linux/macOS
chmod +x deploy.sh
./deploy.sh --type local --env development --auto-setup

# Windows PowerShell
.\deploy.ps1 -DeploymentType local -Environment development -AutoSetup
```

### Option 2: Manual Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Generate sample data
python data/generate_sample_data.py

# 3. Train models
python train_models.py

# 4. Start the application
python start.py --mode web --auto-setup
```

## Local Deployment

### Development Environment

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-repo/medical-ai-system.git
   cd medical-ai-system
   ```

2. **Create virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/macOS
   # or
   venv\Scripts\activate  # Windows
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Setup environment**:
   ```bash
   cp env.development .env
   ```

5. **Generate sample data**:
   ```bash
   python data/generate_sample_data.py
   ```

6. **Train models**:
   ```bash
   python train_models.py
   ```

7. **Start the application**:
   ```bash
   # Web interface
   python start.py --mode web
   
   # API server
   python start.py --mode api
   
   # Interactive demo
   python start.py --mode demo
   ```

### Production Environment

1. **Setup production environment**:
   ```bash
   cp env.production .env
   # Edit .env file with production values
   ```

2. **Install production dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Setup SSL certificates** (if using HTTPS):
   ```bash
   # Place SSL certificates in ssl/ directory
   mkdir ssl
   # Copy cert.pem and key.pem to ssl/
   ```

4. **Start with production settings**:
   ```bash
   python start.py --mode web --env production
   ```

## Docker Deployment

### Single Container Deployment

1. **Build the Docker image**:
   ```bash
   docker build -t medical-ai-system .
   ```

2. **Run the container**:
   ```bash
   docker run -d \
     --name medical-ai-app \
     -p 8000:8000 \
     -p 8501:8501 \
     -v $(pwd)/data:/app/data \
     -v $(pwd)/models:/app/models \
     -v $(pwd)/logs:/app/logs \
     medical-ai-system
   ```

### Multi-Container Deployment

1. **Start all services**:
   ```bash
   docker-compose up -d
   ```

2. **View logs**:
   ```bash
   docker-compose logs -f
   ```

3. **Stop services**:
   ```bash
   docker-compose down
   ```

### Docker Services

The `docker-compose.yml` includes the following services:

- **medical-ai-app**: Main application container
- **redis**: Caching service
- **nginx**: Reverse proxy and load balancer
- **prometheus**: Metrics collection
- **grafana**: Monitoring dashboard

### Access Points

- **Web Interface**: http://localhost:8501
- **API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **Monitoring**: http://localhost:3000 (Grafana)
- **Metrics**: http://localhost:9090 (Prometheus)

## Cloud Deployment

### AWS Deployment

#### Using AWS CloudFormation

1. **Deploy the stack**:
   ```bash
   aws cloudformation create-stack \
     --stack-name medical-ai-system \
     --template-body file://aws-cloudformation.yml \
     --parameters ParameterKey=Environment,ParameterValue=production \
                 ParameterKey=InstanceType,ParameterValue=t3.medium \
                 ParameterKey=KeyPairName,ParameterValue=your-key-pair
   ```

2. **Monitor deployment**:
   ```bash
   aws cloudformation describe-stacks --stack-name medical-ai-system
   ```

3. **Get outputs**:
   ```bash
   aws cloudformation describe-stacks \
     --stack-name medical-ai-system \
     --query 'Stacks[0].Outputs'
   ```

#### Using AWS ECS

1. **Build and push Docker image**:
   ```bash
   # Build image
   docker build -t medical-ai-system .
   
   # Tag for ECR
   docker tag medical-ai-system:latest your-account.dkr.ecr.region.amazonaws.com/medical-ai-system:latest
   
   # Push to ECR
   docker push your-account.dkr.ecr.region.amazonaws.com/medical-ai-system:latest
   ```

2. **Deploy to ECS**:
   ```bash
   aws ecs create-service \
     --cluster your-cluster \
     --service-name medical-ai-service \
     --task-definition medical-ai-task \
     --desired-count 2
   ```

### Azure Deployment

#### Using Azure Resource Manager

1. **Deploy the template**:
   ```bash
   az deployment group create \
     --resource-group your-resource-group \
     --template-file azure-template.json \
     --parameters environment=production vmSize=Standard_B2s
   ```

2. **Get deployment outputs**:
   ```bash
   az deployment group show \
     --resource-group your-resource-group \
     --name deployment-name \
     --query properties.outputs
   ```

#### Using Azure Container Instances

1. **Deploy container**:
   ```bash
   az container create \
     --resource-group your-resource-group \
     --name medical-ai-container \
     --image your-registry.azurecr.io/medical-ai-system:latest \
     --ports 8000 8501 \
     --environment-variables ENVIRONMENT=production
   ```

### Google Cloud Platform Deployment

#### Using Terraform

1. **Initialize Terraform**:
   ```bash
   terraform init
   ```

2. **Plan deployment**:
   ```bash
   terraform plan -var="project_id=your-project-id"
   ```

3. **Apply deployment**:
   ```bash
   terraform apply -var="project_id=your-project-id"
   ```

#### Using Google Cloud Run

1. **Build and push image**:
   ```bash
   # Build image
   docker build -t medical-ai-system .
   
   # Tag for GCR
   docker tag medical-ai-system:latest gcr.io/your-project/medical-ai-system:latest
   
   # Push to GCR
   docker push gcr.io/your-project/medical-ai-system:latest
   ```

2. **Deploy to Cloud Run**:
   ```bash
   gcloud run deploy medical-ai-system \
     --image gcr.io/your-project/medical-ai-system:latest \
     --platform managed \
     --region us-central1 \
     --allow-unauthenticated
   ```

## Environment Configuration

### Environment Variables

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `ENVIRONMENT` | Environment name | development | No |
| `MODEL_CACHE` | Enable model caching | false | No |
| `SECURITY_LEVEL` | Security level | low | No |
| `LOG_LEVEL` | Logging level | DEBUG | No |
| `CORS_ORIGINS` | CORS allowed origins | ["*"] | No |
| `RATE_LIMIT_ENABLED` | Enable rate limiting | false | No |
| `MAX_FILE_SIZE` | Maximum file size | 104857600 | No |
| `SESSION_TIMEOUT` | Session timeout | 7200 | No |
| `API_HOST` | API host | 0.0.0.0 | No |
| `API_PORT` | API port | 8000 | No |
| `WEB_HOST` | Web host | 0.0.0.0 | No |
| `WEB_PORT` | Web port | 8501 | No |

### Configuration Files

- **env.development**: Development environment settings
- **env.staging**: Staging environment settings
- **env.production**: Production environment settings

### Security Configuration

For production deployments, ensure:

1. **Change default secrets**:
   ```bash
   # Generate secure secrets
   openssl rand -hex 32  # For SECRET_KEY
   openssl rand -hex 32  # For ENCRYPTION_KEY
   openssl rand -hex 32  # For JWT_SECRET
   ```

2. **Configure SSL certificates**:
   ```bash
   # Place certificates in ssl/ directory
   mkdir ssl
   cp your-cert.pem ssl/cert.pem
   cp your-key.pem ssl/key.pem
   ```

3. **Set up firewall rules**:
   ```bash
   # Allow only necessary ports
   ufw allow 22    # SSH
   ufw allow 80    # HTTP
   ufw allow 443   # HTTPS
   ufw enable
   ```

## Monitoring and Logging

### Prometheus Metrics

The system exposes metrics at `/metrics` endpoint:

- **System metrics**: CPU, memory, disk usage
- **Application metrics**: Request rate, response time, error rate
- **Business metrics**: Prediction count, model performance

### Grafana Dashboards

Import the provided dashboard configuration:

1. **Access Grafana**: http://localhost:3000
2. **Login**: admin/admin (change default password)
3. **Import dashboard**: Use `grafana-dashboard.json`

### Logging Configuration

Logs are stored in the `logs/` directory:

- **medical_detection.log**: General application logs
- **medical_detection_errors.log**: Error logs only
- **medical_detection_security.log**: Security-related logs
- **medical_detection_performance.log**: Performance logs
- **medical_detection_audit.log**: Audit logs

### Alerting Rules

Prometheus alerting rules are defined in `medical_ai_rules.yml`:

- **System alerts**: High CPU, memory, disk usage
- **Application alerts**: Service down, high error rate
- **Security alerts**: Failed logins, suspicious activity
- **Performance alerts**: Slow responses, high load

## Security Considerations

### Data Protection

1. **Encrypt sensitive data**:
   ```python
   # Use encryption for stored data
   from cryptography.fernet import Fernet
   key = Fernet.generate_key()
   cipher = Fernet(key)
   encrypted_data = cipher.encrypt(sensitive_data)
   ```

2. **Secure file uploads**:
   ```python
   # Validate file types and sizes
   ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'dcm', 'nii', 'csv'}
   MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB
   ```

3. **Implement access controls**:
   ```python
   # Role-based access control
   @require_permission('medical_data_access')
   def access_patient_data(patient_id):
       # Implementation
   ```

### Network Security

1. **Use HTTPS**:
   ```nginx
   # Nginx SSL configuration
   ssl_certificate /etc/ssl/certs/cert.pem;
   ssl_certificate_key /etc/ssl/private/key.pem;
   ```

2. **Implement rate limiting**:
   ```python
   # Rate limiting
   from flask_limiter import Limiter
   limiter = Limiter(app, key_func=get_remote_address)
   
   @app.route('/predict')
   @limiter.limit("10 per minute")
   def predict():
       # Implementation
   ```

3. **Use secure headers**:
   ```python
   # Security headers
   from flask_talisman import Talisman
   Talisman(app, force_https=True)
   ```

### Compliance

- **HIPAA**: Implement data encryption, access controls, audit logging
- **GDPR**: Implement data anonymization, right to deletion
- **SOC 2**: Implement security controls, monitoring, incident response

## Troubleshooting

### Common Issues

#### 1. Port Already in Use

```bash
# Check what's using the port
lsof -i :8000
# Kill the process
kill -9 <PID>
```

#### 2. Permission Denied

```bash
# Fix file permissions
chmod +x deploy.sh
chmod 755 logs/
chmod 644 *.py
```

#### 3. Docker Build Fails

```bash
# Clean Docker cache
docker system prune -a
# Rebuild without cache
docker build --no-cache -t medical-ai-system .
```

#### 4. Model Loading Errors

```bash
# Check model files
ls -la models/saved_models/
# Regenerate models
python train_models.py
```

#### 5. Database Connection Issues

```bash
# Check database status
systemctl status postgresql
# Check connection
psql -h localhost -U medical_ai_user -d medical_ai
```

### Log Analysis

```bash
# View recent logs
tail -f logs/medical_detection.log

# Search for errors
grep "ERROR" logs/medical_detection.log

# View security logs
tail -f logs/medical_detection_security.log
```

### Performance Issues

1. **Check system resources**:
   ```bash
   # CPU usage
   top
   # Memory usage
   free -h
   # Disk usage
   df -h
   ```

2. **Monitor application metrics**:
   ```bash
   # Access Prometheus metrics
   curl http://localhost:9090/metrics
   ```

3. **Optimize configuration**:
   ```python
   # Increase worker processes
   API_WORKERS = 4
   # Enable caching
   CACHE_ENABLED = True
   ```

## Maintenance

### Regular Tasks

#### Daily
- Monitor system health
- Check error logs
- Verify backup completion

#### Weekly
- Review security logs
- Update dependencies
- Performance analysis

#### Monthly
- Security updates
- Model retraining
- Capacity planning

### Backup Strategy

1. **Database backup**:
   ```bash
   # PostgreSQL backup
   pg_dump medical_ai > backup_$(date +%Y%m%d).sql
   ```

2. **Model backup**:
   ```bash
   # Backup trained models
   tar -czf models_backup_$(date +%Y%m%d).tar.gz models/saved_models/
   ```

3. **Configuration backup**:
   ```bash
   # Backup configuration files
   cp -r .env* config/ backup/
   ```

### Updates and Upgrades

1. **Application updates**:
   ```bash
   # Pull latest changes
   git pull origin main
   # Update dependencies
   pip install -r requirements.txt --upgrade
   # Restart services
   docker-compose restart
   ```

2. **Model updates**:
   ```bash
   # Retrain models
   python train_models.py
   # Reload models
   curl -X POST http://localhost:8000/models/reload
   ```

### Scaling

#### Horizontal Scaling

1. **Load balancer configuration**:
   ```nginx
   upstream medical_ai_backend {
       server app1:8000;
       server app2:8000;
       server app3:8000;
   }
   ```

2. **Auto-scaling groups**:
   ```yaml
   # Docker Compose scaling
   docker-compose up --scale medical-ai-app=3
   ```

#### Vertical Scaling

1. **Increase resources**:
   ```yaml
   # Docker Compose resource limits
   deploy:
     resources:
       limits:
         memory: 4G
         cpus: '2.0'
   ```

2. **Optimize configuration**:
   ```python
   # Increase worker processes
   API_WORKERS = 8
   # Enable parallel processing
   PARALLEL_PROCESSING = True
   MAX_WORKERS = 8
   ```

## Support

### Documentation

- **API Documentation**: http://localhost:8000/docs
- **User Manual**: See README.md
- **Developer Guide**: See PROJECT_STRUCTURE.md

### Community

- **GitHub Issues**: Report bugs and feature requests
- **Discussions**: Technical discussions and support
- **Contributions**: Community code contributions

### Professional Support

For enterprise deployments and professional support:

- **Email**: support@medical-ai-system.com
- **Phone**: +1-800-MEDICAL-AI
- **Documentation**: https://docs.medical-ai-system.com

---

**Note**: This deployment guide is regularly updated. Please check for the latest version and ensure you're following the most current instructions.
