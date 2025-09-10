# AI Medical Disease Detection System - Quick Start Guide

## 🚀 Quick Deployment Options

### Option 1: One-Command Deployment (Recommended)

**Linux/macOS:**
```bash
chmod +x deploy.sh && ./deploy.sh --type local --env development --auto-setup
```

**Windows PowerShell:**
```powershell
.\deploy.ps1 -DeploymentType local -Environment development -AutoSetup
```

### Option 2: Docker Deployment

```bash
# Start all services with Docker Compose
docker-compose up -d

# Access the application
# Web Interface: http://localhost:8501
# API: http://localhost:8000
# Monitoring: http://localhost:3000
```

### Option 3: Manual Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Generate sample data and train models
python start.py --mode check --auto-setup

# 3. Start the application
python start.py --mode web
```

## 🌐 Access Points

After successful deployment, you can access:

- **Web Interface**: http://localhost:8501
- **API Server**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health
- **Monitoring Dashboard**: http://localhost:3000 (if using Docker)

## 📋 System Requirements

- **Python**: 3.8 or higher
- **Memory**: 4GB RAM minimum (8GB+ recommended)
- **Storage**: 10GB free space minimum
- **OS**: Windows 10/11, macOS 10.15+, or Linux

## 🔧 Troubleshooting

### Common Issues

1. **Port already in use**:
   ```bash
   # Kill process using port 8000
   lsof -ti:8000 | xargs kill -9
   ```

2. **Permission denied**:
   ```bash
   # Make scripts executable
   chmod +x deploy.sh
   ```

3. **Dependencies not found**:
   ```bash
   # Reinstall requirements
   pip install -r requirements.txt --force-reinstall
   ```

### Getting Help

- Check the full [Deployment Guide](DEPLOYMENT_GUIDE.md)
- Review [Project Structure](PROJECT_STRUCTURE.md)
- Open an issue on GitHub

## 🎯 Next Steps

1. **Explore the Web Interface**: Upload sample medical data and run predictions
2. **Test the API**: Use the interactive documentation at `/docs`
3. **Monitor Performance**: Check the Grafana dashboard
4. **Customize Configuration**: Edit environment files for your needs

## 📚 Additional Resources

- [Full Deployment Guide](DEPLOYMENT_GUIDE.md)
- [API Documentation](http://localhost:8000/docs)
- [Project Structure](PROJECT_STRUCTURE.md)
- [README](README.md)

---

**Need help?** Check the troubleshooting section or open an issue on GitHub!
