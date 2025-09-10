# AI-Powered Chronic Disease Detection System

A comprehensive machine learning system for detecting chronic diseases using multimodal medical data including images, lab results, and patient demographics.

## Features

- **Multimodal Data Processing**: Handles medical images, lab results, and patient information
- **Multiple Disease Detection**: Supports diabetes, heart disease, cancer, and more
- **Advanced AI Models**: Uses ensemble methods and deep learning
- **Interactive Web Interface**: Streamlit-based dashboard for easy interaction
- **API Endpoints**: FastAPI backend for integration with other systems
- **Comprehensive Analytics**: Detailed reports and visualizations

## Installation

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Web Interface
```bash
streamlit run app.py
```

### API Server
```bash
uvicorn api:app --reload
```

## Project Structure

- `app.py` - Main Streamlit application
- `api.py` - FastAPI backend
- `models/` - Pre-trained models and model definitions
- `utils/` - Utility functions for data processing
- `data/` - Sample data and preprocessing scripts
- `notebooks/` - Jupyter notebooks for model development

## Supported Diseases
- Diabetes
- Cardiovascular Disease
- Cancer (Breast, Lung, Skin)
- Chronic Kidney Disease
- Liver Disease

## Data Types

- Medical Images (X-rays, CT scans, MRI)
- Laboratory Results
- Patient Demographics
- Vital Signs
- Medical History
