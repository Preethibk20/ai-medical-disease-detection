"""
Configuration file for AI Medical Disease Detection System
"""

import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
UTILS_DIR = BASE_DIR / "utils"
REPORTS_DIR = BASE_DIR / "reports"
DEMO_OUTPUT_DIR = BASE_DIR / "demo_output"

# Create directories if they don't exist
for directory in [DATA_DIR, MODELS_DIR, UTILS_DIR, REPORTS_DIR, DEMO_OUTPUT_DIR]:
    directory.mkdir(exist_ok=True)

# Model configuration
MODEL_CONFIG = {
    "diseases": [
        "diabetes",
        "cardiovascular", 
        "cancer",
        "kidney_disease",
        "liver_disease"
    ],
    
    "supported_image_formats": [
        ".png", ".jpg", ".jpeg", ".tiff", ".bmp",
        ".dcm", ".nii", ".nii.gz"  # Medical formats
    ],
    
    "supported_lab_formats": [
        ".csv", ".xlsx", ".xls", ".json"
    ],
    
    "image_processing": {
        "target_size": (224, 224),
        "normalization": "min_max",  # or "z_score"
        "augmentation": True
    },
    
    "feature_engineering": {
        "use_image_features": True,
        "use_lab_features": True,
        "use_demographic_features": True,
        "feature_selection": True,
        "dimensionality_reduction": False
    }
}

# AI Models configuration
AI_MODELS_CONFIG = {
    "ensemble_methods": [
        "random_forest",
        "gradient_boosting", 
        "logistic_regression",
        "svm",
        "mlp",
        "xgboost",
        "lightgbm"
    ],
    
    "deep_learning": {
        "enabled": True,
        "architecture": "mlp",  # or "cnn", "transformer"
        "layers": [256, 128, 64],
        "dropout_rate": 0.3,
        "activation": "relu",
        "optimizer": "adam",
        "learning_rate": 0.001,
        "batch_size": 32,
        "epochs": 50,
        "early_stopping": True,
        "patience": 10
    },
    
    "training": {
        "test_size": 0.2,
        "validation_size": 0.2,
        "random_state": 42,
        "cross_validation": True,
        "cv_folds": 5,
        "class_balance": True
    }
}

# Data processing configuration
DATA_PROCESSING_CONFIG = {
    "missing_value_strategy": "mean",  # or "median", "mode", "drop"
    "outlier_detection": True,
    "outlier_method": "iqr",  # or "z_score", "isolation_forest"
    "scaling_method": "standard",  # or "min_max", "robust"
    "categorical_encoding": "label",  # or "one_hot", "target"
    
    "lab_results": {
        "normalization": True,
        "reference_ranges": {
            "glucose": {"min": 70, "max": 100, "unit": "mg/dL"},
            "hba1c": {"min": 4.0, "max": 5.6, "unit": "%"},
            "cholesterol_total": {"min": 125, "max": 200, "unit": "mg/dL"},
            "cholesterol_hdl": {"min": 40, "max": 60, "unit": "mg/dL"},
            "cholesterol_ldl": {"min": 0, "max": 100, "unit": "mg/dL"},
            "triglycerides": {"min": 0, "max": 150, "unit": "mg/dL"},
            "creatinine": {"min": 0.6, "max": 1.2, "unit": "mg/dL"},
            "egfr": {"min": 90, "max": 120, "unit": "mL/min/1.73m²"},
            "alt": {"min": 7, "max": 55, "unit": "U/L"},
            "ast": {"min": 8, "max": 48, "unit": "U/L"},
            "bilirubin": {"min": 0.3, "max": 1.2, "unit": "mg/dL"},
            "albumin": {"min": 3.4, "max": 5.4, "unit": "g/dL"}
        }
    },
    
    "demographics": {
        "age_groups": {
            "young": {"min": 18, "max": 35},
            "middle": {"min": 36, "max": 65},
            "elderly": {"min": 66, "max": 100}
        },
        "bmi_categories": {
            "underweight": {"min": 0, "max": 18.5},
            "normal": {"min": 18.5, "max": 24.9},
            "overweight": {"min": 25.0, "max": 29.9},
            "obese": {"min": 30.0, "max": 100}
        },
        "blood_pressure_categories": {
            "normal": {"systolic": {"min": 90, "max": 120}, "diastolic": {"min": 60, "max": 80}},
            "elevated": {"systolic": {"min": 120, "max": 129}, "diastolic": {"min": 60, "max": 80}},
            "high": {"systolic": {"min": 130, "max": 200}, "diastolic": {"min": 80, "max": 130}}
        }
    }
}

# Risk assessment configuration
RISK_ASSESSMENT_CONFIG = {
    "risk_levels": {
        "low": {"min": 0, "max": 0.3, "color": "green", "action": "monitor"},
        "medium": {"min": 0.3, "max": 0.7, "color": "orange", "action": "closer_monitoring"},
        "high": {"min": 0.7, "max": 1.0, "color": "red", "action": "immediate_attention"}
    },
    
    "disease_specific_thresholds": {
        "diabetes": {
            "low": 0.3,
            "medium": 0.6,
            "high": 0.8
        },
        "cardiovascular": {
            "low": 0.25,
            "medium": 0.55,
            "high": 0.75
        },
        "cancer": {
            "low": 0.2,
            "medium": 0.5,
            "high": 0.7
        },
        "kidney_disease": {
            "low": 0.3,
            "medium": 0.6,
            "high": 0.8
        },
        "liver_disease": {
            "low": 0.25,
            "medium": 0.55,
            "high": 0.75
        }
    }
}

# Web interface configuration
WEB_INTERFACE_CONFIG = {
    "streamlit": {
        "page_title": "AI Medical Disease Detection",
        "page_icon": "🏥",
        "layout": "wide",
        "initial_sidebar_state": "expanded",
        "theme": {
            "primaryColor": "#1f77b4",
            "backgroundColor": "#ffffff",
            "secondaryBackgroundColor": "#f0f2f6",
            "textColor": "#262730"
        }
    },
    
    "api": {
        "title": "AI Medical Disease Detection API",
        "description": "API for detecting chronic diseases using multimodal medical data",
        "version": "1.0.0",
        "host": "0.0.0.0",
        "port": 8000,
        "reload": True,
        "cors_origins": ["*"]
    }
}

# Reporting configuration
REPORTING_CONFIG = {
    "report_formats": ["pdf", "html", "csv", "json"],
    "include_visualizations": True,
    "include_recommendations": True,
    "include_risk_factors": True,
    "include_preventive_measures": True,
    
    "templates": {
        "patient_report": "templates/patient_report.html",
        "batch_report": "templates/batch_report.html",
        "training_report": "templates/training_report.html"
    }
}

# Security and privacy configuration
SECURITY_CONFIG = {
    "data_encryption": True,
    "secure_file_upload": True,
    "max_file_size": 100 * 1024 * 1024,  # 100MB
    "allowed_file_types": [
        "image/*", "application/dicom", "text/csv", 
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    ],
    "session_timeout": 3600,  # 1 hour
    "rate_limiting": True,
    "max_requests_per_minute": 60
}

# Logging configuration
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "file": "logs/medical_detection.log",
    "max_file_size": 10 * 1024 * 1024,  # 10MB
    "backup_count": 5
}

# Performance configuration
PERFORMANCE_CONFIG = {
    "batch_processing": True,
    "max_batch_size": 100,
    "parallel_processing": True,
    "max_workers": 4,
    "cache_enabled": True,
    "cache_ttl": 3600,  # 1 hour
    "model_caching": True
}

# Validation configuration
VALIDATION_CONFIG = {
    "input_validation": True,
    "data_quality_checks": True,
    "range_validation": True,
    "format_validation": True,
    
    "validation_rules": {
        "age": {"min": 0, "max": 120},
        "weight": {"min": 20, "max": 300},
        "height": {"min": 100, "max": 250},
        "glucose": {"min": 20, "max": 1000},
        "hba1c": {"min": 3.0, "max": 20.0},
        "systolic_bp": {"min": 70, "max": 250},
        "diastolic_bp": {"min": 40, "max": 150}
    }
}

# Environment-specific configurations
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")

if ENVIRONMENT == "production":
    SECURITY_CONFIG["data_encryption"] = True
    SECURITY_CONFIG["secure_file_upload"] = True
    LOGGING_CONFIG["level"] = "WARNING"
    PERFORMANCE_CONFIG["cache_enabled"] = True
elif ENVIRONMENT == "testing":
    SECURITY_CONFIG["data_encryption"] = False
    SECURITY_CONFIG["secure_file_upload"] = False
    LOGGING_CONFIG["level"] = "DEBUG"
    PERFORMANCE_CONFIG["cache_enabled"] = False

# Export all configurations
__all__ = [
    "BASE_DIR", "DATA_DIR", "MODELS_DIR", "UTILS_DIR", "REPORTS_DIR", "DEMO_OUTPUT_DIR",
    "MODEL_CONFIG", "AI_MODELS_CONFIG", "DATA_PROCESSING_CONFIG", "RISK_ASSESSMENT_CONFIG",
    "WEB_INTERFACE_CONFIG", "REPORTING_CONFIG", "SECURITY_CONFIG", "LOGGING_CONFIG",
    "PERFORMANCE_CONFIG", "VALIDATION_CONFIG", "ENVIRONMENT"
]
