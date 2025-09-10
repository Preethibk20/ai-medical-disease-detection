# Logging configuration for AI Medical Disease Detection System
import logging
import logging.handlers
import os
from datetime import datetime
from pathlib import Path

class MedicalAILogger:
    """Centralized logging configuration for Medical AI System"""
    
    def __init__(self, log_level="INFO", log_file="logs/medical_detection.log"):
        self.log_level = getattr(logging, log_level.upper())
        self.log_file = log_file
        self.setup_logging()
    
    def setup_logging(self):
        """Setup logging configuration"""
        # Create logs directory if it doesn't exist
        Path("logs").mkdir(exist_ok=True)
        
        # Create formatters
        detailed_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
        )
        
        simple_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        )
        
        # Root logger configuration
        root_logger = logging.getLogger()
        root_logger.setLevel(self.log_level)
        
        # Clear existing handlers
        root_logger.handlers.clear()
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(self.log_level)
        console_handler.setFormatter(simple_formatter)
        root_logger.addHandler(console_handler)
        
        # File handler with rotation
        file_handler = logging.handlers.RotatingFileHandler(
            self.log_file,
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        )
        file_handler.setLevel(self.log_level)
        file_handler.setFormatter(detailed_formatter)
        root_logger.addHandler(file_handler)
        
        # Error file handler
        error_handler = logging.handlers.RotatingFileHandler(
            "logs/medical_detection_errors.log",
            maxBytes=5*1024*1024,  # 5MB
            backupCount=3
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(detailed_formatter)
        root_logger.addHandler(error_handler)
        
        # Security log handler
        security_handler = logging.handlers.RotatingFileHandler(
            "logs/medical_detection_security.log",
            maxBytes=5*1024*1024,  # 5MB
            backupCount=10
        )
        security_handler.setLevel(logging.WARNING)
        security_handler.setFormatter(detailed_formatter)
        
        # Create security logger
        security_logger = logging.getLogger('security')
        security_logger.addHandler(security_handler)
        security_logger.setLevel(logging.WARNING)
        security_logger.propagate = False
        
        # Performance log handler
        performance_handler = logging.handlers.RotatingFileHandler(
            "logs/medical_detection_performance.log",
            maxBytes=5*1024*1024,  # 5MB
            backupCount=5
        )
        performance_handler.setLevel(logging.INFO)
        performance_handler.setFormatter(detailed_formatter)
        
        # Create performance logger
        performance_logger = logging.getLogger('performance')
        performance_logger.addHandler(performance_handler)
        performance_logger.setLevel(logging.INFO)
        performance_logger.propagate = False
        
        # Audit log handler
        audit_handler = logging.handlers.RotatingFileHandler(
            "logs/medical_detection_audit.log",
            maxBytes=10*1024*1024,  # 10MB
            backupCount=30
        )
        audit_handler.setLevel(logging.INFO)
        audit_handler.setFormatter(detailed_formatter)
        
        # Create audit logger
        audit_logger = logging.getLogger('audit')
        audit_logger.addHandler(audit_handler)
        audit_logger.setLevel(logging.INFO)
        audit_logger.propagate = False

class SecurityLogger:
    """Security-specific logging"""
    
    def __init__(self):
        self.logger = logging.getLogger('security')
    
    def log_login_attempt(self, username, ip_address, success):
        """Log login attempts"""
        status = "SUCCESS" if success else "FAILED"
        self.logger.warning(f"LOGIN_ATTEMPT - User: {username}, IP: {ip_address}, Status: {status}")
    
    def log_file_upload(self, filename, file_size, user_id, ip_address):
        """Log file uploads"""
        self.logger.info(f"FILE_UPLOAD - File: {filename}, Size: {file_size}, User: {user_id}, IP: {ip_address}")
    
    def log_suspicious_activity(self, activity_type, details, ip_address):
        """Log suspicious activities"""
        self.logger.error(f"SUSPICIOUS_ACTIVITY - Type: {activity_type}, Details: {details}, IP: {ip_address}")
    
    def log_data_access(self, user_id, data_type, action, ip_address):
        """Log data access"""
        self.logger.info(f"DATA_ACCESS - User: {user_id}, Data: {data_type}, Action: {action}, IP: {ip_address}")

class PerformanceLogger:
    """Performance-specific logging"""
    
    def __init__(self):
        self.logger = logging.getLogger('performance')
    
    def log_prediction_time(self, model_name, prediction_time, input_size):
        """Log prediction performance"""
        self.logger.info(f"PREDICTION_TIME - Model: {model_name}, Time: {prediction_time:.3f}s, Input Size: {input_size}")
    
    def log_model_loading(self, model_name, loading_time, model_size):
        """Log model loading performance"""
        self.logger.info(f"MODEL_LOADING - Model: {model_name}, Time: {loading_time:.3f}s, Size: {model_size}")
    
    def log_api_response_time(self, endpoint, method, response_time, status_code):
        """Log API response times"""
        self.logger.info(f"API_RESPONSE - {method} {endpoint}, Time: {response_time:.3f}s, Status: {status_code}")
    
    def log_memory_usage(self, component, memory_usage):
        """Log memory usage"""
        self.logger.info(f"MEMORY_USAGE - Component: {component}, Usage: {memory_usage}MB")

class AuditLogger:
    """Audit-specific logging"""
    
    def __init__(self):
        self.logger = logging.getLogger('audit')
    
    def log_user_action(self, user_id, action, resource, ip_address):
        """Log user actions"""
        self.logger.info(f"USER_ACTION - User: {user_id}, Action: {action}, Resource: {resource}, IP: {ip_address}")
    
    def log_system_change(self, component, change_type, details, user_id):
        """Log system changes"""
        self.logger.info(f"SYSTEM_CHANGE - Component: {component}, Type: {change_type}, Details: {details}, User: {user_id}")
    
    def log_data_modification(self, data_type, operation, user_id, ip_address):
        """Log data modifications"""
        self.logger.info(f"DATA_MODIFICATION - Type: {data_type}, Operation: {operation}, User: {user_id}, IP: {ip_address}")
    
    def log_model_update(self, model_name, version, user_id, performance_metrics):
        """Log model updates"""
        self.logger.info(f"MODEL_UPDATE - Model: {model_name}, Version: {version}, User: {user_id}, Metrics: {performance_metrics}")

# Initialize loggers
def setup_logging(environment="development"):
    """Setup logging based on environment"""
    log_level = "DEBUG" if environment == "development" else "INFO"
    logger = MedicalAILogger(log_level=log_level)
    
    # Create specialized loggers
    security_logger = SecurityLogger()
    performance_logger = PerformanceLogger()
    audit_logger = AuditLogger()
    
    return logger, security_logger, performance_logger, audit_logger

# Logging decorators
def log_function_call(func):
    """Decorator to log function calls"""
    def wrapper(*args, **kwargs):
        logger = logging.getLogger(func.__module__)
        logger.debug(f"Calling {func.__name__} with args: {args}, kwargs: {kwargs}")
        try:
            result = func(*args, **kwargs)
            logger.debug(f"Function {func.__name__} completed successfully")
            return result
        except Exception as e:
            logger.error(f"Function {func.__name__} failed with error: {str(e)}")
            raise
    return wrapper

def log_prediction(func):
    """Decorator to log predictions"""
    def wrapper(*args, **kwargs):
        logger = logging.getLogger('performance')
        start_time = datetime.now()
        try:
            result = func(*args, **kwargs)
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            logger.info(f"Prediction completed in {duration:.3f} seconds")
            return result
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            raise
    return wrapper

# Structured logging for JSON output
import json

class StructuredLogger:
    """Structured logging for JSON output"""
    
    def __init__(self, name):
        self.logger = logging.getLogger(name)
    
    def log_event(self, event_type, **kwargs):
        """Log structured event"""
        event_data = {
            "timestamp": datetime.now().isoformat(),
            "event_type": event_type,
            **kwargs
        }
        self.logger.info(json.dumps(event_data))
    
    def log_error(self, error_type, error_message, **kwargs):
        """Log structured error"""
        error_data = {
            "timestamp": datetime.now().isoformat(),
            "event_type": "error",
            "error_type": error_type,
            "error_message": error_message,
            **kwargs
        }
        self.logger.error(json.dumps(error_data))

# Example usage
if __name__ == "__main__":
    # Setup logging
    logger, security_logger, performance_logger, audit_logger = setup_logging("development")
    
    # Test logging
    logging.info("Medical AI System logging initialized")
    
    # Test security logging
    security_logger.log_login_attempt("user123", "192.168.1.1", True)
    security_logger.log_file_upload("medical_image.jpg", 1024000, "user123", "192.168.1.1")
    
    # Test performance logging
    performance_logger.log_prediction_time("diabetes_model", 0.245, 100)
    performance_logger.log_api_response_time("/predict", "POST", 0.156, 200)
    
    # Test audit logging
    audit_logger.log_user_action("user123", "predict", "diabetes", "192.168.1.1")
    audit_logger.log_model_update("diabetes_model", "v2.1", "admin", {"accuracy": 0.95})
    
    print("Logging system initialized successfully!")
