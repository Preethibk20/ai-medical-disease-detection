import numpy as np
import pandas as pd
from PIL import Image
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import joblib
import os

class MedicalDataProcessor:
    def __init__(self):
        self.image_scaler = None
        self.lab_scaler = None
        self.label_encoders = {}
        self.imputer = SimpleImputer(strategy='mean')
        
    def process_medical_image(self, image_path, target_size=(224, 224)):
        """Process medical images using PIL only"""
        try:
            # Use PIL for all image processing
            image = Image.open(image_path).convert('L')  # Convert to grayscale
            image = image.resize(target_size)
            image = np.array(image)
            
            # Normalize
            image = (image - image.min()) / (image.max() - image.min())
            image = image.astype(np.float32)
            return image
                
        except Exception as e:
            print(f"Error processing image {image_path}: {e}")
            return None
    
    def process_lab_results(self, lab_data):
        """Process laboratory test results"""
        if isinstance(lab_data, dict):
            lab_data = pd.DataFrame([lab_data])
        
        # Handle missing values
        lab_data = pd.DataFrame(self.imputer.fit_transform(lab_data), 
                              columns=lab_data.columns)
        
        # Standardize numerical values
        if self.lab_scaler is None:
            self.lab_scaler = StandardScaler()
            lab_data = pd.DataFrame(self.lab_scaler.fit_transform(lab_data),
                                  columns=lab_data.columns)
        else:
            lab_data = pd.DataFrame(self.lab_scaler.transform(lab_data),
                                  columns=lab_data.columns)
        
        return lab_data
    
    def process_patient_demographics(self, demographics):
        """Process patient demographic information to match training data format"""
        if isinstance(demographics, dict):
            demographics = pd.DataFrame([demographics])
        
        # Create the same 14 features as training data
        processed_data = pd.DataFrame()
        
        # Basic demographics
        processed_data['age'] = demographics['age']
        processed_data['gender'] = demographics['gender'].map({'Male': 1, 'Female': 0, 'Other': 0})
        processed_data['weight'] = demographics['weight']
        processed_data['height'] = demographics['height']
        
        # Calculate BMI
        processed_data['bmi'] = demographics['weight'] / ((demographics['height']/100) ** 2)
        
        # Lab results (use provided values or defaults)
        processed_data['glucose'] = demographics.get('glucose', 100.0)
        processed_data['hba1c'] = demographics.get('hba1c', 5.0)
        processed_data['systolic_bp'] = demographics.get('systolic_bp', 120.0)
        processed_data['diastolic_bp'] = demographics.get('diastolic_bp', 80.0)
        processed_data['cholesterol_total'] = demographics.get('cholesterol_total', 200.0)
        processed_data['creatinine'] = demographics.get('creatinine', 1.0)
        processed_data['egfr'] = demographics.get('egfr', 90.0)
        processed_data['alt'] = demographics.get('alt', 25.0)
        processed_data['ast'] = demographics.get('ast', 25.0)
        
        return processed_data
    
    def combine_modalities(self, image_data, lab_data, demographics):
        """Combine different data modalities"""
        # For now, just return demographics as it contains all 14 features
        if demographics is not None:
            return demographics.values
        else:
            # Create default features if no demographics
            default_features = np.array([[50, 1, 70, 170, 24.2, 100, 5.0, 120, 80, 200, 1.0, 90, 25, 25]])
            return default_features
    
    def save_preprocessors(self, save_path):
        """Save fitted preprocessors"""
        os.makedirs(save_path, exist_ok=True)
        
        if self.image_scaler:
            joblib.dump(self.image_scaler, os.path.join(save_path, 'image_scaler.pkl'))
        if self.lab_scaler:
            joblib.dump(self.lab_scaler, os.path.join(save_path, 'lab_scaler.pkl'))
        if self.label_encoders:
            joblib.dump(self.label_encoders, os.path.join(save_path, 'label_encoders.pkl'))
        if self.imputer:
            joblib.dump(self.imputer, os.path.join(save_path, 'imputer.pkl'))
    
    def load_preprocessors(self, load_path):
        """Load fitted preprocessors"""
        if os.path.exists(os.path.join(load_path, 'image_scaler.pkl')):
            self.image_scaler = joblib.load(os.path.join(load_path, 'image_scaler.pkl'))
        if os.path.exists(os.path.join(load_path, 'lab_scaler.pkl')):
            self.lab_scaler = joblib.load(os.path.join(load_path, 'lab_scaler.pkl'))
        if os.path.exists(os.path.join(load_path, 'label_encoders.pkl')):
            self.label_encoders = joblib.load(os.path.join(load_path, 'label_encoders.pkl'))
        if os.path.exists(os.path.join(load_path, 'imputer.pkl')):
            self.imputer = joblib.load(os.path.join(load_path, 'imputer.pkl'))
