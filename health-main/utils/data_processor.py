import numpy as np
import pandas as pd
import cv2
from PIL import Image
import pydicom
import nibabel as nib
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
        """Process medical images (X-ray, CT, MRI)"""
        try:
            # Handle different image formats
            if image_path.endswith('.dcm'):
                # DICOM file
                dcm = pydicom.dcmread(image_path)
                image = dcm.pixel_array
            elif image_path.endswith('.nii') or image_path.endswith('.nii.gz'):
                # NIfTI file
                nii = nib.load(image_path)
                image = nii.get_fdata()
                # Take middle slice for 3D volumes
                if len(image.shape) == 3:
                    image = image[:, :, image.shape[2]//2]
            else:
                # Regular image file
                image = cv2.imread(image_path)
                if image is not None:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                else:
                    image = np.array(Image.open(image_path).convert('L'))
            
            # Normalize and resize
            if image is not None:
                image = cv2.resize(image, target_size)
                image = (image - image.min()) / (image.max() - image.min())
                image = image.astype(np.float32)
                return image
            else:
                return None
                
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
        """Process patient demographic information"""
        if isinstance(demographics, dict):
            demographics = pd.DataFrame([demographics])
        
        # Encode categorical variables
        categorical_cols = demographics.select_dtypes(include=['object']).columns
        
        for col in categorical_cols:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                demographics[col] = self.label_encoders[col].fit_transform(demographics[col])
            else:
                demographics[col] = self.label_encoders[col].transform(demographics[col])
        
        # Handle missing values
        demographics = pd.DataFrame(self.imputer.fit_transform(demographics),
                                 columns=demographics.columns)
        
        return demographics
    
    def combine_modalities(self, image_data, lab_data, demographics):
        """Combine different data modalities"""
        features = []
        
        if image_data is not None:
            # Flatten image features
            image_features = image_data.flatten()
            features.extend(image_features)
        
        if lab_data is not None:
            lab_features = lab_data.values.flatten()
            features.extend(lab_features)
        
        if demographics is not None:
            demo_features = demographics.values.flatten()
            features.extend(demo_features)
        
        return np.array(features).reshape(1, -1)
    
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
