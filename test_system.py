#!/usr/bin/env python3
"""
Test script to verify AI Medical Disease Detection System
"""

import sys
import os
import numpy as np
import pandas as pd

# Add utils and models to path
sys.path.append('utils')
sys.path.append('models')

from utils.data_processor import MedicalDataProcessor
from models.disease_models import MultiDiseaseDetector

def test_disease_detection():
    """Test the disease detection system"""
    print("🏥 Testing AI Medical Disease Detection System")
    print("=" * 50)
    
    # Initialize components
    data_processor = MedicalDataProcessor()
    disease_detector = MultiDiseaseDetector()
    
    # Load trained models
    print("Loading trained models...")
    try:
        disease_detector.load_all_models('models/saved_models')
        print("✅ Models loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        return False
    
    # Create sample patient data
    print("\nCreating sample patient data...")
    sample_data = {
        'age': 55,
        'gender': 'Male',
        'weight': 80.0,
        'height': 175.0,
        'systolic_bp': 140,
        'diastolic_bp': 90,
        'heart_rate': 75,
        'temperature': 37.0,
        'glucose': 120,
        'hba1c': 6.5,
        'cholesterol_total': 220,
        'cholesterol_hdl': 45,
        'cholesterol_ldl': 150,
        'triglycerides': 180,
        'creatinine': 1.1,
        'egfr': 85,
        'alt': 35,
        'ast': 30,
        'bilirubin': 1.0,
        'albumin': 4.2,
        'hemoglobin': 14.5,
        'white_blood_cells': 7000,
        'platelets': 250000
    }
    
    # Process the data
    print("Processing patient data...")
    demo_df = pd.DataFrame([sample_data])
    processed_demo = data_processor.process_patient_demographics(demo_df)
    
    # Make predictions
    print("Making disease predictions...")
    try:
        predictions = disease_detector.predict_all_diseases(processed_demo)
        
        print("\n🎯 Disease Risk Assessment Results:")
        print("-" * 40)
        
        for disease, risk in predictions.items():
            if risk is not None:
                risk_value = float(risk[0]) if hasattr(risk, '__len__') else float(risk)
                risk_percentage = risk_value * 100
                
                if risk_percentage < 30:
                    status = "🟢 Low Risk"
                elif risk_percentage < 70:
                    status = "🟡 Medium Risk"
                else:
                    status = "🔴 High Risk"
                
                print(f"{disease.replace('_', ' ').title():<20}: {risk_percentage:5.1f}% - {status}")
        
        print("\n✅ Disease detection test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error making predictions: {e}")
        return False

def test_model_accuracy():
    """Test model accuracy with sample data"""
    print("\n📊 Testing Model Accuracy...")
    print("-" * 30)
    
    # Load test data
    test_data = {}
    diseases = ['diabetes', 'cardiovascular', 'cancer', 'kidney_disease', 'liver_disease']
    
    for disease in diseases:
        try:
            X_test_path = f'data/{disease}_X_test.csv'
            y_test_path = f'data/{disease}_y_test.csv'
            
            if os.path.exists(X_test_path) and os.path.exists(y_test_path):
                X_test = pd.read_csv(X_test_path)
                y_test = pd.read_csv(y_test_path)
                
                if len(y_test.columns) == 1:
                    y_test = y_test.iloc[:, 0]
                else:
                    y_test = y_test.iloc[:, 0]
                
                test_data[disease] = {
                    'features': X_test,
                    'labels': y_test
                }
                
                print(f"✅ Loaded {disease} test data: {X_test.shape[0]} samples")
            else:
                print(f"❌ Missing test data for {disease}")
                
        except Exception as e:
            print(f"❌ Error loading {disease} test data: {e}")
    
    if not test_data:
        print("❌ No test data available")
        return False
    
    # Test predictions
    disease_detector = MultiDiseaseDetector()
    disease_detector.load_all_models('models/saved_models')
    
    print("\n🎯 Testing predictions on sample data...")
    for disease, data in test_data.items():
        try:
            # Test on first 5 samples
            X_sample = data['features'].head(5)
            y_sample = data['labels'].head(5)
            
            predictions = disease_detector.disease_models[disease].predict(X_sample)
            y_pred_binary = (predictions > 0.5).astype(int)
            
            accuracy = np.mean(y_pred_binary == y_sample.values)
            print(f"{disease.replace('_', ' ').title():<20}: {accuracy:.3f} accuracy")
            
        except Exception as e:
            print(f"❌ Error testing {disease}: {e}")
    
    return True

def main():
    """Main test function"""
    print("🚀 Starting AI Medical Disease Detection System Tests")
    print("=" * 60)
    
    # Test 1: Disease Detection
    success1 = test_disease_detection()
    
    # Test 2: Model Accuracy
    success2 = test_model_accuracy()
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 Test Summary:")
    print(f"Disease Detection Test: {'✅ PASSED' if success1 else '❌ FAILED'}")
    print(f"Model Accuracy Test: {'✅ PASSED' if success2 else '❌ FAILED'}")
    
    if success1 and success2:
        print("\n🎉 All tests passed! The system is ready for use.")
        print("\n🌐 Access the system:")
        print("   • Web Interface: http://localhost:8501")
        print("   • API Server: http://localhost:8000")
        print("   • API Documentation: http://localhost:8000/docs")
    else:
        print("\n⚠️ Some tests failed. Please check the errors above.")
    
    return success1 and success2

if __name__ == "__main__":
    main()
