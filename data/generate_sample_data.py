import numpy as np
import pandas as pd
import os
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

def generate_diabetes_data(n_samples=1000):
    """Generate synthetic diabetes data"""
    # Features: age, gender, weight, height, BMI, glucose, HbA1c, family_history
    np.random.seed(42)
    
    # Generate features
    age = np.random.normal(55, 15, n_samples)
    age = np.clip(age, 18, 90)
    
    gender = np.random.choice([0, 1], n_samples)  # 0: Female, 1: Male
    
    weight = np.random.normal(75, 15, n_samples)
    weight = np.clip(weight, 40, 150)
    
    height = np.random.normal(170, 10, n_samples)
    height = np.clip(height, 150, 200)
    
    bmi = weight / ((height/100) ** 2)
    
    glucose = np.random.normal(120, 40, n_samples)
    glucose = np.clip(glucose, 70, 300)
    
    hba1c = glucose / 20 + np.random.normal(0, 1, n_samples)
    hba1c = np.clip(hba1c, 4, 12)
    
    family_history = np.random.choice([0, 1], n_samples, p=[0.7, 0.3])
    
    # Generate labels (diabetes risk)
    # Higher risk for: older age, high BMI, high glucose, high HbA1c, family history
    risk_score = (
        0.3 * (age - 18) / (90 - 18) +
        0.2 * (bmi - 18.5) / (40 - 18.5) +
        0.3 * (glucose - 70) / (300 - 70) +
        0.15 * (hba1c - 4) / (12 - 4) +
        0.05 * family_history +
        np.random.normal(0, 0.1, n_samples)
    )
    
    labels = (risk_score > 0.5).astype(int)
    
    # Create DataFrame
    data = pd.DataFrame({
        'age': age,
        'gender': gender,
        'weight': weight,
        'height': height,
        'bmi': bmi,
        'glucose': glucose,
        'hba1c': hba1c,
        'family_history': family_history,
        'diabetes_risk': labels
    })
    
    return data

def generate_cardiovascular_data(n_samples=1000):
    """Generate synthetic cardiovascular disease data"""
    np.random.seed(42)
    
    # Features: age, gender, weight, height, BMI, BP, cholesterol, smoking, exercise
    age = np.random.normal(60, 15, n_samples)
    age = np.clip(age, 25, 85)
    
    gender = np.random.choice([0, 1], n_samples)
    
    weight = np.random.normal(80, 18, n_samples)
    weight = np.clip(weight, 45, 160)
    
    height = np.random.normal(172, 12, n_samples)
    height = np.clip(height, 155, 205)
    
    bmi = weight / ((height/100) ** 2)
    
    systolic_bp = np.random.normal(130, 25, n_samples)
    systolic_bp = np.clip(systolic_bp, 90, 200)
    
    diastolic_bp = systolic_bp * 0.7 + np.random.normal(0, 8, n_samples)
    diastolic_bp = np.clip(diastolic_bp, 60, 120)
    
    cholesterol_total = np.random.normal(200, 40, n_samples)
    cholesterol_total = np.clip(cholesterol_total, 120, 350)
    
    cholesterol_hdl = np.random.normal(50, 15, n_samples)
    cholesterol_hdl = np.clip(cholesterol_hdl, 25, 100)
    
    cholesterol_ldl = cholesterol_total - cholesterol_hdl - np.random.normal(30, 10, n_samples)
    cholesterol_ldl = np.clip(cholesterol_ldl, 50, 250)
    
    triglycerides = np.random.normal(150, 80, n_samples)
    triglycerides = np.clip(triglycerides, 50, 500)
    
    smoking = np.random.choice([0, 1], n_samples, p=[0.8, 0.2])
    
    exercise_hours = np.random.exponential(2, n_samples)
    exercise_hours = np.clip(exercise_hours, 0, 8)
    
    # Generate labels (cardiovascular disease risk)
    risk_score = (
        0.25 * (age - 25) / (85 - 25) +
        0.15 * (bmi - 18.5) / (40 - 18.5) +
        0.2 * (systolic_bp - 90) / (200 - 90) +
        0.15 * (cholesterol_total - 120) / (350 - 120) +
        0.1 * (1 - cholesterol_hdl / 100) +
        0.1 * smoking +
        0.05 * (1 - exercise_hours / 8) +
        np.random.normal(0, 0.1, n_samples)
    )
    
    labels = (risk_score > 0.6).astype(int)
    
    data = pd.DataFrame({
        'age': age,
        'gender': gender,
        'weight': weight,
        'height': height,
        'bmi': bmi,
        'systolic_bp': systolic_bp,
        'diastolic_bp': diastolic_bp,
        'cholesterol_total': cholesterol_total,
        'cholesterol_hdl': cholesterol_hdl,
        'cholesterol_ldl': cholesterol_ldl,
        'triglycerides': triglycerides,
        'smoking': smoking,
        'exercise_hours': exercise_hours,
        'cardiovascular_risk': labels
    })
    
    return data

def generate_cancer_data(n_samples=1000):
    """Generate synthetic cancer screening data"""
    np.random.seed(42)
    
    # Features: age, gender, family_history, lifestyle_factors, screening_results
    age = np.random.normal(50, 15, n_samples)
    age = np.clip(age, 30, 80)
    
    gender = np.random.choice([0, 1], n_samples)
    
    family_history = np.random.choice([0, 1], n_samples, p=[0.8, 0.2])
    
    smoking_years = np.random.exponential(10, n_samples)
    smoking_years = np.clip(smoking_years, 0, 50)
    
    alcohol_consumption = np.random.exponential(5, n_samples)
    alcohol_consumption = np.clip(alcohol_consumption, 0, 30)
    
    exercise_frequency = np.random.choice([0, 1, 2, 3], n_samples, p=[0.3, 0.3, 0.25, 0.15])
    
    bmi = np.random.normal(26, 5, n_samples)
    bmi = np.clip(bmi, 18, 45)
    
    # Generate labels (cancer risk)
    risk_score = (
        0.3 * (age - 30) / (80 - 30) +
        0.2 * family_history +
        0.15 * (smoking_years / 50) +
        0.1 * (alcohol_consumption / 30) +
        0.1 * (1 - exercise_frequency / 3) +
        0.15 * (bmi - 18) / (45 - 18) +
        np.random.normal(0, 0.1, n_samples)
    )
    
    labels = (risk_score > 0.5).astype(int)
    
    data = pd.DataFrame({
        'age': age,
        'gender': gender,
        'family_history': family_history,
        'smoking_years': smoking_years,
        'alcohol_consumption': alcohol_consumption,
        'exercise_frequency': exercise_frequency,
        'bmi': bmi,
        'cancer_risk': labels
    })
    
    return data

def generate_kidney_disease_data(n_samples=1000):
    """Generate synthetic kidney disease data"""
    np.random.seed(42)
    
    # Features: age, gender, weight, height, BP, creatinine, eGFR, proteinuria
    age = np.random.normal(65, 15, n_samples)
    age = np.clip(age, 25, 90)
    
    gender = np.random.choice([0, 1], n_samples)
    
    weight = np.random.normal(75, 15, n_samples)
    weight = np.clip(weight, 45, 140)
    
    height = np.random.normal(170, 10, n_samples)
    height = np.clip(height, 150, 200)
    
    bmi = weight / ((height/100) ** 2)
    
    systolic_bp = np.random.normal(140, 25, n_samples)
    systolic_bp = np.clip(systolic_bp, 90, 200)
    
    diastolic_bp = systolic_bp * 0.7 + np.random.normal(0, 8, n_samples)
    diastolic_bp = np.clip(diastolic_bp, 60, 120)
    
    creatinine = np.random.normal(1.2, 0.5, n_samples)
    creatinine = np.clip(creatinine, 0.5, 3.0)
    
    # eGFR calculation (simplified)
    egfr = 140 * (0.993 ** age) * np.where(gender == 0, 0.85, 1.0) / creatinine
    egfr = np.clip(egfr, 10, 120)
    
    proteinuria = np.random.exponential(50, n_samples)
    proteinuria = np.clip(proteinuria, 0, 500)
    
    diabetes = np.random.choice([0, 1], n_samples, p=[0.7, 0.3])
    
    # Generate labels (kidney disease risk)
    risk_score = (
        0.2 * (age - 25) / (90 - 25) +
        0.15 * (bmi - 18.5) / (40 - 18.5) +
        0.2 * (systolic_bp - 90) / (200 - 90) +
        0.25 * (creatinine - 0.5) / (3.0 - 0.5) +
        0.1 * (1 - egfr / 120) +
        0.1 * (proteinuria / 500) +
        np.random.normal(0, 0.1, n_samples)
    )
    
    labels = (risk_score > 0.5).astype(int)
    
    data = pd.DataFrame({
        'age': age,
        'gender': gender,
        'weight': weight,
        'height': height,
        'bmi': bmi,
        'systolic_bp': systolic_bp,
        'diastolic_bp': diastolic_bp,
        'creatinine': creatinine,
        'egfr': egfr,
        'proteinuria': proteinuria,
        'diabetes': diabetes,
        'kidney_disease_risk': labels
    })
    
    return data

def generate_liver_disease_data(n_samples=1000):
    """Generate synthetic liver disease data"""
    np.random.seed(42)
    
    # Features: age, gender, weight, height, alcohol, medications, liver enzymes
    age = np.random.normal(55, 15, n_samples)
    age = np.clip(age, 25, 85)
    
    gender = np.random.choice([0, 1], n_samples)
    
    weight = np.random.normal(75, 15, n_samples)
    weight = np.clip(weight, 45, 140)
    
    height = np.random.normal(170, 10, n_samples)
    height = np.clip(height, 150, 200)
    
    bmi = weight / ((height/100) ** 2)
    
    alcohol_consumption = np.random.exponential(8, n_samples)
    alcohol_consumption = np.clip(alcohol_consumption, 0, 40)
    
    alt = np.random.normal(25, 15, n_samples)
    alt = np.clip(alt, 5, 100)
    
    ast = alt * 0.8 + np.random.normal(0, 8, n_samples)
    ast = np.clip(ast, 5, 100)
    
    bilirubin = np.random.normal(0.8, 0.4, n_samples)
    bilirubin = np.clip(bilirubin, 0.1, 3.0)
    
    albumin = np.random.normal(4.0, 0.5, n_samples)
    albumin = np.clip(albumin, 2.5, 5.5)
    
    hepatitis_b = np.random.choice([0, 1], n_samples, p=[0.95, 0.05])
    
    hepatitis_c = np.random.choice([0, 1], n_samples, p=[0.97, 0.03])
    
    # Generate labels (liver disease risk)
    risk_score = (
        0.15 * (age - 25) / (85 - 25) +
        0.1 * (bmi - 18.5) / (40 - 18.5) +
        0.25 * (alcohol_consumption / 40) +
        0.2 * (alt - 5) / (100 - 5) +
        0.15 * (bilirubin - 0.1) / (3.0 - 0.1) +
        0.1 * (1 - albumin / 5.5) +
        0.05 * hepatitis_b +
        0.05 * hepatitis_c +
        np.random.normal(0, 0.1, n_samples)
    )
    
    labels = (risk_score > 0.5).astype(int)
    
    data = pd.DataFrame({
        'age': age,
        'gender': gender,
        'weight': weight,
        'height': height,
        'bmi': bmi,
        'alcohol_consumption': alcohol_consumption,
        'alt': alt,
        'ast': ast,
        'bilirubin': bilirubin,
        'albumin': albumin,
        'hepatitis_b': hepatitis_b,
        'hepatitis_c': hepatitis_c,
        'liver_disease_risk': labels
    })
    
    return data

def generate_combined_dataset():
    """Generate combined dataset for multimodal analysis"""
    print("Generating sample medical datasets...")
    
    # Create data directory
    os.makedirs('data', exist_ok=True)
    
    # Generate individual datasets
    diabetes_data = generate_diabetes_data(1000)
    cardiovascular_data = generate_cardiovascular_data(1000)
    cancer_data = generate_cancer_data(1000)
    kidney_data = generate_kidney_disease_data(1000)
    liver_data = generate_liver_disease_data(1000)
    
    # Save individual datasets
    diabetes_data.to_csv('data/diabetes_data.csv', index=False)
    cardiovascular_data.to_csv('data/cardiovascular_data.csv', index=False)
    cancer_data.to_csv('data/cancer_data.csv', index=False)
    kidney_data.to_csv('data/kidney_disease_data.csv', index=False)
    liver_data.to_csv('data/liver_disease_data.csv', index=False)
    
    # Create combined features dataset
    print("Creating combined features dataset...")
    
    # Common features across all datasets
    common_features = ['age', 'gender', 'weight', 'height', 'bmi']
    
    # Combine all datasets
    combined_data = pd.DataFrame()
    
    # Start with diabetes data as base
    combined_data = diabetes_data[common_features].copy()
    
    # Add disease-specific features
    combined_data['glucose'] = diabetes_data['glucose']
    combined_data['hba1c'] = diabetes_data['hba1c']
    combined_data['systolic_bp'] = cardiovascular_data['systolic_bp']
    combined_data['diastolic_bp'] = cardiovascular_data['diastolic_bp']
    combined_data['cholesterol_total'] = cardiovascular_data['cholesterol_total']
    combined_data['creatinine'] = kidney_data['creatinine']
    combined_data['egfr'] = kidney_data['egfr']
    combined_data['alt'] = liver_data['alt']
    combined_data['ast'] = liver_data['ast']
    
    # Add all disease labels
    combined_data['diabetes'] = diabetes_data['diabetes_risk']
    combined_data['cardiovascular'] = cardiovascular_data['cardiovascular_risk']
    combined_data['cancer'] = cancer_data['cancer_risk']
    combined_data['kidney_disease'] = kidney_data['kidney_disease_risk']
    combined_data['liver_disease'] = liver_data['liver_disease_risk']
    
    # Save combined dataset
    combined_data.to_csv('data/combined_medical_data.csv', index=False)
    
    # Create training and testing splits
    print("Creating training/testing splits...")
    
    # Features (excluding disease labels)
    feature_cols = [col for col in combined_data.columns if col not in 
                   ['diabetes', 'cardiovascular', 'cancer', 'kidney_disease', 'liver_disease']]
    
    X = combined_data[feature_cols]
    
    # Create splits for each disease
    for disease in ['diabetes', 'cardiovascular', 'cancer', 'kidney_disease', 'liver_disease']:
        y = combined_data[disease]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Save splits
        X_train.to_csv(f'data/{disease}_X_train.csv', index=False)
        X_test.to_csv(f'data/{disease}_X_test.csv', index=False)
        y_train.to_csv(f'data/{disease}_y_train.csv', index=False)
        y_test.to_csv(f'data/{disease}_y_test.csv', index=False)
    
    print("Sample data generation completed!")
    print(f"Generated {len(combined_data)} samples")
    print("Files saved in 'data/' directory")
    
    return combined_data

if __name__ == "__main__":
    # Generate sample data
    combined_data = generate_combined_dataset()
    
    # Display summary
    print("\nDataset Summary:")
    print(f"Total samples: {len(combined_data)}")
    print(f"Features: {len(combined_data.columns) - 5}")  # Excluding 5 disease labels
    print(f"Diseases: 5")
    
    print("\nDisease prevalence:")
    for disease in ['diabetes', 'cardiovascular', 'cancer', 'kidney_disease', 'liver_disease']:
        prevalence = combined_data[disease].mean() * 100
        print(f"{disease.replace('_', ' ').title()}: {prevalence:.1f}%")
    
    print("\nFeature statistics:")
    print(combined_data.describe())
