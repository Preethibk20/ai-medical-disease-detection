#!/usr/bin/env python3
"""
Demo script for AI Medical Disease Detection System
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Add utils and models to path
sys.path.append('utils')
sys.path.append('models')

from utils.data_processor import MedicalDataProcessor
from models.disease_models import MultiDiseaseDetector
from utils.model_prep import ensure_models_ready
from models.demo_models import DemoDiseaseDetector

def create_sample_patient():
    """Create a sample patient for demonstration"""
    print("👤 Creating sample patient data...")
    
    # Sample patient demographics
    demographics = {
        'age': 58,
        'gender': 'Male',
        'weight': 85.0,  # kg
        'height': 175.0,  # cm
        'systolic_bp': 145,
        'diastolic_bp': 95,
        'heart_rate': 78,
        'temperature': 37.2
    }
    
    # Calculate BMI
    bmi = demographics['weight'] / ((demographics['height']/100) ** 2)
    demographics['bmi'] = round(bmi, 1)
    
    # Sample lab results
    lab_results = {
        'glucose': 135,  # mg/dL (elevated)
        'hba1c': 6.8,   # % (elevated)
        'cholesterol_total': 220,  # mg/dL (elevated)
        'cholesterol_hdl': 45,     # mg/dL (low)
        'cholesterol_ldl': 150,    # mg/dL (elevated)
        'triglycerides': 180,      # mg/dL (elevated)
        'creatinine': 1.4,         # mg/dL (elevated)
        'egfr': 65,                # mL/min/1.73m² (reduced)
        'alt': 45,                 # U/L (elevated)
        'ast': 42,                 # U/L (elevated)
        'bilirubin': 1.2,          # mg/dL (normal)
        'albumin': 3.8,            # g/dL (low)
        'hemoglobin': 13.5,        # g/dL (normal)
        'white_blood_cells': 7.2,  # K/µL (normal)
        'platelets': 250           # K/µL (normal)
    }
    
    print("✅ Sample patient created:")
    print(f"   Age: {demographics['age']} years")
    print(f"   Gender: {demographics['gender']}")
    print(f"   BMI: {demographics['bmi']} (Overweight)")
    print(f"   Blood Pressure: {demographics['systolic_bp']}/{demographics['diastolic_bp']} mmHg (High)")
    print(f"   Glucose: {lab_results['glucose']} mg/dL (Elevated)")
    print(f"   HbA1c: {lab_results['hba1c']}% (Elevated)")
    
    return demographics, lab_results

def run_disease_detection_demo():
    """Run the complete disease detection demo"""
    print("\n🏥 AI Medical Disease Detection - Demo")
    print("=" * 50)
    
    # Step 1: Create sample patient
    demographics, lab_results = create_sample_patient()
    
    # Step 2: Initialize data processor
    print("\n🔧 Initializing data processor...")
    data_processor = MedicalDataProcessor()
    
    # Step 3: Process data
    print("📊 Processing patient data...")
    
    # Process demographics
    demo_df = pd.DataFrame([demographics])
    processed_demo = data_processor.process_patient_demographics(demo_df)
    
    # Process lab results
    lab_df = pd.DataFrame([lab_results])
    processed_lab = data_processor.process_lab_results(lab_df)
    
    # Combine modalities
    combined_features = data_processor.combine_modalities(
        None,  # No image for this demo
        processed_lab,
        processed_demo
    )
    
    print(f"✅ Combined features shape: {combined_features.shape}")
    
    # Step 4: Initialize and run disease detection
    print("\n🤖 Running disease detection...")
    
    # Use pretrained models when available; otherwise demo fallback
    disease_detector = ensure_models_ready()
    demo_fallback = DemoDiseaseDetector()
    
    # Get predictions
    try:
        predictions = disease_detector.predict_all_diseases(combined_features)
    except Exception:
        predictions = None
    if not predictions or any(v is None for v in predictions.items() if isinstance(predictions, dict)):
        # Build raw dicts for demo predictor
        predictions = demo_fallback.predict_all_diseases_from_raw(
            image_features=None,
            lab=lab_results,
            demo=demographics
        )
    
    # Step 5: Display results
    print("\n📊 Disease Detection Results:")
    print("-" * 40)
    
    results_summary = []
    
    for disease, risk in predictions.items():
        if risk is not None:
            risk_value = float(risk[0]) if hasattr(risk, '__len__') else float(risk)
            risk_percentage = risk_value * 100
            
            # Determine risk level
            if risk_percentage < 30:
                risk_level = "🟢 Low Risk"
                color = "green"
            elif risk_percentage < 70:
                risk_level = "🟡 Medium Risk"
                color = "orange"
            else:
                risk_level = "🔴 High Risk"
                color = "red"
            
            results_summary.append({
                'Disease': disease.replace('_', ' ').title(),
                'Risk Score': f"{risk_percentage:.1f}%",
                'Risk Level': risk_level,
                'Risk Value': risk_percentage
            })
            
            print(f"{disease.replace('_', ' ').title():<20} {risk_percentage:>6.1f}% {risk_level}")
    
    # Step 6: Generate recommendations
    print("\n💡 Medical Recommendations:")
    print("-" * 40)
    
    high_risk_diseases = [r for r in results_summary if r['Risk Value'] >= 70]
    medium_risk_diseases = [r for r in results_summary if 30 <= r['Risk Value'] < 70]
    
    if high_risk_diseases:
        print("⚠️  IMMEDIATE ATTENTION REQUIRED:")
        for disease in high_risk_diseases:
            print(f"   • {disease['Disease']}: Consult specialist immediately")
    
    if medium_risk_diseases:
        print("\nℹ️  MONITOR CLOSELY:")
        for disease in medium_risk_diseases:
            print(f"   • {disease['Disease']}: Regular check-ups recommended")
    
    # General recommendations
    print("\n📋 GENERAL HEALTH RECOMMENDATIONS:")
    print("   • Maintain healthy diet and exercise routine")
    print("   • Monitor blood pressure regularly")
    print("   • Reduce salt and saturated fat intake")
    print("   • Regular blood glucose monitoring")
    print("   • Annual comprehensive health check-up")
    
    return results_summary

def create_visualization(results_summary):
    """Create visualization of the results"""
    print("\n📈 Creating results visualization...")
    
    if not results_summary:
        print("❌ No results to visualize")
        return
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Bar chart of risk scores
    diseases = [r['Disease'] for r in results_summary]
    risk_values = [r['Risk Value'] for r in results_summary]
    colors = ['red' if r['Risk Value'] >= 70 else 'orange' if r['Risk Value'] >= 30 else 'green' 
              for r in results_summary]
    
    bars = ax1.bar(diseases, risk_values, color=colors, alpha=0.7)
    ax1.set_title('Disease Risk Assessment', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Risk Score (%)', fontsize=12)
    ax1.set_ylim(0, 100)
    
    # Add value labels on bars
    for bar, value in zip(bars, risk_values):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # Rotate x-axis labels
    ax1.tick_params(axis='x', rotation=45)
    
    # Pie chart of risk distribution
    risk_categories = {'Low Risk': 0, 'Medium Risk': 0, 'High Risk': 0}
    for result in results_summary:
        if result['Risk Value'] < 30:
            risk_categories['Low Risk'] += 1
        elif result['Risk Value'] < 70:
            risk_categories['Medium Risk'] += 1
        else:
            risk_categories['High Risk'] += 1
    
    # Only show categories with values > 0
    categories = [k for k, v in risk_categories.items() if v > 0]
    values = [risk_categories[k] for k in categories]
    colors_pie = ['green', 'orange', 'red']
    
    ax2.pie(values, labels=categories, colors=colors_pie[:len(categories)], 
            autopct='%1.0f', startangle=90)
    ax2.set_title('Risk Distribution', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    # Save plot
    os.makedirs('demo_output', exist_ok=True)
    plt.savefig('demo_output/disease_risk_assessment.png', dpi=300, bbox_inches='tight')
    print("✅ Visualization saved to demo_output/disease_risk_assessment.png")
    
    plt.show()

def run_comparison_demo():
    """Run comparison demo with multiple patients"""
    print("\n🔍 Comparison Demo - Multiple Patients")
    print("=" * 40)
    
    # Create multiple sample patients
    patients = [
        {
            'name': 'Patient A',
            'demographics': {'age': 35, 'gender': 'Female', 'weight': 60, 'height': 165, 'bmi': 22.0},
            'lab_results': {'glucose': 95, 'hba1c': 5.2, 'cholesterol_total': 180, 'creatinine': 0.8}
        },
        {
            'name': 'Patient B',
            'demographics': {'age': 65, 'gender': 'Male', 'weight': 90, 'height': 175, 'bmi': 29.4},
            'lab_results': {'glucose': 160, 'hba1c': 7.5, 'cholesterol_total': 250, 'creatinine': 1.6}
        },
        {
            'name': 'Patient C',
            'demographics': {'age': 45, 'gender': 'Female', 'weight': 70, 'height': 160, 'bmi': 27.3},
            'lab_results': {'glucose': 110, 'hba1c': 5.8, 'cholesterol_total': 200, 'creatinine': 0.9}
        }
    ]
    
    # Initialize data processor
    data_processor = MedicalDataProcessor()
    disease_detector = ensure_models_ready()
    demo_fallback = DemoDiseaseDetector()
    
    comparison_results = []
    
    for patient in patients:
        print(f"\n👤 Analyzing {patient['name']}...")
        
        # Process data
        demo_df = pd.DataFrame([patient['demographics']])
        lab_df = pd.DataFrame([patient['lab_results']])
        
        processed_demo = data_processor.process_patient_demographics(demo_df)
        processed_lab = data_processor.process_lab_results(lab_df)
        
        combined_features = data_processor.combine_modalities(
            None, processed_lab, processed_demo
        )
        
        # Get predictions
        try:
            predictions = disease_detector.predict_all_diseases(combined_features)
        except Exception:
            predictions = None
        if not predictions or any(v is None for v in predictions.items() if isinstance(predictions, dict)):
            predictions = demo_fallback.predict_all_diseases_from_raw(
                image_features=None,
                lab=patient['lab_results'],
                demo=patient['demographics']
            )
        
        # Store results
        patient_results = {'Patient': patient['name']}
        for disease, risk in predictions.items():
            if risk is not None:
                risk_value = float(risk[0]) if hasattr(risk, '__len__') else float(risk)
                patient_results[disease] = risk_value * 100
        
        comparison_results.append(patient_results)
    
    # Create comparison DataFrame
    comparison_df = pd.DataFrame(comparison_results)
    
    print("\n📊 Comparison Results:")
    print(comparison_df.to_string(index=False, float_format='%.1f'))
    
    # Save comparison results
    comparison_df.to_csv('demo_output/patient_comparison.csv', index=False)
    print("\n✅ Comparison results saved to demo_output/patient_comparison.csv")
    
    return comparison_df

def main():
    """Main demo function"""
    print("🚀 AI Medical Disease Detection System - Interactive Demo")
    print("=" * 60)
    
    while True:
        print("\n📋 Demo Options:")
        print("1. Single Patient Analysis")
        print("2. Multiple Patient Comparison")
        print("3. Exit")
        
        choice = input("\nSelect an option (1-3): ").strip()
        
        if choice == '1':
            print("\n" + "="*50)
            results = run_disease_detection_demo()
            create_visualization(results)
            
        elif choice == '2':
            print("\n" + "="*50)
            run_comparison_demo()
            
        elif choice == '3':
            print("\n👋 Thank you for using the AI Medical Disease Detection Demo!")
            break
            
        else:
            print("❌ Invalid choice. Please select 1, 2, or 3.")
    
    print("\n💡 To run the full system:")
    print("   • Web Interface: streamlit run app.py")
    print("   • API Server: uvicorn api:app --reload")
    print("   • Training: python train_models.py")

if __name__ == "__main__":
    main()
