#!/usr/bin/env python3
"""
Training script for AI Medical Disease Detection Models
"""

import os
import sys
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

# Add utils and models to path
sys.path.append('utils')
sys.path.append('models')

from utils.data_processor import MedicalDataProcessor
from models.disease_models import MultiDiseaseDetector

def load_training_data():
    """Load training data for all diseases"""
    print("Loading training data...")
    
    data_dir = 'data'
    if not os.path.exists(data_dir):
        print("Data directory not found. Please run data/generate_sample_data.py first.")
        return None
    
    training_data = {}
    
    diseases = ['diabetes', 'cardiovascular', 'cancer', 'kidney_disease', 'liver_disease']
    
    for disease in diseases:
        try:
            X_train_path = os.path.join(data_dir, f'{disease}_X_train.csv')
            y_train_path = os.path.join(data_dir, f'{disease}_y_train.csv')
            
            if os.path.exists(X_train_path) and os.path.exists(y_train_path):
                X_train = pd.read_csv(X_train_path)
                y_train = pd.read_csv(y_train_path)
                
                # Handle single column DataFrames
                if len(y_train.columns) == 1:
                    y_train = y_train.iloc[:, 0]
                else:
                    y_train = y_train.iloc[:, 0]  # Take first column
                
                training_data[disease] = {
                    'features': X_train,
                    'labels': y_train
                }
                
                print(f"Loaded {disease} data: {X_train.shape[0]} samples, {X_train.shape[1]} features")
            else:
                print(f"Missing data files for {disease}")
                
        except Exception as e:
                            print(f"Error loading {disease} data: {e}")
    
    return training_data

def train_models(training_data):
    """Train all disease detection models"""
    print("\nStarting model training...")
    
    # Initialize disease detector
    disease_detector = MultiDiseaseDetector()
    disease_detector.initialize_models()
    
    # Train models for each disease
    for disease, data in training_data.items():
        print(f"\nTraining {disease} models...")
        
        try:
            X = data['features'].values
            y = data['labels'].values
            
            # Train models
            disease_detector.disease_models[disease].train_models(X, y)
            
            print(f"{disease} models trained successfully")
            
        except Exception as e:
            print(f"Error training {disease} models: {e}")
    
    return disease_detector

def evaluate_models(disease_detector, training_data):
    """Evaluate all trained models"""
    print("\nEvaluating models...")
    
    results = {}
    
    for disease, data in training_data.items():
        print(f"\nEvaluating {disease} models...")
        
        try:
            X = data['features'].values
            y = data['labels'].values
            
            # Get predictions
            predictions = disease_detector.disease_models[disease].predict(X)
            
            # Convert to binary predictions
            y_pred_binary = (predictions > 0.5).astype(int)
            
            # Calculate metrics
            accuracy = np.mean(y_pred_binary == y)
            auc = roc_auc_score(y, predictions)
            
            # Classification report
            report = classification_report(y, y_pred_binary, output_dict=True)
            
            # Confusion matrix
            cm = confusion_matrix(y, y_pred_binary)
            
            results[disease] = {
                'accuracy': accuracy,
                'auc': auc,
                'precision': report['weighted avg']['precision'],
                'recall': report['weighted avg']['recall'],
                'f1_score': report['weighted avg']['f1-score'],
                'confusion_matrix': cm,
                'predictions': predictions,
                'true_labels': y
            }
            
            print(f"{disease} - Accuracy: {accuracy:.3f}, AUC: {auc:.3f}")
            
        except Exception as e:
            print(f"Error evaluating {disease} models: {e}")
    
    return results

def save_models(disease_detector, save_path='models/saved_models'):
    """Save all trained models"""
    print(f"\nSaving models to {save_path}...")
    
    try:
        disease_detector.save_all_models(save_path)
        print("Models saved successfully")
    except Exception as e:
        print(f"Error saving models: {e}")

def generate_training_report(results, save_path='training_report'):
    """Generate comprehensive training report"""
    print(f"\nGenerating training report...")
    
    os.makedirs(save_path, exist_ok=True)
    
    # Summary statistics
    summary_data = []
    for disease, metrics in results.items():
        summary_data.append({
            'Disease': disease.replace('_', ' ').title(),
            'Accuracy': f"{metrics['accuracy']:.3f}",
            'AUC': f"{metrics['auc']:.3f}",
            'Precision': f"{metrics['precision']:.3f}",
            'Recall': f"{metrics['recall']:.3f}",
            'F1-Score': f"{metrics['f1_score']:.3f}"
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(os.path.join(save_path, 'model_performance_summary.csv'), index=False)
    
    # Create performance visualization
    plt.figure(figsize=(15, 10))
    
    # Accuracy comparison
    plt.subplot(2, 3, 1)
    diseases = [d.replace('_', ' ').title() for d in results.keys()]
    accuracies = [results[d]['accuracy'] for d in results.keys()]
    plt.bar(diseases, accuracies, color='skyblue')
    plt.title('Model Accuracy Comparison')
    plt.ylabel('Accuracy')
    plt.xticks(rotation=45)
    
    # AUC comparison
    plt.subplot(2, 3, 2)
    aucs = [results[d]['auc'] for d in results.keys()]
    plt.bar(diseases, aucs, color='lightgreen')
    plt.title('Model AUC Comparison')
    plt.ylabel('AUC')
    plt.xticks(rotation=45)
    
    # F1-Score comparison
    plt.subplot(2, 3, 3)
    f1_scores = [results[d]['f1_score'] for d in results.keys()]
    plt.bar(diseases, f1_scores, color='lightcoral')
    plt.title('Model F1-Score Comparison')
    plt.ylabel('F1-Score')
    plt.xticks(rotation=45)
    
    # Confusion matrices
    for i, (disease, metrics) in enumerate(results.items()):
        plt.subplot(2, 3, i + 4)
        cm = metrics['confusion_matrix']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'{disease.replace("_", " ").title()} Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'model_performance_visualization.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save detailed results
    with open(os.path.join(save_path, 'detailed_results.txt'), 'w') as f:
        f.write("AI Medical Disease Detection - Training Results\n")
        f.write("=" * 50 + "\n\n")
        
        for disease, metrics in results.items():
            f.write(f"{disease.replace('_', ' ').title()}:\n")
            f.write(f"  Accuracy: {metrics['accuracy']:.3f}\n")
            f.write(f"  AUC: {metrics['auc']:.3f}\n")
            f.write(f"  Precision: {metrics['precision']:.3f}\n")
            f.write(f"  Recall: {metrics['recall']:.3f}\n")
            f.write(f"  F1-Score: {metrics['f1_score']:.3f}\n")
            f.write(f"  Confusion Matrix:\n{metrics['confusion_matrix']}\n\n")
    
    print(f"Training report saved to {save_path}")
    
    return summary_df

def main():
    """Main training pipeline"""
    print("AI Medical Disease Detection - Model Training Pipeline")
    print("=" * 60)
    
    # Step 1: Load training data
    training_data = load_training_data()
    if not training_data:
        print("No training data available. Exiting.")
        return
    
    print(f"\nLoaded data for {len(training_data)} diseases")
    
    # Step 2: Train models
    disease_detector = train_models(training_data)
    
    # Step 3: Evaluate models
    results = evaluate_models(disease_detector, training_data)
    
    if not results:
        print("No results to report. Exiting.")
        return
    
    # Step 4: Save models
    save_models(disease_detector)
    
    # Step 5: Generate report
    summary_df = generate_training_report(results)
    
    # Final summary
    print("\nTraining Pipeline Completed!")
    print("=" * 40)
    
    print("\nModel Performance Summary:")
    print(summary_df.to_string(index=False))
    
    # Calculate overall performance
    avg_accuracy = np.mean([r['accuracy'] for r in results.values()])
    avg_auc = np.mean([r['auc'] for r in results.values()])
    
    print(f"\nOverall Performance:")
    print(f"  Average Accuracy: {avg_accuracy:.3f}")
    print(f"  Average AUC: {avg_auc:.3f}")
    
    print(f"\nModels saved to: models/saved_models/")
    print(f"Report saved to: training_report/")
    
    print("\nReady to use the AI Medical Disease Detection System!")

if __name__ == "__main__":
    main()
