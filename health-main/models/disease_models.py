import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import xgboost as xgb
import lightgbm as lgb
import tensorflow as tf
from tensorflow.keras import layers, models
import joblib
import os
from sklearn.preprocessing import LabelEncoder

class DiseaseDetectionModel:
    def __init__(self, disease_type):
        self.disease_type = disease_type
        self.models = {}
        self.ensemble_weights = None
        self.feature_names = None
        
    def create_models(self):
        """Create various ML models for ensemble"""
        self.models = {
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'gradient_boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
            'logistic_regression': LogisticRegression(random_state=42, max_iter=1000),
            'svm': SVC(probability=True, random_state=42),
            'mlp': MLPClassifier(hidden_layer_sizes=(100, 50), random_state=42, max_iter=1000),
            'xgboost': xgb.XGBClassifier(random_state=42),
            'lightgbm': lgb.LGBMClassifier(random_state=42)
        }
        
    def create_deep_learning_model(self, input_shape, num_classes=2):
        """Create a deep neural network for medical data"""
        model = models.Sequential([
            layers.Dense(256, activation='relu', input_shape=input_shape),
            layers.Dropout(0.3),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train_models(self, X_train, y_train, X_val=None, y_val=None):
        """Train all models"""
        if X_val is None:
            X_train, X_val, y_train, y_val = train_test_split(
                X_train, y_train, test_size=0.2, random_state=42
            )
        
        # Train traditional ML models
        for name, model in self.models.items():
            print(f"Training {name}...")
            model.fit(X_train, y_train)
        
        # Train deep learning model
        if len(X_train.shape) == 1:
            X_train = X_train.reshape(-1, 1)
        if len(X_val.shape) == 1:
            X_val = X_val.reshape(-1, 1)
            
        self.deep_model = self.create_deep_learning_model((X_train.shape[1],))
        
        # Convert labels to integers if needed
        if isinstance(y_train[0], str):
            label_encoder = LabelEncoder()
            y_train_encoded = label_encoder.fit_transform(y_train)
            y_val_encoded = label_encoder.transform(y_val)
        else:
            y_train_encoded = y_train
            y_val_encoded = y_val
        
        self.deep_model.fit(
            X_train, y_train_encoded,
            validation_data=(X_val, y_val_encoded),
            epochs=50,
            batch_size=32,
            verbose=0
        )
        
        # Calculate ensemble weights based on validation performance
        self.calculate_ensemble_weights(X_val, y_val)
        
    def calculate_ensemble_weights(self, X_val, y_val):
        """Calculate optimal weights for ensemble prediction"""
        predictions = {}
        scores = {}
        
        # Get predictions from traditional models
        for name, model in self.models.items():
            pred = model.predict_proba(X_val)[:, 1]
            predictions[name] = pred
            scores[name] = accuracy_score(y_val, model.predict(X_val))
        
        # Get predictions from deep learning model
        if hasattr(self, 'deep_model'):
            deep_pred = self.deep_model.predict(X_val)
            if len(deep_pred.shape) > 1:
                deep_pred = deep_pred[:, 1]
            predictions['deep_learning'] = deep_pred
            scores['deep_learning'] = accuracy_score(y_val, np.argmax(deep_pred, axis=1))
        
        # Calculate weights based on validation accuracy
        total_score = sum(scores.values())
        self.ensemble_weights = {name: score/total_score for name, score in scores.items()}
        
    def predict(self, X, use_ensemble=True):
        """Make predictions using ensemble or individual models"""
        if use_ensemble and self.ensemble_weights:
            predictions = {}
            
            # Get predictions from traditional models
            for name, model in self.models.items():
                pred = model.predict_proba(X)[:, 1]
                predictions[name] = pred
            
            # Get predictions from deep learning model
            if hasattr(self, 'deep_model'):
                deep_pred = self.deep_model.predict(X)
                if len(deep_pred.shape) > 1:
                    deep_pred = deep_pred[:, 1]
                predictions['deep_learning'] = deep_pred
            
            # Weighted ensemble prediction
            final_pred = np.zeros(len(X))
            for name, pred in predictions.items():
                final_pred += self.ensemble_weights[name] * pred
            
            return final_pred
        else:
            # Use best individual model
            best_model_name = max(self.ensemble_weights, key=self.ensemble_weights.get)
            if best_model_name == 'deep_learning':
                pred = self.deep_model.predict(X)
                if len(pred.shape) > 1:
                    pred = pred[:, 1]
                return pred
            else:
                return self.models[best_model_name].predict_proba(X)[:, 1]
    
    def evaluate(self, X_test, y_test):
        """Evaluate model performance"""
        y_pred = self.predict(X_test)
        y_pred_binary = (y_pred > 0.5).astype(int)
        
        accuracy = accuracy_score(y_test, y_pred_binary)
        report = classification_report(y_test, y_pred_binary)
        conf_matrix = confusion_matrix(y_test, y_pred_binary)
        
        return {
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': conf_matrix,
            'predictions': y_pred
        }
    
    def save_models(self, save_path):
        """Save all trained models"""
        os.makedirs(save_path, exist_ok=True)
        
        # Save traditional ML models
        for name, model in self.models.items():
            joblib.dump(model, os.path.join(save_path, f'{name}.pkl'))
        
        # Save deep learning model
        if hasattr(self, 'deep_model'):
            self.deep_model.save(os.path.join(save_path, 'deep_model.h5'))
        
        # Save ensemble weights
        joblib.dump(self.ensemble_weights, os.path.join(save_path, 'ensemble_weights.pkl'))
    
    def load_models(self, load_path):
        """Load trained models"""
        # Ensure model definitions exist before loading
        if not self.models:
            self.create_models()
        
        # Load traditional ML models
        for name in list(self.models.keys()):
            model_path = os.path.join(load_path, f'{name}.pkl')
            if os.path.exists(model_path):
                self.models[name] = joblib.load(model_path)
        
        # Load deep learning model
        deep_model_path = os.path.join(load_path, 'deep_model.h5')
        if os.path.exists(deep_model_path):
            self.deep_model = tf.keras.models.load_model(deep_model_path)
        
        # Load ensemble weights
        weights_path = os.path.join(load_path, 'ensemble_weights.pkl')
        if os.path.exists(weights_path):
            self.ensemble_weights = joblib.load(weights_path)

class MultiDiseaseDetector:
    def __init__(self):
        self.disease_models = {}
        self.diseases = [
            'diabetes', 'cardiovascular', 'cancer', 'kidney_disease', 'liver_disease'
        ]
        
    def initialize_models(self):
        """Initialize models for all diseases"""
        for disease in self.diseases:
            self.disease_models[disease] = DiseaseDetectionModel(disease)
            self.disease_models[disease].create_models()
    
    def train_all_models(self, data_dict):
        """Train models for all diseases"""
        for disease in self.diseases:
            if disease in data_dict:
                print(f"Training {disease} model...")
                X = data_dict[disease]['features']
                y = data_dict[disease]['labels']
                self.disease_models[disease].train_models(X, y)
    
    def predict_all_diseases(self, features):
        """Predict all diseases for given features"""
        predictions = {}
        for disease, model in self.disease_models.items():
            try:
                pred = model.predict(features)
                predictions[disease] = pred
            except Exception as e:
                print(f"Error predicting {disease}: {e}")
                predictions[disease] = None
        
        return predictions
    
    def save_all_models(self, base_path):
        """Save all disease models"""
        for disease, model in self.disease_models.items():
            save_path = os.path.join(base_path, disease)
            model.save_models(save_path)
    
    def load_all_models(self, base_path):
        """Load all disease models"""
        for disease in self.diseases:
            load_path = os.path.join(base_path, disease)
            if os.path.exists(load_path):
                self.disease_models[disease] = DiseaseDetectionModel(disease)
                # Ensure model definitions exist before loading
                self.disease_models[disease].create_models()
                self.disease_models[disease].load_models(load_path)
