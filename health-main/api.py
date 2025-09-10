from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import numpy as np
import pandas as pd
import json
import io
import os
import sys

# Add utils and models to path
sys.path.append('utils')
sys.path.append('models')

from utils.data_processor import MedicalDataProcessor
from utils.model_prep import ensure_models_ready
from models.demo_models import DemoDiseaseDetector

# Initialize FastAPI app
app = FastAPI(
    title="AI Medical Disease Detection API",
    description="API for detecting chronic diseases using multimodal medical data",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components with pretrained models auto-load
data_processor = MedicalDataProcessor()
disease_detector = ensure_models_ready()
demo_detector = DemoDiseaseDetector()

# Pydantic models for request/response
class PatientDemographics(BaseModel):
    age: int
    gender: str
    weight: float
    height: float
    systolic_bp: Optional[int] = None
    diastolic_bp: Optional[int] = None
    heart_rate: Optional[int] = None
    temperature: Optional[float] = None
    medical_history: Optional[List[str]] = None
    medications: Optional[List[str]] = None

class LabResults(BaseModel):
    glucose: Optional[float] = None
    hba1c: Optional[float] = None
    cholesterol_total: Optional[float] = None
    cholesterol_hdl: Optional[float] = None
    cholesterol_ldl: Optional[float] = None
    triglycerides: Optional[float] = None
    creatinine: Optional[float] = None
    egfr: Optional[float] = None
    alt: Optional[float] = None
    ast: Optional[float] = None
    bilirubin: Optional[float] = None
    albumin: Optional[float] = None
    hemoglobin: Optional[float] = None
    white_blood_cells: Optional[float] = None
    platelets: Optional[float] = None

class DiseasePredictionRequest(BaseModel):
    demographics: PatientDemographics
    lab_results: Optional[LabResults] = None
    image_data: Optional[str] = None  # Base64 encoded image

class DiseasePredictionResponse(BaseModel):
    patient_id: str
    predictions: Dict[str, float]
    risk_levels: Dict[str, str]
    confidence_scores: Dict[str, float]
    recommendations: List[str]
    timestamp: str

class HealthReport(BaseModel):
    patient_id: str
    overall_health_score: float
    risk_factors: List[str]
    preventive_measures: List[str]
    follow_up_schedule: Dict[str, str]

# Health check endpoint
@app.get("/")
async def root():
    return {"message": "AI Medical Disease Detection API", "status": "healthy"}

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "models_loaded": os.path.exists('models/saved_models'),
        "supported_diseases": disease_detector.diseases,
        "api_version": "1.0.0"
    }

# Disease prediction endpoint
@app.post("/predict", response_model=DiseasePredictionResponse)
async def predict_diseases(request: DiseasePredictionRequest):
    """
    Predict diseases based on multimodal medical data
    """
    try:
        # Process demographics
        demo_data = request.demographics.dict()
        demo_df = pd.DataFrame([demo_data])
        processed_demo = data_processor.process_patient_demographics(demo_df)
        
        # Process lab results if provided
        processed_lab = None
        if request.lab_results:
            lab_data = request.lab_results.dict()
            lab_df = pd.DataFrame([lab_data])
            processed_lab = data_processor.process_lab_results(lab_df)
        
        # Process image if provided
        processed_image = None
        if request.image_data:
            # Decode base64 image
            try:
                import base64
                image_bytes = base64.b64decode(request.image_data)
                with open("temp_api_image", "wb") as f:
                    f.write(image_bytes)
                
                processed_image = data_processor.process_medical_image("temp_api_image")
                os.remove("temp_api_image")
            except Exception as e:
                print(f"Error processing image: {e}")
        
        # Combine modalities
        combined_features = data_processor.combine_modalities(
            processed_image, processed_lab, processed_demo
        )
        
        # Predict with pretrained; fallback to demo if needed
        try:
            predictions = disease_detector.predict_all_diseases(combined_features)
        except Exception:
            predictions = None
        if not predictions or any(v is None for v in predictions.values()):
            # Build raw dicts for demo predictor
            image_feats = None
            if processed_image is not None:
                mean_intensity = float(processed_image.mean())
                import cv2 as _cv2
                edge_map = _cv2.Canny((processed_image * 255).astype('uint8'), 50, 150)
                edge_density = float(edge_map.mean() / 255.0)
                image_feats = {
                    'mean_intensity': mean_intensity,
                    'edge_density': edge_density
                }
            lab_dict = None
            if processed_lab is not None:
                try:
                    lab_dict = processed_lab.to_dict('records')[0]
                except Exception:
                    lab_dict = None
            demo_dict = request.demographics.dict() if request and request.demographics else None
            predictions = demo_detector.predict_all_diseases_from_raw(image_feats, lab_dict, demo_dict)
        
        # Process results
        risk_levels = {}
        confidence_scores = {}
        recommendations = []
        
        for disease, risk in predictions.items():
            if risk is not None:
                risk_value = float(risk[0]) if hasattr(risk, '__len__') else float(risk)
                
                # Determine risk level
                if risk_value < 0.3:
                    risk_levels[disease] = "low"
                elif risk_value < 0.7:
                    risk_levels[disease] = "medium"
                else:
                    risk_levels[disease] = "high"
                
                # Calculate confidence (simplified)
                confidence_scores[disease] = min(0.95, 0.7 + risk_value * 0.25)
                
                # Generate recommendations
                if risk_value >= 0.7:
                    recommendations.append(f"Immediate consultation required for {disease.replace('_', ' ')}")
                elif risk_value >= 0.3:
                    recommendations.append(f"Regular monitoring recommended for {disease.replace('_', ' ')}")
        
        # Generate patient ID
        import uuid
        patient_id = str(uuid.uuid4())[:8]
        
        # Convert predictions to float values
        predictions_float = {}
        for disease, risk in predictions.items():
            if risk is not None:
                predictions_float[disease] = float(risk[0]) if hasattr(risk, '__len__') else float(risk)
        
        return DiseasePredictionResponse(
            patient_id=patient_id,
            predictions=predictions_float,
            risk_levels=risk_levels,
            confidence_scores=confidence_scores,
            recommendations=recommendations,
            timestamp=pd.Timestamp.now().isoformat()
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

# File upload endpoint for medical images
@app.post("/upload-image")
async def upload_medical_image(file: UploadFile = File(...)):
    """
    Upload and process medical images
    """
    try:
        # Validate file type
        allowed_types = ['image/png', 'image/jpeg', 'image/jpg', 'application/dicom']
        if file.content_type not in allowed_types:
            raise HTTPException(status_code=400, detail="Invalid file type")
        
        # Read file content
        content = await file.read()
        
        # Save temporarily and process
        temp_path = f"temp_upload_{file.filename}"
        with open(temp_path, "wb") as f:
            f.write(content)
        
        # Process image
        processed_image = data_processor.process_medical_image(temp_path)
        
        # Clean up
        os.remove(temp_path)
        
        if processed_image is not None:
            # Convert to base64 for response
            import base64
            import cv2
            
            # Normalize image for display
            img_normalized = ((processed_image - processed_image.min()) / 
                            (processed_image.max() - processed_image.min()) * 255).astype(np.uint8)
            
            # Encode as base64
            _, buffer = cv2.imencode('.png', img_normalized)
            img_base64 = base64.b64encode(buffer).decode('utf-8')
            
            return {
                "message": "Image processed successfully",
                "filename": file.filename,
                "processed_image": img_base64,
                "image_shape": processed_image.shape
            }
        else:
            raise HTTPException(status_code=400, detail="Failed to process image")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Image processing error: {str(e)}")

# Batch prediction endpoint
@app.post("/predict-batch")
async def predict_batch_diseases(requests: List[DiseasePredictionRequest]):
    """
    Predict diseases for multiple patients
    """
    try:
        results = []
        
        for i, request in enumerate(requests):
            try:
                # Process each request
                demo_data = request.demographics.dict()
                demo_df = pd.DataFrame([demo_data])
                processed_demo = data_processor.process_patient_demographics(demo_df)
                
                # Process lab results if provided
                processed_lab = None
                if request.lab_results:
                    lab_data = request.lab_results.dict()
                    lab_df = pd.DataFrame([lab_data])
                    processed_lab = data_processor.process_lab_results(lab_df)
                
                # Combine modalities
                combined_features = data_processor.combine_modalities(
                    None, processed_lab, processed_demo
                )
                
                # Predict
                disease_detector.initialize_models()
                predictions = disease_detector.predict_all_diseases(combined_features)
                
                # Convert predictions
                predictions_float = {}
                for disease, risk in predictions.items():
                    if risk is not None:
                        predictions_float[disease] = float(risk[0]) if hasattr(risk, '__len__') else float(risk)
                
                results.append({
                    "patient_index": i,
                    "predictions": predictions_float,
                    "status": "success"
                })
                
            except Exception as e:
                results.append({
                    "patient_index": i,
                    "error": str(e),
                    "status": "failed"
                })
        
        return {"batch_results": results}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")

# Health report generation endpoint
@app.post("/generate-report", response_model=HealthReport)
async def generate_health_report(request: DiseasePredictionRequest):
    """
    Generate comprehensive health report
    """
    try:
        # Get predictions first
        demo_data = request.demographics.dict()
        demo_df = pd.DataFrame([demo_data])
        processed_demo = data_processor.process_patient_demographics(demo_df)
        
        processed_lab = None
        if request.lab_results:
            lab_data = request.lab_results.dict()
            lab_df = pd.DataFrame([lab_data])
            processed_lab = data_processor.process_lab_results(lab_df)
        
        combined_features = data_processor.combine_modalities(
            None, processed_lab, processed_demo
        )
        
        disease_detector.initialize_models()
        predictions = disease_detector.predict_all_diseases(combined_features)
        
        # Calculate overall health score
        if predictions:
            valid_predictions = [v for v in predictions.values() if v is not None]
            if valid_predictions:
                avg_risk = np.mean(valid_predictions)
                overall_health_score = max(0, 100 - (avg_risk * 100))
            else:
                overall_health_score = 50.0
        else:
            overall_health_score = 50.0
        
        # Identify risk factors
        risk_factors = []
        if request.demographics.age > 65:
            risk_factors.append("Advanced age")
        if request.demographics.weight / ((request.demographics.height/100) ** 2) > 30:
            risk_factors.append("High BMI")
        if request.demographics.systolic_bp and request.demographics.systolic_bp > 140:
            risk_factors.append("High blood pressure")
        
        # Generate preventive measures
        preventive_measures = [
            "Regular exercise (30 minutes daily)",
            "Balanced diet with fruits and vegetables",
            "Regular health check-ups",
            "Stress management techniques"
        ]
        
        # Follow-up schedule
        follow_up_schedule = {
            "diabetes": "Every 3 months" if any(p > 0.5 for p in predictions.values() if p is not None) else "Annually",
            "cardiovascular": "Every 6 months" if any(p > 0.4 for p in predictions.values() if p is not None) else "Annually",
            "general": "Every 6 months" if overall_health_score < 70 else "Annually"
        }
        
        # Generate patient ID
        import uuid
        patient_id = str(uuid.uuid4())[:8]
        
        return HealthReport(
            patient_id=patient_id,
            overall_health_score=overall_health_score,
            risk_factors=risk_factors,
            preventive_measures=preventive_measures,
            follow_up_schedule=follow_up_schedule
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Report generation error: {str(e)}")

# Model management endpoints
@app.get("/models/status")
async def get_models_status():
    """
    Get status of all disease detection models
    """
    try:
        models_status = {}
        
        for disease in disease_detector.diseases:
            model_path = f"models/saved_models/{disease}"
            models_status[disease] = {
                "loaded": os.path.exists(model_path),
                "path": model_path,
                "last_updated": None  # Would be implemented with actual model metadata
            }
        
        return {
            "models_status": models_status,
            "total_models": len(disease_detector.diseases),
            "models_loaded": sum(1 for status in models_status.values() if status["loaded"])
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting models status: {str(e)}")

@app.post("/models/reload")
async def reload_models():
    """
    Reload all disease detection models
    """
    try:
        base_path = "models/saved_models"
        if os.path.exists(base_path):
            disease_detector.load_all_models(base_path)
            return {"message": "Models reloaded successfully"}
        else:
            raise HTTPException(status_code=404, detail="No saved models found")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reloading models: {str(e)}")

# Data processing endpoints
@app.post("/process/lab-results")
async def process_lab_results(lab_data: LabResults):
    """
    Process laboratory results data
    """
    try:
        lab_df = pd.DataFrame([lab_data.dict()])
        processed_lab = data_processor.process_lab_results(lab_df)
        
        return {
            "message": "Lab results processed successfully",
            "processed_data": processed_lab.to_dict('records')[0],
            "shape": processed_lab.shape
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lab results processing error: {str(e)}")

@app.post("/process/demographics")
async def process_demographics(demographics: PatientDemographics):
    """
    Process patient demographics data
    """
    try:
        demo_df = pd.DataFrame([demographics.dict()])
        processed_demo = data_processor.process_patient_demographics(demo_df)
        
        return {
            "message": "Demographics processed successfully",
            "processed_data": processed_demo.to_dict('records')[0],
            "shape": processed_demo.shape
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Demographics processing error: {str(e)}")

# Statistics and analytics endpoints
@app.get("/analytics/summary")
async def get_analytics_summary():
    """
    Get system analytics summary
    """
    try:
        return {
            "total_diseases_supported": len(disease_detector.diseases),
            "supported_diseases": disease_detector.diseases,
            "data_modalities": ["images", "lab_results", "demographics"],
            "ai_models": ["random_forest", "gradient_boosting", "logistic_regression", 
                         "svm", "mlp", "xgboost", "lightgbm", "deep_learning"],
            "api_endpoints": [
                "/predict", "/predict-batch", "/generate-report", 
                "/upload-image", "/process/lab-results", "/process/demographics"
            ]
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting analytics: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
