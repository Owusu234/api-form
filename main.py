"""
FastAPI Diabetes Prediction Service
====================================
REST API for diabetes prediction using trained SVM model
"""

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
import joblib
import json
import numpy as np
import pandas as pd
from datetime import datetime
import os
import uvicorn


app = FastAPI(
    title="Diabetes Prediction API",
    description="Predict diabetes risk using machine learning model",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


model = None
feature_names = None
model_loaded = False


class PatientData(BaseModel):
    """Input schema for patient data"""
    Pregnancies: int = Field(..., ge=0, le=20, description="Number of pregnancies")
    Glucose: int = Field(..., ge=0, le=300, description="Plasma glucose concentration")
    BloodPressure: int = Field(..., ge=0, le=200, description="Diastolic blood pressure (mm Hg)")
    SkinThickness: int = Field(..., ge=0, le=100, description="Triceps skin fold thickness (mm)")
    Insulin: int = Field(..., ge=0, le=900, description="2-Hour serum insulin (mu U/ml)")
    BMI: float = Field(..., ge=0.0, le=70.0, description="Body mass index")
    DiabetesPedigreeFunction: float = Field(..., ge=0.0, le=2.5, description="Diabetes pedigree function")
    Age: int = Field(..., ge=1, le=120, description="Age (years)")

class PredictionResponse(BaseModel):
    """Response schema for prediction"""
    prediction: int
    probability: float
    diagnosis: str
    confidence: str
    timestamp: str
    model_version: str

class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    model_loaded: bool
    timestamp: str

class BatchPredictionRequest(BaseModel):
    """Batch prediction request"""
    patients: List[PatientData]

class BatchPredictionResponse(BaseModel):
    """Batch prediction response"""
    predictions: List[PredictionResponse]
    total_processed: int


def load_model():
    """Load the trained model and feature names"""
    global model, feature_names, model_loaded
    
    try:
        model_path = "diabetes_model.pkl"
        features_path = "feature_names.json"
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        if not os.path.exists(features_path):
            raise FileNotFoundError(f"Feature names file not found: {features_path}")
        
        model = joblib.load(model_path)
        
        with open(features_path, 'r') as f:
            feature_names = json.load(f)
        
        model_loaded = True
        print(f"✓ Model loaded successfully from {model_path}")
        print(f"✓ Features: {feature_names}")
        
    except Exception as e:
        model_loaded = False
        print(f"✗ Error loading model: {str(e)}")
        raise

# Load model on startup
@app.on_event("startup")
async def startup_event():
    """Initialize model on application startup"""
    load_model()


def calculate_engineered_features(patient_data: dict) -> dict:
    """Calculate engineered features (Glucose_Age, BMI_Insulin)"""
    data = patient_data.copy()
    data['Glucose_Age'] = (data['Glucose'] * data['Age']) / 100
    data['BMI_Insulin'] = (data['BMI'] * data['Insulin']) / 100
    return data

def get_confidence_level(probability: float) -> str:
    """Determine confidence level based on probability"""
    if probability >= 0.9:
        return "Very High"
    elif probability >= 0.7:
        return "High"
    elif probability >= 0.5:
        return "Medium"
    elif probability >= 0.3:
        return "Low"
    else:
        return "Very Low"


@app.get("/", response_model=HealthResponse)
async def root():
    """Root endpoint - Health check"""
    return HealthResponse(
        status="healthy" if model_loaded else "unhealthy",
        model_loaded=model_loaded,
        timestamp=datetime.now().isoformat()
    )

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy" if model_loaded else "unhealthy",
        model_loaded=model_loaded,
        timestamp=datetime.now().isoformat()
    )

@app.post("/predict", response_model=PredictionResponse)
async def predict_diabetes(patient: PatientData):
    """
    Predict diabetes for a single patient
    
    - **Pregnancies**: Number of pregnancies
    - **Glucose**: Plasma glucose concentration
    - **BloodPressure**: Diastolic blood pressure
    - **SkinThickness**: Triceps skin fold thickness
    - **Insulin**: 2-Hour serum insulin
    - **BMI**: Body mass index
    - **DiabetesPedigreeFunction**: Diabetes pedigree function
    - **Age**: Age in years
    """
    if not model_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Please check server logs."
        )
    
    try:
        # Convert to dict and add engineered features
        patient_dict = patient.model_dump()
        patient_dict = calculate_engineered_features(patient_dict)
        
        # Create DataFrame with correct column order
        patient_df = pd.DataFrame([patient_dict], columns=feature_names)
        
        # Make prediction
        prediction = int(model.predict(patient_df)[0])
        probabilities = model.predict_proba(patient_df)[0]
        probability = float(probabilities[1])  # Probability of class 1 (diabetic)
        
        # Determine diagnosis and confidence
        diagnosis = "Diabetic" if prediction == 1 else "Non-Diabetic"
        confidence = get_confidence_level(probability)
        
        return PredictionResponse(
            prediction=prediction,
            probability=round(probability, 4),
            diagnosis=diagnosis,
            confidence=confidence,
            timestamp=datetime.now().isoformat(),
            model_version="1.0.0"
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )

@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_diabetes_batch(request: BatchPredictionRequest):
    """
    Predict diabetes for multiple patients
    
    Accepts a list of patient records and returns predictions for all
    """
    if not model_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Please check server logs."
        )
    
    if len(request.patients) > 100:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Maximum 100 patients per batch request"
        )
    
    try:
        predictions = []
        
        for patient in request.patients:
            patient_dict = patient.model_dump()
            patient_dict = calculate_engineered_features(patient_dict)
            patient_df = pd.DataFrame([patient_dict], columns=feature_names)
            
            prediction = int(model.predict(patient_df)[0])
            probabilities = model.predict_proba(patient_df)[0]
            probability = float(probabilities[1])
            
            predictions.append(PredictionResponse(
                prediction=prediction,
                probability=round(probability, 4),
                diagnosis="Diabetic" if prediction == 1 else "Non-Diabetic",
                confidence=get_confidence_level(probability),
                timestamp=datetime.now().isoformat(),
                model_version="1.0.0"
            ))
        
        return BatchPredictionResponse(
            predictions=predictions,
            total_processed=len(predictions)
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch prediction failed: {str(e)}"
        )

@app.get("/model/info")
async def model_info():
    """Get information about the loaded model"""
    if not model_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )
    
    return {
        "model_type": type(model).__name__,
        "features": feature_names,
        "feature_count": len(feature_names),
        "version": "1.0.0",
        "loaded_at": datetime.now().isoformat()
    }


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler"""
    return {
        "error": "Internal Server Error",
        "detail": str(exc),
        "timestamp": datetime.now().isoformat()
    }


if __name__ == "__main__":

    
    port = int(os.environ.get("PORT", 8000))
    
    print(f" Starting server on 0.0.0.0:{port}")
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",  # Required: listen on all interfaces
        port=port,
        reload=False,    # Never use reload in production
        workers=1,       # Render handles horizontal scaling
        log_level="info",
        access_log=True
    )

