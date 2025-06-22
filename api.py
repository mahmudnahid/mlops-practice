#!/usr/bin/env python3
"""
FastAPI ML Model Service
Serves the trained ML model via REST API
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import numpy as np
import json
import os
from typing import List, Dict, Any
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="ML Model API",
    description="Binary Classification Model Service",
    version="1.0.0"
)

# Global variables for model and metadata
model = None
model_metadata = {}

class PredictionRequest(BaseModel):
    features: List[float] = Field(..., description="List of feature values", min_items=10, max_items=10)
    
    class Config:
        schema_extra = {
            "example": {
                "features": [0.1, -0.5, 1.2, 0.8, -1.1, 0.3, -0.7, 1.5, 0.2, -0.9]
            }
        }

class PredictionResponse(BaseModel):
    prediction: int = Field(..., description="Predicted class (0 or 1)")
    probability: List[float] = Field(..., description="Class probabilities")
    model_version: str = Field(..., description="Model version timestamp")

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: str = None

def load_model(model_path: str = "models/model.joblib", metrics_path: str = "models/metrics.json"):
    """Load the trained model and its metadata"""
    global model, model_metadata
    
    try:
        # Load model
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        model = joblib.load(model_path)
        logger.info(f"Model loaded from: {model_path}")
        
        # Load metadata
        if os.path.exists(metrics_path):
            with open(metrics_path, 'r') as f:
                model_metadata = json.load(f)
            logger.info(f"Model metadata loaded from: {metrics_path}")
        else:
            logger.warning(f"Metadata file not found: {metrics_path}")
            model_metadata = {"timestamp": "unknown"}
        
        return True
    
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        return False

@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    success = load_model()
    if not success:
        logger.error("Failed to load model on startup")

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with basic info"""
    return {
        "message": "ML Model API",
        "version": "1.0.0",
        "status": "running"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy" if model is not None else "unhealthy",
        model_loaded=model is not None,
        model_version=model_metadata.get("timestamp", "unknown")
    )

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Make prediction using the loaded model"""
    global model, model_metadata
    
    # Check if model is loaded
    if model is None:
        raise HTTPException(
            status_code=503, 
            detail="Model not loaded. Please check server logs."
        )
    
    try:
        # Convert features to numpy array and reshape for single prediction
        features_array = np.array(request.features).reshape(1, -1)
        
        # Make prediction
        prediction = model.predict(features_array)[0]
        probabilities = model.predict_proba(features_array)[0].tolist()
        
        return PredictionResponse(
            prediction=int(prediction),
            probability=probabilities,
            model_version=model_metadata.get("timestamp", "unknown")
        )
    
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )

@app.post("/predict_batch")
async def predict_batch(features_list: List[List[float]]):
    """Make batch predictions"""
    global model
    
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please check server logs."
        )
    
    try:
        # Validate input
        for i, features in enumerate(features_list):
            if len(features) != 10:
                raise ValueError(f"Sample {i}: Expected 10 features, got {len(features)}")
        
        # Convert to numpy array
        features_array = np.array(features_list)
        
        # Make predictions
        predictions = model.predict(features_array).tolist()
        probabilities = model.predict_proba(features_array).tolist()
        
        return {
            "predictions": predictions,
            "probabilities": probabilities,
            "count": len(predictions),
            "model_version": model_metadata.get("timestamp", "unknown")
        }
    
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Batch prediction failed: {str(e)}"
        )

@app.get("/model_info")
async def get_model_info():
    """Get model information and metrics"""
    global model_metadata
    
    if not model_metadata:
        raise HTTPException(
            status_code=503,
            detail="Model metadata not available"
        )
    
    return {
        "model_metadata": model_metadata,
        "model_loaded": model is not None
    }

@app.post("/reload_model")
async def reload_model():
    """Reload the model (useful for model updates)"""
    success = load_model()
    
    if success:
        return {
            "message": "Model reloaded successfully",
            "model_version": model_metadata.get("timestamp", "unknown")
        }
    else:
        raise HTTPException(
            status_code=500,
            detail="Failed to reload model"
        )

if __name__ == "__main__":
    import uvicorn
    
    # Load model on startup
    load_model()
    
    # Run the API
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )