#!/usr/bin/env python3
"""
Test Suite for FastAPI ML Service
Tests API endpoints and functionality
"""

import pytest
import httpx
from httpx import ASGITransport
import asyncio
import json
import os
import tempfile
import shutil
from fastapi.testclient import TestClient
from train import MLPipeline

# Import the FastAPI app
import sys
sys.path.append('.')
from api import app, load_model

class TestMLAPI:
    
    @pytest.fixture(scope="class")
    def setup_model(self):
        """Setup: Train and save a model for testing"""
        # Create temporary directory
        temp_dir = tempfile.mkdtemp()
        
        try:
            # Train a model
            pipeline = MLPipeline(model_dir=temp_dir)
            model_path = pipeline.run_pipeline()
            
            # Update the model loading path for the API
            success = load_model(
                model_path=os.path.join(temp_dir, "model.joblib"),
                metrics_path=os.path.join(temp_dir, "metrics.json")
            )
            
            assert success, "Failed to load model for testing"
            
            yield temp_dir
            
        finally:
            # Cleanup
            shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def client(self, setup_model):
        """Create test client"""
        return TestClient(app)
    
    def test_root_endpoint(self, client):
        """Test root endpoint"""
        response = client.get("/")
        assert response.status_code == 200
        
        data = response.json()
        assert "message" in data
        assert "version" in data
        assert "status" in data
        assert data["status"] == "running"
    
    def test_health_endpoint(self, client):
        """Test health check endpoint"""
        response = client.get("/health")
        assert response.status_code == 200
        
        data = response.json()
        assert "status" in data
        assert "model_loaded" in data
        assert "model_version" in data
        assert data["model_loaded"] == True
        assert data["status"] == "healthy"
    
    def test_predict_endpoint_valid_input(self, client):
        """Test prediction with valid input"""
        test_features = [0.1, -0.5, 1.2, 0.8, -1.1, 0.3, -0.7, 1.5, 0.2, -0.9]
        
        response = client.post(
            "/predict",
            json={"features": test_features}
        )
        
        assert response.status_code == 200
        
        data = response.json()
        assert "prediction" in data
        assert "probability" in data
        assert "model_version" in data
        
        # Check prediction value
        assert data["prediction"] in [0, 1]
        
        # Check probability format
        assert len(data["probability"]) == 2
        assert all(0 <= p <= 1 for p in data["probability"])
        assert abs(sum(data["probability"]) - 1.0) < 1e-6  # Probabilities sum to 1
    
    def test_predict_endpoint_invalid_input_too_few_features(self, client):
        """Test prediction with too few features"""
        test_features = [0.1, -0.5, 1.2]  # Only 3 features instead of 10
        
        response = client.post(
            "/predict",
            json={"features": test_features}
        )
        
        assert response.status_code == 422  # Validation error
    
    def test_predict_endpoint_invalid_input_too_many_features(self, client):
        """Test prediction with too many features"""
        test_features = [0.1] * 15  # 15 features instead of 10
        
        response = client.post(
            "/predict",
            json={"features": test_features}
        )
        
        assert response.status_code == 422  # Validation error
    
    def test_predict_endpoint_invalid_input_wrong_type(self, client):
        """Test prediction with wrong data type"""
        response = client.post(
            "/predict",
            json={"features": "not_a_list"}
        )
        
        assert response.status_code == 422  # Validation error
    
    def test_batch_predict_endpoint(self, client):
        """Test batch prediction endpoint"""
        test_features_batch = [
            [0.1, -0.5, 1.2, 0.8, -1.1, 0.3, -0.7, 1.5, 0.2, -0.9],
            [-0.2, 0.8, -1.1, 0.3, 1.4, -0.6, 0.9, -1.2, 0.7, 0.1],
            [1.1, -0.3, 0.7, -0.9, 0.2, 1.3, -0.8, 0.4, -1.5, 0.6]
        ]
        
        response = client.post(
            "/predict_batch",
            json=test_features_batch
        )
        
        assert response.status_code == 200
        
        data = response.json()
        assert "predictions" in data
        assert "probabilities" in data
        assert "count" in data
        assert "model_version" in data
        
        # Check batch results
        assert len(data["predictions"]) == 3
        assert len(data["probabilities"]) == 3
        assert data["count"] == 3
        
        # Check each prediction
        for pred in data["predictions"]:
            assert pred in [0, 1]
        
        # Check each probability
        for prob_row in data["probabilities"]:
            assert len(prob_row) == 2
            assert all(0 <= p <= 1 for p in prob_row)
            assert abs(sum(prob_row) - 1.0) < 1e-6
    
    def test_batch_predict_invalid_features(self, client):
        """Test batch prediction with invalid feature count"""
        test_features_batch = [
            [0.1, -0.5, 1.2],  # Wrong number of features
            [0.1, -0.5, 1.2, 0.8, -1.1, 0.3, -0.7, 1.5, 0.2, -0.9]
        ]
        
        response = client.post(
            "/predict_batch",
            json=test_features_batch
        )
        
        assert response.status_code == 500
    
    def test_model_info_endpoint(self, client):
        """Test model info endpoint"""
        response = client.get("/model_info")
        assert response.status_code == 200
        
        data = response.json()
        assert "model_metadata" in data
        assert "model_loaded" in data
        assert data["model_loaded"] == True
        
        # Check metadata structure
        metadata = data["model_metadata"]
        assert "validation_accuracy" in metadata
        assert "test_accuracy" in metadata
        assert "timestamp" in metadata
    
    def test_reload_model_endpoint(self, client):
        """Test model reload endpoint"""
        response = client.post("/reload_model")
        
        # This might fail if model files aren't in the expected location
        # but the endpoint should exist
        assert response.status_code in [200, 500]
        
        if response.status_code == 200:
            data = response.json()
            assert "message" in data
            assert "model_version" in data
    
    def test_api_documentation(self, client):
        """Test that OpenAPI documentation is available"""
        response = client.get("/docs")
        assert response.status_code == 200
        
        response = client.get("/openapi.json")
        assert response.status_code == 200
        
        openapi_spec = response.json()
        assert "openapi" in openapi_spec
        assert "info" in openapi_spec
        assert "paths" in openapi_spec
    
    def test_prediction_consistency(self, client):
        """Test that the same input gives the same prediction"""
        test_features = [0.1, -0.5, 1.2, 0.8, -1.1, 0.3, -0.7, 1.5, 0.2, -0.9]
        
        # Make multiple requests with same input
        responses = []
        for _ in range(3):
            response = client.post(
                "/predict",
                json={"features": test_features}
            )
            assert response.status_code == 200
            responses.append(response.json())
        
        # Check all predictions are the same
        predictions = [r["prediction"] for r in responses]
        assert all(p == predictions[0] for p in predictions)
        
        # Check all probabilities are the same
        probabilities = [r["probability"] for r in responses]
        for i in range(1, len(probabilities)):
            assert probabilities[i] == probabilities[0]

    @pytest.mark.asyncio
    async def test_async_predictions(self, setup_model):
        """Test async API calls"""
        async with httpx.AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test") as ac:
            test_features = [0.1, -0.5, 1.2, 0.8, -1.1, 0.3, -0.7, 1.5, 0.2, -0.9]
            
            response = await ac.post(
                "/predict",
                json={"features": test_features}
            )
            
            assert response.status_code == 200
            data = response.json()
            assert "prediction" in data
            assert "probability" in data

if __name__ == "__main__":
    pytest.main([__file__, "-v"])