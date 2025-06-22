#!/usr/bin/env python3
"""
Test Suite for ML Pipeline
Tests training pipeline and model functionality
"""

import pytest
import pandas as pd
import numpy as np
import os
import json
import tempfile
import shutil
from train import MLPipeline

class TestMLPipeline:
    
    @pytest.fixture
    def temp_model_dir(self):
        """Create temporary directory for testing"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def pipeline(self, temp_model_dir):
        """Create pipeline instance for testing"""
        return MLPipeline(model_dir=temp_model_dir, random_state=42)
    
    def test_data_generation(self, pipeline):
        """Test synthetic data generation"""
        df = pipeline.generate_data(n_samples=100, n_features=10)
        
        # Check shape
        assert df.shape == (100, 11)  # 10 features + 1 target
        
        # Check columns
        expected_cols = [f"feature_{i}" for i in range(10)] + ['target']
        assert list(df.columns) == expected_cols
        
        # Check target values
        assert set(df['target'].unique()) == {0, 1}
        
        # Check data types
        assert df.dtypes['target'] == np.int64
    
    def test_data_preparation(self, pipeline):
        """Test data splitting"""
        df = pipeline.generate_data(n_samples=100, n_features=10)
        X_train, X_val, X_test, y_train, y_val, y_test = pipeline.prepare_data(df)
        
        # Check shapes
        assert X_train.shape[1] == 10  # 10 features
        assert X_val.shape[1] == 10
        assert X_test.shape[1] == 10
        
        # Check total samples
        total_samples = len(X_train) + len(X_val) + len(X_test)
        assert total_samples == 100
        
        # Check that splits are reasonable
        assert len(X_test) == 20  # 20% of 100
        assert len(X_val) == 16   # ~20% of remaining 80
        assert len(X_train) == 64 # remaining
        
        # Check no data leakage (unique indices)
        train_idx = set(X_train.index)
        val_idx = set(X_val.index)
        test_idx = set(X_test.index)
        
        assert len(train_idx.intersection(val_idx)) == 0
        assert len(train_idx.intersection(test_idx)) == 0
        assert len(val_idx.intersection(test_idx)) == 0
    
    def test_model_training(self, pipeline):
        """Test model training"""
        df = pipeline.generate_data(n_samples=200, n_features=10)
        X_train, X_val, X_test, y_train, y_val, y_test = pipeline.prepare_data(df)
        
        # Train model
        model = pipeline.train_model(X_train, y_train, X_val, y_val)
        
        # Check model exists
        assert model is not None
        assert pipeline.model is not None
        
        # Check metrics exist
        assert 'validation_accuracy' in pipeline.metrics
        assert isinstance(pipeline.metrics['validation_accuracy'], float)
        assert 0 <= pipeline.metrics['validation_accuracy'] <= 1
    
    def test_model_testing(self, pipeline):
        """Test model evaluation"""
        df = pipeline.generate_data(n_samples=200, n_features=10)
        X_train, X_val, X_test, y_train, y_val, y_test = pipeline.prepare_data(df)
        
        # Train and test model
        pipeline.train_model(X_train, y_train, X_val, y_val)
        test_accuracy = pipeline.test_model(X_test, y_test)
        
        # Check test results
        assert isinstance(test_accuracy, float)
        assert 0 <= test_accuracy <= 1
        assert 'test_accuracy' in pipeline.metrics
        assert 'test_report' in pipeline.metrics
        assert 'confusion_matrix' in pipeline.metrics
    
    def test_model_saving_and_loading(self, pipeline, temp_model_dir):
        """Test model persistence"""
        df = pipeline.generate_data(n_samples=200, n_features=10)
        X_train, X_val, X_test, y_train, y_val, y_test = pipeline.prepare_data(df)
        
        # Train model
        pipeline.train_model(X_train, y_train, X_val, y_val)
        pipeline.test_model(X_test, y_test)
        
        # Save model
        model_path = pipeline.save_model("test_model.joblib")
        
        # Check files exist
        assert os.path.exists(model_path)
        assert os.path.exists(os.path.join(temp_model_dir, "metrics.json"))
        
        # Check metrics file content
        with open(os.path.join(temp_model_dir, "metrics.json"), 'r') as f:
            saved_metrics = json.load(f)
        
        assert 'validation_accuracy' in saved_metrics
        assert 'test_accuracy' in saved_metrics
        assert 'timestamp' in saved_metrics
    
    def test_full_pipeline(self, pipeline):
        """Test complete pipeline execution"""
        model_path = pipeline.run_pipeline()
        
        # Check model was saved
        assert os.path.exists(model_path)
        
        # Check metrics were generated
        assert 'validation_accuracy' in pipeline.metrics
        assert 'test_accuracy' in pipeline.metrics
        
        # Check accuracy is reasonable (should be > 0.5 for synthetic data)
        assert pipeline.metrics['test_accuracy'] > 0.5
    
    def test_untrained_model_error(self, pipeline):
        """Test error handling for untrained model"""
        with pytest.raises(ValueError, match="Model not trained yet"):
            pipeline.test_model(np.random.rand(10, 10), np.random.randint(0, 2, 10))
        
        with pytest.raises(ValueError, match="Model not trained yet"):
            pipeline.save_model()
    
    def test_model_predictions(self, pipeline):
        """Test model prediction functionality"""
        df = pipeline.generate_data(n_samples=200, n_features=10)
        X_train, X_val, X_test, y_train, y_val, y_test = pipeline.prepare_data(df)
        
        # Train model
        pipeline.train_model(X_train, y_train, X_val, y_val)
        
        # Test prediction shapes
        predictions = pipeline.model.predict(X_test)
        probabilities = pipeline.model.predict_proba(X_test)
        
        assert len(predictions) == len(X_test)
        assert probabilities.shape == (len(X_test), 2)  # 2 classes
        assert all(pred in [0, 1] for pred in predictions)
        assert all(0 <= prob <= 1 for row in probabilities for prob in row)

if __name__ == "__main__":
    pytest.main([__file__, "-v"])