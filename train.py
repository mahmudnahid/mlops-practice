#!/usr/bin/env python3
"""
ML Training Pipeline
Trains a simple binary classifier and saves the model
"""

import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os
import json
from datetime import datetime

class MLPipeline:
    def __init__(self, model_dir="models", random_state=42):
        self.model_dir = model_dir
        self.random_state = random_state
        self.model = None
        self.metrics = {}
        
        # Create model directory if it doesn't exist
        os.makedirs(model_dir, exist_ok=True)
    
    def generate_data(self, n_samples=1000, n_features=10):
        """Generate synthetic dataset for binary classification"""
        X, y = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=5,
            n_redundant=2,
            n_clusters_per_class=1,
            random_state=self.random_state
        )
        
        # Create feature names
        feature_names = [f"feature_{i}" for i in range(n_features)]
        
        # Convert to DataFrame
        df = pd.DataFrame(X, columns=feature_names)
        df['target'] = y
        
        return df
    
    def prepare_data(self, df, test_size=0.2, val_size=0.2):
        """Split data into train, validation, and test sets"""
        X = df.drop('target', axis=1)
        y = df['target']
        
        # First split: separate test set
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )
        
        # Second split: separate train and validation from remaining data
        # val_size_adjusted = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size, 
            random_state=self.random_state, stratify=y_temp
        )
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def train_model(self, X_train, y_train, X_val, y_val):
        """Train the model and validate"""
        self.model = RandomForestClassifier(
            n_estimators=100,
            random_state=self.random_state,
            max_depth=10
        )
        
        # Train the model
        print("Training model...")
        self.model.fit(X_train, y_train)
        
        # Validate the model
        print("Validating model...")
        val_predictions = self.model.predict(X_val)
        val_accuracy = accuracy_score(y_val, val_predictions)
        
        self.metrics['validation_accuracy'] = val_accuracy
        self.metrics['validation_report'] = classification_report(y_val, val_predictions, output_dict=True)
        
        print(f"Validation Accuracy: {val_accuracy:.4f}")
        
        return self.model
    
    def test_model(self, X_test, y_test):
        """Test the final model"""
        if self.model is None:
            raise ValueError("Model not trained yet!")
        
        print("Testing model...")
        test_predictions = self.model.predict(X_test)
        test_accuracy = accuracy_score(y_test, test_predictions)
        
        self.metrics['test_accuracy'] = test_accuracy
        self.metrics['test_report'] = classification_report(y_test, test_predictions, output_dict=True)
        self.metrics['confusion_matrix'] = confusion_matrix(y_test, test_predictions).tolist()
        
        print(f"Test Accuracy: {test_accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, test_predictions))
        
        return test_accuracy
    
    def save_model(self, model_name="model.joblib"):
        """Save the trained model and metrics"""
        if self.model is None:
            raise ValueError("Model not trained yet!")
        
        model_path = os.path.join(self.model_dir, model_name)
        metrics_path = os.path.join(self.model_dir, "metrics.json")
        
        # Save model
        joblib.dump(self.model, model_path)
        print(f"Model saved to: {model_path}")
        
        # Save metrics with timestamp
        self.metrics['timestamp'] = datetime.now().isoformat()
        self.metrics['model_path'] = model_path
        
        with open(metrics_path, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        print(f"Metrics saved to: {metrics_path}")
        
        return model_path
    
    def run_pipeline(self):
        """Run the complete ML pipeline"""
        print("=== Starting ML Pipeline ===")
        
        # Generate data
        print("\n1. Generating synthetic data...")
        df = self.generate_data()
        print(f"Generated dataset with shape: {df.shape}")
        
        # Prepare data
        print("\n2. Preparing data...")
        X_train, X_val, X_test, y_train, y_val, y_test = self.prepare_data(df)
        print(f"Train set: {X_train.shape}, Val set: {X_val.shape}, Test set: {X_test.shape}")
        
        # Train model
        print("\n3. Training model...")
        self.train_model(X_train, y_train, X_val, y_val)
        
        # Test model
        print("\n4. Testing model...")
        self.test_model(X_test, y_test)
        
        # Save model
        print("\n5. Saving model...")
        model_path = self.save_model()
        
        print("\n=== Pipeline Complete ===")
        return model_path

def main():
    pipeline = MLPipeline()
    model_path = pipeline.run_pipeline()
    print(f"\nModel ready for deployment at: {model_path}")

if __name__ == "__main__":
    main()