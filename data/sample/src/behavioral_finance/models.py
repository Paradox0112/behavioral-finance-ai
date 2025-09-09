"""
Machine learning models for behavioral finance analysis.

This module contains various AI models for detecting behavioral patterns
and predicting market anomalies in financial data.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from typing import Optional, Dict, Any, Tuple
import logging
import joblib

logger = logging.getLogger(__name__)


class BehavioralModel(BaseEstimator, ClassifierMixin):
    """
    Base class for behavioral finance AI models.
    """
    
    def __init__(self, model_type: str = 'random_forest', **kwargs):
        """
        Initialize BehavioralModel.
        
        Args:
            model_type: Type of underlying model ('random_forest', 'gradient_boosting')
            **kwargs: Additional parameters for the underlying model
        """
        self.model_type = model_type
        self.model_params = kwargs
        self.model = None
        self.scaler = StandardScaler()
        self.is_fitted = False
        
        self._initialize_model()
        
    def _initialize_model(self):
        """Initialize the underlying ML model."""
        if self.model_type == 'random_forest':
            self.model = RandomForestClassifier(**self.model_params)
        elif self.model_type == 'gradient_boosting':
            self.model = GradientBoostingClassifier(**self.model_params)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
            
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'BehavioralModel':
        """
        Train the behavioral model.
        
        Args:
            X: Feature matrix
            y: Target variable
            
        Returns:
            Fitted model instance
        """
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Fit the model
        self.model.fit(X_scaled, y)
        self.is_fitted = True
        
        logger.info(f"Model fitted with {len(X)} samples")
        return self
        
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using the fitted model.
        
        Args:
            X: Feature matrix
            
        Returns:
            Predictions array
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
            
        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)
        
        return predictions
        
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict class probabilities.
        
        Args:
            X: Feature matrix
            
        Returns:
            Probability estimates
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
            
        X_scaled = self.scaler.transform(X)
        probabilities = self.model.predict_proba(X_scaled)
        
        return probabilities
        
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance scores.
        
        Returns:
            Dictionary mapping feature names to importance scores
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted to get feature importance")
            
        if hasattr(self.model, 'feature_importances_'):
            return dict(zip(
                range(len(self.model.feature_importances_)),
                self.model.feature_importances_
            ))
        else:
            logger.warning("Model does not support feature importance")
            return {}
            
    def save_model(self, file_path: str):
        """
        Save the fitted model to disk.
        
        Args:
            file_path: Path to save the model
        """
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted model")
            
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'model_type': self.model_type,
            'model_params': self.model_params
        }
        
        joblib.dump(model_data, file_path)
        logger.info(f"Model saved to {file_path}")
        
    def load_model(self, file_path: str):
        """
        Load a fitted model from disk.
        
        Args:
            file_path: Path to the saved model
        """
        model_data = joblib.load(file_path)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.model_type = model_data['model_type']
        self.model_params = model_data['model_params']
        self.is_fitted = True
        
        logger.info(f"Model loaded from {file_path}")


class BiasDetectionModel(BehavioralModel):
    """
    Specialized model for detecting cognitive biases in trading behavior.
    """
    
    def __init__(self, bias_types: Optional[List[str]] = None, **kwargs):
        """
        Initialize bias detection model.
        
        Args:
            bias_types: List of bias types to detect
            **kwargs: Additional model parameters
        """
        self.bias_types = bias_types or [
            'anchoring_bias',
            'confirmation_bias', 
            'herding_behavior',
            'loss_aversion',
            'overconfidence'
        ]
        super().__init__(**kwargs)
        
    def detect_biases(self, X: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Detect multiple types of biases in the data.
        
        Args:
            X: Feature matrix
            
        Returns:
            Dictionary mapping bias types to detection results
        """
        # This is a placeholder implementation
        # In practice, you would have separate models for each bias type
        bias_predictions = {}
        
        for bias_type in self.bias_types:
            # For now, use the same model for all bias types
            predictions = self.predict_proba(X)
            bias_predictions[bias_type] = predictions
            
        return bias_predictions
