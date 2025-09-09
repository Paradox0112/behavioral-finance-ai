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
from typing import Optional, Dict, List
import logging

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
