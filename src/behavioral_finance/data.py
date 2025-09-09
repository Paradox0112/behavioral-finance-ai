"""
Data processing module for behavioral finance analysis.

This module handles legacy data ingestion, cleaning, and preprocessing
for behavioral finance AI models.
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, List
import logging

logger = logging.getLogger(__name__)


class DataProcessor:
    """
    Handles processing of legacy financial data for behavioral analysis.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize DataProcessor with optional configuration.
        
        Args:
            config: Configuration dictionary for data processing parameters
        """
        self.config = config or {}
        self.data = None
        
    def load_legacy_data(self, file_path: str, format: str = 'csv') -> pd.DataFrame:
        """
        Load legacy financial data from various formats.
        
        Args:
            file_path: Path to the data file
            format: Data format ('csv', 'excel', 'json')
            
        Returns:
            DataFrame containing the loaded data
        """
        try:
            if format.lower() == 'csv':
                data = pd.read_csv(file_path)
            elif format.lower() in ['excel', 'xlsx']:
                data = pd.read_excel(file_path)
            elif format.lower() == 'json':
                data = pd.read_json(file_path)
            else:
                raise ValueError(f"Unsupported format: {format}")
                
            logger.info(f"Loaded {len(data)} records from {file_path}")
            self.data = data
            return data
            
        except Exception as e:
            logger.error(f"Error loading data from {file_path}: {str(e)}")
            raise
            
    def clean_data(self, data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Clean and preprocess financial data.
        
        Args:
            data: DataFrame to clean (uses self.data if None)
            
        Returns:
            Cleaned DataFrame
        """
        if data is None:
            data = self.data
            
        if data is None:
            raise ValueError("No data available for cleaning")
            
        # Remove duplicates
        data = data.drop_duplicates()
        
        # Handle missing values
        data = data.fillna(method='ffill')
        
        # Convert date columns
        date_columns = data.select_dtypes(include=['object']).columns
        for col in date_columns:
            try:
                data[col] = pd.to_datetime(data[col])
            except:
                continue
                
        logger.info(f"Data cleaned: {len(data)} records remaining")
        return data
        
    def extract_behavioral_features(self, data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Extract behavioral finance features from the data.
        
        Args:
            data: DataFrame to process (uses self.data if None)
            
        Returns:
            DataFrame with behavioral features
        """
        if data is None:
            data = self.data
            
        if data is None:
            raise ValueError("No data available for feature extraction")
            
        # This is a placeholder for behavioral feature extraction
        # In practice, this would implement specific behavioral finance indicators
        features = data.copy()
        
        # Add momentum indicators
        if 'price' in data.columns:
            features['momentum_5d'] = data['price'].pct_change(5)
            features['momentum_20d'] = data['price'].pct_change(20)
            
        # Add volatility measures
        if 'returns' in data.columns:
            features['volatility_5d'] = data['returns'].rolling(5).std()
            features['volatility_20d'] = data['returns'].rolling(20).std()
            
        logger.info("Behavioral features extracted")
        return features
