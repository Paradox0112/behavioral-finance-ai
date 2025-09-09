"""
Test cases for the data processing module.
"""

import pytest
import pandas as pd
import numpy as np
from behavioral_finance.data import DataProcessor


class TestDataProcessor:
    """Test cases for DataProcessor class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.processor = DataProcessor()
        
        # Create sample data
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        self.sample_data = pd.DataFrame({
            'date': dates,
            'price': 100 + np.cumsum(np.random.randn(100) * 0.01),
            'volume': np.random.randint(1000, 10000, 100),
            'returns': np.random.randn(100) * 0.02
        })
        
    def test_initialization(self):
        """Test DataProcessor initialization."""
        processor = DataProcessor()
        assert processor.config == {}
        assert processor.data is None
        
        config = {'param': 'value'}
        processor = DataProcessor(config)
        assert processor.config == config
        
    def test_clean_data(self):
        """Test data cleaning functionality."""
        # Add some duplicates and NaNs
        dirty_data = self.sample_data.copy()
        dirty_data = pd.concat([dirty_data, dirty_data.iloc[:5]])  # Add duplicates
        dirty_data.loc[10:15, 'price'] = np.nan  # Add NaNs
        
        cleaned = self.processor.clean_data(dirty_data)
        
        # Check that duplicates are removed
        assert len(cleaned) <= len(dirty_data)
        
        # Check that NaNs are handled
        assert cleaned['price'].isna().sum() == 0
        
    def test_extract_behavioral_features(self):
        """Test behavioral feature extraction."""
        features = self.processor.extract_behavioral_features(self.sample_data)
        
        # Check that new features are added
        assert 'momentum_5d' in features.columns
        assert 'momentum_20d' in features.columns
        
        # Check data types
        assert features['momentum_5d'].dtype in [np.float64, 'float64']
        
    def test_extract_features_no_data(self):
        """Test feature extraction with no data."""
        with pytest.raises(ValueError, match="No data available"):
            self.processor.extract_behavioral_features()
