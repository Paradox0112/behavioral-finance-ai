"""
Analysis module for behavioral finance patterns.

This module provides tools for analyzing behavioral patterns,
market anomalies, and cognitive biases in financial data.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class BehavioralAnalyzer:
    """
    Main analyzer for behavioral finance patterns and anomalies.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize BehavioralAnalyzer.
        
        Args:
            config: Configuration dictionary for analysis parameters
        """
        self.config = config or {}
        self.results = {}
        
    def analyze_momentum_patterns(self, data: pd.DataFrame, 
                                price_col: str = 'price') -> Dict:
        """
        Analyze momentum patterns in the data.
        
        Args:
            data: DataFrame containing financial data
            price_col: Name of the price column
            
        Returns:
            Dictionary containing momentum analysis results
        """
        if price_col not in data.columns:
            raise ValueError(f"Column '{price_col}' not found in data")
            
        results = {}
        
        # Calculate returns
        data['returns'] = data[price_col].pct_change()
        
        # Short-term momentum (5-day)
        data['momentum_5d'] = data['returns'].rolling(5).mean()
        
        # Medium-term momentum (20-day)
        data['momentum_20d'] = data['returns'].rolling(20).mean()
        
        # Momentum persistence analysis
        results['momentum_persistence'] = {
            'short_term_autocorr': data['momentum_5d'].autocorr(),
            'medium_term_autocorr': data['momentum_20d'].autocorr(),
            'cross_correlation': data['momentum_5d'].corr(data['momentum_20d'])
        }
        
        logger.info("Momentum pattern analysis completed")
        return results
        
    def detect_herding_behavior(self, data: pd.DataFrame,
                               volume_col: str = 'volume',
                               price_col: str = 'price') -> Dict:
        """
        Detect herding behavior patterns.
        
        Args:
            data: DataFrame containing financial data
            volume_col: Name of the volume column
            price_col: Name of the price column
            
        Returns:
            Dictionary containing herding behavior analysis
        """
        results = {}
        
        # Calculate price volatility
        data['returns'] = data[price_col].pct_change()
        data['volatility'] = data['returns'].rolling(20).std()
        
        # Volume analysis
        if volume_col in data.columns:
            data['volume_ma'] = data[volume_col].rolling(20).mean()
            data['volume_ratio'] = data[volume_col] / data['volume_ma']
            
            # Herding indicator: high volume with low volatility
            herding_conditions = (
                (data['volume_ratio'] > 1.5) & 
                (data['volatility'] < data['volatility'].quantile(0.3))
            )
            
            results['herding_periods'] = data[herding_conditions].index.tolist()
            results['herding_frequency'] = len(results['herding_periods']) / len(data)
            
        logger.info("Herding behavior analysis completed")
        return results
        
    def analyze_loss_aversion(self, data: pd.DataFrame,
                            price_col: str = 'price') -> Dict:
        """
        Analyze loss aversion patterns.
        
        Args:
            data: DataFrame containing financial data
            price_col: Name of the price column
            
        Returns:
            Dictionary containing loss aversion analysis
        """
        results = {}
        
        data['returns'] = data[price_col].pct_change()
        
        # Separate gains and losses
        gains = data['returns'][data['returns'] > 0]
        losses = data['returns'][data['returns'] < 0]
        
        if len(gains) > 0 and len(losses) > 0:
            results['gain_loss_ratio'] = len(gains) / len(losses)
            results['avg_gain'] = gains.mean()
            results['avg_loss'] = losses.mean()
            results['loss_aversion_ratio'] = abs(results['avg_loss']) / results['avg_gain']
            
            # Disposition effect: tendency to hold losers too long
            results['disposition_effect'] = self._calculate_disposition_effect(data, price_col)
            
        logger.info("Loss aversion analysis completed")
        return results
        
    def _calculate_disposition_effect(self, data: pd.DataFrame, 
                                    price_col: str) -> float:
        """
        Calculate disposition effect metric.
        
        Args:
            data: DataFrame containing financial data
            price_col: Name of the price column
            
        Returns:
            Disposition effect score
        """
        # Simplified disposition effect calculation
        # In practice, this would require more sophisticated analysis
        # of individual trading decisions
        
        data['returns'] = data[price_col].pct_change()
        data['cumulative_returns'] = (1 + data['returns']).cumprod() - 1
        
        # Count periods where positions are held during losses vs gains
        losing_periods = data[data['cumulative_returns'] < 0]
        gaining_periods = data[data['cumulative_returns'] > 0]
        
        if len(gaining_periods) > 0:
            return len(losing_periods) / len(gaining_periods)
        else:
            return 0.0
            
    def generate_report(self, data: pd.DataFrame) -> str:
        """
        Generate comprehensive behavioral analysis report.
        
        Args:
            data: DataFrame containing financial data
            
        Returns:
            Formatted analysis report string
        """
        report = []
        report.append("=" * 50)
        report.append("BEHAVIORAL FINANCE ANALYSIS REPORT")
        report.append("=" * 50)
        
        # Run all analyses
        momentum_results = self.analyze_momentum_patterns(data)
        herding_results = self.detect_herding_behavior(data)
        loss_aversion_results = self.analyze_loss_aversion(data)
        
        # Format results
        report.append("\n1. MOMENTUM ANALYSIS:")
        report.append(f"   Short-term autocorrelation: {momentum_results['momentum_persistence']['short_term_autocorr']:.4f}")
        report.append(f"   Medium-term autocorrelation: {momentum_results['momentum_persistence']['medium_term_autocorr']:.4f}")
        
        report.append("\n2. HERDING BEHAVIOR:")
        if 'herding_frequency' in herding_results:
            report.append(f"   Herding frequency: {herding_results['herding_frequency']:.2%}")
            report.append(f"   Number of herding periods: {len(herding_results['herding_periods'])}")
        
        report.append("\n3. LOSS AVERSION:")
        if 'loss_aversion_ratio' in loss_aversion_results:
            report.append(f"   Loss aversion ratio: {loss_aversion_results['loss_aversion_ratio']:.4f}")
            report.append(f"   Disposition effect: {loss_aversion_results['disposition_effect']:.4f}")
        
        return "\n".join(report)
