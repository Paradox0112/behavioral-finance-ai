"""
Behavioral Finance AI Engine

A sophisticated AI engine for analyzing legacy financial data through behavioral finance principles.
"""

__version__ = "0.1.0"
__author__ = "Behavioral Finance AI Team"
__email__ = "team@behavioral-finance-ai.com"

from .data import DataProcessor
from .models import BehavioralModel
from .analysis import BehavioralAnalyzer

__all__ = [
    "DataProcessor",
    "BehavioralModel", 
    "BehavioralAnalyzer"
]
