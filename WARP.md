# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Project Overview

The Behavioral Finance AI Engine is a sophisticated Python application designed to analyze legacy financial data through behavioral finance principles. It uses machine learning models to detect cognitive biases, behavioral patterns, and market anomalies in historical financial datasets.

## Common Development Commands

### Environment Setup
```bash
# Install core dependencies
pip install -r requirements.txt

# Install with development dependencies
pip install -e ".[dev]"

# Install all optional dependencies for full functionality
pip install -e ".[dev,ml,financial,nlp,viz,stats,db]"
```

### Testing
```bash
# Run all tests
pytest

# Run tests with coverage report
pytest --cov=src/behavioral_finance --cov-report=html --cov-report=term-missing

# Run specific test file
pytest tests/test_data.py

# Run specific test method
pytest tests/test_data.py::TestDataProcessor::test_clean_data
```

### Code Quality
```bash
# Format code with Black
black src/ tests/

# Lint with flake8
flake8 src/ tests/

# Type checking with mypy
mypy src/behavioral_finance/
```

### Running the Application
```bash
# Using the CLI entry point (if available)
bf-analyze --config config/config.yaml --data data/sample/

# Running as Python module
python -m behavioral_finance

# Interactive usage
python -c "from behavioral_finance import DataProcessor, BehavioralModel, BehavioralAnalyzer"
```

## Architecture Overview

### Core Components

**DataProcessor** (`src/behavioral_finance/data.py`)
- Handles legacy financial data ingestion from CSV, Excel, and JSON formats
- Provides data cleaning (duplicate removal, missing value handling, date conversion)
- Extracts behavioral finance features (momentum indicators, volatility measures)
- Key methods: `load_legacy_data()`, `clean_data()`, `extract_behavioral_features()`

**BehavioralModel** (`src/behavioral_finance/models.py`)
- Base ML model class supporting Random Forest and Gradient Boosting
- Includes specialized `BiasDetectionModel` for cognitive bias identification
- Handles feature scaling and model persistence
- Supports detection of: anchoring bias, confirmation bias, herding behavior, loss aversion, overconfidence

**BehavioralAnalyzer** (`src/behavioral_finance/analysis.py`)
- Main analysis engine for behavioral pattern detection
- Key analysis methods:
  - `analyze_momentum_patterns()` - Short/medium-term momentum analysis
  - `detect_herding_behavior()` - High volume + low volatility patterns
  - `analyze_loss_aversion()` - Gain/loss ratios and disposition effect
- Generates comprehensive HTML/PDF reports via `generate_report()`

### Data Flow Architecture

1. **Data Ingestion**: Legacy financial data → DataProcessor.load_legacy_data()
2. **Data Cleaning**: Raw data → DataProcessor.clean_data() → Clean dataset
3. **Feature Engineering**: Clean data → DataProcessor.extract_behavioral_features() → Feature matrix
4. **Model Training**: Features + labels → BehavioralModel.fit() → Trained model
5. **Analysis**: Features → BehavioralAnalyzer → Pattern detection results
6. **Reporting**: Analysis results → BehavioralAnalyzer.generate_report() → Formatted reports

### Configuration System

The application uses YAML-based configuration in `config/config.yaml`:
- **Data processing parameters**: Cleaning methods, outlier detection thresholds
- **Model configurations**: Algorithm selection, hyperparameters for each model type
- **Analysis parameters**: Window sizes for momentum, herding detection thresholds
- **External API configurations**: Yahoo Finance, Alpha Vantage integration settings
- **Performance settings**: Parallel processing, memory limits, caching

### Package Structure

```
src/behavioral_finance/
├── __init__.py          # Main package exports
├── data.py              # DataProcessor class
├── models.py            # ML models (BehavioralModel, BiasDetectionModel)
└── analysis.py          # BehavioralAnalyzer class
```

## Key Development Patterns

### Error Handling
- All major operations include try-catch blocks with logging
- ValueError raised for invalid inputs (unsupported formats, missing data)
- Comprehensive logging at INFO level for operation tracking

### Data Validation
- Automatic column existence checks before processing
- Type validation for numeric columns
- Date column auto-detection and conversion

### Model Interface
- Scikit-learn compatible base classes (BaseEstimator, ClassifierMixin)
- Consistent fit/predict interface across all models
- Automatic feature scaling with StandardScaler

### Testing Strategy
- Class-based test organization (TestDataProcessor, etc.)
- Setup methods creating realistic sample data
- Comprehensive edge case testing (missing data, invalid formats)

## Configuration Management

Key configuration sections to modify:
- `data.features.momentum_windows`: Adjust timeframes for momentum analysis [5, 10, 20]
- `models.behavioral_model.parameters`: Tune ML model hyperparameters
- `analysis.patterns.herding.volume_threshold`: Sensitivity for herding detection (default: 1.5x)
- `external_data.apis`: Enable/configure financial data sources

## Dependencies and Optional Features

**Core Dependencies**: pandas, numpy, scipy, scikit-learn, matplotlib, seaborn, pyyaml
**ML Extensions**: xgboost, lightgbm, tensorflow, torch (optional)
**Financial Data**: yfinance, pandas-datareader, ta-lib (optional)
**NLP Analysis**: nltk, textblob, vaderSentiment (optional)
**Statistical Analysis**: statsmodels, pingouin (optional)

Install optional feature groups as needed:
```bash
pip install -e ".[ml,financial,nlp]"  # Install ML, financial, and NLP extensions
```
