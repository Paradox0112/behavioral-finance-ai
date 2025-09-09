# Behavioral Finance AI Engine Documentation

This documentation provides an overview of the project's purpose, architecture, and usage.

## Purpose
Analyze legacy financial datasets to uncover behavioral patterns, cognitive biases, and market anomalies using ML/AI.

## Architecture
- Data processing and feature extraction
- Modeling (classification of anomalies/biases)
- Analysis and reporting

## Quickstart
1. Install dependencies: `pip install -r requirements.txt`
2. Configure settings in `config/config.yaml`
3. Use the package:

```python
from behavioral_finance import DataProcessor, BehavioralModel, BehavioralAnalyzer

processor = DataProcessor()
# df = processor.load_legacy_data('data/raw/legacy.csv')
# df_clean = processor.clean_data(df)
# features = processor.extract_behavioral_features(df_clean)

# model = BehavioralModel(model_type='random_forest')
# model.fit(features.dropna(), target)

# analyzer = BehavioralAnalyzer()
# report = analyzer.generate_report(features)
# print(report)
```

## Contributing
- Run tests: `pytest`
- Format code: `black .`
- Lint: `flake8`

