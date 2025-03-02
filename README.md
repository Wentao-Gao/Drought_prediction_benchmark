# Drought Prediction Benchmark

## Overview
This repository provides a benchmark framework for predicting the Standardized Precipitation Evapotranspiration Index (SPEI) in South Australia for future one-month and three-month periods. The goal is to evaluate various forecasting models and methodologies for drought prediction using both traditional and machine learning-based approaches.

## Features
- **Predicting SPEI**: Focuses on one-month and three-month ahead forecasts.
- **Benchmarking Models**: Includes classical statistical models, machine learning models, and foundation models.
- **Data Sources**: Incorporates climate model outputs and observational data.
- **Evaluation Metrics**: Standardized evaluation framework to compare model performance.
- **Feature Selection**: Utilizes techniques such as Granger causality and PCMCI for causal discovery.
- **Foundation Model Exploration**: Investigates in-context learning and reinforcement learning-based approaches.

## Data
The dataset consists of:
- **NCEP-NCAR Reanalysis 1**: Meteorological features.
- **SPEI-GD Dataset**: The first global multi-scale daily SPEI dataset (Zenodo DOI: [10.5281/zenodo.8060268](https://doi.org/10.5281/zenodo.8060268)).
  - We specifically use the **30-day SPEI (spei30)** data at 0.25° spatial resolution from 1982 to 2021.
  - Data is based on ERA5 precipitation and Singer's potential evapotranspiration.
  - Available in NetCDF format.

## Models
The repository includes the following categories of models:
- **Baseline Models**: ARIMA, Persistence Model, Linear Regression.
- **Machine Learning Models**: Random Forest, XGBoost, LSTM, Transformer-based architectures.
- **Foundation Models**: TimesFM and other large-scale pre-trained models.
- **Hybrid Approaches**: Combining domain adaptation techniques with traditional forecasting.

## Installation
```bash
# Clone the repository
git clone https://github.com/yourusername/Drought_prediction_benchmark.git
cd Drought_prediction_benchmark

# Create a virtual environment
python -m venv env
source env/bin/activate  # On Windows use `env\Scripts\activate`

# Install dependencies
pip install -r requirements.txt
```

## Usage
1. **Data Preparation**: Download and preprocess datasets.
2. **Model Training**: Run benchmark models with default or custom configurations.
3. **Evaluation**: Compare model performance using standardized metrics.

Example command to run a baseline model:
```bash
python train.py --model baseline --target spei_1m
```

## Evaluation Metrics
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- Spearman's Rank Correlation Coefficient
- Skill Score Relative to Climatology

## Contributions
We welcome contributions! Feel free to submit pull requests for:
- Adding new models
- Improving preprocessing pipelines
- Enhancing evaluation metrics

## Contact
For any inquiries(including questions or potential colabration), please reach out to gaowy014@mymail.unisa.edu.au.

