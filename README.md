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
- **NINO Index**: ENSO-related index from NOAA PSL ([Link](https://psl.noaa.gov/enso/dashboard.html)) used as an additional predictor for drought forecasting.

## Models
The repository includes the following categories of models:

### Traditional Models
| Method | Description |
|--------|-------------|
| ARIMA | AutoRegressive Integrated Moving Average, a widely used time-series forecasting model. |
| Persistence Model | Assumes future values remain the same as the last observed value. |
| Linear Regression | A simple statistical model using past data trends. |
| Holt-Winters | Exponential smoothing model capturing trend and seasonality. |
| SARIMA | Seasonal ARIMA, an extension of ARIMA for handling seasonality. |
| Prophet | Developed by Facebook, designed for time-series forecasting with strong seasonality. |

### Machine Learning Models
| Method | Description |
|--------|-------------|
| Random Forest | Ensemble learning method using multiple decision trees. |
| XGBoost | Extreme Gradient Boosting, highly efficient tree-based model. |
| LightGBM | A faster and scalable gradient boosting method optimized for large datasets. |

### Deep Learning Models
| Method | Description |
|--------|-------------|
| LSTM | Long Short-Term Memory, a type of recurrent neural network (RNN) suitable for sequential data. |
| GRU | Gated Recurrent Unit, a variant of LSTM with fewer parameters. |
| TCN | Temporal Convolutional Network, a CNN-based model for sequential tasks. |

### Transformer-based Models
| Method | Description | MSE | MAE |
|--------|-------------|----|----|
| Autoformer | Transformer-based model with autocorrelation mechanism for long-term forecasting. | 0.6995 | 0.6669 |
| Crossformer | Cross-scale attention-based Transformer for multiscale time-series learning. | 0.6030 | 0.6575 |
| DLinear | Decomposition-based Linear Transformer alternative. | 0.5197 | 0.5974 |
| FiLM | Feature-wise Linear Modulation-based Transformer. | 0.7527 | 0.7505 |
| iTransformer | Improved Transformer model for time-series applications. | 0.4974 | 0.5704 |
| MICN | Multi-Instance Contrastive Network for time-series analysis. | 1.4579 | 0.9533 |
| PatchTST | Patch-based Transformer for long-sequence forecasting. | 0.5336 | 0.6158 |
| Pyraformer | Pyramid attention-based Transformer. | 0.5815 | 0.6325 |
| TimeMixer | Mixer-based model optimized for time-series forecasting. | 0.5876 | 0.6394 |
| TimeXer | Transformer variant designed for extreme-scale time-series learning. | 0.4124 | 0.5171 |
| TimesNet | Neural network specifically designed for time-series prediction. | 0.6823 | 0.6957 |
| Transformer | Standard Transformer architecture applied to time-series. | 0.6832 | 0.6758 |
| TSMixer | Mixing model for time-series. | 0.4985 | 0.5974 |

### Foundation Models
| Method | Description |
|--------|-------------|
| TimesFM | Large-scale foundation model for time-series forecasting. |
| TabPFN_TS | Transformer-based foundation model for probabilistic time-series forecasting. |
| TimeGPT | Pre-trained generative time-series model. |
| ClimaX | Foundation model specialized in climate forecasting. |
| Other Pre-trained Models | Exploration of pre-trained foundation models for drought prediction. |

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
For any inquiries (including questions or potential collaboration), please reach out to gaowy014@mymail.unisa.edu.au.

