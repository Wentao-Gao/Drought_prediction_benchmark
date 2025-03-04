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
The dataset consists of meteorological and climate-related variables used for drought prediction. The main data sources are:

### 📍 **Primary Data Sources**
1. **NCEP-NCAR Reanalysis 1** ([Link](https://www.psl.noaa.gov/data/gridded/data.ncep.reanalysis.html))
   - Global reanalysis dataset providing historical climate and atmospheric variables.
   - Features include surface radiation fluxes, temperature, wind speed, and pressure.

2. **SPEI-GD Dataset** ([DOI: 10.5281/zenodo.8060268](https://doi.org/10.5281/zenodo.8060268))
   - The first **global multi-scale daily SPEI dataset**.
   - We use the **30-day SPEI (SPEI-30)** at **0.25° spatial resolution**.
   - Time range: **1982–2021**.
   - Based on ERA5 precipitation and Singer’s potential evapotranspiration.

3. **NINO Index** ([NOAA PSL](https://psl.noaa.gov/enso/dashboard.html))
   - ENSO-related indices used as additional predictors for drought forecasting.
   - Includes **NINO12, NINO34, NINO4** and the **Dipole Mode Index (DMI)**.

---

### 🌍 **Dataset Structure**
Our dataset is stored in **NetCDF format**, containing:
- **Time Dimension** (`time`): Daily timestamps from 1982 to 2019.
- **Spatial Dimensions** (`lat, lon`): Grid points covering South Australia.
- **Meteorological Variables** (38 features):
  
Here is the **revised dataset description** with proper **units** and **clarifications** for the NINO indices:

**🌍 Meteorological Variables**
| Variable | Description | Unit |
|----------|------------|------|
| **spei** | Standardized Precipitation Evapotranspiration Index | Dimensionless |
| **uswrf** | Upward Shortwave Radiation Flux | W/m² |
| **csusf** | Clear-Sky Upward Shortwave Flux | W/m² |
| **lhtfl** | Latent Heat Flux | W/m² |
| **vddsf** | Downward Diffuse Shortwave Flux | W/m² |
| **tmax** | Maximum 2m Temperature | K |
| **shtfl** | Sensible Heat Flux | W/m² |
| **vgwd** | Gravity Wave Drag (V component) | m/s² |
| **air** | Air Temperature at 2m | K |
| **tmin** | Minimum 2m Temperature | K |
| **dswrf** | Downward Shortwave Radiation Flux | W/m² |
| **vbdsf** | Diffuse Shortwave Flux at Surface | W/m² |
| **nswrs** | Net Shortwave Radiation at Surface | W/m² |
| **pres** | Surface Pressure | Pa |
| **skt** | Skin Temperature | K |
| **nlwrs** | Net Longwave Radiation at Surface | W/m² |
| **uwnd** | Zonal Wind (U component) at 10m | m/s |
| **nbdsf** | Near-Surface Downward Shortwave Flux | W/m² |
| **vwnd** | Meridional Wind (V component) at 10m | m/s |
| **weasd** | Water Equivalent of Snow Depth | kg/m² |
| **prate** | Precipitation Rate | kg/m²/s |
| **cfnlf** | Cloud Forcing Net Longwave Flux | W/m² |
| **shum** | Specific Humidity at 2m | kg/kg |
| **dlwrf** | Downward Longwave Radiation Flux | W/m² |
| **csdsf** | Clear-Sky Downward Shortwave Flux | W/m² |
| **vflx** | Meridional Surface Momentum Flux | N/m² |
| **nddsf** | Net Downward Diffuse Shortwave Flux | W/m² |
| **cfnsf** | Cloud Forcing Net Shortwave Flux | W/m² |
| **ulwrf** | Upward Longwave Radiation Flux | W/m² |
| **icec** | Sea Ice Concentration | % |
| **cprat** | Convective Precipitation Rate | kg/m²/s |
| **uflx** | Zonal Surface Momentum Flux | N/m² |
| **csdlf** | Clear-Sky Downward Longwave Flux | W/m² |
| **ugwd** | Gravity Wave Drag (U component) | m/s² |

---

**🌏 ENSO-Related Indices**
| Variable | Description | Unit |
|----------|------------|------|
| **NINA12** | NINO 1+2 Sea Surface Temperature Anomaly | °C |
| **NINA34** | NINO 3.4 Sea Surface Temperature Anomaly | °C |
| **NINA4** | NINO 4 Sea Surface Temperature Anomaly | °C |
| **DMI_HadISST1.1** | Dipole Mode Index (Indian Ocean SST Anomaly) | °C |

**📌 Explanation of ENSO Indices:**
- **NINO1+2 (NINA12)**: Covers the easternmost equatorial Pacific region (80°W-90°W, 0°-10°S), mainly used for near-coastal El Niño monitoring.
- **NINO3.4 (NINA34)**: Located in the central Pacific (120°W-170°W, 5°N-5°S), most commonly used for defining El Niño and La Niña events.
- **NINO4 (NINA4)**: Represents the western equatorial Pacific (160°E-150°W, 5°N-5°S), capturing SST variability in the western Pacific.
- **DMI (Indian Ocean Dipole Mode Index)**: Measures the difference in SST anomalies between the western and eastern Indian Ocean.

---


Would you like to add any **other metadata** (e.g., data collection period, spatial resolution, or specific preprocessing steps)? 🚀


### 📊 **Example Data Format (CSV)**
The dataset can also be saved in CSV format:
```csv
time,lat,lon,spei,uswrf,csusf,lhtfl,vddsf,tmax,shtfl,vgwd,air,tmin,dswrf,vbdsf,nswrs,pres,skt,nlwrs,uwnd,nbdsf,vwnd,weasd,prate,cfnlf,shum,dlwrf,csdsf,vflx,nddsf,cfnsf,ulwrf,icec,cprat,uflx,csdlf,ugwd,NINA12,NINA34,NINA4,DMI_HadISST1.1
2021-01-01,-34.0,138.5,-1.2,189.4,167.2,23.4,98.6,42.1,15.3,-1.2,20.4,14.2,210.1,180.3,190.2,1012.5,28.3,87.4,3.4,88.1,2.1,0.0,2.8,0.95,12.4,305.2,135.5,5.3,88.2,85.3,250.6,0.0,0.1,15.3,305.1,2.5,0.1,-0.2,1.2,0.8
```

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

