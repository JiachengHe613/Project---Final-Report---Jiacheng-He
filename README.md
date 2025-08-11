# Satellite Orbit Maneuver Detection System

## Project Overview
This project implements a complete workflow to analyze and automatically detect orbital anomalies for multiple satellites (CryoSat‑2, Fengyun‑2F, Sentinel‑3A). The project experiments with XGBoost, ARIMA, Autoencoder, and LSTM algorithms, and evaluates different models based on PR curves and F1‑score. It covers EDA, feature engineering, anomaly‑detection model building, and a complete evaluation pipeline.

## Satellites
- CryoSat‑2
- Fengyun‑2F
- Sentinel‑3A

## Models
- Baseline: 3‑day EMA
- ARIMA
- LSTM
- Autoencoder
- XGBoost

## Directory Structure
```text
├── satellite_data/                   # Raw orbital‑element data and maneuver logs
│   ├── manoeuvres/                   # Maneuver record text files (per satellite)
│   └── orbital_elements/             # Orbital‑element data (per satellite)
├── cs2/                              # Processed data for CryoSat‑2
├── fy2f/                             # Processed data for Fengyun‑2F
├── s3a/                              # Processed data for Sentinel‑3A
├── cs2_EMA/ | cs2_ARIMA/ | cs2_LSTM/ | cs2_XGBoost/ | cs2_Autoencoder/
|   # Detection outputs for each model on CryoSat‑2
├── fy2f_EMA/ | fy2f_ARIMA/ | fy2f_LSTM/ | fy2f_XGBoost/ | fy2f_Autoencoder/
│   # Detection outputs for each model on Fengyun‑2F
├── s3a_EMA/ | s3a_ARIMA/ | s3a_LSTM/ | s3a_XGBoost/ | s3a_Autoencoder/
│   # Detection outputs for each model on Sentinel‑3A
├── cs2_figure/ | fy2f_figure/ | s3a_figure/        # EDA figures per satellite
├── combo_figure/                                   # Multi‑satellite comparison figures
├── 1_process_maneuvers.py             # Process raw maneuver data files
├── 1_process_maneuvers_FY2F.py        # Process Fengyun‑2F maneuver data
├── 2_maneuver_timeline.py             # Build multi‑satellite maneuver timeline: maneuver_timeline.png
├── 3_data_check.py                    # Data preprocessing
├── 4_orbital_feature_analysis.py      # Orbital feature EDA
├── 5_features_F1.py                   # Feature selection via 3‑day EMA F1‑score
├── 6_baseline_EMA.py                  # 3‑day EMA baseline training and evaluation
├── 7_XGBoost.py                       # XGBoost training and evaluation
├── 8_ARIMA_arg.ipynb                  # ARIMA parameter search
├── 8_ARIMA_model.py                   # ARIMA training and evaluation
├── 9_Autoencoder.py                   # Autoencoder training and evaluation
├── 10_LSTM.py                         # LSTM training and evaluation
└── 11_PR.py                           # PR curve generation
```

Each satellite‑model directory contains prediction plots, residual plots, combined residual plots, and `PRdata.csv` for threshold and PR evaluation.
 