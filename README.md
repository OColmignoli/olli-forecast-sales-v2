# OLLI Forecast Sales V2

A comprehensive sales forecasting system using Azure ML with multiple advanced forecasting models.

## Project Overview

This project implements multiple forecasting models to predict sales in weekly buckets, including:
- LSTM (TensorFlow/Keras)
- Azure AutoML Time Series Forecasting
- Transformer-Based Models (TFT)
- DeepAR+
- CNNs
- Facebook Prophet
- Stacked Ensemble Model

## Project Structure

```
olli-forecast-sales-v2/
├── src/
│   ├── models/         # Individual model implementations
│   ├── api/           # FastAPI backend
│   ├── utils/         # Shared utilities
│   └── web/          # Frontend application
├── data/             # Data storage
├── notebooks/        # Jupyter notebooks for model development
├── tests/           # Unit and integration tests
└── config/          # Configuration files
```

## Setup Instructions

1. Configure Azure ML workspace
2. Install dependencies
3. Set up environment variables
4. Run the application

## Azure ML Configuration

- Workspace: olli-forecast-ml
- Resource Group: olli-forecast-rg
- Region: West US 2
- Private Network Access Configured

## Features

- Weekly sales forecasting using multiple models
- Web interface for model interaction
- Data upload functionality
- Excel export capability
- Automated model retraining
- Model performance monitoring
