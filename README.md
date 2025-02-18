# OLLI Forecast Sales V2

A comprehensive sales forecasting system leveraging Azure ML and multiple advanced forecasting models for accurate weekly sales predictions.

## 🎯 Project Overview

This project implements an ensemble of state-of-the-art forecasting models to predict sales in weekly buckets, featuring:
- LSTM Networks (TensorFlow/Keras)
- Azure AutoML Time Series Forecasting
- Transformer-Based Models (TFT)
- DeepAR+ (Probabilistic Forecasting)
- CNNs for Time Series
- Facebook Prophet
- Stacking Ensemble Model

## 🏗️ Project Structure

```
olli-forecast-sales-v2/
├── src/
│   ├── models/          # Model implementations
│   │   ├── lstm/       # LSTM model
│   │   ├── transformer/ # Transformer models
│   │   ├── prophet/    # Prophet implementation
│   │   └── ensemble/   # Stacking ensemble
│   ├── monitoring/     # Performance monitoring
│   ├── deployment/     # Deployment utilities
│   ├── training/       # Training pipelines
│   ├── utils/          # Shared utilities
│   └── web/           # Web interface
├── data/
│   ├── raw/           # Original data
│   ├── processed/     # Processed datasets
│   └── interim/       # Intermediate data
├── tests/
│   ├── unit/         # Unit tests
│   └── integration/  # Integration tests
├── config/           # Configuration files
└── .github/
    └── workflows/    # CI/CD pipelines
```

## 🚀 Quick Start

1. **Clone the Repository**
   ```bash
   git clone https://github.com/OColmignoli/olli-forecast-sales-v2.git
   cd olli-forecast-sales-v2
   ```

2. **Set Up Python Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Configure Azure Credentials**
   - Create an Azure ML workspace
   - Set up Service Principal authentication
   - Add credentials to GitHub Secrets or local environment

4. **Prepare Data**
   - Place your sales data in `data/raw/`
   - Run data preprocessing pipeline
   ```bash
   python src/utils/data_ingestion.py
   ```

5. **Run the Application**
   ```bash
   python src/web/backend/app.py
   ```

## 🔧 Configuration

### Azure ML Settings
- Workspace: OLLI_ML_Forecast
- Resource Group: OLLI-resource
- Region: westus2
- Subscription ID: c828c783-7a28-48f4-b56f-a6c189437d77

### Required Environment Variables
```bash
AZURE_SUBSCRIPTION_ID=<your-subscription-id>
AZURE_RESOURCE_GROUP=<your-resource-group>
AZURE_WORKSPACE_NAME=<your-workspace-name>
```

## 🌟 Features

### Machine Learning
- Multiple model architectures for robust predictions
- Automated model selection and ensemble learning
- Real-time prediction capabilities
- Drift detection and automated retraining

### Monitoring & Operations
- Real-time performance monitoring
- Resource usage tracking
- Automated alerts
- Model versioning and rollback

### Web Interface
- Interactive dashboards
- Data upload/download functionality
- Model training controls
- Performance visualization

### CI/CD Pipeline
- Automated testing
- Code quality checks
- Continuous deployment
- Environment management

## 📊 Data Schema

The system expects sales data with the following structure:
- Customer Information (Number, Name, Brand)
- Product Details (Code, Name, Group)
- Transaction Data (Year, Week, Weight, Volume)
- Financial Metrics (Gross/Net Sales, COGS, Profit)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🔗 Links

- [Azure ML Documentation](https://docs.microsoft.com/en-us/azure/machine-learning/)
- [Project Wiki](https://github.com/OColmignoli/olli-forecast-sales-v2/wiki)
- [Issue Tracker](https://github.com/OColmignoli/olli-forecast-sales-v2/issues)
