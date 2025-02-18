"""
Training pipeline for stacking ensemble model.
"""
import os
import logging
import pandas as pd
from pathlib import Path
from typing import Dict, Optional
import mlflow
from datetime import datetime

from ...utils.data_ingestion import load_and_preprocess_data
from ...utils.azure_workspace import AzureWorkspaceManager
from ..prophet.model import ProphetSalesForecaster
from ..lstm.model import LSTMSalesForecaster
from ..transformer.model import TransformerSalesForecaster
from ..deepar.model import DeepARSalesForecaster
from ..cnn.model import CNNSalesForecaster
from ..automl.model import AutoMLSalesForecaster
from .model import StackingEnsemble

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_base_models(
    model_paths: Dict[str, str],
    workspace: Optional[object] = None
) -> Dict[str, object]:
    """
    Load trained base models.
    
    Args:
        model_paths: Dictionary of model paths
        workspace: Optional Azure ML workspace
        
    Returns:
        Dictionary of loaded models
    """
    base_models = {}
    
    try:
        # Load Prophet model
        prophet_model = ProphetSalesForecaster()
        prophet_model.load_model(model_paths['prophet'])
        base_models['prophet'] = prophet_model
        
        # Load LSTM model
        lstm_model = LSTMSalesForecaster()
        lstm_model.load_model(model_paths['lstm'])
        base_models['lstm'] = lstm_model
        
        # Load Transformer model
        transformer_model = TransformerSalesForecaster()
        transformer_model.load_model(model_paths['transformer'])
        base_models['transformer'] = transformer_model
        
        # Load DeepAR model
        deepar_model = DeepARSalesForecaster()
        deepar_model.load_model(model_paths['deepar'])
        base_models['deepar'] = deepar_model
        
        # Load CNN model
        cnn_model = CNNSalesForecaster()
        cnn_model.load_model(model_paths['cnn'])
        base_models['cnn'] = cnn_model
        
        # Load AutoML model
        automl_model = AutoMLSalesForecaster(workspace=workspace)
        automl_model.load_model(model_paths['automl'])
        base_models['automl'] = automl_model
        
        return base_models
        
    except Exception as e:
        logger.error(f"Error loading base models: {str(e)}")
        raise

def train_ensemble_model(
    data_path: str,
    model_paths: Dict[str, str],
    target_col: str = 'CV Gross Sales',
    meta_model_type: str = 'xgboost',
    forecast_horizon: int = 13,
    feature_engineering_params: Optional[Dict] = None
) -> StackingEnsemble:
    """
    Train stacking ensemble model.
    
    Args:
        data_path: Path to training data
        model_paths: Dictionary of base model paths
        target_col: Target column name
        meta_model_type: Type of meta-model
        forecast_horizon: Forecast horizon
        feature_engineering_params: Optional feature engineering parameters
        
    Returns:
        Trained ensemble model
    """
    try:
        # Load and preprocess data
        success, df, results = load_and_preprocess_data(data_path)
        if not success:
            raise ValueError(f"Data loading failed: {results}")
        
        # Initialize Azure workspace
        azure_manager = AzureWorkspaceManager()
        workspace_success, workspace = azure_manager.initialize_workspace()
        if not workspace_success:
            raise ValueError("Azure workspace initialization failed")
        
        # Load base models
        base_models = load_base_models(model_paths, workspace)
        
        # Set up MLflow tracking
        mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI', 'local'))
        mlflow.set_experiment('stacking_ensemble')
        
        with mlflow.start_run() as run:
            # Initialize ensemble model
            model = StackingEnsemble(
                base_models=base_models,
                meta_model_type=meta_model_type,
                forecast_horizon=forecast_horizon,
                feature_engineering_params=feature_engineering_params
            )
            
            # Train model
            metrics = model.train(df, target_col)
            
            # Log parameters
            mlflow.log_param('meta_model_type', meta_model_type)
            mlflow.log_param('forecast_horizon', forecast_horizon)
            if feature_engineering_params:
                mlflow.log_params(feature_engineering_params)
            
            # Log metrics
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)
            
            # Generate predictions
            historical_preds, future_preds = model.predict(df, target_col)
            
            # Save predictions
            output_dir = Path('outputs/ensemble')
            output_dir.mkdir(parents=True, exist_ok=True)
            
            historical_preds.to_csv(output_dir / 'historical_predictions.csv')
            future_preds.to_csv(output_dir / 'future_predictions.csv')
            
            # Log artifacts
            mlflow.log_artifacts(str(output_dir), artifact_path='predictions')
            
            # Save model
            model_path = output_dir / 'model'
            model.save_model(str(model_path))
            mlflow.log_artifacts(str(model_path), artifact_path='model')
            
            # Register dataset
            azure_manager.register_dataset(
                data_path=data_path,
                dataset_name=f"ensemble_data_{datetime.now().strftime('%Y%m%d')}",
                dataset_description="Sales forecasting ensemble training data"
            )
            
            logger.info(f"Ensemble training completed successfully. Metrics: {metrics}")
            return model
            
    except Exception as e:
        logger.error(f"Error in ensemble training pipeline: {str(e)}")
        raise

def main():
    """Main training script."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train stacking ensemble model')
    parser.add_argument('--data-path', type=str, required=True, help='Path to training data')
    parser.add_argument('--model-paths', type=str, required=True, help='JSON file containing paths to base models')
    parser.add_argument('--target-col', type=str, default='CV Gross Sales', help='Target column to forecast')
    parser.add_argument('--meta-model', type=str, default='xgboost', help='Meta-model type (xgboost, rf, or gbm)')
    parser.add_argument('--forecast-horizon', type=int, default=13, help='Forecast horizon')
    
    args = parser.parse_args()
    
    # Load model paths
    with open(args.model_paths, 'r') as f:
        import json
        model_paths = json.load(f)
    
    # Train model
    model = train_ensemble_model(
        data_path=args.data_path,
        model_paths=model_paths,
        target_col=args.target_col,
        meta_model_type=args.meta_model,
        forecast_horizon=args.forecast_horizon
    )
    
    logger.info("Ensemble model training completed successfully")

if __name__ == '__main__':
    main()
