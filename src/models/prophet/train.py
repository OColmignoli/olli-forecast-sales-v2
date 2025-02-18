"""
Training pipeline for Prophet sales forecasting model.
"""
import os
import logging
import pandas as pd
from pathlib import Path
from typing import Dict, Optional, Tuple
import mlflow
from datetime import datetime

from ...utils.data_ingestion import load_and_preprocess_data
from ...utils.azure_workspace import AzureWorkspaceManager
from .model import ProphetSalesForecaster

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train_prophet_model(
    data_path: str,
    target_col: str = 'CV Gross Sales',
    model_params: Optional[Dict] = None,
    azure_upload: bool = True
) -> Tuple[ProphetSalesForecaster, Dict[str, float]]:
    """
    Train Prophet model and track with MLflow.
    
    Args:
        data_path: Path to training data
        target_col: Column to forecast
        model_params: Optional model parameters
        azure_upload: Whether to upload model to Azure ML
        
    Returns:
        Tuple of (trained_model, metrics)
    """
    try:
        # Load and preprocess data
        success, df, results = load_and_preprocess_data(data_path)
        if not success:
            raise ValueError(f"Data loading failed: {results}")
        
        # Initialize Azure ML workspace if needed
        if azure_upload:
            azure_manager = AzureWorkspaceManager()
            workspace_success, error = azure_manager.initialize_workspace()
            if not workspace_success:
                raise ValueError(f"Azure workspace initialization failed: {error}")
        
        # Set up MLflow tracking
        mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI', 'local'))
        mlflow.set_experiment('prophet_sales_forecast')
        
        with mlflow.start_run() as run:
            # Initialize and train model
            model_params = model_params or {}
            model = ProphetSalesForecaster(target_col=target_col, **model_params)
            
            # Train model
            metrics = model.train(df, run_id=run.info.run_id)
            
            # Log metrics
            mlflow.log_metrics(metrics)
            
            # Generate and log predictions
            historical_preds, future_preds = model.predict(df)
            
            # Save predictions
            output_dir = Path('outputs/prophet')
            output_dir.mkdir(parents=True, exist_ok=True)
            
            historical_preds.to_csv(output_dir / 'historical_predictions.csv')
            future_preds.to_csv(output_dir / 'future_predictions.csv')
            
            # Log artifacts
            mlflow.log_artifacts(str(output_dir), artifact_path='predictions')
            
            # Save model
            model_path = output_dir / 'model'
            model.save_model(str(model_path))
            mlflow.log_artifacts(str(model_path), artifact_path='model')
            
            # Upload to Azure ML if requested
            if azure_upload:
                # Register dataset
                azure_manager.register_dataset(
                    data_path=data_path,
                    dataset_name=f"sales_data_{datetime.now().strftime('%Y%m%d')}",
                    dataset_description="Sales forecasting training data"
                )
                
                # Set up model environment
                azure_manager.setup_model_environment(
                    model_name='prophet',
                    environment_name='prophet-forecast-env'
                )
            
            logger.info(f"Training completed successfully. Metrics: {metrics}")
            return model, metrics
            
    except Exception as e:
        logger.error(f"Error in training pipeline: {str(e)}")
        raise

def main():
    """Main training script."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train Prophet sales forecasting model')
    parser.add_argument('--data-path', type=str, required=True, help='Path to training data')
    parser.add_argument('--target-col', type=str, default='CV Gross Sales', help='Target column to forecast')
    parser.add_argument('--no-azure', action='store_true', help='Disable Azure ML upload')
    
    args = parser.parse_args()
    
    # Train model
    model, metrics = train_prophet_model(
        data_path=args.data_path,
        target_col=args.target_col,
        azure_upload=not args.no_azure
    )
    
    logger.info(f"Training metrics: {metrics}")

if __name__ == '__main__':
    main()
