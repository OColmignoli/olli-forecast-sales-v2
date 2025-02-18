"""
Training pipeline for AutoML sales forecasting model.
"""
import os
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple, List
import mlflow
from datetime import datetime
from azureml.core import Workspace
from azureml.core.authentication import ServicePrincipalAuthentication
from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core.compute_target import ComputeTargetException

from ...utils.data_ingestion import load_and_preprocess_data
from ...utils.azure_workspace import AzureWorkspaceManager
from .model import AutoMLSalesForecaster

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_compute_target(
    workspace: Workspace,
    compute_name: str = 'sales-forecast-cluster',
    vm_size: str = 'Standard_DS3_v2',
    min_nodes: int = 0,
    max_nodes: int = 4,
    idle_seconds_before_scaledown: int = 120
) -> ComputeTarget:
    """
    Set up Azure ML compute target.
    
    Args:
        workspace: Azure ML workspace
        compute_name: Name of compute target
        vm_size: VM size
        min_nodes: Minimum number of nodes
        max_nodes: Maximum number of nodes
        idle_seconds_before_scaledown: Idle time before scaling down
        
    Returns:
        Compute target
    """
    try:
        compute_target = ComputeTarget(workspace=workspace, name=compute_name)
        logger.info("Found existing compute target")
    except ComputeTargetException:
        logger.info("Creating new compute target...")
        
        config = AmlCompute.provisioning_configuration(
            vm_size=vm_size,
            min_nodes=min_nodes,
            max_nodes=max_nodes,
            idle_seconds_before_scaledown=idle_seconds_before_scaledown
        )
        
        compute_target = ComputeTarget.create(
            workspace,
            compute_name,
            config
        )
        
        compute_target.wait_for_completion(show_output=True)
    
    return compute_target

def prepare_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    Prepare features for AutoML model.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Tuple of (processed_df, feature_columns)
    """
    # Select numeric columns as features
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Remove target-related columns that shouldn't be features
    exclude_cols = [
        'Transaction Year', 'Transaction Week',
        'year', 'month', 'week', 'quarter'
    ]
    feature_cols = [col for col in numeric_cols if col not in exclude_cols]
    
    return df, feature_cols

def train_automl_model(
    data_path: str,
    target_col: str = 'CV Gross Sales',
    model_params: Optional[Dict] = None,
    training_params: Optional[Dict] = None
) -> Tuple[AutoMLSalesForecaster, Dict[str, float]]:
    """
    Train AutoML model and track with MLflow.
    
    Args:
        data_path: Path to training data
        target_col: Column to forecast
        model_params: Optional model parameters
        training_params: Optional training parameters
        
    Returns:
        Tuple of (trained_model, metrics)
    """
    try:
        # Load and preprocess data
        success, df, results = load_and_preprocess_data(data_path)
        if not success:
            raise ValueError(f"Data loading failed: {results}")
        
        # Prepare features
        df, feature_cols = prepare_features(df)
        
        # Initialize Azure ML workspace
        azure_manager = AzureWorkspaceManager()
        workspace_success, workspace = azure_manager.initialize_workspace()
        if not workspace_success:
            raise ValueError(f"Azure workspace initialization failed")
        
        # Set up compute target
        compute_target = setup_compute_target(workspace)
        
        # Set up MLflow tracking
        mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI', 'local'))
        mlflow.set_experiment('automl_sales_forecast')
        
        with mlflow.start_run() as run:
            # Set default parameters
            default_model_params = {
                'time_column_name': 'date',
                'target_column_name': target_col,
                'forecast_horizon': 13,
                'max_horizon': 52,
                'experiment_timeout_hours': 3.0,
                'max_concurrent_iterations': 4,
                'primary_metric': 'normalized_root_mean_squared_error'
            }
            
            # Update with provided parameters
            if model_params:
                default_model_params.update(model_params)
            
            # Initialize model
            model = AutoMLSalesForecaster(
                workspace=workspace,
                compute_target=compute_target,
                **default_model_params
            )
            
            # Train model
            metrics = model.train(
                df,
                experiment_name=f'automl_sales_forecast_{run.info.run_id}'
            )
            
            # Log parameters and metrics
            mlflow.log_params(default_model_params)
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)
            
            # Generate and log predictions
            historical_preds, future_preds = model.predict(df)
            
            # Save predictions
            output_dir = Path('outputs/automl')
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
                dataset_name=f"sales_data_{datetime.now().strftime('%Y%m%d')}",
                dataset_description="Sales forecasting training data"
            )
            
            logger.info(f"Training completed successfully. Metrics: {metrics}")
            return model, metrics
            
    except Exception as e:
        logger.error(f"Error in training pipeline: {str(e)}")
        raise

def main():
    """Main training script."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train AutoML sales forecasting model')
    parser.add_argument('--data-path', type=str, required=True, help='Path to training data')
    parser.add_argument('--target-col', type=str, default='CV Gross Sales', help='Target column to forecast')
    parser.add_argument('--forecast-horizon', type=int, default=13, help='Forecast horizon')
    parser.add_argument('--max-horizon', type=int, default=52, help='Maximum forecast horizon')
    parser.add_argument('--timeout-hours', type=float, default=3.0, help='Experiment timeout in hours')
    parser.add_argument('--max-concurrent', type=int, default=4, help='Maximum concurrent iterations')
    
    args = parser.parse_args()
    
    # Set up parameters
    model_params = {
        'target_column_name': args.target_col,
        'forecast_horizon': args.forecast_horizon,
        'max_horizon': args.max_horizon,
        'experiment_timeout_hours': args.timeout_hours,
        'max_concurrent_iterations': args.max_concurrent
    }
    
    # Train model
    model, metrics = train_automl_model(
        data_path=args.data_path,
        model_params=model_params
    )
    
    logger.info(f"Training metrics: {metrics}")

if __name__ == '__main__':
    main()
