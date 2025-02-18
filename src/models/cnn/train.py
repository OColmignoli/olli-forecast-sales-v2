"""
Training pipeline for CNN sales forecasting model.
"""
import os
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple, List
import mlflow
import tensorflow as tf
from datetime import datetime

from ...utils.data_ingestion import load_and_preprocess_data
from ...utils.azure_workspace import AzureWorkspaceManager
from .model import CNNSalesForecaster

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def prepare_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    Prepare features for CNN model.
    
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

def train_cnn_model(
    data_path: str,
    target_col: str = 'CV Gross Sales',
    model_params: Optional[Dict] = None,
    training_params: Optional[Dict] = None,
    azure_upload: bool = True
) -> Tuple[CNNSalesForecaster, Dict[str, float]]:
    """
    Train CNN model and track with MLflow.
    
    Args:
        data_path: Path to training data
        target_col: Column to forecast
        model_params: Optional model parameters
        training_params: Optional training parameters
        azure_upload: Whether to upload model to Azure ML
        
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
        
        # Initialize Azure ML workspace if needed
        if azure_upload:
            azure_manager = AzureWorkspaceManager()
            workspace_success, error = azure_manager.initialize_workspace()
            if not workspace_success:
                raise ValueError(f"Azure workspace initialization failed: {error}")
        
        # Set up MLflow tracking
        mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI', 'local'))
        mlflow.set_experiment('cnn_sales_forecast')
        
        with mlflow.start_run() as run:
            # Set default parameters
            default_model_params = {
                'sequence_length': 52,
                'forecast_horizon': 13,
                'n_filters': [64, 32, 16],
                'kernel_sizes': [3, 3, 3],
                'dense_units': [64, 32],
                'dropout_rate': 0.2,
                'learning_rate': 1e-3
            }
            
            default_training_params = {
                'epochs': 100,
                'batch_size': 32,
                'validation_split': 0.2
            }
            
            # Update with provided parameters
            if model_params:
                default_model_params.update(model_params)
            if training_params:
                default_training_params.update(training_params)
            
            # Initialize model
            model = CNNSalesForecaster(**default_model_params)
            
            # Set up callbacks
            callbacks = [
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=10,
                    restore_best_weights=True
                ),
                tf.keras.callbacks.ModelCheckpoint(
                    filepath='checkpoints/cnn_model_{epoch:02d}_{val_loss:.2f}.h5',
                    monitor='val_loss',
                    save_best_only=True
                ),
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=5,
                    min_lr=1e-6
                )
            ]
            
            # Train model
            metrics = model.train(
                df,
                target_col=target_col,
                features=feature_cols,
                callbacks=callbacks,
                **default_training_params
            )
            
            # Log parameters
            mlflow.log_params(default_model_params)
            mlflow.log_params(default_training_params)
            
            # Generate and log predictions
            historical_preds, future_preds = model.predict(
                df,
                target_col=target_col,
                features=feature_cols
            )
            
            # Save predictions
            output_dir = Path('outputs/cnn')
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
                    model_name='cnn',
                    environment_name='cnn-forecast-env'
                )
            
            logger.info(f"Training completed successfully. Metrics: {metrics}")
            return model, metrics
            
    except Exception as e:
        logger.error(f"Error in training pipeline: {str(e)}")
        raise

def main():
    """Main training script."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train CNN sales forecasting model')
    parser.add_argument('--data-path', type=str, required=True, help='Path to training data')
    parser.add_argument('--target-col', type=str, default='CV Gross Sales', help='Target column to forecast')
    parser.add_argument('--sequence-length', type=int, default=52, help='Sequence length')
    parser.add_argument('--forecast-horizon', type=int, default=13, help='Forecast horizon')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Training batch size')
    parser.add_argument('--no-azure', action='store_true', help='Disable Azure ML upload')
    
    args = parser.parse_args()
    
    # Set up parameters
    model_params = {
        'sequence_length': args.sequence_length,
        'forecast_horizon': args.forecast_horizon
    }
    
    training_params = {
        'epochs': args.epochs,
        'batch_size': args.batch_size
    }
    
    # Train model
    model, metrics = train_cnn_model(
        data_path=args.data_path,
        target_col=args.target_col,
        model_params=model_params,
        training_params=training_params,
        azure_upload=not args.no_azure
    )
    
    logger.info(f"Training metrics: {metrics}")

if __name__ == '__main__':
    main()
