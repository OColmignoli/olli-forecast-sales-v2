"""
Training pipeline for Transformer sales forecasting model.
"""
import os
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple, List
import mlflow
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from datetime import datetime

from ...utils.data_ingestion import load_and_preprocess_data
from ...utils.azure_workspace import AzureWorkspaceManager
from .model import TransformerSalesForecaster

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SalesDataModule(pl.LightningDataModule):
    """PyTorch Lightning data module for sales data."""
    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        batch_size: int = 32,
        val_split: float = 0.2
    ):
        super().__init__()
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.val_split = val_split

    def setup(self, stage: Optional[str] = None):
        """Prepare data for training/validation."""
        val_size = int(len(self.X) * self.val_split)
        train_size = len(self.X) - val_size
        
        self.X_train, self.X_val = self.X[:train_size], self.X[train_size:]
        self.y_train, self.y_val = self.y[:train_size], self.y[train_size:]

    def train_dataloader(self):
        """Get training dataloader."""
        return torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(self.X_train, self.y_train),
            batch_size=self.batch_size,
            shuffle=True
        )

    def val_dataloader(self):
        """Get validation dataloader."""
        return torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(self.X_val, self.y_val),
            batch_size=self.batch_size
        )

def prepare_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    Prepare features for Transformer model.
    
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

def train_transformer_model(
    data_path: str,
    target_col: str = 'CV Gross Sales',
    model_params: Optional[Dict] = None,
    training_params: Optional[Dict] = None,
    azure_upload: bool = True
) -> Tuple[TransformerSalesForecaster, Dict[str, float]]:
    """
    Train Transformer model and track with MLflow.
    
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
        mlflow.set_experiment('transformer_sales_forecast')
        
        with mlflow.start_run() as run:
            # Set default parameters
            default_model_params = {
                'n_features': len(feature_cols),
                'd_model': 64,
                'nhead': 4,
                'num_layers': 2,
                'dropout': 0.1,
                'learning_rate': 1e-4,
                'sequence_length': 52,
                'forecast_horizon': 13
            }
            
            default_training_params = {
                'max_epochs': 100,
                'batch_size': 32,
                'val_split': 0.2,
                'patience': 10
            }
            
            # Update with provided parameters
            if model_params:
                default_model_params.update(model_params)
            if training_params:
                default_training_params.update(training_params)
            
            # Initialize model
            model = TransformerSalesForecaster(**default_model_params)
            
            # Prepare data
            X, y = model.prepare_data(df, target_col, feature_cols)
            data_module = SalesDataModule(
                X, y,
                batch_size=default_training_params['batch_size'],
                val_split=default_training_params['val_split']
            )
            
            # Set up callbacks
            callbacks = [
                ModelCheckpoint(
                    monitor='val_loss',
                    dirpath='checkpoints',
                    filename='transformer-{epoch:02d}-{val_loss:.2f}',
                    save_top_k=3,
                    mode='min'
                ),
                EarlyStopping(
                    monitor='val_loss',
                    patience=default_training_params['patience'],
                    mode='min'
                )
            ]
            
            # Initialize trainer
            trainer = pl.Trainer(
                max_epochs=default_training_params['max_epochs'],
                callbacks=callbacks,
                accelerator='auto',
                devices=1
            )
            
            # Train model
            trainer.fit(model, data_module)
            
            # Log parameters
            mlflow.log_params(default_model_params)
            mlflow.log_params(default_training_params)
            
            # Generate and log predictions
            historical_preds, future_preds = model.predict(
                df,
                features=feature_cols,
                target_col=target_col
            )
            
            # Save predictions
            output_dir = Path('outputs/transformer')
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
                    model_name='transformer',
                    environment_name='transformer-forecast-env'
                )
            
            # Calculate metrics
            metrics = {
                'train_loss': trainer.callback_metrics.get('train_loss', 0.0).item(),
                'val_loss': trainer.callback_metrics.get('val_loss', 0.0).item()
            }
            
            logger.info(f"Training completed successfully. Metrics: {metrics}")
            return model, metrics
            
    except Exception as e:
        logger.error(f"Error in training pipeline: {str(e)}")
        raise

def main():
    """Main training script."""
    import argparse
    import torch
    
    parser = argparse.ArgumentParser(description='Train Transformer sales forecasting model')
    parser.add_argument('--data-path', type=str, required=True, help='Path to training data')
    parser.add_argument('--target-col', type=str, default='CV Gross Sales', help='Target column to forecast')
    parser.add_argument('--d-model', type=int, default=64, help='Model dimension')
    parser.add_argument('--nhead', type=int, default=4, help='Number of attention heads')
    parser.add_argument('--num-layers', type=int, default=2, help='Number of transformer layers')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Training batch size')
    parser.add_argument('--no-azure', action='store_true', help='Disable Azure ML upload')
    
    args = parser.parse_args()
    
    # Set up parameters
    model_params = {
        'd_model': args.d_model,
        'nhead': args.nhead,
        'num_layers': args.num_layers
    }
    
    training_params = {
        'max_epochs': args.epochs,
        'batch_size': args.batch_size
    }
    
    # Train model
    model, metrics = train_transformer_model(
        data_path=args.data_path,
        target_col=args.target_col,
        model_params=model_params,
        training_params=training_params,
        azure_upload=not args.no_azure
    )
    
    logger.info(f"Training metrics: {metrics}")

if __name__ == '__main__':
    main()
