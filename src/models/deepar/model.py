"""
DeepAR+ model for probabilistic sales forecasting using GluonTS/PyTorch.
"""
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import logging
from pathlib import Path
from gluonts.torch.model.deepar import DeepARModel
from gluonts.torch.util import copy_parameters
from gluonts.dataset.common import ListDataset
from gluonts.transform import (
    AddObservedValuesIndicator,
    AddTimeFeatures,
    AsNumpyArray,
    Chain,
    SelectFields,
    SetField,
    TestSplitSampler,
    Transformation,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DeepARSalesForecaster:
    """
    DeepAR+ model for sales forecasting.
    """
    def __init__(
        self,
        prediction_length: int = 13,
        context_length: int = 52,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout_rate: float = 0.1,
        learning_rate: float = 1e-3,
        num_parallel_samples: int = 100
    ):
        """
        Initialize DeepAR+ model.
        
        Args:
            prediction_length: Number of time steps to predict
            context_length: Number of time steps to use as context
            hidden_size: Hidden size of the LSTM layers
            num_layers: Number of LSTM layers
            dropout_rate: Dropout rate
            learning_rate: Learning rate
            num_parallel_samples: Number of samples for prediction
        """
        self.prediction_length = prediction_length
        self.context_length = context_length
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.num_parallel_samples = num_parallel_samples
        
        self.model = None
        self.transformation = None
        
    def create_transformation(self, freq: str = "W") -> Transformation:
        """
        Create data transformation pipeline.
        
        Args:
            freq: Time series frequency
            
        Returns:
            GluonTS transformation chain
        """
        self.transformation = Chain([
            AddTimeFeatures(
                start_field="start",
                target_field="target",
                output_field="time_features",
                time_features=["week_of_year", "month_of_year"],
                pred_length=self.prediction_length
            ),
            AddObservedValuesIndicator(
                target_field="target",
                output_field="observed_values"
            ),
            AsNumpyArray(
                field="target",
                expected_ndim=1
            ),
            AsNumpyArray(
                field="time_features",
                expected_ndim=2
            )
        ])
        
        return self.transformation

    def create_training_data(
        self,
        df: pd.DataFrame,
        target_col: str,
        freq: str = "W"
    ) -> ListDataset:
        """
        Create GluonTS training dataset.
        
        Args:
            df: Input DataFrame
            target_col: Target column name
            freq: Time series frequency
            
        Returns:
            GluonTS ListDataset
        """
        # Ensure data is sorted by time
        df = df.sort_index()
        
        # Create GluonTS dataset
        data = [{
            "start": df.index[0],
            "target": df[target_col].values,
            "feat_static_cat": [0],  # Single time series
            "item_id": "sales"  # Identifier for the time series
        }]
        
        return ListDataset(data, freq=freq)

    def create_model(self, input_dim: int = 1) -> DeepARModel:
        """
        Create DeepAR+ model.
        
        Args:
            input_dim: Input dimension
            
        Returns:
            DeepAR model
        """
        self.model = DeepARModel(
            freq="W",
            prediction_length=self.prediction_length,
            context_length=self.context_length,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout_rate=self.dropout_rate,
            learning_rate=self.learning_rate,
            num_parallel_samples=self.num_parallel_samples,
            input_size=input_dim
        )
        
        return self.model

    def train(
        self,
        df: pd.DataFrame,
        target_col: str,
        num_epochs: int = 100,
        batch_size: int = 32,
        num_batches_per_epoch: int = 50,
        freq: str = "W"
    ) -> Dict[str, float]:
        """
        Train the model.
        
        Args:
            df: Input DataFrame
            target_col: Target column name
            num_epochs: Number of training epochs
            batch_size: Batch size
            num_batches_per_epoch: Number of batches per epoch
            freq: Time series frequency
            
        Returns:
            Dictionary of training metrics
        """
        try:
            # Create transformation
            if self.transformation is None:
                self.create_transformation(freq)
            
            # Create training data
            train_data = self.create_training_data(df, target_col, freq)
            
            # Create model if not exists
            if self.model is None:
                self.create_model()
            
            # Train model
            self.model.train(
                training_data=train_data,
                num_epochs=num_epochs,
                batch_size=batch_size,
                num_batches_per_epoch=num_batches_per_epoch
            )
            
            # Calculate training metrics
            metrics = {
                "train_loss": self.model.trainer.epoch_loss
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error in training: {str(e)}")
            raise

    def predict(
        self,
        df: pd.DataFrame,
        target_col: str,
        freq: str = "W"
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Generate predictions.
        
        Args:
            df: Input DataFrame
            target_col: Target column name
            freq: Time series frequency
            
        Returns:
            Tuple of (historical_predictions, future_predictions)
        """
        try:
            # Create prediction dataset
            pred_data = self.create_training_data(df, target_col, freq)
            
            # Generate forecasts
            forecast_it = self.model.predict(pred_data)
            forecast = next(forecast_it)
            
            # Extract predictions
            prediction_samples = forecast.samples
            mean_prediction = prediction_samples.mean(axis=0)
            
            # Create historical predictions DataFrame
            historical_dates = pd.date_range(
                start=df.index[-self.prediction_length],
                periods=self.prediction_length,
                freq=freq
            )
            
            historical_preds = pd.DataFrame(
                {
                    'prediction': mean_prediction,
                    'actual': df[target_col].values[-self.prediction_length:]
                },
                index=historical_dates
            )
            
            # Create future predictions DataFrame
            future_dates = pd.date_range(
                start=df.index[-1] + pd.Timedelta(weeks=1),
                periods=self.prediction_length,
                freq=freq
            )
            
            future_preds = pd.DataFrame(
                {
                    'prediction': mean_prediction,
                    'lower_bound': np.quantile(prediction_samples, 0.1, axis=0),
                    'upper_bound': np.quantile(prediction_samples, 0.9, axis=0)
                },
                index=future_dates
            )
            
            return historical_preds, future_preds
            
        except Exception as e:
            logger.error(f"Error in prediction: {str(e)}")
            raise

    def save_model(self, path: str) -> None:
        """Save model to disk."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save model parameters
        torch.save(self.model.state_dict(), path / 'model.pt')
        
        # Save model configuration
        config = {
            'prediction_length': self.prediction_length,
            'context_length': self.context_length,
            'hidden_size': self.hidden_size,
            'num_layers': self.num_layers,
            'dropout_rate': self.dropout_rate,
            'learning_rate': self.learning_rate,
            'num_parallel_samples': self.num_parallel_samples
        }
        torch.save(config, path / 'config.pt')

    @classmethod
    def load_model(cls, path: str) -> 'DeepARSalesForecaster':
        """Load model from disk."""
        path = Path(path)
        
        # Load configuration
        config = torch.load(path / 'config.pt')
        
        # Create model instance
        model = cls(**config)
        model.create_model()
        
        # Load model parameters
        model.model.load_state_dict(torch.load(path / 'model.pt'))
        
        return model
