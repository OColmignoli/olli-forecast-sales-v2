"""
LSTM model implementation for sales forecasting.
Handles model architecture, training, prediction, and evaluation.
"""
import tensorflow as tf
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import logging
from pathlib import Path
import json
import mlflow
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LSTMSalesForecaster:
    """LSTM-based sales forecasting model."""
    
    def __init__(
        self,
        target_col: str = 'CV Gross Sales',
        sequence_length: int = 52,
        n_features: int = 1,
        lstm_units: List[int] = [64, 32],
        dropout_rate: float = 0.2,
        learning_rate: float = 0.001
    ):
        """
        Initialize the LSTM forecaster.
        
        Args:
            target_col: Column to forecast
            sequence_length: Number of time steps to look back
            n_features: Number of input features
            lstm_units: List of units in each LSTM layer
            dropout_rate: Dropout rate for regularization
            learning_rate: Learning rate for optimization
        """
        self.target_col = target_col
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        
        # Build model
        self.model = self._build_model()
        
        # Store parameters
        self.model_params = {
            'target_col': target_col,
            'sequence_length': sequence_length,
            'n_features': n_features,
            'lstm_units': lstm_units,
            'dropout_rate': dropout_rate,
            'learning_rate': learning_rate
        }
    
    def _build_model(self) -> tf.keras.Model:
        """
        Build LSTM model architecture.
        
        Returns:
            Compiled Keras model
        """
        model = tf.keras.Sequential()
        
        # First LSTM layer
        model.add(tf.keras.layers.LSTM(
            units=self.lstm_units[0],
            return_sequences=len(self.lstm_units) > 1,
            input_shape=(self.sequence_length, self.n_features)
        ))
        model.add(tf.keras.layers.Dropout(self.dropout_rate))
        
        # Additional LSTM layers
        for units in self.lstm_units[1:]:
            model.add(tf.keras.layers.LSTM(
                units=units,
                return_sequences=False
            ))
            model.add(tf.keras.layers.Dropout(self.dropout_rate))
        
        # Dense layers
        model.add(tf.keras.layers.Dense(16, activation='relu'))
        model.add(tf.keras.layers.Dense(1))
        
        # Compile model
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='mse',
            metrics=['mae', 'mape']
        )
        
        return model
    
    def prepare_sequences(
        self,
        data: np.ndarray,
        target: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Create sequences for LSTM input.
        
        Args:
            data: Input data array
            target: Optional target array
            
        Returns:
            Tuple of (X, y) where y is None if target is None
        """
        X = []
        y = [] if target is not None else None
        
        for i in range(len(data) - self.sequence_length):
            X.append(data[i:(i + self.sequence_length)])
            if target is not None:
                y.append(target[i + self.sequence_length])
        
        X = np.array(X)
        if y is not None:
            y = np.array(y)
        
        return X, y
    
    def prepare_data(
        self,
        df: pd.DataFrame,
        features: Optional[List[str]] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare data for LSTM model.
        
        Args:
            df: Input DataFrame
            features: Optional list of feature columns
            
        Returns:
            Tuple of (X, y) arrays
        """
        # Get features
        if features is None:
            features = [self.target_col]
        
        # Extract arrays
        X_data = df[features].values
        y_data = df[self.target_col].values
        
        # Create sequences
        X, y = self.prepare_sequences(X_data, y_data)
        
        return X, y
    
    def train(
        self,
        df: pd.DataFrame,
        features: Optional[List[str]] = None,
        epochs: int = 100,
        batch_size: int = 32,
        validation_split: float = 0.2,
        run_id: Optional[str] = None
    ) -> Dict[str, float]:
        """
        Train the LSTM model.
        
        Args:
            df: Training data
            features: Optional list of feature columns
            epochs: Number of training epochs
            batch_size: Training batch size
            validation_split: Validation data fraction
            run_id: Optional MLflow run ID
            
        Returns:
            Dictionary with training metrics
        """
        try:
            # Prepare data
            X, y = self.prepare_data(df, features)
            
            # Start MLflow run if run_id provided
            if run_id:
                mlflow.start_run(run_id=run_id)
                mlflow.log_params(self.model_params)
            
            # Train model
            history = self.model.fit(
                X, y,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=validation_split,
                verbose=1
            )
            
            # Get metrics
            metrics = {
                'loss': history.history['loss'][-1],
                'val_loss': history.history['val_loss'][-1],
                'mae': history.history['mae'][-1],
                'val_mae': history.history['val_mae'][-1],
                'mape': history.history['mape'][-1],
                'val_mape': history.history['val_mape'][-1]
            }
            
            # Log metrics if using MLflow
            if run_id:
                mlflow.log_metrics(metrics)
                mlflow.end_run()
            
            logger.info(f"Model training completed. Metrics: {metrics}")
            return metrics
            
        except Exception as e:
            logger.error(f"Error in model training: {str(e)}")
            if run_id:
                mlflow.end_run(status='FAILED')
            raise
    
    def predict(
        self,
        df: pd.DataFrame,
        features: Optional[List[str]] = None,
        forecast_horizon: int = 52
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Generate forecasts using the trained model.
        
        Args:
            df: Input data
            features: Optional list of feature columns
            forecast_horizon: Number of steps to forecast
            
        Returns:
            Tuple of (historical_predictions, future_predictions)
        """
        try:
            # Prepare data for historical predictions
            X, _ = self.prepare_data(df, features)
            
            # Generate historical predictions
            historical_preds = self.model.predict(X)
            
            # Create DataFrame with historical predictions
            historical_dates = df.index[self.sequence_length:]
            historical_predictions = pd.DataFrame(
                historical_preds,
                index=historical_dates,
                columns=['yhat']
            )
            
            # Generate future predictions
            future_preds = []
            last_sequence = X[-1]
            
            for _ in range(forecast_horizon):
                # Predict next value
                next_pred = self.model.predict(last_sequence.reshape(1, self.sequence_length, -1))
                future_preds.append(next_pred[0, 0])
                
                # Update sequence
                last_sequence = np.roll(last_sequence, -1, axis=0)
                last_sequence[-1] = next_pred
            
            # Create future dates
            future_dates = pd.date_range(
                start=df.index[-1],
                periods=forecast_horizon + 1,
                freq='W'
            )[1:]
            
            # Create DataFrame with future predictions
            future_predictions = pd.DataFrame(
                future_preds,
                index=future_dates,
                columns=['yhat']
            )
            
            return historical_predictions, future_predictions
            
        except Exception as e:
            logger.error(f"Error in prediction: {str(e)}")
            raise
    
    def evaluate(self, df: pd.DataFrame, features: Optional[List[str]] = None) -> Dict[str, float]:
        """
        Evaluate model performance.
        
        Args:
            df: Data to evaluate on
            features: Optional list of feature columns
            
        Returns:
            Dictionary of evaluation metrics
        """
        try:
            # Get predictions
            historical_preds, _ = self.predict(df, features)
            
            # Calculate metrics
            y_true = df[self.target_col][self.sequence_length:]
            y_pred = historical_preds['yhat']
            
            # Root Mean Squared Error
            rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
            
            # Mean Absolute Error
            mae = np.mean(np.abs(y_true - y_pred))
            
            # Mean Absolute Percentage Error
            mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
            
            # R-squared
            ss_res = np.sum((y_true - y_pred) ** 2)
            ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
            r2 = 1 - (ss_res / ss_tot)
            
            metrics = {
                'rmse': rmse,
                'mae': mae,
                'mape': mape,
                'r2': r2
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error in evaluation: {str(e)}")
            raise
    
    def save_model(self, path: str) -> None:
        """
        Save the trained model.
        
        Args:
            path: Path to save the model
        """
        try:
            # Create directory if it doesn't exist
            save_path = Path(path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save model parameters
            with open(save_path / 'model_params.json', 'w') as f:
                json.dump(self.model_params, f)
            
            # Save Keras model
            self.model.save(save_path / 'lstm_model')
            
            logger.info(f"Model saved to {path}")
            
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            raise
    
    @classmethod
    def load_model(cls, path: str) -> 'LSTMSalesForecaster':
        """
        Load a trained model.
        
        Args:
            path: Path to load the model from
            
        Returns:
            Loaded LSTMSalesForecaster instance
        """
        try:
            # Load model parameters
            with open(Path(path) / 'model_params.json', 'r') as f:
                model_params = json.load(f)
            
            # Create instance with loaded parameters
            instance = cls(**model_params)
            
            # Load Keras model
            instance.model = tf.keras.models.load_model(Path(path) / 'lstm_model')
            
            logger.info(f"Model loaded from {path}")
            return instance
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
