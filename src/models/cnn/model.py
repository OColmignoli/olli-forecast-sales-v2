"""
CNN model for sales forecasting using TensorFlow/Keras.
"""
import tensorflow as tf
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CNNSalesForecaster:
    """
    CNN model for sales forecasting.
    """
    def __init__(
        self,
        sequence_length: int = 52,
        forecast_horizon: int = 13,
        n_features: Optional[int] = None,
        n_filters: List[int] = [64, 32, 16],
        kernel_sizes: List[int] = [3, 3, 3],
        dense_units: List[int] = [64, 32],
        dropout_rate: float = 0.2,
        learning_rate: float = 1e-3
    ):
        """
        Initialize CNN model.
        
        Args:
            sequence_length: Length of input sequence
            forecast_horizon: Number of time steps to forecast
            n_features: Number of input features
            n_filters: Number of filters in each conv layer
            kernel_sizes: Kernel sizes for each conv layer
            dense_units: Number of units in dense layers
            dropout_rate: Dropout rate
            learning_rate: Learning rate
        """
        self.sequence_length = sequence_length
        self.forecast_horizon = forecast_horizon
        self.n_features = n_features
        self.n_filters = n_filters
        self.kernel_sizes = kernel_sizes
        self.dense_units = dense_units
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        
        self.model = None
        self.history = None
        
    def build_model(self) -> tf.keras.Model:
        """
        Build CNN model architecture.
        
        Returns:
            Compiled Keras model
        """
        if self.n_features is None:
            raise ValueError("n_features must be set before building model")
        
        # Input layer
        inputs = tf.keras.layers.Input(shape=(self.sequence_length, self.n_features))
        
        # CNN layers
        x = inputs
        for filters, kernel_size in zip(self.n_filters, self.kernel_sizes):
            x = tf.keras.layers.Conv1D(
                filters=filters,
                kernel_size=kernel_size,
                padding='same',
                activation='relu'
            )(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.MaxPooling1D(pool_size=2)(x)
        
        # Flatten and dense layers
        x = tf.keras.layers.Flatten()(x)
        
        for units in self.dense_units:
            x = tf.keras.layers.Dense(units, activation='relu')(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.Dropout(self.dropout_rate)(x)
        
        # Output layer
        outputs = tf.keras.layers.Dense(self.forecast_horizon)(x)
        
        # Create model
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        
        # Compile model
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='mse',
            metrics=['mae']
        )
        
        self.model = model
        return model

    def prepare_sequences(
        self,
        df: pd.DataFrame,
        target_col: str,
        features: List[str]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare sequences for training.
        
        Args:
            df: Input DataFrame
            target_col: Target column name
            features: Feature columns
            
        Returns:
            Tuple of (X, y) arrays
        """
        # Extract features and target
        X = df[features].values
        y = df[target_col].values
        
        # Create sequences
        X_sequences = []
        y_sequences = []
        
        for i in range(len(df) - self.sequence_length - self.forecast_horizon + 1):
            X_sequences.append(X[i:i + self.sequence_length])
            y_sequences.append(y[i + self.sequence_length:i + self.sequence_length + self.forecast_horizon])
        
        return np.array(X_sequences), np.array(y_sequences)

    def train(
        self,
        df: pd.DataFrame,
        target_col: str,
        features: List[str],
        epochs: int = 100,
        batch_size: int = 32,
        validation_split: float = 0.2,
        callbacks: Optional[List[tf.keras.callbacks.Callback]] = None
    ) -> Dict[str, float]:
        """
        Train the model.
        
        Args:
            df: Input DataFrame
            target_col: Target column name
            features: Feature columns
            epochs: Number of epochs
            batch_size: Batch size
            validation_split: Validation split ratio
            callbacks: Optional list of Keras callbacks
            
        Returns:
            Dictionary of training metrics
        """
        try:
            # Set number of features
            self.n_features = len(features)
            
            # Build model if not exists
            if self.model is None:
                self.build_model()
            
            # Prepare sequences
            X, y = self.prepare_sequences(df, target_col, features)
            
            # Train model
            self.history = self.model.fit(
                X, y,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=validation_split,
                callbacks=callbacks,
                verbose=1
            )
            
            # Calculate metrics
            metrics = {
                'train_loss': self.history.history['loss'][-1],
                'train_mae': self.history.history['mae'][-1],
                'val_loss': self.history.history['val_loss'][-1],
                'val_mae': self.history.history['val_mae'][-1]
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error in training: {str(e)}")
            raise

    def predict(
        self,
        df: pd.DataFrame,
        target_col: str,
        features: List[str]
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Generate predictions.
        
        Args:
            df: Input DataFrame
            target_col: Target column name
            features: Feature columns
            
        Returns:
            Tuple of (historical_predictions, future_predictions)
        """
        try:
            # Prepare sequences
            X, y = self.prepare_sequences(df, target_col, features)
            
            # Generate predictions
            predictions = self.model.predict(X)
            
            # Create historical predictions DataFrame
            historical_dates = pd.date_range(
                start=df.index[self.sequence_length],
                periods=len(predictions),
                freq='W-MON'
            )
            
            historical_preds = pd.DataFrame(
                {
                    'prediction': predictions[:, 0],
                    'actual': df[target_col].values[self.sequence_length:self.sequence_length + len(predictions)]
                },
                index=historical_dates
            )
            
            # Generate future predictions
            last_sequence = X[-1:]
            future_preds = self.model.predict(last_sequence)
            
            future_dates = pd.date_range(
                start=df.index[-1] + pd.Timedelta(weeks=1),
                periods=self.forecast_horizon,
                freq='W-MON'
            )
            
            future_predictions = pd.DataFrame(
                {
                    'prediction': future_preds.reshape(-1)
                },
                index=future_dates
            )
            
            return historical_preds, future_predictions
            
        except Exception as e:
            logger.error(f"Error in prediction: {str(e)}")
            raise

    def save_model(self, path: str) -> None:
        """Save model to disk."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save model architecture and weights
        self.model.save(path / 'model.h5')
        
        # Save model configuration
        config = {
            'sequence_length': self.sequence_length,
            'forecast_horizon': self.forecast_horizon,
            'n_features': self.n_features,
            'n_filters': self.n_filters,
            'kernel_sizes': self.kernel_sizes,
            'dense_units': self.dense_units,
            'dropout_rate': self.dropout_rate,
            'learning_rate': self.learning_rate
        }
        
        np.save(path / 'config.npy', config)

    @classmethod
    def load_model(cls, path: str) -> 'CNNSalesForecaster':
        """Load model from disk."""
        path = Path(path)
        
        # Load configuration
        config = np.load(path / 'config.npy', allow_pickle=True).item()
        
        # Create model instance
        model = cls(**config)
        
        # Load model architecture and weights
        model.model = tf.keras.models.load_model(path / 'model.h5')
        
        return model
