"""
Transformer model for sales forecasting using PyTorch Lightning.
"""
import torch
import torch.nn as nn
import pytorch_lightning as pl
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PositionalEncoding(nn.Module):
    """Positional encoding for transformer model."""
    def __init__(self, d_model: int, max_seq_length: int = 100):
        super().__init__()
        position = torch.arange(max_seq_length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe = torch.zeros(max_seq_length, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input tensor."""
        return x + self.pe[:x.size(0)]

class TransformerSalesForecaster(pl.LightningModule):
    """
    Transformer model for sales forecasting.
    """
    def __init__(
        self,
        n_features: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        learning_rate: float = 1e-4,
        sequence_length: int = 52,
        forecast_horizon: int = 13
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # Model parameters
        self.n_features = n_features
        self.d_model = d_model
        self.sequence_length = sequence_length
        self.forecast_horizon = forecast_horizon
        self.learning_rate = learning_rate
        
        # Input projection
        self.input_projection = nn.Linear(n_features, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, sequence_length)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        # Output projection
        self.output_projection = nn.Linear(d_model, 1)
        
        # Loss function
        self.criterion = nn.MSELoss()
        
        # Metrics
        self.training_step_outputs = []
        self.validation_step_outputs = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model."""
        # Input shape: (batch_size, seq_len, n_features)
        x = self.input_projection(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        output = self.output_projection(x)
        return output

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training step."""
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log('train_loss', loss)
        self.training_step_outputs.append(loss)
        return loss

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Validation step."""
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log('val_loss', loss)
        self.validation_step_outputs.append(loss)
        return loss

    def on_train_epoch_end(self) -> None:
        """Called at the end of training epoch."""
        epoch_loss = torch.stack(self.training_step_outputs).mean()
        self.log('train_epoch_loss', epoch_loss)
        self.training_step_outputs.clear()

    def on_validation_epoch_end(self) -> None:
        """Called at the end of validation epoch."""
        epoch_loss = torch.stack(self.validation_step_outputs).mean()
        self.log('val_epoch_loss', epoch_loss)
        self.validation_step_outputs.clear()

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Configure optimizer."""
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def prepare_data(self, df: pd.DataFrame, target_col: str, features: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare data for training.
        
        Args:
            df: Input DataFrame
            target_col: Target column name
            features: List of feature column names
            
        Returns:
            Tuple of (X, y) tensors
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
        
        return (
            torch.FloatTensor(np.array(X_sequences)),
            torch.FloatTensor(np.array(y_sequences)).unsqueeze(-1)
        )

    def predict(
        self,
        df: pd.DataFrame,
        features: List[str],
        target_col: str
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Generate predictions.
        
        Args:
            df: Input DataFrame
            features: Feature columns
            target_col: Target column name
            
        Returns:
            Tuple of (historical_predictions, future_predictions)
        """
        self.eval()
        with torch.no_grad():
            # Prepare data
            X, _ = self.prepare_data(df, target_col, features)
            
            # Generate predictions
            predictions = self(X)
            predictions = predictions.numpy()
            
            # Create DataFrames
            dates = pd.date_range(
                start=df.index[self.sequence_length],
                periods=len(predictions),
                freq='W-MON'
            )
            
            historical_preds = pd.DataFrame(
                predictions.reshape(-1, 1),
                index=dates,
                columns=['prediction']
            )
            historical_preds['actual'] = df[target_col].values[self.sequence_length:self.sequence_length + len(predictions)]
            
            # Generate future predictions
            last_sequence = X[-1:]
            future_preds = self(last_sequence).numpy()
            
            future_dates = pd.date_range(
                start=dates[-1] + pd.Timedelta(weeks=1),
                periods=self.forecast_horizon,
                freq='W-MON'
            )
            
            future_predictions = pd.DataFrame(
                future_preds.reshape(-1, 1),
                index=future_dates,
                columns=['prediction']
            )
            
            return historical_preds, future_predictions

    def save_model(self, path: str) -> None:
        """Save model to disk."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path / 'model.pt')
        torch.save(self.hparams, path / 'hparams.pt')

    @classmethod
    def load_model(cls, path: str) -> 'TransformerSalesForecaster':
        """Load model from disk."""
        path = Path(path)
        hparams = torch.load(path / 'hparams.pt')
        model = cls(**hparams)
        model.load_state_dict(torch.load(path / 'model.pt'))
        return model
