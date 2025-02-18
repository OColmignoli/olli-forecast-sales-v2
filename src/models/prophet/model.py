"""
Prophet model implementation for sales forecasting.
Handles model training, prediction, and evaluation.
"""
import pandas as pd
import numpy as np
from prophet import Prophet
from typing import Dict, List, Optional, Tuple
import logging
from pathlib import Path
import json
import mlflow
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ProphetSalesForecaster:
    """Prophet-based sales forecasting model."""
    
    def __init__(
        self,
        target_col: str = 'CV Gross Sales',
        yearly_seasonality: bool = True,
        weekly_seasonality: bool = True,
        daily_seasonality: bool = False,
        seasonality_mode: str = 'multiplicative',
        changepoint_prior_scale: float = 0.05,
        seasonality_prior_scale: float = 10.0
    ):
        """
        Initialize the Prophet forecaster.
        
        Args:
            target_col: Column to forecast
            yearly_seasonality: Whether to include yearly seasonality
            weekly_seasonality: Whether to include weekly seasonality
            daily_seasonality: Whether to include daily seasonality
            seasonality_mode: Either 'additive' or 'multiplicative'
            changepoint_prior_scale: Flexibility of the trend
            seasonality_prior_scale: Strength of the seasonality
        """
        self.target_col = target_col
        self.model = Prophet(
            yearly_seasonality=yearly_seasonality,
            weekly_seasonality=weekly_seasonality,
            daily_seasonality=daily_seasonality,
            seasonality_mode=seasonality_mode,
            changepoint_prior_scale=changepoint_prior_scale,
            seasonality_prior_scale=seasonality_prior_scale
        )
        self.model_params = {
            'target_col': target_col,
            'yearly_seasonality': yearly_seasonality,
            'weekly_seasonality': weekly_seasonality,
            'daily_seasonality': daily_seasonality,
            'seasonality_mode': seasonality_mode,
            'changepoint_prior_scale': changepoint_prior_scale,
            'seasonality_prior_scale': seasonality_prior_scale
        }
    
    def prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare data for Prophet model.
        
        Args:
            df: Input DataFrame with datetime index
            
        Returns:
            DataFrame in Prophet format
        """
        # Prophet requires columns named 'ds' and 'y'
        prophet_df = pd.DataFrame({
            'ds': df.index,
            'y': df[self.target_col]
        })
        
        # Add additional regressors if available
        for col in ['month_sin', 'month_cos', 'week_sin', 'week_cos']:
            if col in df.columns:
                prophet_df[col] = df[col]
                self.model.add_regressor(col)
        
        return prophet_df
    
    def train(
        self,
        df: pd.DataFrame,
        run_id: Optional[str] = None
    ) -> Dict[str, float]:
        """
        Train the Prophet model.
        
        Args:
            df: Training data
            run_id: Optional MLflow run ID
            
        Returns:
            Dictionary with training metrics
        """
        try:
            # Prepare data
            train_df = self.prepare_data(df)
            
            # Start MLflow run if run_id provided
            if run_id:
                mlflow.start_run(run_id=run_id)
                mlflow.log_params(self.model_params)
            
            # Fit model
            self.model.fit(train_df)
            
            # Calculate training metrics
            train_metrics = self.evaluate(df)
            
            # Log metrics if using MLflow
            if run_id:
                mlflow.log_metrics(train_metrics)
                mlflow.end_run()
            
            logger.info(f"Model training completed. Metrics: {train_metrics}")
            return train_metrics
            
        except Exception as e:
            logger.error(f"Error in model training: {str(e)}")
            if run_id:
                mlflow.end_run(status='FAILED')
            raise
    
    def predict(
        self,
        df: pd.DataFrame,
        forecast_horizon: int = 52
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Generate forecasts using the trained model.
        
        Args:
            df: Input data
            forecast_horizon: Number of weeks to forecast
            
        Returns:
            Tuple of (historical_predictions, future_predictions)
        """
        try:
            # Prepare data
            prophet_df = self.prepare_data(df)
            
            # Make predictions for historical data
            historical_forecast = self.model.predict(prophet_df)
            
            # Create future dataframe
            future_dates = pd.date_range(
                start=df.index[-1],
                periods=forecast_horizon + 1,
                freq='W'
            )[1:]  # Exclude the last historical date
            
            future_df = pd.DataFrame({'ds': future_dates})
            
            # Add regressors for future dates if they were used in training
            if 'month_sin' in prophet_df.columns:
                future_df['month_sin'] = np.sin(2 * np.pi * future_dates.month/12.0)
                future_df['month_cos'] = np.cos(2 * np.pi * future_dates.month/12.0)
                future_df['week_sin'] = np.sin(2 * np.pi * future_dates.isocalendar().week/52.0)
                future_df['week_cos'] = np.cos(2 * np.pi * future_dates.isocalendar().week/52.0)
            
            # Generate future predictions
            future_forecast = self.model.predict(future_df)
            
            # Format outputs
            historical_predictions = pd.DataFrame({
                'ds': historical_forecast['ds'],
                'yhat': historical_forecast['yhat'],
                'yhat_lower': historical_forecast['yhat_lower'],
                'yhat_upper': historical_forecast['yhat_upper']
            }).set_index('ds')
            
            future_predictions = pd.DataFrame({
                'ds': future_forecast['ds'],
                'yhat': future_forecast['yhat'],
                'yhat_lower': future_forecast['yhat_lower'],
                'yhat_upper': future_forecast['yhat_upper']
            }).set_index('ds')
            
            return historical_predictions, future_predictions
            
        except Exception as e:
            logger.error(f"Error in prediction: {str(e)}")
            raise
    
    def evaluate(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Evaluate model performance.
        
        Args:
            df: Data to evaluate on
            
        Returns:
            Dictionary of evaluation metrics
        """
        try:
            # Get predictions
            historical_preds, _ = self.predict(df)
            
            # Calculate metrics
            y_true = df[self.target_col]
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
            
            # Save Prophet model
            with open(save_path / 'prophet_model.json', 'w') as f:
                self.model.serialize_model(f)
            
            logger.info(f"Model saved to {path}")
            
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            raise
    
    @classmethod
    def load_model(cls, path: str) -> 'ProphetSalesForecaster':
        """
        Load a trained model.
        
        Args:
            path: Path to load the model from
            
        Returns:
            Loaded ProphetSalesForecaster instance
        """
        try:
            # Load model parameters
            with open(Path(path) / 'model_params.json', 'r') as f:
                model_params = json.load(f)
            
            # Create instance with loaded parameters
            instance = cls(**model_params)
            
            # Load Prophet model
            with open(Path(path) / 'prophet_model.json', 'r') as f:
                instance.model = Prophet.deserialize_model(f)
            
            logger.info(f"Model loaded from {path}")
            return instance
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
