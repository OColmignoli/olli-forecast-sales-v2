"""
AutoML model for sales forecasting using Azure AutoML.
"""
import os
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from datetime import datetime
from azureml.core import Workspace, Experiment, Dataset
from azureml.train.automl import AutoMLConfig
from azureml.automl.core.forecasting_parameters import ForecastingParameters
from azureml.core.compute import ComputeTarget
from azureml.train.automl.run import AutoMLRun
from azureml.exceptions import UserErrorException

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AutoMLSalesForecaster:
    """
    AutoML model for sales forecasting using Azure AutoML.
    """
    def __init__(
        self,
        workspace: Workspace,
        compute_target: ComputeTarget,
        time_column_name: str = 'date',
        target_column_name: str = 'CV Gross Sales',
        forecast_horizon: int = 13,
        max_horizon: int = 52,
        experiment_timeout_hours: float = 3.0,
        max_concurrent_iterations: int = 4,
        primary_metric: str = 'normalized_root_mean_squared_error'
    ):
        """
        Initialize AutoML forecaster.
        
        Args:
            workspace: Azure ML workspace
            compute_target: Azure ML compute target
            time_column_name: Name of time column
            target_column_name: Name of target column
            forecast_horizon: Number of periods to forecast
            max_horizon: Maximum forecast horizon
            experiment_timeout_hours: Maximum experiment runtime
            max_concurrent_iterations: Maximum concurrent iterations
            primary_metric: Primary metric for model selection
        """
        self.workspace = workspace
        self.compute_target = compute_target
        self.time_column_name = time_column_name
        self.target_column_name = target_column_name
        self.forecast_horizon = forecast_horizon
        self.max_horizon = max_horizon
        self.experiment_timeout_hours = experiment_timeout_hours
        self.max_concurrent_iterations = max_concurrent_iterations
        self.primary_metric = primary_metric
        
        self.best_run = None
        self.fitted_model = None

    def prepare_training_data(
        self,
        df: pd.DataFrame,
        freq: str = 'W'
    ) -> Tuple[Dataset, Dict[str, List[str]]]:
        """
        Prepare data for AutoML training.
        
        Args:
            df: Input DataFrame
            freq: Time series frequency
            
        Returns:
            Tuple of (dataset, feature_columns)
        """
        try:
            # Ensure datetime index
            if not isinstance(df.index, pd.DatetimeIndex):
                raise ValueError("DataFrame must have DatetimeIndex")
            
            # Reset index to make date a column
            df = df.reset_index()
            df.rename(columns={'index': 'date'}, inplace=True)
            
            # Select feature columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            exclude_cols = ['Transaction Year', 'Transaction Week']
            feature_cols = [col for col in numeric_cols if col not in exclude_cols]
            
            # Create feature definitions
            feature_definitions = {
                'features': feature_cols,
                'time_series_id_columns': [],  # Single time series
                'time_column_name': self.time_column_name,
                'target_column_name': self.target_column_name
            }
            
            # Create Azure ML dataset
            datastore = self.workspace.get_default_datastore()
            temp_path = f'automl_data_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
            
            df.to_csv(temp_path + '.csv', index=False)
            dataset = Dataset.Tabular.from_delimited_files(
                path=[(datastore, temp_path + '.csv')]
            )
            
            # Clean up temporary file
            os.remove(temp_path + '.csv')
            
            return dataset, feature_definitions
            
        except Exception as e:
            logger.error(f"Error in data preparation: {str(e)}")
            raise

    def create_automl_config(
        self,
        training_data: Dataset,
        feature_definitions: Dict[str, List[str]]
    ) -> AutoMLConfig:
        """
        Create AutoML configuration.
        
        Args:
            training_data: Training dataset
            feature_definitions: Feature column definitions
            
        Returns:
            AutoML configuration
        """
        try:
            # Create forecasting parameters
            forecasting_parameters = ForecastingParameters(
                time_column_name=self.time_column_name,
                forecast_horizon=self.forecast_horizon,
                time_series_id_column_names=feature_definitions['time_series_id_columns'],
                target_rolling_window_size=self.max_horizon
            )
            
            # Create AutoML config
            automl_config = AutoMLConfig(
                task='forecasting',
                primary_metric=self.primary_metric,
                training_data=training_data,
                label_column_name=self.target_column_name,
                compute_target=self.compute_target,
                experiment_timeout_minutes=int(self.experiment_timeout_hours * 60),
                max_concurrent_iterations=self.max_concurrent_iterations,
                enable_early_stopping=True,
                enable_ensembling=True,
                enable_stack_ensembling=True,
                verbosity=logging.INFO,
                path='.',
                forecasting_parameters=forecasting_parameters
            )
            
            return automl_config
            
        except Exception as e:
            logger.error(f"Error in AutoML configuration: {str(e)}")
            raise

    def train(
        self,
        df: pd.DataFrame,
        experiment_name: str = 'automl_sales_forecast'
    ) -> Dict[str, float]:
        """
        Train AutoML model.
        
        Args:
            df: Input DataFrame
            experiment_name: Name of experiment
            
        Returns:
            Dictionary of training metrics
        """
        try:
            # Prepare training data
            training_data, feature_definitions = self.prepare_training_data(df)
            
            # Create AutoML config
            automl_config = self.create_automl_config(
                training_data,
                feature_definitions
            )
            
            # Create experiment
            experiment = Experiment(self.workspace, experiment_name)
            
            # Submit experiment
            logger.info("Submitting AutoML experiment...")
            remote_run = experiment.submit(automl_config)
            
            # Wait for completion
            remote_run.wait_for_completion(show_output=True)
            
            # Get best run and model
            self.best_run = remote_run.get_best_run_by_primary_metric()
            self.fitted_model = self.best_run.get_output()
            
            # Get metrics
            metrics = self.best_run.get_metrics()
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error in training: {str(e)}")
            raise

    def predict(
        self,
        df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Generate predictions.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Tuple of (historical_predictions, future_predictions)
        """
        try:
            if self.fitted_model is None:
                raise ValueError("Model must be trained before prediction")
            
            # Prepare data for prediction
            X_test = df.copy()
            
            # Generate predictions
            y_pred = self.fitted_model.forecast(X_test)
            
            # Create historical predictions DataFrame
            historical_dates = df.index[-len(y_pred):]
            historical_preds = pd.DataFrame(
                {
                    'prediction': y_pred,
                    'actual': df[self.target_column_name].values[-len(y_pred):]
                },
                index=historical_dates
            )
            
            # Generate future predictions
            future_dates = pd.date_range(
                start=df.index[-1] + pd.Timedelta(weeks=1),
                periods=self.forecast_horizon,
                freq='W-MON'
            )
            
            future_preds = self.fitted_model.forecast_quantiles(
                X_test,
                quantiles=[0.025, 0.5, 0.975]
            )
            
            future_predictions = pd.DataFrame(
                {
                    'prediction': future_preds[0.5],
                    'lower_bound': future_preds[0.025],
                    'upper_bound': future_preds[0.975]
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
        
        if self.fitted_model is None:
            raise ValueError("Model must be trained before saving")
        
        # Save model
        self.fitted_model.save(path / 'model.pkl')
        
        # Save configuration
        config = {
            'time_column_name': self.time_column_name,
            'target_column_name': self.target_column_name,
            'forecast_horizon': self.forecast_horizon,
            'max_horizon': self.max_horizon,
            'experiment_timeout_hours': self.experiment_timeout_hours,
            'max_concurrent_iterations': self.max_concurrent_iterations,
            'primary_metric': self.primary_metric
        }
        
        pd.to_pickle(config, path / 'config.pkl')

    @classmethod
    def load_model(
        cls,
        path: str,
        workspace: Workspace,
        compute_target: ComputeTarget
    ) -> 'AutoMLSalesForecaster':
        """Load model from disk."""
        path = Path(path)
        
        # Load configuration
        config = pd.read_pickle(path / 'config.pkl')
        
        # Create model instance
        model = cls(workspace, compute_target, **config)
        
        # Load model
        model.fitted_model = AutoMLRun.load_model(path / 'model.pkl')
        
        return model
