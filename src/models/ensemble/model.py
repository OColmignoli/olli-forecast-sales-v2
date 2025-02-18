"""
Stacking ensemble model for sales forecasting.
"""
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import mlflow
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import xgboost as xgb
from datetime import datetime

from ..prophet.model import ProphetSalesForecaster
from ..lstm.model import LSTMSalesForecaster
from ..transformer.model import TransformerSalesForecaster
from ..deepar.model import DeepARSalesForecaster
from ..cnn.model import CNNSalesForecaster
from ..automl.model import AutoMLSalesForecaster

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StackingEnsemble:
    """
    Stacking ensemble model combining predictions from multiple base models.
    """
    def __init__(
        self,
        base_models: Dict[str, Any],
        meta_model_type: str = 'xgboost',
        forecast_horizon: int = 13,
        feature_engineering_params: Optional[Dict] = None
    ):
        """
        Initialize stacking ensemble.
        
        Args:
            base_models: Dictionary of base models
            meta_model_type: Type of meta-model ('xgboost', 'rf', or 'gbm')
            forecast_horizon: Number of periods to forecast
            feature_engineering_params: Optional feature engineering parameters
        """
        self.base_models = base_models
        self.meta_model_type = meta_model_type
        self.forecast_horizon = forecast_horizon
        self.feature_engineering_params = feature_engineering_params or {}
        
        self.meta_model = None
        self.base_predictions = {}
        self.feature_columns = []

    def _create_meta_model(self) -> Any:
        """Create meta-model based on specified type."""
        if self.meta_model_type == 'xgboost':
            return xgb.XGBRegressor(
                objective='reg:squarederror',
                n_estimators=100,
                learning_rate=0.1,
                max_depth=4,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42
            )
        elif self.meta_model_type == 'rf':
            return RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                random_state=42
            )
        elif self.meta_model_type == 'gbm':
            return GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=4,
                random_state=42
            )
        else:
            raise ValueError(f"Unsupported meta-model type: {self.meta_model_type}")

    def _generate_base_predictions(
        self,
        df: pd.DataFrame,
        target_col: str
    ) -> pd.DataFrame:
        """
        Generate predictions from all base models.
        
        Args:
            df: Input DataFrame
            target_col: Target column name
            
        Returns:
            DataFrame with base model predictions
        """
        predictions = []
        
        for model_name, model in self.base_models.items():
            logger.info(f"Generating predictions for {model_name}")
            
            try:
                # Get predictions from base model
                historical_preds, _ = model.predict(df, target_col=target_col)
                
                # Store predictions
                predictions.append(
                    historical_preds['prediction'].rename(f'{model_name}_pred')
                )
                
                self.base_predictions[model_name] = historical_preds
                
            except Exception as e:
                logger.error(f"Error generating predictions for {model_name}: {str(e)}")
                continue
        
        # Combine all predictions
        return pd.concat(predictions, axis=1)

    def _engineer_meta_features(
        self,
        df: pd.DataFrame,
        base_preds: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Engineer features for meta-model.
        
        Args:
            df: Input DataFrame
            base_preds: Base model predictions
            
        Returns:
            DataFrame with engineered features
        """
        features = base_preds.copy()
        
        # Add time-based features
        features['week_of_year'] = df.index.isocalendar().week
        features['month'] = df.index.month
        features['quarter'] = df.index.quarter
        
        # Add rolling statistics of base predictions
        for model in base_preds.columns:
            # Rolling mean
            features[f'{model}_roll_mean_4'] = base_preds[model].rolling(4).mean()
            features[f'{model}_roll_mean_12'] = base_preds[model].rolling(12).mean()
            
            # Rolling std
            features[f'{model}_roll_std_4'] = base_preds[model].rolling(4).std()
            features[f'{model}_roll_std_12'] = base_preds[model].rolling(12).std()
        
        # Add model agreement features
        features['pred_std'] = features[base_preds.columns].std(axis=1)
        features['pred_range'] = features[base_preds.columns].max(axis=1) - features[base_preds.columns].min(axis=1)
        
        # Store feature columns
        self.feature_columns = features.columns.tolist()
        
        return features

    def train(
        self,
        df: pd.DataFrame,
        target_col: str
    ) -> Dict[str, float]:
        """
        Train the ensemble model.
        
        Args:
            df: Input DataFrame
            target_col: Target column name
            
        Returns:
            Dictionary of training metrics
        """
        try:
            # Generate base model predictions
            base_preds = self._generate_base_predictions(df, target_col)
            
            # Engineer meta-features
            meta_features = self._engineer_meta_features(df, base_preds)
            
            # Prepare target
            y = df[target_col].loc[meta_features.index]
            
            # Handle missing values
            meta_features = meta_features.fillna(method='ffill').fillna(method='bfill')
            
            # Create and train meta-model
            self.meta_model = self._create_meta_model()
            self.meta_model.fit(meta_features, y)
            
            # Calculate metrics
            y_pred = self.meta_model.predict(meta_features)
            mse = np.mean((y - y_pred) ** 2)
            mae = np.mean(np.abs(y - y_pred))
            
            metrics = {
                'train_mse': mse,
                'train_mae': mae,
                'train_rmse': np.sqrt(mse)
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error in ensemble training: {str(e)}")
            raise

    def predict(
        self,
        df: pd.DataFrame,
        target_col: str
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Generate ensemble predictions.
        
        Args:
            df: Input DataFrame
            target_col: Target column name
            
        Returns:
            Tuple of (historical_predictions, future_predictions)
        """
        try:
            # Generate base model predictions
            base_preds = self._generate_base_predictions(df, target_col)
            
            # Engineer meta-features
            meta_features = self._engineer_meta_features(df, base_preds)
            
            # Generate predictions
            predictions = self.meta_model.predict(meta_features)
            
            # Create historical predictions DataFrame
            historical_preds = pd.DataFrame(
                {
                    'prediction': predictions,
                    'actual': df[target_col].loc[meta_features.index]
                },
                index=meta_features.index
            )
            
            # Generate future predictions
            future_dates = pd.date_range(
                start=df.index[-1] + pd.Timedelta(weeks=1),
                periods=self.forecast_horizon,
                freq='W-MON'
            )
            
            # Collect future predictions from base models
            future_base_preds = {}
            prediction_intervals = []
            
            for model_name, model in self.base_models.items():
                _, future_pred = model.predict(df, target_col=target_col)
                future_base_preds[f'{model_name}_pred'] = future_pred['prediction']
                
                # Collect uncertainty intervals if available
                if 'lower_bound' in future_pred.columns:
                    prediction_intervals.append(
                        (future_pred['lower_bound'], future_pred['upper_bound'])
                    )
            
            # Create future meta-features
            future_meta_features = pd.DataFrame(future_base_preds, index=future_dates)
            future_meta_features['week_of_year'] = future_dates.isocalendar().week
            future_meta_features['month'] = future_dates.month
            future_meta_features['quarter'] = future_dates.quarter
            
            # Add other meta-features
            for col in self.feature_columns:
                if col not in future_meta_features.columns:
                    future_meta_features[col] = future_meta_features[future_base_preds.keys()].mean(axis=1)
            
            # Generate future predictions
            future_predictions = pd.DataFrame(
                {
                    'prediction': self.meta_model.predict(future_meta_features[self.feature_columns])
                },
                index=future_dates
            )
            
            # Add uncertainty bounds if available
            if prediction_intervals:
                lower_bounds = np.mean([lb for lb, _ in prediction_intervals], axis=0)
                upper_bounds = np.mean([ub for _, ub in prediction_intervals], axis=0)
                
                future_predictions['lower_bound'] = lower_bounds
                future_predictions['upper_bound'] = upper_bounds
            
            return historical_preds, future_predictions
            
        except Exception as e:
            logger.error(f"Error in ensemble prediction: {str(e)}")
            raise

    def save_model(self, path: str) -> None:
        """Save model to disk."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save meta-model
        if self.meta_model_type == 'xgboost':
            self.meta_model.save_model(str(path / 'meta_model.json'))
        else:
            pd.to_pickle(self.meta_model, path / 'meta_model.pkl')
        
        # Save configuration
        config = {
            'meta_model_type': self.meta_model_type,
            'forecast_horizon': self.forecast_horizon,
            'feature_engineering_params': self.feature_engineering_params,
            'feature_columns': self.feature_columns
        }
        
        pd.to_pickle(config, path / 'config.pkl')

    @classmethod
    def load_model(
        cls,
        path: str,
        base_models: Dict[str, Any]
    ) -> 'StackingEnsemble':
        """Load model from disk."""
        path = Path(path)
        
        # Load configuration
        config = pd.read_pickle(path / 'config.pkl')
        
        # Create model instance
        model = cls(
            base_models=base_models,
            meta_model_type=config['meta_model_type'],
            forecast_horizon=config['forecast_horizon'],
            feature_engineering_params=config['feature_engineering_params']
        )
        
        # Load meta-model
        if config['meta_model_type'] == 'xgboost':
            model.meta_model = xgb.XGBRegressor()
            model.meta_model.load_model(str(path / 'meta_model.json'))
        else:
            model.meta_model = pd.read_pickle(path / 'meta_model.pkl')
        
        # Set feature columns
        model.feature_columns = config['feature_columns']
        
        return model
