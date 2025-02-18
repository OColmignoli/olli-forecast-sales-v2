import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.models.prophet_model import ProphetModel
from src.models.lstm_model import LSTMModel
from src.models.ensemble_model import StackingEnsemble
from src.utils.data_preprocessing import preprocess_data
from src.monitoring.manager import MonitoringManager

class TestModelPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Create sample data
        dates = pd.date_range(start='2024-01-01', periods=500, freq='D')
        cls.data = pd.DataFrame({
            'Transaction_Date': dates,
            'Volume': np.random.normal(1000, 100, 500),
            'CV_Gross_Sales': np.random.normal(5000, 500, 500),
            'CV_Net_Sales': np.random.normal(4500, 450, 500),
            'CV_COGS': np.random.normal(3000, 300, 500),
            'CV_Gross_Profit': np.random.normal(1500, 150, 500)
        })
        
        # Initialize monitoring
        cls.monitoring = MonitoringManager()

    def test_end_to_end_pipeline(self):
        # 1. Data Preprocessing
        X_train, y_train, X_test, y_test = preprocess_data(
            self.data,
            target_col='CV_Gross_Sales',
            test_size=0.2,
            sequence_length=10
        )
        
        self.assertIsNotNone(X_train)
        self.assertIsNotNone(y_train)
        self.assertTrue(len(X_train) > len(X_test))

        # 2. Model Training
        prophet_model = ProphetModel()
        lstm_model = LSTMModel(
            input_dim=X_train.shape[-1],
            hidden_dim=32,
            num_layers=2,
            output_dim=1
        )

        # Train base models
        prophet_model.fit(self.data[['Transaction_Date', 'CV_Gross_Sales']])
        lstm_model.fit(X_train, y_train)

        # Create and train ensemble
        ensemble = StackingEnsemble(
            base_models=[prophet_model, lstm_model],
            meta_model='xgboost'
        )
        ensemble.fit(X_train, y_train)

        # 3. Model Prediction
        predictions = ensemble.predict(X_test)
        self.assertEqual(len(predictions), len(y_test))

        # 4. Model Evaluation
        metrics = {
            'mae': np.mean(np.abs(predictions - y_test)),
            'rmse': np.sqrt(np.mean((predictions - y_test) ** 2)),
            'r2': 1 - np.sum((y_test - predictions) ** 2) / np.sum((y_test - np.mean(y_test)) ** 2)
        }

        # Log metrics
        self.monitoring.log_model_metrics('ensemble_model', metrics)

        # Assertions for model performance
        self.assertLess(metrics['mae'], 1000)  # Adjust threshold as needed
        self.assertLess(metrics['rmse'], 1500)  # Adjust threshold as needed
        self.assertGreater(metrics['r2'], 0.5)  # Adjust threshold as needed

    def test_data_drift_detection(self):
        # Split data into reference and current
        reference_data = self.data.iloc[:250]
        current_data = self.data.iloc[250:]

        # Check for data drift
        drift_detected = self.monitoring.detect_data_drift(
            reference_data['CV_Gross_Sales'].values,
            current_data['CV_Gross_Sales'].values
        )

        self.assertIsInstance(drift_detected, bool)

    def test_model_retraining_trigger(self):
        # Simulate performance degradation
        poor_metrics = {
            'mae': 2000,
            'rmse': 2500,
            'r2': 0.3
        }

        # Log poor metrics
        self.monitoring.log_model_metrics('ensemble_model', poor_metrics)

        # Check if retraining is needed
        retraining_needed = self.monitoring.check_retraining_needed(
            model_name='ensemble_model',
            metric_thresholds={
                'mae': 1500,
                'rmse': 2000,
                'r2': 0.4
            }
        )

        self.assertTrue(retraining_needed)

if __name__ == '__main__':
    unittest.main()
