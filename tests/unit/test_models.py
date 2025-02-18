import unittest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.models.prophet_model import ProphetModel
from src.models.lstm_model import LSTMModel
from src.models.transformer_model import TransformerModel
from src.models.deepar_model import DeepARModel
from src.models.cnn_model import CNNModel
from src.models.ensemble_model import StackingEnsemble

class TestProphetModel(unittest.TestCase):
    def setUp(self):
        self.model = ProphetModel()
        self.sample_data = pd.DataFrame({
            'ds': pd.date_range(start='2024-01-01', periods=100, freq='D'),
            'y': np.random.normal(0, 1, 100)
        })

    def test_model_initialization(self):
        self.assertIsNotNone(self.model)
        self.assertTrue(hasattr(self.model, 'model'))

    def test_model_fit(self):
        with patch('prophet.Prophet.fit') as mock_fit:
            self.model.fit(self.sample_data)
            mock_fit.assert_called_once()

    def test_model_predict(self):
        with patch('prophet.Prophet.predict') as mock_predict:
            mock_predict.return_value = pd.DataFrame({
                'ds': pd.date_range(start='2024-04-10', periods=30, freq='D'),
                'yhat': np.random.normal(0, 1, 30)
            })
            forecast = self.model.predict(periods=30)
            self.assertEqual(len(forecast), 30)

class TestLSTMModel(unittest.TestCase):
    def setUp(self):
        self.model = LSTMModel(
            input_dim=1,
            hidden_dim=32,
            num_layers=2,
            output_dim=1
        )
        self.sample_data = np.random.normal(0, 1, (100, 1))

    def test_model_initialization(self):
        self.assertIsNotNone(self.model)
        self.assertTrue(hasattr(self.model, 'lstm'))
        self.assertTrue(hasattr(self.model, 'linear'))

    @patch('tensorflow.keras.Model.fit')
    def test_model_fit(self, mock_fit):
        X = np.random.normal(0, 1, (90, 10, 1))
        y = np.random.normal(0, 1, (90, 1))
        self.model.fit(X, y, epochs=10)
        mock_fit.assert_called_once()

    @patch('tensorflow.keras.Model.predict')
    def test_model_predict(self, mock_predict):
        mock_predict.return_value = np.random.normal(0, 1, (10, 1))
        X = np.random.normal(0, 1, (10, 10, 1))
        predictions = self.model.predict(X)
        self.assertEqual(predictions.shape, (10, 1))

class TestTransformerModel(unittest.TestCase):
    def setUp(self):
        self.model = TransformerModel(
            input_dim=1,
            d_model=32,
            nhead=4,
            num_layers=2,
            output_dim=1
        )
        self.sample_data = np.random.normal(0, 1, (100, 1))

    def test_model_initialization(self):
        self.assertIsNotNone(self.model)
        self.assertTrue(hasattr(self.model, 'transformer'))
        self.assertTrue(hasattr(self.model, 'decoder'))

    @patch('pytorch_lightning.Trainer.fit')
    def test_model_fit(self, mock_fit):
        X = np.random.normal(0, 1, (90, 10, 1))
        y = np.random.normal(0, 1, (90, 1))
        self.model.fit(X, y)
        mock_fit.assert_called_once()

    def test_model_predict(self):
        X = np.random.normal(0, 1, (10, 10, 1))
        with patch.object(self.model, 'transformer') as mock_transformer:
            mock_transformer.return_value = np.random.normal(0, 1, (10, 1))
            predictions = self.model.predict(X)
            self.assertEqual(predictions.shape, (10, 1))

class TestStackingEnsemble(unittest.TestCase):
    def setUp(self):
        self.base_models = [
            Mock(name='prophet'),
            Mock(name='lstm'),
            Mock(name='transformer')
        ]
        self.meta_model = Mock(name='xgboost')
        self.ensemble = StackingEnsemble(
            base_models=self.base_models,
            meta_model=self.meta_model
        )

    def test_ensemble_initialization(self):
        self.assertEqual(len(self.ensemble.base_models), 3)
        self.assertIsNotNone(self.ensemble.meta_model)

    def test_ensemble_fit(self):
        X = np.random.normal(0, 1, (100, 10))
        y = np.random.normal(0, 1, 100)
        
        # Mock base model predictions
        for model in self.base_models:
            model.predict.return_value = np.random.normal(0, 1, 100)
        
        self.ensemble.fit(X, y)
        
        # Verify base models were called
        for model in self.base_models:
            model.fit.assert_called_once()
        
        # Verify meta model was called
        self.meta_model.fit.assert_called_once()

    def test_ensemble_predict(self):
        X = np.random.normal(0, 1, (10, 10))
        
        # Mock base model predictions
        for model in self.base_models:
            model.predict.return_value = np.random.normal(0, 1, 10)
        
        # Mock meta model prediction
        self.meta_model.predict.return_value = np.random.normal(0, 1, 10)
        
        predictions = self.ensemble.predict(X)
        self.assertEqual(len(predictions), 10)

if __name__ == '__main__':
    unittest.main()
