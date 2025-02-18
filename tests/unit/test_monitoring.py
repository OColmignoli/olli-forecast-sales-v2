import unittest
from unittest.mock import Mock, patch
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.monitoring.manager import MonitoringManager
from src.monitoring.config import MonitoringConfig

class TestMonitoringManager(unittest.TestCase):
    def setUp(self):
        self.config = MonitoringConfig()
        self.manager = MonitoringManager(self.config)

    @patch('src.monitoring.manager.MonitoringManager._initialize_app_insights')
    def test_log_model_metrics(self, mock_init_insights):
        # Arrange
        metrics = {
            'mae': 0.5,
            'rmse': 0.7,
            'r2': 0.85
        }
        
        # Act
        self.manager.log_model_metrics('test_model', metrics)
        
        # Assert
        self.assertTrue(True)  # Replace with actual assertions

    @patch('src.monitoring.manager.MonitoringManager._initialize_app_insights')
    def test_track_resource_usage(self, mock_init_insights):
        # Act
        metrics = self.manager.get_resource_metrics()
        
        # Assert
        self.assertIn('cpu_percent', metrics)
        self.assertIn('memory_percent', metrics)
        
    @patch('src.monitoring.manager.MonitoringManager._initialize_app_insights')
    def test_detect_data_drift(self, mock_init_insights):
        # Arrange
        reference_data = [1, 2, 3, 4, 5]
        current_data = [1.1, 2.2, 3.3, 4.4, 5.5]
        
        # Act
        drift_detected = self.manager.detect_data_drift(reference_data, current_data)
        
        # Assert
        self.assertIsInstance(drift_detected, bool)

if __name__ == '__main__':
    unittest.main()
