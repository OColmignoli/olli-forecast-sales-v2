"""
Background monitoring service.
"""
import time
import logging
import threading
from typing import Optional
from datetime import datetime, timedelta
import pandas as pd

from .manager import MonitoringManager
from .alerts import AlertManager
from .config import MonitoringConfig

class MonitoringService:
    """Background service for continuous monitoring."""
    
    def __init__(self, env: str = 'development'):
        """
        Initialize monitoring service.
        
        Args:
            env: Environment name
        """
        self.config = MonitoringConfig(env)
        self.monitoring_manager = MonitoringManager(env)
        self.alert_manager = AlertManager(env)
        self.logger = logging.getLogger(__name__)
        
        self._stop_event = threading.Event()
        self._resource_thread: Optional[threading.Thread] = None
        self._drift_thread: Optional[threading.Thread] = None
        
        # Store reference data for drift detection
        self._reference_data: Optional[pd.DataFrame] = None
        self._last_drift_check: Optional[datetime] = None
    
    def start(self) -> None:
        """Start monitoring service."""
        try:
            self.logger.info("Starting monitoring service")
            
            # Start resource monitoring thread
            self._resource_thread = threading.Thread(
                target=self._monitor_resources,
                daemon=True
            )
            self._resource_thread.start()
            
            # Start drift monitoring thread
            self._drift_thread = threading.Thread(
                target=self._monitor_drift,
                daemon=True
            )
            self._drift_thread.start()
            
            self.logger.info("Monitoring service started successfully")
            
        except Exception as e:
            self.logger.error(f"Error starting monitoring service: {str(e)}")
            raise
    
    def stop(self) -> None:
        """Stop monitoring service."""
        try:
            self.logger.info("Stopping monitoring service")
            
            # Set stop event
            self._stop_event.set()
            
            # Wait for threads to finish
            if self._resource_thread:
                self._resource_thread.join(timeout=5.0)
            if self._drift_thread:
                self._drift_thread.join(timeout=5.0)
            
            self.logger.info("Monitoring service stopped successfully")
            
        except Exception as e:
            self.logger.error(f"Error stopping monitoring service: {str(e)}")
            raise
    
    def _monitor_resources(self) -> None:
        """Monitor system resources."""
        resource_config = self.config.resource_monitoring_config
        
        while not self._stop_event.is_set():
            try:
                # Get resource metrics
                metrics = self.monitoring_manager.log_resource_metrics()
                
                # Check thresholds
                for metric in resource_config['metrics']:
                    current_value = metrics.get(metric['name'])
                    if current_value is not None:
                        if current_value >= metric['threshold']:
                            self.alert_manager.send_metric_alert(
                                metric_name=metric['name'],
                                metric_value=current_value,
                                threshold=metric['threshold'],
                                details={'window': metric['window']}
                            )
                
                # Sleep for sampling interval
                time.sleep(resource_config['sampling_interval'])
                
            except Exception as e:
                self.logger.error(f"Error in resource monitoring: {str(e)}")
                self.alert_manager.send_error_alert(
                    error_type="ResourceMonitoringError",
                    error_message=str(e)
                )
                time.sleep(60)  # Sleep longer on error
    
    def _monitor_drift(self) -> None:
        """Monitor data drift."""
        drift_config = self.config.model_monitoring_config['data_drift']
        
        while not self._stop_event.is_set():
            try:
                # Check if drift detection is enabled
                if not drift_config['enabled']:
                    time.sleep(3600)  # Sleep for an hour if disabled
                    continue
                
                # Check if it's time for drift detection
                current_time = datetime.now()
                if (self._last_drift_check and 
                    current_time - self._last_drift_check < 
                    timedelta(seconds=drift_config['check_interval'])):
                    time.sleep(60)
                    continue
                
                # Get current data
                if hasattr(self, 'get_current_data'):
                    current_data = self.get_current_data()
                else:
                    time.sleep(drift_config['check_interval'])
                    continue
                
                # Initialize reference data if needed
                if self._reference_data is None:
                    self._reference_data = current_data
                    self._last_drift_check = current_time
                    continue
                
                # Check for drift
                drift_metrics = self.monitoring_manager.check_data_drift(
                    reference_data=self._reference_data,
                    current_data=current_data,
                    features=drift_config['features_to_monitor']
                )
                
                # Alert if significant drift detected
                for feature in drift_config['features_to_monitor']:
                    ks_statistic = drift_metrics.get(f"{feature}_ks_statistic")
                    if ks_statistic is not None:
                        if ks_statistic > drift_config.get('threshold', 0.1):
                            self.alert_manager.send_metric_alert(
                                metric_name=f"data_drift_{feature}",
                                metric_value=ks_statistic,
                                threshold=drift_config['threshold'],
                                details={
                                    'feature': feature,
                                    'p_value': drift_metrics[f"{feature}_p_value"]
                                }
                            )
                
                # Update reference data and last check time
                self._reference_data = current_data
                self._last_drift_check = current_time
                
                # Sleep until next check
                time.sleep(drift_config['check_interval'])
                
            except Exception as e:
                self.logger.error(f"Error in drift monitoring: {str(e)}")
                self.alert_manager.send_error_alert(
                    error_type="DriftMonitoringError",
                    error_message=str(e)
                )
                time.sleep(300)  # Sleep for 5 minutes on error
    
    def set_reference_data(self, data: pd.DataFrame) -> None:
        """
        Set reference data for drift detection.
        
        Args:
            data: Reference data
        """
        self._reference_data = data.copy()
        self._last_drift_check = datetime.now()
        self.logger.info("Updated reference data for drift detection")
    
    def get_current_data(self) -> pd.DataFrame:
        """
        Get current data for drift detection.
        This method should be implemented by the user.
        
        Returns:
            Current data
        """
        raise NotImplementedError(
            "get_current_data method must be implemented to use drift detection"
        )
