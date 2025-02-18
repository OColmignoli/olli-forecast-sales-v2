"""
Monitoring manager for the sales forecasting system.
"""
import os
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
import psutil
import GPUtil
import pandas as pd
from opencensus.ext.azure.log_exporter import AzureLogHandler
from opencensus.ext.azure.metrics_exporter import MetricsExporter
from opencensus.stats import stats as opencensus_stats
from opencensus.trace import config_integration
from opencensus.ext.azure import metrics_exporter
import mlflow

from .config import MonitoringConfig

# Configure OpenCensus integration
config_integration.trace_integrations(['logging', 'requests'])

class MonitoringManager:
    """Manager for monitoring and logging operations."""
    
    def __init__(self, env: str = 'development'):
        """
        Initialize monitoring manager.
        
        Args:
            env: Environment name
        """
        self.config = MonitoringConfig(env)
        self._setup_logging()
        self._setup_metrics_exporter()
        self.logger = logging.getLogger(__name__)
        
    def _setup_logging(self) -> None:
        """Set up logging configuration."""
        config = self.config.logging_config
        
        # Create logs directory if it doesn't exist
        if config['file']['enabled']:
            log_dir = Path(config['file']['path'])
            log_dir.mkdir(parents=True, exist_ok=True)
        
        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(config['level'])
        
        # Configure formatter
        formatter = logging.Formatter(
            fmt=config['format'],
            datefmt=config['date_format']
        )
        
        # Add file handler
        if config['file']['enabled']:
            from logging.handlers import RotatingFileHandler
            
            file_handler = RotatingFileHandler(
                filename=log_dir / f"{datetime.now().strftime('%Y%m%d')}.log",
                maxBytes=config['file']['max_size_mb'] * 1024 * 1024,
                backupCount=config['file']['backup_count']
            )
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)
        
        # Add console handler
        if config['console']['enabled']:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            root_logger.addHandler(console_handler)
        
        # Add Azure Application Insights handler
        if config['app_insights']['enabled']:
            app_insights_handler = AzureLogHandler(
                connection_string=self.config.app_insights_config['connection_string']
            )
            app_insights_handler.setFormatter(formatter)
            root_logger.addHandler(app_insights_handler)
    
    def _setup_metrics_exporter(self) -> None:
        """Set up metrics exporter for Azure Application Insights."""
        self.metrics_exporter = MetricsExporter(
            connection_string=self.config.app_insights_config['connection_string']
        )
        
        # Create measurement map for custom metrics
        self.mmap = self.metrics_exporter.get_metrics_map()
    
    def log_model_metrics(
        self,
        model_name: str,
        metrics: Dict[str, float],
        run_id: Optional[str] = None
    ) -> None:
        """
        Log model performance metrics.
        
        Args:
            model_name: Name of the model
            metrics: Dictionary of metric names and values
            run_id: Optional MLflow run ID
        """
        try:
            # Log to MLflow if enabled
            if self.config.logging_config['mlflow']['enabled']:
                if run_id:
                    with mlflow.start_run(run_id=run_id):
                        mlflow.log_metrics(metrics)
                else:
                    with mlflow.start_run():
                        mlflow.log_metrics(metrics)
            
            # Log to Application Insights
            for metric_name, value in metrics.items():
                self.mmap.measure_float_put(f"{model_name}_{metric_name}", value)
            
            self.metrics_exporter.export_metrics()
            self.logger.info(f"Logged metrics for model {model_name}: {metrics}")
            
        except Exception as e:
            self.logger.error(f"Error logging model metrics: {str(e)}")
            raise
    
    def log_resource_metrics(self) -> Dict[str, float]:
        """
        Log system resource metrics.
        
        Returns:
            Dictionary of resource metrics
        """
        try:
            metrics = {}
            
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            metrics['cpu_usage'] = cpu_percent
            
            # Memory metrics
            memory = psutil.virtual_memory()
            metrics['memory_usage'] = memory.percent
            metrics['memory_available_gb'] = memory.available / (1024 ** 3)
            
            # Disk metrics
            disk = psutil.disk_usage('/')
            metrics['disk_usage'] = disk.percent
            metrics['disk_available_gb'] = disk.free / (1024 ** 3)
            
            # GPU metrics if available
            try:
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu_metrics = {
                        'gpu_usage': sum(gpu.load * 100 for gpu in gpus) / len(gpus),
                        'gpu_memory_usage': sum(gpu.memoryUtil * 100 for gpu in gpus) / len(gpus)
                    }
                    metrics.update(gpu_metrics)
            except Exception as gpu_error:
                self.logger.warning(f"Could not get GPU metrics: {str(gpu_error)}")
            
            # Log metrics to Application Insights
            for metric_name, value in metrics.items():
                self.mmap.measure_float_put(f"resource_{metric_name}", value)
            
            self.metrics_exporter.export_metrics()
            self.logger.debug(f"Logged resource metrics: {metrics}")
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error logging resource metrics: {str(e)}")
            raise
    
    def check_data_drift(
        self,
        reference_data: pd.DataFrame,
        current_data: pd.DataFrame,
        features: List[str]
    ) -> Dict[str, float]:
        """
        Check for data drift between reference and current data.
        
        Args:
            reference_data: Reference data
            current_data: Current data
            features: List of features to monitor
            
        Returns:
            Dictionary of drift metrics
        """
        try:
            from scipy.stats import ks_2samp
            
            drift_metrics = {}
            
            for feature in features:
                if feature in reference_data.columns and feature in current_data.columns:
                    # Perform Kolmogorov-Smirnov test
                    ks_statistic, p_value = ks_2samp(
                        reference_data[feature],
                        current_data[feature]
                    )
                    
                    drift_metrics[f"{feature}_ks_statistic"] = ks_statistic
                    drift_metrics[f"{feature}_p_value"] = p_value
                    
                    # Log drift metrics
                    self.mmap.measure_float_put(
                        f"drift_{feature}_ks_statistic",
                        ks_statistic
                    )
                    self.mmap.measure_float_put(
                        f"drift_{feature}_p_value",
                        p_value
                    )
            
            self.metrics_exporter.export_metrics()
            self.logger.info(f"Data drift metrics: {drift_metrics}")
            
            return drift_metrics
            
        except Exception as e:
            self.logger.error(f"Error checking data drift: {str(e)}")
            raise
    
    def log_prediction_metrics(
        self,
        model_name: str,
        latency: float,
        error: Optional[str] = None
    ) -> None:
        """
        Log prediction performance metrics.
        
        Args:
            model_name: Name of the model
            latency: Prediction latency in milliseconds
            error: Optional error message
        """
        try:
            # Log latency
            self.mmap.measure_float_put(
                f"{model_name}_prediction_latency",
                latency
            )
            
            # Log error rate
            error_value = 1.0 if error else 0.0
            self.mmap.measure_float_put(
                f"{model_name}_prediction_error_rate",
                error_value
            )
            
            # Export metrics
            self.metrics_exporter.export_metrics()
            
            # Log details
            if error:
                self.logger.error(
                    f"Prediction error for {model_name}: {error}. "
                    f"Latency: {latency}ms"
                )
            else:
                self.logger.info(
                    f"Successful prediction for {model_name}. "
                    f"Latency: {latency}ms"
                )
                
        except Exception as e:
            self.logger.error(f"Error logging prediction metrics: {str(e)}")
            raise
    
    def log_user_activity(
        self,
        user_id: str,
        activity_type: str,
        details: Dict[str, Any]
    ) -> None:
        """
        Log user activity.
        
        Args:
            user_id: User identifier
            activity_type: Type of activity
            details: Activity details
        """
        try:
            # Create activity log
            activity_log = {
                'user_id': user_id,
                'activity_type': activity_type,
                'timestamp': datetime.utcnow().isoformat(),
                'details': details
            }
            
            # Log to Application Insights
            self.logger.info(
                f"User activity: {activity_log}",
                extra={'custom_dimensions': activity_log}
            )
            
        except Exception as e:
            self.logger.error(f"Error logging user activity: {str(e)}")
            raise
