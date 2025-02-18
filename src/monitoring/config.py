"""
Configuration for monitoring and logging system.
"""
import os
from pathlib import Path
from typing import Dict, Any
import yaml
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class MonitoringConfig:
    """Configuration for monitoring and logging."""
    
    def __init__(self, env: str = 'development'):
        """
        Initialize monitoring configuration.
        
        Args:
            env: Environment name
        """
        self.env = env
        self.config = self._load_config()
        
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        config_path = Path(__file__).parent / 'config' / f'{self.env}.yaml'
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Override with environment variables
        config['app_insights'] = {
            'connection_string': os.getenv('APPINSIGHTS_CONNECTION_STRING', 
                                         config['app_insights']['connection_string']),
            'instrumentation_key': os.getenv('APPINSIGHTS_INSTRUMENTATION_KEY',
                                           config['app_insights']['instrumentation_key'])
        }
        
        return config
    
    @property
    def app_insights_config(self) -> Dict[str, str]:
        """Get Application Insights configuration."""
        return self.config['app_insights']
    
    @property
    def logging_config(self) -> Dict[str, Any]:
        """Get logging configuration."""
        return self.config['logging']
    
    @property
    def model_monitoring_config(self) -> Dict[str, Any]:
        """Get model monitoring configuration."""
        return self.config['model_monitoring']
    
    @property
    def resource_monitoring_config(self) -> Dict[str, Any]:
        """Get resource monitoring configuration."""
        return self.config['resource_monitoring']
    
    @property
    def alert_config(self) -> Dict[str, Any]:
        """Get alerting configuration."""
        return self.config['alerts']
