"""
Deployment configuration for sales forecasting models.
"""
import os
from pathlib import Path
from typing import Dict, Any
import yaml
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class DeploymentConfig:
    """Configuration for model deployment."""
    
    def __init__(self, env: str = 'development'):
        """
        Initialize deployment configuration.
        
        Args:
            env: Environment ('development', 'staging', or 'production')
        """
        self.env = env
        self.config = self._load_config()
        
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        config_path = Path(__file__).parent / 'config' / f'{self.env}.yaml'
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            
        # Override with environment variables if present
        config['azure'] = {
            'subscription_id': os.getenv('AZURE_SUBSCRIPTION_ID', config['azure']['subscription_id']),
            'resource_group': os.getenv('AZURE_RESOURCE_GROUP', config['azure']['resource_group']),
            'workspace_name': os.getenv('AZURE_WORKSPACE_NAME', config['azure']['workspace_name']),
            'region': os.getenv('AZURE_REGION', config['azure']['region'])
        }
        
        return config
    
    @property
    def azure_config(self) -> Dict[str, str]:
        """Get Azure configuration."""
        return self.config['azure']
    
    @property
    def compute_config(self) -> Dict[str, Any]:
        """Get compute configuration."""
        return self.config['compute']
    
    @property
    def deployment_config(self) -> Dict[str, Any]:
        """Get deployment configuration."""
        return self.config['deployment']
    
    @property
    def monitoring_config(self) -> Dict[str, Any]:
        """Get monitoring configuration."""
        return self.config['monitoring']
