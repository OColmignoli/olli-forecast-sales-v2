"""
Deployment manager for sales forecasting models.
"""
import os
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple
import mlflow
from azureml.core import Workspace, Model, Environment
from azureml.core.model import InferenceConfig
from azureml.core.webservice import AciWebservice, LocalWebservice
from azureml.exceptions import WebserviceException
from azureml.core.authentication import ServicePrincipalAuthentication

from .config import DeploymentConfig
from ..utils.azure_workspace import AzureWorkspaceManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelDeploymentManager:
    """Manager for model deployment operations."""
    
    def __init__(self, env: str = 'development'):
        """
        Initialize deployment manager.
        
        Args:
            env: Environment name
        """
        self.config = DeploymentConfig(env)
        self.azure_manager = AzureWorkspaceManager()
        
    def _get_latest_model(self, model_name: str) -> Tuple[str, str]:
        """
        Get latest model version from MLflow.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Tuple of (run_id, model_path)
        """
        client = mlflow.tracking.MlflowClient()
        
        # Get latest model version
        versions = client.search_model_versions(f"name='{model_name}'")
        if not versions:
            raise ValueError(f"No versions found for model {model_name}")
            
        latest_version = sorted(versions, key=lambda x: x.version, reverse=True)[0]
        
        return latest_version.run_id, latest_version.source

    def _create_inference_config(
        self,
        model_path: str,
        score_script: str = 'score.py'
    ) -> InferenceConfig:
        """
        Create inference configuration.
        
        Args:
            model_path: Path to model
            score_script: Name of scoring script
            
        Returns:
            Inference configuration
        """
        env = Environment.from_conda_specification(
            name='sales_forecast_env',
            file_path=str(Path(model_path) / 'conda.yaml')
        )
        
        return InferenceConfig(
            entry_script=score_script,
            source_directory=model_path,
            environment=env
        )

    def _create_deployment_config(
        self,
        deployment_type: str
    ) -> Dict:
        """
        Create deployment configuration.
        
        Args:
            deployment_type: Type of deployment ('AzureML' or 'Local')
            
        Returns:
            Deployment configuration
        """
        compute_config = self.config.compute_config
        
        if deployment_type == 'AzureML':
            return AciWebservice.deploy_configuration(
                cpu_cores=compute_config['cpu_cores'],
                memory_gb=compute_config['memory_gb'],
                auth_enabled=self.config.deployment_config['auth_enabled'],
                ssl_enabled=self.config.deployment_config['ssl_enabled'],
                enable_app_insights=self.config.deployment_config['app_insights_enabled'],
                tags={
                    'environment': self.config.env,
                    'project': 'sales_forecast'
                }
            )
        else:
            return LocalWebservice.deploy_configuration(
                port=8890
            )

    def deploy_model(
        self,
        model_name: str,
        deployment_name: Optional[str] = None,
        deployment_type: str = 'AzureML'
    ) -> str:
        """
        Deploy model to specified environment.
        
        Args:
            model_name: Name of the model to deploy
            deployment_name: Optional name for deployment
            deployment_type: Type of deployment ('AzureML' or 'Local')
            
        Returns:
            Endpoint URL
        """
        try:
            # Initialize Azure workspace
            success, workspace = self.azure_manager.initialize_workspace()
            if not success:
                raise ValueError("Failed to initialize Azure workspace")
            
            # Get latest model
            run_id, model_path = self._get_latest_model(model_name)
            
            # Create or update deployment name
            if not deployment_name:
                deployment_name = (
                    f"{self.config.deployment_config['endpoint_name']}-"
                    f"{model_name}-{run_id[:8]}"
                )
            
            # Create inference config
            inference_config = self._create_inference_config(model_path)
            
            # Create deployment config
            deployment_config = self._create_deployment_config(deployment_type)
            
            try:
                # Try to update existing service
                logger.info(f"Updating existing service: {deployment_name}")
                service = Model.deploy(
                    workspace=workspace,
                    name=deployment_name,
                    models=[model_path],
                    inference_config=inference_config,
                    deployment_config=deployment_config,
                    overwrite=True
                )
            except WebserviceException:
                # Create new service if update fails
                logger.info(f"Creating new service: {deployment_name}")
                service = Model.deploy(
                    workspace=workspace,
                    name=deployment_name,
                    models=[model_path],
                    inference_config=inference_config,
                    deployment_config=deployment_config
                )
            
            service.wait_for_deployment(show_output=True)
            
            if service.state != "Healthy":
                raise ValueError(
                    f"Deployment failed. Service state: {service.state}"
                )
            
            # Enable monitoring if configured
            if self.config.monitoring_config['enable_model_monitoring']:
                self._enable_monitoring(service)
            
            logger.info(f"Model deployed successfully. Endpoint: {service.scoring_uri}")
            return service.scoring_uri
            
        except Exception as e:
            logger.error(f"Error in model deployment: {str(e)}")
            raise

    def _enable_monitoring(self, service: AciWebservice) -> None:
        """
        Enable monitoring for deployed service.
        
        Args:
            service: Deployed web service
        """
        try:
            monitoring_config = self.config.monitoring_config
            
            # Enable data collection
            service.update(
                enable_app_insights=True,
                sampling_rate=monitoring_config['sampling_rate']
            )
            
            # Set up alerts
            if monitoring_config.get('performance_alert_threshold'):
                thresholds = monitoring_config['performance_alert_threshold']
                
                service.update_monitoring(
                    latency_threshold=thresholds['latency_ms'],
                    error_threshold=thresholds['error_rate']
                )
            
            logger.info("Monitoring enabled successfully")
            
        except Exception as e:
            logger.error(f"Error enabling monitoring: {str(e)}")
            raise

    def delete_deployment(
        self,
        deployment_name: str,
        deployment_type: str = 'AzureML'
    ) -> None:
        """
        Delete model deployment.
        
        Args:
            deployment_name: Name of deployment to delete
            deployment_type: Type of deployment ('AzureML' or 'Local')
        """
        try:
            # Initialize Azure workspace
            success, workspace = self.azure_manager.initialize_workspace()
            if not success:
                raise ValueError("Failed to initialize Azure workspace")
            
            # Get service
            if deployment_type == 'AzureML':
                service = AciWebservice(workspace, deployment_name)
            else:
                service = LocalWebservice(workspace, deployment_name)
            
            # Delete service
            service.delete()
            logger.info(f"Deployment {deployment_name} deleted successfully")
            
        except Exception as e:
            logger.error(f"Error deleting deployment: {str(e)}")
            raise

    def list_deployments(self) -> Dict:
        """
        List all active deployments.
        
        Returns:
            Dictionary of deployment information
        """
        try:
            # Initialize Azure workspace
            success, workspace = self.azure_manager.initialize_workspace()
            if not success:
                raise ValueError("Failed to initialize Azure workspace")
            
            # Get all services
            aci_services = AciWebservice.list(workspace)
            local_services = LocalWebservice.list(workspace)
            
            deployments = {}
            
            # Process ACI services
            for service in aci_services:
                deployments[service.name] = {
                    'type': 'AzureML',
                    'state': service.state,
                    'endpoint': service.scoring_uri,
                    'created_time': service.created_time,
                    'compute': {
                        'cpu_cores': service.compute_type.cpu_cores,
                        'memory_gb': service.compute_type.memory_gb
                    }
                }
            
            # Process local services
            for service in local_services:
                deployments[service.name] = {
                    'type': 'Local',
                    'state': service.state,
                    'endpoint': service.scoring_uri,
                    'created_time': service.created_time
                }
            
            return deployments
            
        except Exception as e:
            logger.error(f"Error listing deployments: {str(e)}")
            raise
