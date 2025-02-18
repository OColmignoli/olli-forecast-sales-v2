"""
Azure ML workspace initialization and management.
Handles workspace setup, data registration, and experiment tracking.
"""
import os
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple
from azure.ai.ml import MLClient, Input, Output
from azure.ai.ml.entities import Data
from azure.ai.ml.constants import AssetTypes
from azure.identity import DefaultAzureCredential
from .azure_setup import get_ml_client, setup_compute_cluster

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AzureWorkspaceManager:
    """Manages Azure ML workspace operations."""
    
    def __init__(self):
        """Initialize the workspace manager."""
        self.client = None
        self.compute_target = None
    
    def initialize_workspace(self) -> Tuple[bool, Optional[str]]:
        """
        Initialize the Azure ML workspace and required resources.
        
        Returns:
            Tuple of (success, error_message)
        """
        try:
            # Get ML client
            self.client = get_ml_client()
            
            # Set up compute cluster
            self.compute_target = setup_compute_cluster(
                self.client,
                compute_name="sales-forecast-cluster"
            )
            
            logger.info("Successfully initialized Azure ML workspace")
            return True, None
            
        except Exception as e:
            error_msg = f"Failed to initialize workspace: {str(e)}"
            logger.error(error_msg)
            return False, error_msg
    
    def register_dataset(
        self,
        data_path: str,
        dataset_name: str,
        dataset_description: str
    ) -> Tuple[bool, Optional[Data]]:
        """
        Register a dataset in the Azure ML workspace.
        
        Args:
            data_path: Path to the data file
            dataset_name: Name for the dataset
            dataset_description: Description of the dataset
            
        Returns:
            Tuple of (success, dataset)
        """
        try:
            # Create data asset
            my_data = Data(
                path=data_path,
                type=AssetTypes.URI_FILE,
                description=dataset_description,
                name=dataset_name,
                version='1.0.0'
            )
            
            # Register the data asset
            registered_data = self.client.data.create_or_update(my_data)
            
            logger.info(f"Successfully registered dataset: {dataset_name}")
            return True, registered_data
            
        except Exception as e:
            logger.error(f"Failed to register dataset: {str(e)}")
            return False, None
    
    def create_experiment(
        self,
        experiment_name: str,
        model_name: str,
        compute_name: Optional[str] = None
    ) -> Dict[str, any]:
        """
        Create an experiment configuration for model training.
        
        Args:
            experiment_name: Name of the experiment
            model_name: Name of the model being trained
            compute_name: Optional compute target name
            
        Returns:
            Dictionary with experiment configuration
        """
        try:
            # Use specified compute or default
            compute_name = compute_name or "sales-forecast-cluster"
            
            # Create experiment config
            experiment_config = {
                "experiment_name": experiment_name,
                "compute_name": compute_name,
                "model_name": model_name,
                "tags": {
                    "model_type": model_name,
                    "project": "sales_forecast"
                },
                "properties": {
                    "pipeline_name": f"{model_name}_training",
                    "created_by": "azure_workspace_manager"
                }
            }
            
            logger.info(f"Created experiment configuration for {experiment_name}")
            return experiment_config
            
        except Exception as e:
            logger.error(f"Failed to create experiment config: {str(e)}")
            raise
    
    def setup_model_environment(
        self,
        model_name: str,
        environment_name: Optional[str] = None
    ) -> Tuple[bool, Optional[str]]:
        """
        Set up the training environment for a specific model.
        
        Args:
            model_name: Name of the model
            environment_name: Optional custom environment name
            
        Returns:
            Tuple of (success, error_message)
        """
        try:
            from azure.ai.ml.entities import Environment, BuildContext
            
            # Use model-specific or default environment name
            env_name = environment_name or f"{model_name.lower()}-env"
            
            # Create environment
            env = Environment(
                name=env_name,
                description=f"Environment for {model_name} model",
                build=BuildContext(path=".")
            )
            
            # Register environment
            self.client.environments.create_or_update(env)
            
            logger.info(f"Successfully set up environment: {env_name}")
            return True, None
            
        except Exception as e:
            error_msg = f"Failed to set up model environment: {str(e)}"
            logger.error(error_msg)
            return False, error_msg
    
    def get_workspace_info(self) -> Dict[str, any]:
        """
        Get information about the current workspace.
        
        Returns:
            Dictionary with workspace information
        """
        try:
            workspace = self.client.workspace
            
            info = {
                "name": workspace.name,
                "location": workspace.location,
                "resource_group": workspace.resource_group,
                "compute_targets": [
                    compute.name for compute in self.client.compute.list()
                ],
                "datasets": [
                    dataset.name for dataset in self.client.data.list()
                ],
                "environments": [
                    env.name for env in self.client.environments.list()
                ]
            }
            
            return info
            
        except Exception as e:
            logger.error(f"Failed to get workspace info: {str(e)}")
            raise
