"""
Azure ML workspace setup and management utilities.
Handles workspace configuration, compute resources, and environment setup.
"""
import os
from typing import Dict, Optional
import logging
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
from azure.ai.ml.entities import AmlCompute
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

def get_ml_client() -> MLClient:
    """
    Get or create Azure ML client using environment variables.
    
    Returns:
        MLClient: Authenticated Azure ML client
    """
    try:
        # Get credentials and client
        credential = DefaultAzureCredential()
        client = MLClient(
            credential=credential,
            subscription_id=os.getenv('AZURE_SUBSCRIPTION_ID'),
            resource_group_name=os.getenv('AZURE_RESOURCE_GROUP'),
            workspace_name=os.getenv('AZURE_WORKSPACE_NAME')
        )
        logger.info(f"Successfully connected to workspace: {os.getenv('AZURE_WORKSPACE_NAME')}")
        return client
    except Exception as e:
        logger.error(f"Error connecting to Azure ML workspace: {str(e)}")
        raise

def setup_compute_cluster(
    client: MLClient,
    compute_name: str,
    vm_size: str = "Standard_DS3_v2",
    min_instances: int = 0,
    max_instances: int = 4,
    idle_time_before_scale_down: int = 120
) -> AmlCompute:
    """
    Create or get a compute cluster for training.
    
    Args:
        client: Azure ML client
        compute_name: Name of the compute cluster
        vm_size: VM size to use
        min_instances: Minimum number of nodes
        max_instances: Maximum number of nodes
        idle_time_before_scale_down: Idle time before scaling down in seconds
        
    Returns:
        AmlCompute: Compute cluster
    """
    try:
        # Check if compute target already exists
        try:
            compute = client.compute.get(compute_name)
            logger.info(f"Found existing compute target: {compute_name}")
            return compute
        except Exception:
            logger.info(f"Creating new compute target: {compute_name}")
            
        # Define compute configuration
        compute_config = AmlCompute(
            name=compute_name,
            type="amlcompute",
            size=vm_size,
            min_instances=min_instances,
            max_instances=max_instances,
            idle_time_before_scale_down=idle_time_before_scale_down
        )
        
        # Create compute target
        compute = client.compute.begin_create_or_update(compute_config).result()
        
        logger.info(f"Created compute target: {compute_name}")
        return compute
        
    except Exception as e:
        logger.error(f"Error setting up compute cluster: {str(e)}")
        raise

def validate_workspace_connection() -> Tuple[bool, Optional[str]]:
    """
    Validate connection to Azure ML workspace.
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        client = get_ml_client()
        workspace = client.workspace
        return True, None
    except Exception as e:
        return False, str(e)

def setup_training_environment(
    client: MLClient,
    environment_name: str,
    conda_file: str = "conda.yaml"
) -> None:
    """
    Set up the training environment with required dependencies.
    
    Args:
        client: Azure ML client
        environment_name: Name of the environment
        conda_file: Path to conda environment file
    """
    try:
        from azure.ai.ml.entities import Environment, BuildContext
        
        # Create environment
        env = Environment(
            name=environment_name,
            description="Environment for sales forecasting models",
            build=BuildContext(path=".")
        )
        
        # Register environment
        client.environments.create_or_update(env)
        logger.info(f"Created/updated environment: {environment_name}")
        
    except Exception as e:
        logger.error(f"Error setting up training environment: {str(e)}")
        raise

def initialize_azure_ml() -> Tuple[bool, Dict[str, any]]:
    """
    Initialize Azure ML workspace and required resources.
    
    Returns:
        Tuple of (success, resources)
    """
    try:
        # Validate workspace connection
        is_valid, error = validate_workspace_connection()
        if not is_valid:
            return False, {"error": error}
        
        # Get ML client
        client = get_ml_client()
        
        # Set up compute cluster
        compute = setup_compute_cluster(
            client,
            compute_name="sales-forecast-cluster"
        )
        
        # Set up training environment
        setup_training_environment(
            client,
            environment_name="sales-forecast-env"
        )
        
        return True, {
            "client": client,
            "compute": compute
        }
        
    except Exception as e:
        logger.error(f"Error initializing Azure ML: {str(e)}")
        return False, {"error": str(e)}
