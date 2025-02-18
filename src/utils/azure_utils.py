from azureml.core import Workspace
from azureml.core.authentication import InteractiveLoginAuthentication
import yaml
import os
from pathlib import Path

def load_config():
    config_path = Path(__file__).parents[2] / "config" / "azure_config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)

def get_workspace():
    """Get or create Azure ML workspace"""
    config = load_config()
    ws_config = config["workspace"]
    
    try:
        ws = Workspace.get(
            name=ws_config["name"],
            resource_group=ws_config["resource_group"],
            subscription_id=os.getenv("AZURE_SUBSCRIPTION_ID"),
            auth=InteractiveLoginAuthentication()
        )
        print("Found existing workspace.")
        return ws
    except Exception as e:
        print(f"Workspace not found: {e}")
        print("Please ensure you have the correct permissions and subscription ID.")
        return None

def setup_compute(workspace, config):
    """Set up compute clusters if they don't exist"""
    from azureml.core.compute import AmlCompute, ComputeTarget
    
    for cluster_type, cluster_config in config["compute"].items():
        if cluster_type not in workspace.compute_targets:
            compute_config = AmlCompute.provisioning_configuration(
                vm_size=cluster_config["vm_size"],
                min_nodes=cluster_config["min_nodes"],
                max_nodes=cluster_config["max_nodes"],
                idle_seconds_before_scaledown=cluster_config["idle_seconds_before_scaledown"]
            )
            
            cluster = ComputeTarget.create(workspace, cluster_type, compute_config)
            cluster.wait_for_completion(show_output=True)

def main():
    workspace = get_workspace()
    if workspace:
        config = load_config()
        setup_compute(workspace, config)
        print("Azure ML workspace and compute resources are ready.")

if __name__ == "__main__":
    main()
