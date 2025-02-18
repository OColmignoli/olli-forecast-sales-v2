from azureml.core import Workspace, Experiment, Environment, ScriptRunConfig
from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core.conda_dependencies import CondaDependencies
import os
import yaml

def setup_azure_environment():
    # Get workspace
    ws = Workspace.get(
        name="olli-forecast-ml",
        resource_group="olli-forecast-rg",
        subscription_id=os.getenv("AZURE_SUBSCRIPTION_ID")
    )

    # Create compute target if it doesn't exist
    compute_name = "forecast-cluster"
    if compute_name not in ws.compute_targets:
        compute_config = AmlCompute.provisioning_configuration(
            vm_size="STANDARD_DS3_V2",
            min_nodes=0,
            max_nodes=4
        )
        compute_target = ComputeTarget.create(ws, compute_name, compute_config)
        compute_target.wait_for_completion(show_output=True)
    else:
        compute_target = ws.compute_targets[compute_name]

    # Create Azure ML environment
    env = Environment("forecast-env")
    conda_dep = CondaDependencies()

    # Add required packages
    packages = [
        "tensorflow==2.13.0",
        "torch==2.0.1",
        "pytorch-lightning==2.0.2",
        "transformers==4.30.2",
        "gluonts==0.13.2",
        "prophet==1.1.4",
        "pandas",
        "numpy",
        "scikit-learn",
        "azureml-sdk",
        "azureml-defaults"
    ]
    
    for package in packages:
        conda_dep.add_pip_package(package)

    env.python.conda_dependencies = conda_dep

    # Register environment
    env.register(workspace=ws)

    return ws, compute_target, env

def create_experiment(ws, name):
    """Create or get an experiment"""
    return Experiment(workspace=ws, name=name)

def main():
    # Set up Azure environment
    ws, compute_target, env = setup_azure_environment()
    
    # Create experiments for each model type
    experiments = {
        "lstm": create_experiment(ws, "lstm-forecast"),
        "transformer": create_experiment(ws, "transformer-forecast"),
        "deepar": create_experiment(ws, "deepar-forecast"),
        "prophet": create_experiment(ws, "prophet-forecast"),
        "cnn": create_experiment(ws, "cnn-forecast"),
        "automl": create_experiment(ws, "automl-forecast"),
        "ensemble": create_experiment(ws, "ensemble-forecast")
    }
    
    print("Azure ML workspace and experiments are ready!")
    return ws, compute_target, env, experiments

if __name__ == "__main__":
    main()
