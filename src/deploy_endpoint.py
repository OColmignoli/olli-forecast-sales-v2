from azureml.core import Workspace, Model, Environment
from azureml.core.webservice import AciWebservice
from azureml.core.model import InferenceConfig
import os

def deploy_model(workspace, model_name, entry_script):
    """Deploy a model as a web service"""
    
    # Get the registered model
    model = Model(workspace, model_name)
    
    # Create inference config
    env = Environment.get(workspace, "forecast-env")
    inference_config = InferenceConfig(
        entry_script=entry_script,
        environment=env
    )
    
    # Create deployment config
    deployment_config = AciWebservice.deploy_configuration(
        cpu_cores=1,
        memory_gb=1,
        auth_enabled=True,
        enable_app_insights=True,
        collect_model_data=True
    )
    
    # Deploy the web service
    service = Model.deploy(
        workspace=workspace,
        name=f"{model_name}-service",
        models=[model],
        inference_config=inference_config,
        deployment_config=deployment_config,
        overwrite=True
    )
    
    service.wait_for_deployment(show_output=True)
    return service

def main():
    # Get workspace
    ws = Workspace.get(
        name="olli-forecast-ml",
        resource_group="olli-forecast-rg",
        subscription_id=os.getenv("AZURE_SUBSCRIPTION_ID")
    )
    
    # Deploy each model type
    models = {
        "lstm": "score_lstm.py",
        "transformer": "score_transformer.py",
        "deepar": "score_deepar.py",
        "prophet": "score_prophet.py",
        "cnn": "score_cnn.py",
        "ensemble": "score_ensemble.py"
    }
    
    for model_name, entry_script in models.items():
        print(f"Deploying {model_name} model...")
        service = deploy_model(ws, model_name, entry_script)
        print(f"Deployment successful. Scoring URI: {service.scoring_uri}")

if __name__ == "__main__":
    main()
