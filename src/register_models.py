from azureml.core import Model
from azureml.core.model import Model
from azureml.core import Workspace
import os

def register_model(workspace, model_path, model_name, tags=None):
    """Register a model in Azure ML workspace"""
    return Model.register(
        workspace=workspace,
        model_path=model_path,
        model_name=model_name,
        tags=tags,
        description=f"Sales forecasting model: {model_name}"
    )

def get_latest_model(workspace, model_name):
    """Get the latest version of a registered model"""
    try:
        model = Model(workspace, model_name)
        return model
    except Exception as e:
        print(f"Model {model_name} not found: {e}")
        return None

def list_registered_models(workspace):
    """List all registered models in the workspace"""
    models = Model.list(workspace)
    for model in models:
        print(f"Name: {model.name}, Version: {model.version}, Status: {model.status}")
    return models

def main():
    # Get workspace
    ws = Workspace.get(
        name="olli-forecast-ml",
        resource_group="olli-forecast-rg",
        subscription_id=os.getenv("AZURE_SUBSCRIPTION_ID")
    )
    
    # List existing models
    print("Currently registered models:")
    list_registered_models(ws)

if __name__ == "__main__":
    main()
