from azure.ai.ml import MLClient, dsl
from azure.ai.ml.entities import Component
from azure.identity import DefaultAzureCredential
from azure.ai.ml.constants import AssetTypes

# Connect to the workspace
credential = DefaultAzureCredential()
ml_client = MLClient(
    credential=credential,
    subscription_id="c828c783-7a28-48f4-b56f-a6c189437d77",
    resource_group_name="OLLI-resource",
    workspace_name="OLLI_ML_Forecast"
)

# Create the component
@dsl.component
def sales_forecast_training(
    input_data: dsl.Input(type=AssetTypes.URI_FOLDER),
    model_output: dsl.Output(type=AssetTypes.URI_FOLDER)
):
    """Train a sales forecasting model using Prophet.
    
    Args:
        input_data: Input sales history data
        model_output: Output directory for trained model and forecasts
    """
    return dsl.component(
        name="sales_forecast_training",
        version="1.0.0",
        display_name="Sales Forecast Training",
        type="command",
        code="./train_src",
        environment="azureml:forecast-env:1",
        command="python train.py --input_data ${{inputs.input_data}} --output_dir ${{outputs.model_output}}"
    )

# Register the component
component = sales_forecast_training()
registered_component = ml_client.components.create_or_update(component)
print(f"Registered component: {registered_component.name} (version {registered_component.version})")
