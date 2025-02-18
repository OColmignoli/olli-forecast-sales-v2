# Pipeline configuration for Azure ML Studio
from azureml.core import Workspace
from azureml.pipeline.core import Pipeline
from azureml.pipeline.steps import PythonScriptStep
from azureml.core.runconfig import RunConfiguration
from azureml.core.compute import ComputeTarget

# Get workspace
ws = Workspace.from_config()

# Get compute target
compute_target = ComputeTarget(workspace=ws, name="forecast-cpu-cluster")

# Create run configuration
run_config = RunConfiguration()
run_config.target = compute_target
run_config.environment.name = "forecast-env"

# Configure the training step
train_step = PythonScriptStep(
    name="train_forecast_model",
    source_directory=".",
    script_name="train.py",
    compute_target=compute_target,
    runconfig=run_config
)

# Create and publish pipeline
pipeline = Pipeline(workspace=ws, steps=[train_step])
published_pipeline = pipeline.publish(
    name="Sales_Forecast_Training_Pipeline",
    description="Pipeline for training sales forecasting models"
)
