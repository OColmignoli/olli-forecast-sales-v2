from azureml.core import Workspace, Environment
from azureml.core.conda_dependencies import CondaDependencies

# Get workspace
ws = Workspace(
    subscription_id="c828c783-7a28-48f4-b56f-a6c189437d77",
    resource_group="OLLI-resource",
    workspace_name="OLLI_ML_Forecast"
)

# Create environment
env = Environment(name="forecast-env")
env.python.conda_dependencies = CondaDependencies.create(
    python_version="3.8",
    pip_packages=[
        "prophet",
        "scikit-learn",
        "pandas",
        "numpy",
        "matplotlib",
        "statsmodels",
        "azureml-core",
        "azureml-dataset-runtime",
        "azureml-defaults",
        "azureml-telemetry",
        "azureml-train-automl-client"
    ]
)

# Register environment
env.register(workspace=ws)
print("Environment registered successfully!")
