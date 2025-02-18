"""
FastAPI backend for sales forecasting web interface.
"""
import os
import logging
from typing import Dict, List, Optional
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from datetime import datetime
import pandas as pd
import mlflow
from pathlib import Path

from ...utils.data_ingestion import load_and_preprocess_data
from ...utils.azure_workspace import AzureWorkspaceManager
from ...models.ensemble.model import StackingEnsemble

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="Sales Forecasting API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Azure workspace manager
azure_manager = AzureWorkspaceManager()

# Model cache
MODEL_CACHE = {}

class TrainingRequest(BaseModel):
    """Training request schema."""
    target_column: str
    forecast_horizon: int = 13
    meta_model_type: str = "xgboost"
    feature_engineering_params: Optional[Dict] = None

class PredictionRequest(BaseModel):
    """Prediction request schema."""
    model_id: str
    target_column: str
    forecast_horizon: Optional[int] = None

class ModelMetadata(BaseModel):
    """Model metadata schema."""
    model_id: str
    training_date: str
    metrics: Dict[str, float]
    parameters: Dict[str, any]

@app.on_event("startup")
async def startup_event():
    """Initialize resources on startup."""
    # Initialize MLflow
    mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI', 'local'))
    
    # Initialize Azure workspace
    workspace_success, _ = azure_manager.initialize_workspace()
    if not workspace_success:
        logger.error("Failed to initialize Azure workspace")

def load_model(model_id: str) -> StackingEnsemble:
    """Load model from cache or storage."""
    if model_id in MODEL_CACHE:
        return MODEL_CACHE[model_id]
    
    try:
        # Get model path from MLflow
        model_path = mlflow.get_run(model_id).info.artifact_uri + "/model"
        
        # Load base models (paths should be configured)
        base_model_paths = {
            'prophet': 'models/prophet/latest',
            'lstm': 'models/lstm/latest',
            'transformer': 'models/transformer/latest',
            'deepar': 'models/deepar/latest',
            'cnn': 'models/cnn/latest',
            'automl': 'models/automl/latest'
        }
        
        # Initialize workspace
        _, workspace = azure_manager.initialize_workspace()
        
        # Load ensemble model
        model = StackingEnsemble.load_model(
            path=model_path,
            base_models=base_model_paths
        )
        
        # Cache model
        MODEL_CACHE[model_id] = model
        return model
        
    except Exception as e:
        logger.error(f"Error loading model {model_id}: {str(e)}")
        raise HTTPException(status_code=404, detail=f"Model {model_id} not found")

@app.post("/api/train")
async def train_model(
    file: UploadFile = File(...),
    request: TrainingRequest = None
) -> ModelMetadata:
    """
    Train a new ensemble model.
    
    Args:
        file: CSV file with training data
        request: Training parameters
        
    Returns:
        Model metadata
    """
    try:
        # Save uploaded file
        temp_path = Path("temp") / file.filename
        temp_path.parent.mkdir(exist_ok=True)
        
        with temp_path.open("wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Load and preprocess data
        success, df, results = load_and_preprocess_data(str(temp_path))
        if not success:
            raise HTTPException(status_code=400, detail=f"Data loading failed: {results}")
        
        # Start MLflow run
        with mlflow.start_run() as run:
            # Train model
            model = StackingEnsemble(
                base_models={},  # Will be loaded during training
                meta_model_type=request.meta_model_type,
                forecast_horizon=request.forecast_horizon,
                feature_engineering_params=request.feature_engineering_params
            )
            
            metrics = model.train(df, request.target_column)
            
            # Log parameters and metrics
            mlflow.log_params({
                'target_column': request.target_column,
                'forecast_horizon': request.forecast_horizon,
                'meta_model_type': request.meta_model_type
            })
            
            if request.feature_engineering_params:
                mlflow.log_params(request.feature_engineering_params)
            
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)
            
            # Save model
            model_path = Path("models") / "ensemble" / run.info.run_id
            model.save_model(str(model_path))
            mlflow.log_artifacts(str(model_path), "model")
            
            # Cache model
            MODEL_CACHE[run.info.run_id] = model
            
            # Clean up
            temp_path.unlink()
            
            return ModelMetadata(
                model_id=run.info.run_id,
                training_date=datetime.now().isoformat(),
                metrics=metrics,
                parameters={
                    'target_column': request.target_column,
                    'forecast_horizon': request.forecast_horizon,
                    'meta_model_type': request.meta_model_type,
                    'feature_engineering_params': request.feature_engineering_params
                }
            )
            
    except Exception as e:
        logger.error(f"Error in model training: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/predict")
async def predict(
    file: UploadFile = File(...),
    request: PredictionRequest = None
) -> Dict:
    """
    Generate predictions using trained model.
    
    Args:
        file: CSV file with prediction data
        request: Prediction parameters
        
    Returns:
        Dictionary with predictions
    """
    try:
        # Save uploaded file
        temp_path = Path("temp") / file.filename
        temp_path.parent.mkdir(exist_ok=True)
        
        with temp_path.open("wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Load and preprocess data
        success, df, results = load_and_preprocess_data(str(temp_path))
        if not success:
            raise HTTPException(status_code=400, detail=f"Data loading failed: {results}")
        
        # Load model
        model = load_model(request.model_id)
        
        # Override forecast horizon if specified
        if request.forecast_horizon:
            model.forecast_horizon = request.forecast_horizon
        
        # Generate predictions
        historical_preds, future_preds = model.predict(df, request.target_column)
        
        # Clean up
        temp_path.unlink()
        
        return {
            'historical_predictions': historical_preds.to_dict(orient='records'),
            'future_predictions': future_preds.to_dict(orient='records')
        }
        
    except Exception as e:
        logger.error(f"Error in prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/models")
async def list_models() -> List[ModelMetadata]:
    """List all trained models."""
    try:
        # Get all runs from MLflow
        runs = mlflow.search_runs(experiment_ids=[mlflow.get_experiment_by_name("stacking_ensemble").experiment_id])
        
        models = []
        for _, run in runs.iterrows():
            models.append(ModelMetadata(
                model_id=run.run_id,
                training_date=run.start_time.isoformat(),
                metrics={
                    k: v for k, v in run.items()
                    if k.startswith(('metrics.', 'params.'))
                },
                parameters={
                    k.replace('params.', ''): v for k, v in run.items()
                    if k.startswith('params.')
                }
            ))
        
        return models
        
    except Exception as e:
        logger.error(f"Error listing models: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}
