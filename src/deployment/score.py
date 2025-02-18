"""
Scoring script for sales forecasting model deployment.
"""
import os
import json
import logging
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, Any
import mlflow

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def init():
    """Initialize model. This will be called once when the container starts."""
    global model
    
    try:
        # Get model path
        model_path = os.getenv('AZUREML_MODEL_DIR')
        
        # Load ensemble model
        model = mlflow.pyfunc.load_model(model_path)
        
        logger.info("Model loaded successfully")
        
    except Exception as e:
        logger.error(f"Error initializing model: {str(e)}")
        raise

def validate_input(data: Dict[str, Any]) -> bool:
    """
    Validate input data.
    
    Args:
        data: Input data dictionary
        
    Returns:
        True if valid, False otherwise
    """
    required_fields = [
        'Transaction Year',
        'Transaction Week',
        'Volume',
        'CV Gross Sales',
        'CV Net Sales',
        'CV COGS',
        'CV Gross Profit'
    ]
    
    try:
        # Check if all required fields are present
        if not all(field in data.columns for field in required_fields):
            logger.error("Missing required fields in input data")
            return False
        
        # Check data types
        for field in required_fields:
            if not np.issubdtype(data[field].dtype, np.number):
                logger.error(f"Field {field} must be numeric")
                return False
        
        # Check for missing values
        if data.isnull().any().any():
            logger.error("Input data contains missing values")
            return False
        
        return True
        
    except Exception as e:
        logger.error(f"Error validating input: {str(e)}")
        return False

def preprocess_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess input data.
    
    Args:
        data: Input DataFrame
        
    Returns:
        Preprocessed DataFrame
    """
    try:
        # Create date column
        data['date'] = pd.to_datetime(
            data['Transaction Year'].astype(str) + '-W' + 
            data['Transaction Week'].astype(str) + '-1',
            format='%Y-W%W-%w'
        )
        
        # Sort by date
        data = data.sort_values('date')
        
        # Set date as index
        data = data.set_index('date')
        
        return data
        
    except Exception as e:
        logger.error(f"Error preprocessing data: {str(e)}")
        raise

def run(raw_data: str) -> str:
    """
    Generate predictions for input data.
    
    Args:
        raw_data: JSON string containing input data
        
    Returns:
        JSON string with predictions
    """
    try:
        # Parse input data
        data = pd.DataFrame(json.loads(raw_data))
        
        # Validate input
        if not validate_input(data):
            raise ValueError("Invalid input data")
        
        # Preprocess data
        processed_data = preprocess_data(data)
        
        # Generate predictions
        predictions = model.predict(processed_data)
        
        # Format response
        response = {
            'predictions': predictions.to_dict(orient='records'),
            'forecast_start': predictions.index[0].isoformat(),
            'forecast_end': predictions.index[-1].isoformat()
        }
        
        return json.dumps(response)
        
    except Exception as e:
        error_response = {
            'error': str(e),
            'error_type': type(e).__name__
        }
        return json.dumps(error_response)
