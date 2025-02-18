import os
import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
import argparse
from azureml.core import Run

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_data', type=str, help='Path to input data')
    parser.add_argument('--output_dir', type=str, help='Output directory for model and forecasts')
    # Parse known args only to handle filenames with spaces
    args, _ = parser.parse_known_args()
    return args

def preprocess_data(df):
    """Preprocess data for time series forecasting."""
    # Group by week and calculate total sales
    df['date'] = pd.to_datetime(df['Transaction Year'].astype(str) + '-W' + 
                               df['Transaction Week'].astype(str) + '-1')
    
    weekly_sales = df.groupby('date').agg({
        'Volume': 'sum',
        'CV Gross Sales': 'sum',
        'CV Net Sales': 'sum',
        'CV COGS': 'sum',
        'CV Gross Profit': 'sum'
    }).reset_index()
    
    # Rename columns for Prophet
    weekly_sales = weekly_sales.rename(columns={'date': 'ds', 'Volume': 'y'})
    return weekly_sales

def train_prophet_model(df):
    """Train Facebook Prophet model."""
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        seasonality_mode='multiplicative'
    )
    
    model.fit(df)
    
    # Create future dataframe for predictions
    future = model.make_future_dataframe(periods=13, freq='W')
    forecast = model.predict(future)
    
    return model, forecast

def evaluate_model(y_true, y_pred):
    """Calculate model performance metrics."""
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    
    return {
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse
    }

def main():
    # Get run context
    run = Run.get_context()
    
    # Parse arguments
    args = parse_args()
    
    try:
        # Read input CSV file
        print(f"Reading data from {args.input_data}")
        df = pd.read_csv(args.input_data)
        
        # Preprocess data
        processed_data = preprocess_data(df)
        
        # Train model
        model, forecast = train_prophet_model(processed_data)
        
        # Evaluate model
        metrics = evaluate_model(
            processed_data['y'],
            forecast['yhat'][:len(processed_data)]
        )
        
        # Log metrics
        for metric_name, metric_value in metrics.items():
            run.log(metric_name, metric_value)
        
        # Save outputs
        os.makedirs(args.output_dir, exist_ok=True)
        model.save(os.path.join(args.output_dir, 'prophet_model.json'))
        forecast.to_csv(os.path.join(args.output_dir, 'forecast.csv'), index=False)
        
    except Exception as e:
        print(f'Error in training script: {str(e)}')
        raise

if __name__ == "__main__":
    main()
