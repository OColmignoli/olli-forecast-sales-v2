"""
Data ingestion utilities for sales forecasting data.
Handles loading, preprocessing, and feature engineering for sales data.
"""
import pandas as pd
import numpy as np
from typing import Tuple, Dict, Optional
import logging
from pathlib import Path
from .data_validation import load_and_validate_csv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a datetime index from Transaction Year and Week columns.
    
    Args:
        df: Input DataFrame with Transaction Year and Week columns
        
    Returns:
        DataFrame with datetime index
    """
    try:
        # Convert year and week to datetime
        df['date'] = pd.to_datetime(
            df['Transaction Year'].astype(str) + '-' + 
            df['Transaction Week'].astype(str) + '-1', 
            format='%Y-%W-%w'
        )
        
        # Sort by date
        df = df.sort_values('date')
        
        # Set date as index
        df = df.set_index('date')
        
        return df
        
    except Exception as e:
        logger.error(f"Error creating datetime index: {str(e)}")
        raise

def create_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create time-based features for the dataset.
    
    Args:
        df: Input DataFrame with datetime index
        
    Returns:
        DataFrame with additional time features
    """
    try:
        # Extract basic time features
        df['year'] = df.index.year
        df['month'] = df.index.month
        df['week'] = df.index.isocalendar().week
        df['quarter'] = df.index.quarter
        
        # Create cyclical features for week of year
        df['week_sin'] = np.sin(2 * np.pi * df['week']/52.0)
        df['week_cos'] = np.cos(2 * np.pi * df['week']/52.0)
        
        # Create month cyclical features
        df['month_sin'] = np.sin(2 * np.pi * df['month']/12.0)
        df['month_cos'] = np.cos(2 * np.pi * df['month']/12.0)
        
        return df
        
    except Exception as e:
        logger.error(f"Error creating time features: {str(e)}")
        raise

def create_lag_features(df: pd.DataFrame, columns: list, lags: list) -> pd.DataFrame:
    """
    Create lag features for specified columns.
    
    Args:
        df: Input DataFrame
        columns: List of columns to create lags for
        lags: List of lag periods
        
    Returns:
        DataFrame with lag features
    """
    try:
        for col in columns:
            for lag in lags:
                df[f'{col}_lag_{lag}'] = df[col].shift(lag)
        
        return df
        
    except Exception as e:
        logger.error(f"Error creating lag features: {str(e)}")
        raise

def create_rolling_features(df: pd.DataFrame, columns: list, windows: list) -> pd.DataFrame:
    """
    Create rolling window features for specified columns.
    
    Args:
        df: Input DataFrame
        columns: List of columns to create rolling features for
        windows: List of window sizes
        
    Returns:
        DataFrame with rolling features
    """
    try:
        for col in columns:
            for window in windows:
                df[f'{col}_rolling_mean_{window}'] = df[col].rolling(window=window).mean()
                df[f'{col}_rolling_std_{window}'] = df[col].rolling(window=window).std()
                
        return df
        
    except Exception as e:
        logger.error(f"Error creating rolling features: {str(e)}")
        raise

def preprocess_sales_data(
    df: pd.DataFrame,
    create_lags: bool = True,
    create_rolling: bool = True
) -> pd.DataFrame:
    """
    Main preprocessing function for sales data.
    
    Args:
        df: Input DataFrame
        create_lags: Whether to create lag features
        create_rolling: Whether to create rolling features
        
    Returns:
        Preprocessed DataFrame
    """
    try:
        # Create datetime index
        df = create_datetime_index(df)
        
        # Create time features
        df = create_time_features(df)
        
        # Target columns for feature engineering
        target_cols = ['Volume', 'CV Gross Sales', 'CV Net Sales', 'CV Gross Profit']
        
        # Create lag features
        if create_lags:
            df = create_lag_features(
                df,
                columns=target_cols,
                lags=[1, 2, 3, 4, 52]  # Previous weeks and same week last year
            )
        
        # Create rolling features
        if create_rolling:
            df = create_rolling_features(
                df,
                columns=target_cols,
                windows=[4, 12, 26]  # Monthly, quarterly, semi-annual
            )
        
        return df
        
    except Exception as e:
        logger.error(f"Error in preprocessing: {str(e)}")
        raise

def load_and_preprocess_data(
    file_path: str,
    create_lags: bool = True,
    create_rolling: bool = True
) -> Tuple[bool, Optional[pd.DataFrame], Dict[str, any]]:
    """
    Load and preprocess sales data from CSV file.
    
    Args:
        file_path: Path to CSV file
        create_lags: Whether to create lag features
        create_rolling: Whether to create rolling features
        
    Returns:
        Tuple of (success, dataframe, results)
    """
    try:
        # Load and validate data
        is_valid, df, validation_results = load_and_validate_csv(file_path)
        
        if not is_valid:
            logger.error("Data validation failed")
            return False, None, validation_results
        
        # Preprocess data
        df_processed = preprocess_sales_data(
            df,
            create_lags=create_lags,
            create_rolling=create_rolling
        )
        
        # Log preprocessing results
        feature_counts = {
            'original_features': len(df.columns),
            'processed_features': len(df_processed.columns),
            'samples': len(df_processed)
        }
        
        logger.info(f"Preprocessing completed: {feature_counts}")
        
        return True, df_processed, {
            'feature_counts': feature_counts,
            'validation_results': validation_results
        }
        
    except Exception as e:
        logger.error(f"Error in data loading and preprocessing: {str(e)}")
        return False, None, {'error': str(e)}
