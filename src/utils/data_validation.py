"""
Data validation utilities for sales forecasting data.
Handles validation of CSV data according to required schema and business rules.
"""
import pandas as pd
import numpy as np
from typing import Tuple, List, Dict
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define required columns and their data types
REQUIRED_COLUMNS = {
    'Transaction Year': 'int32',
    'Transaction Week': 'int32',
    'Volume': 'float64',
    'CV Gross Sales': 'float64',
    'CV Net Sales': 'float64',
    'CV COGS': 'float64',
    'CV Gross Profit': 'float64'
}

def validate_column_presence(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """
    Validate that all required columns are present in the DataFrame.
    
    Args:
        df: Input DataFrame to validate
        
    Returns:
        Tuple of (is_valid, missing_columns)
    """
    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    return len(missing_cols) == 0, missing_cols

def validate_data_types(df: pd.DataFrame) -> Tuple[bool, Dict[str, str]]:
    """
    Validate that columns have correct data types.
    
    Args:
        df: Input DataFrame to validate
        
    Returns:
        Tuple of (is_valid, invalid_columns)
    """
    invalid_cols = {}
    
    for col, expected_type in REQUIRED_COLUMNS.items():
        if col in df.columns:
            # Check if column can be converted to expected type
            try:
                df[col].astype(expected_type)
            except (ValueError, TypeError):
                invalid_cols[col] = f"Expected {expected_type}, got {df[col].dtype}"
    
    return len(invalid_cols) == 0, invalid_cols

def validate_value_ranges(df: pd.DataFrame) -> Tuple[bool, Dict[str, str]]:
    """
    Validate that values are within expected ranges.
    
    Args:
        df: Input DataFrame to validate
        
    Returns:
        Tuple of (is_valid, range_violations)
    """
    violations = {}
    
    # Week number should be between 1 and 53
    if 'Transaction Week' in df.columns:
        invalid_weeks = df[~df['Transaction Week'].between(1, 53)]['Transaction Week'].unique()
        if len(invalid_weeks) > 0:
            violations['Transaction Week'] = f"Invalid weeks found: {invalid_weeks}"
    
    # All numeric values should be non-negative
    numeric_cols = ['Volume', 'CV Gross Sales', 'CV Net Sales', 'CV COGS', 'CV Gross Profit']
    for col in numeric_cols:
        if col in df.columns:
            negative_values = df[df[col] < 0][col].count()
            if negative_values > 0:
                violations[col] = f"Found {negative_values} negative values"
    
    return len(violations) == 0, violations

def validate_temporal_consistency(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """
    Validate temporal consistency of the data.
    
    Args:
        df: Input DataFrame to validate
        
    Returns:
        Tuple of (is_valid, inconsistencies)
    """
    issues = []
    
    if all(col in df.columns for col in ['Transaction Year', 'Transaction Week']):
        # Sort by year and week
        df_sorted = df.sort_values(['Transaction Year', 'Transaction Week'])
        
        # Check for gaps in weeks
        df_sorted['next_year'] = df_sorted['Transaction Year'].shift(-1)
        df_sorted['next_week'] = df_sorted['Transaction Week'].shift(-1)
        
        # Find gaps within years
        year_groups = df_sorted.groupby('Transaction Year')
        for year, group in year_groups:
            weeks = group['Transaction Week'].unique()
            expected_weeks = set(range(1, max(weeks) + 1))
            missing_weeks = expected_weeks - set(weeks)
            if missing_weeks:
                issues.append(f"Year {year} missing weeks: {sorted(missing_weeks)}")
        
        # Check year transitions
        invalid_transitions = df_sorted[
            (df_sorted['next_year'] == df_sorted['Transaction Year']) &
            (df_sorted['next_week'] != df_sorted['Transaction Week'] + 1) &
            (df_sorted['next_week'] != 1)  # Allow transition to week 1 for new year
        ]
        
        if len(invalid_transitions) > 0:
            issues.append("Found invalid week transitions between consecutive records")
    
    return len(issues) == 0, issues

def validate_sales_data(df: pd.DataFrame) -> Tuple[bool, Dict[str, any]]:
    """
    Main validation function that runs all checks on the sales data.
    
    Args:
        df: Input DataFrame to validate
        
    Returns:
        Tuple of (is_valid, validation_results)
    """
    validation_results = {}
    
    # Check column presence
    cols_valid, missing_cols = validate_column_presence(df)
    if not cols_valid:
        validation_results['missing_columns'] = missing_cols
    
    # Check data types
    types_valid, invalid_types = validate_data_types(df)
    if not types_valid:
        validation_results['invalid_types'] = invalid_types
    
    # Check value ranges
    ranges_valid, range_violations = validate_value_ranges(df)
    if not ranges_valid:
        validation_results['range_violations'] = range_violations
    
    # Check temporal consistency
    temporal_valid, temporal_issues = validate_temporal_consistency(df)
    if not temporal_valid:
        validation_results['temporal_issues'] = temporal_issues
    
    # Overall validation result
    is_valid = cols_valid and types_valid and ranges_valid and temporal_valid
    
    if is_valid:
        logger.info("Data validation passed all checks")
    else:
        logger.warning("Data validation failed some checks")
        for check, issues in validation_results.items():
            logger.warning(f"{check}: {issues}")
    
    return is_valid, validation_results

def load_and_validate_csv(file_path: str) -> Tuple[bool, pd.DataFrame, Dict[str, any]]:
    """
    Load a CSV file and validate its contents.
    
    Args:
        file_path: Path to the CSV file
        
    Returns:
        Tuple of (is_valid, dataframe, validation_results)
    """
    try:
        # Read CSV with appropriate data types
        df = pd.read_csv(file_path, dtype=REQUIRED_COLUMNS)
        
        # Run validation
        is_valid, validation_results = validate_sales_data(df)
        
        return is_valid, df, validation_results
        
    except Exception as e:
        logger.error(f"Error loading CSV file: {str(e)}")
        return False, None, {'error': str(e)}
