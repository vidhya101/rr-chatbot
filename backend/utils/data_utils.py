import pandas as pd
import numpy as np
import json
from typing import Any, Dict, List, Union

def convert_to_serializable(obj: Any) -> Any:
    """
    Convert an object to a JSON-serializable format by handling NaN values and other special types.
    
    Args:
        obj: The object to convert
        
    Returns:
        JSON-serializable version of the object
    """
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient='records')
    elif isinstance(obj, pd.Series):
        return obj.to_dict()
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_to_serializable(item) for item in obj)
    elif pd.isna(obj) or (isinstance(obj, float) and np.isnan(obj)):
        return None
    return obj

def safe_json_serialize(obj: Any) -> str:
    """
    Safely serialize an object to JSON, handling NaN values and other special types.
    
    Args:
        obj: The object to serialize
        
    Returns:
        JSON string
    """
    return json.dumps(convert_to_serializable(obj)) 