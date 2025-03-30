import pandas as pd
import numpy as np
import json
from typing import Any, Dict, List, Union
from datetime import datetime, date

def convert_to_serializable(obj: Any) -> Any:
    """
    Convert an object to a JSON-serializable format by handling NaN values and other special types.
    
    Args:
        obj: The object to convert
        
    Returns:
        JSON-serializable version of the object
    """
    if obj is None:
        return None
    elif isinstance(obj, (str, int, float, bool)):
        return obj
    elif isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    elif isinstance(obj, (np.float64, np.float32, np.float16)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, pd.Series):
        return obj.tolist()
    elif isinstance(obj, (datetime, date)):
        return obj.isoformat()
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_serializable(item) for item in obj]
    elif hasattr(obj, '__dict__'):
        return convert_to_serializable(obj.__dict__)
    else:
        try:
            # Try to convert to string if no other conversion is possible
            return str(obj)
        except:
            return None

def safe_json_serialize(obj: Any) -> str:
    """
    Safely serialize an object to JSON, handling NaN values and other special types.
    
    Args:
        obj: The object to serialize
        
    Returns:
        JSON string
    """
    return json.dumps(convert_to_serializable(obj)) 