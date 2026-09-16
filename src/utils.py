"""
Utility functions for AI Model Risk & GenAI Evaluation Framework
Provides logging, metrics formatting, and traffic-light status formatting.
"""

import json
import logging
from typing import Any, Dict
import numpy as np
import pandas as pd

# Configure standard logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("ModelRiskFramework")


class NpEncoder(json.JSONEncoder):
    """JSON Encoder that handles NumPy and Pandas scalar/array types."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        return super(NpEncoder, self).default(obj)


def get_traffic_light(status: str) -> str:
    """Return formatted status text with emoji for executive dashboards and reports."""
    status_upper = status.upper()
    if status_upper in ["PASS", "ACCEPTABLE", "STABLE", "LOW_RISK", "GREEN"]:
        return "PASS (Low Risk)"
    elif status_upper in ["WARNING", "REVIEW_REQUIRED", "MODERATE_DRIFT", "AMBER", "MEDIUM_RISK"]:
        return "REVIEW REQUIRED (Medium Risk)"
    elif status_upper in ["FAIL", "HIGH_RISK", "SIGNIFICANT_DRIFT", "RED", "UNACCEPTABLE"]:
        return "CRITICAL ALERT (High Risk)"
    return f"ℹ️ {status}"


def format_pct(val: float, decimals: int = 1) -> str:
    """Format floating point number as human readable percentage."""
    if val is None or np.isnan(val):
        return "N/A"
    return f"{val * 100:.{decimals}f}%"


def format_currency(val: float) -> str:
    """Format numeric currency value."""
    if val is None or np.isnan(val):
        return "$0"
    return f"${val:,.0f}"
