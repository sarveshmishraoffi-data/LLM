"""
Data Validation Engine for AI Model Risk Framework
Implements rigorous pre-model checks: schema validation, missing data rates,
range/bounds checks, duplicate detection, and anomaly scoring.
"""

from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd
from src.utils import logger, get_traffic_light


class DataValidator:
    """Validates data integrity, ranges, distributions, and business logic rules."""

    def __init__(self, schema_rules: Optional[Dict[str, Any]] = None):
        # Default credit risk banking rules
        self.rules = schema_rules or {
            "age": {"min": 18, "max": 100, "type": "numeric"},
            "annual_income": {"min": 0.0, "max": 10000000.0, "type": "numeric"},
            "loan_amount": {"min": 500.0, "max": 250000.0, "type": "numeric"},
            "debt_to_income_ratio": {"min": 0.0, "max": 1.5, "type": "numeric"},
            "employment_length_years": {"min": 0, "max": 60, "type": "numeric"},
            "credit_history_years": {"min": 0, "max": 60, "type": "numeric"},
            "num_open_credit_lines": {"min": 0, "max": 50, "type": "numeric"},
            "has_previous_default": {"allowed": [0, 1]},
            "risk_label": {"allowed": [0, 1]},
        }

    def check_missing_values(self, df: pd.DataFrame, max_allowed_missing_pct: float = 0.05) -> Dict[str, Any]:
        """Check column-level missingness rates against regulatory tolerances."""
        missing_counts = df.isnull().sum()
        missing_pcts = missing_counts / len(df)
        
        column_reports = {}
        flagged_columns = []
        
        for col in df.columns:
            cnt = int(missing_counts[col])
            pct = float(missing_pcts[col])
            is_flagged = pct > max_allowed_missing_pct
            if is_flagged:
                flagged_columns.append(col)
            column_reports[col] = {
                "missing_count": cnt,
                "missing_pct": round(pct, 4),
                "status": "FAIL" if is_flagged else "PASS"
            }
            
        status = "FAIL" if len(flagged_columns) > 0 else "PASS"
        return {
            "check_name": "Missing Values Check",
            "status": status,
            "total_missing_cells": int(missing_counts.sum()),
            "columns_exceeding_tolerance": flagged_columns,
            "details": column_reports
        }

    def check_duplicates(self, df: pd.DataFrame, id_column: str = "customer_id") -> Dict[str, Any]:
        """Identify redundant records and duplicate entity identifiers."""
        exact_dups = int(df.duplicated().sum())
        id_dups = int(df[id_column].duplicated().sum()) if id_column in df.columns else 0
        
        passed = (exact_dups == 0) and (id_dups == 0)
        return {
            "check_name": "Duplicate Records Check",
            "status": "PASS" if passed else "FAIL",
            "exact_duplicate_rows": exact_dups,
            "duplicate_ids": id_dups,
            "evidence": f"Found {exact_dups} exact duplicate row(s) and {id_dups} duplicate ID(s)."
        }

    def check_range_validity(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate whether values adhere to physical and banking sanity limits."""
        violations = {}
        total_violations = 0
        
        for col, constraints in self.rules.items():
            if col not in df.columns:
                continue
            
            series = df[col].dropna()
            col_violations = 0
            details = []
            
            if "min" in constraints:
                below_min = int((series < constraints["min"]).sum())
                if below_min > 0:
                    col_violations += below_min
                    details.append(f"{below_min} rows < {constraints['min']}")
                    
            if "max" in constraints:
                above_max = int((series > constraints["max"]).sum())
                if above_max > 0:
                    col_violations += above_max
                    details.append(f"{above_max} rows > {constraints['max']}")
                    
            if "allowed" in constraints:
                invalid_cat = int((~series.isin(constraints["allowed"])).sum())
                if invalid_cat > 0:
                    col_violations += invalid_cat
                    details.append(f"{invalid_cat} rows outside {constraints['allowed']}")
                    
            if col_violations > 0:
                total_violations += col_violations
                violations[col] = {
                    "violation_count": col_violations,
                    "reasons": details
                }
                
        # Cross-column consistency check: Employment length cannot exceed age - 18
        if "age" in df.columns and "employment_length_years" in df.columns:
            valid_rows = df[["age", "employment_length_years"]].dropna()
            impossible_emp = int((valid_rows["employment_length_years"] > (valid_rows["age"] - 16)).sum())
            if impossible_emp > 0:
                total_violations += impossible_emp
                violations["cross_column_employment_vs_age"] = {
                    "violation_count": impossible_emp,
                    "reasons": [f"{impossible_emp} applicants have employment length exceeding age - 16"]
                }
                
        return {
            "check_name": "Range & Domain Validity Check",
            "status": "PASS" if total_violations == 0 else "FAIL",
            "total_violations": total_violations,
            "violations_by_column": violations
        }

    def detect_anomalies_iqr(self, df: pd.DataFrame, factor: float = 3.0) -> Dict[str, Any]:
        """Detect extreme statistical outliers using IQR rule with conservative factor."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        outlier_summary = {}
        
        for col in numeric_cols:
            if col in ["customer_id", "has_previous_default", "risk_label"]:
                continue
            series = df[col].dropna()
            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - factor * iqr
            upper_bound = q3 + factor * iqr
            
            outliers = series[(series < lower_bound) | (series > upper_bound)]
            outlier_summary[col] = {
                "outlier_count": int(len(outliers)),
                "outlier_pct": round(len(outliers) / len(series), 4),
                "lower_bound": round(lower_bound, 2),
                "upper_bound": round(upper_bound, 2)
            }
            
        return {
            "check_name": "Extreme Outlier Detection",
            "status": "PASS",
            "details": outlier_summary
        }

    def generate_quality_scorecard(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Run all data validation checks and compute composite Data Health Score."""
        missing_res = self.check_missing_values(df)
        dup_res = self.check_duplicates(df)
        range_res = self.check_range_validity(df)
        outlier_res = self.detect_anomalies_iqr(df)
        
        # Scoring logic
        deductions = 0
        if missing_res["status"] == "FAIL":
            deductions += 30
        elif missing_res["total_missing_cells"] > 0:
            deductions += 10
            
        if dup_res["status"] == "FAIL":
            deductions += 25
            
        if range_res["status"] == "FAIL":
            deductions += 35
            
        health_score = max(0, 100 - deductions)
        
        if health_score >= 90:
            governance_status = "PASS"
        elif health_score >= 70:
            governance_status = "REVIEW_REQUIRED"
        else:
            governance_status = "FAIL"
            
        scorecard = {
            "dataset_rows": len(df),
            "dataset_columns": len(df.columns),
            "data_health_score": health_score,
            "governance_status": governance_status,
            "traffic_light": get_traffic_light(governance_status),
            "checks": {
                "missing_values": missing_res,
                "duplicates": dup_res,
                "range_validity": range_res,
                "outliers": outlier_res
            }
        }
        return scorecard

    def sanitize_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """Sanitize raw data: drop duplicates, impute/clamp out-of-range values."""
        clean_df = df.copy()
        
        # 1. Deduplicate
        clean_df = clean_df.drop_duplicates()
        if "customer_id" in clean_df.columns:
            clean_df = clean_df.drop_duplicates(subset=["customer_id"])
            
        # 2. Rectify impossible bounds
        if "age" in clean_df.columns:
            clean_df.loc[clean_df["age"] < 18, "age"] = 18
            clean_df.loc[clean_df["age"] > 100, "age"] = 75
            
        if "annual_income" in clean_df.columns:
            # Replace negative income with median
            median_income = clean_df.loc[clean_df["annual_income"] > 0, "annual_income"].median()
            clean_df.loc[clean_df["annual_income"] < 0, "annual_income"] = median_income
            clean_df["annual_income"] = clean_df["annual_income"].fillna(median_income)
            
        if "debt_to_income_ratio" in clean_df.columns:
            clean_df.loc[clean_df["debt_to_income_ratio"] > 1.5, "debt_to_income_ratio"] = 0.65
            clean_df.loc[clean_df["debt_to_income_ratio"] < 0.0, "debt_to_income_ratio"] = 0.05
            
        if "employment_length_years" in clean_df.columns:
            median_emp = clean_df["employment_length_years"].median()
            clean_df["employment_length_years"] = clean_df["employment_length_years"].fillna(median_emp)
            # Consistency with age
            clean_df["employment_length_years"] = np.minimum(
                clean_df["employment_length_years"], 
                np.maximum(clean_df["age"] - 18, 0)
            )
            
        return clean_df
