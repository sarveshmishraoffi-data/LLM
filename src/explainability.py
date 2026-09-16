"""
Explainability & Attribution Engine (SHAP)
Computes global feature importances, local applicant-level attributions,
and generates Adverse Action reason codes compliant with FCRA and model risk standards.
"""

import os
import sys
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
import shap
from sklearn.pipeline import Pipeline

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.utils import logger


class ModelExplainer:
    """Computes SHAP explanations for Credit Risk models (Global & Local)."""

    def __init__(self, pipeline: Pipeline, background_data: Optional[pd.DataFrame] = None):
        self.pipeline = pipeline
        self.preprocessor = pipeline.named_steps["preprocessor"]
        self.classifier = pipeline.named_steps["classifier"]
        self.feature_names = None
        self.explainer = None
        self._initialize_explainer(background_data)

    def _get_transformed_feature_names(self) -> List[str]:
        """Extract feature names after One-Hot and scaling transformations."""
        names = []
        for name, trans, cols in self.preprocessor.transformers_:
            if name == "remainder" and trans == "drop":
                continue
            if hasattr(trans, "get_feature_names_out"):
                names.extend(trans.get_feature_names_out(cols))
            elif hasattr(trans, "named_steps") and "onehot" in trans.named_steps:
                onehot = trans.named_steps["onehot"]
                names.extend(onehot.get_feature_names_out(cols))
            else:
                names.extend(cols)
        return [str(n).replace("num__", "").replace("cat__", "") for n in names]

    def _initialize_explainer(self, background_data: Optional[pd.DataFrame] = None):
        """Build SHAP TreeExplainer or LinearExplainer based on model class."""
        self.feature_names = self._get_transformed_feature_names()
        
        # Check model type
        model_name = self.classifier.__class__.__name__
        logger.info(f"Initializing SHAP explainer for {model_name}")

        if "XGB" in model_name or "RandomForest" in model_name:
            self.explainer = shap.TreeExplainer(self.classifier)
        else:
            # LinearExplainer or KernelExplainer
            if background_data is not None:
                bg_transformed = self.preprocessor.transform(background_data.iloc[:100])
            else:
                bg_transformed = np.zeros((1, len(self.feature_names)))
            self.explainer = shap.LinearExplainer(self.classifier, bg_transformed)

    def compute_global_importance(self, X: pd.DataFrame) -> pd.DataFrame:
        """Compute mean absolute SHAP values across dataset for global importance ranking."""
        X_trans = self.preprocessor.transform(X)
        shap_values = self.explainer.shap_values(X_trans)

        # Handle binary classification shapes (list or 2D array)
        if isinstance(shap_values, list):
            sv = shap_values[1] # positive class (High Risk)
        elif len(shap_values.shape) == 3:
            sv = shap_values[:, :, 1]
        else:
            sv = shap_values

        mean_abs = np.abs(sv).mean(axis=0)
        df_imp = pd.DataFrame({
            "Feature": self.feature_names,
            "Mean_Abs_SHAP": mean_abs
        }).sort_values(by="Mean_Abs_SHAP", ascending=False).reset_index(drop=True)

        df_imp["Relative_Importance_Pct"] = (df_imp["Mean_Abs_SHAP"] / df_imp["Mean_Abs_SHAP"].sum()).round(4)
        return df_imp

    def explain_instance(
        self,
        applicant_series: pd.Series,
        top_n_reasons: int = 4
    ) -> Dict[str, Any]:
        """
        Produce local explanation for an individual applicant.
        Returns prediction probability, top adverse risk factors, and mitigating factors.
        """
        df_single = pd.DataFrame([applicant_series])
        X_trans = self.preprocessor.transform(df_single)
        
        prob = float(self.classifier.predict_proba(X_trans)[0, 1])
        shap_values = self.explainer.shap_values(X_trans)

        if isinstance(shap_values, list):
            sv = shap_values[1][0]
        elif len(shap_values.shape) == 3:
            sv = shap_values[0, :, 1]
        else:
            sv = shap_values[0]

        # Map to feature names
        contributions = []
        for feat, val in zip(self.feature_names, sv):
            contributions.append({
                "feature": feat,
                "shap_value": float(val),
                "direction": "Increases Risk" if val > 0 else "Decreases Risk"
            })

        df_contrib = pd.DataFrame(contributions)
        
        # Risk increasing factors (Adverse Action Reasons)
        risk_increasing = df_contrib[df_contrib["shap_value"] > 0].sort_values(
            by="shap_value", ascending=False
        ).head(top_n_reasons).to_dict(orient="records")

        # Risk mitigating factors
        risk_decreasing = df_contrib[df_contrib["shap_value"] < 0].sort_values(
            by="shap_value", ascending=True
        ).head(top_n_reasons).to_dict(orient="records")

        risk_tier = "High Risk" if prob >= 0.50 else "Low Risk"

        return {
            "predicted_probability": round(prob, 4),
            "assigned_risk_tier": risk_tier,
            "top_risk_increasing_factors": risk_increasing,
            "top_risk_mitigating_factors": risk_decreasing,
            "raw_shap_vector": sv.tolist(),
            "feature_names": self.feature_names
        }
