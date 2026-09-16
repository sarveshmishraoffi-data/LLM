"""
Model Training & Pipeline Orchestrator
Trains Logistic Regression baseline and XGBoost Champion models.
Enforces Fair Lending compliance by isolating protected attributes (age_group, gender)
from training inputs while preserving them for fairness audits.
"""

import os
import sys
import json
from typing import Dict, Any, Tuple, Optional, List

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from src.utils import logger, NpEncoder


class CreditModelTrainer:
    """Manages feature preprocessing, training, and persistence of Credit Risk models."""

    NUMERICAL_FEATURES = [
        "annual_income",
        "loan_amount",
        "employment_length_years",
        "credit_history_years",
        "num_open_credit_lines",
        "debt_to_income_ratio",
        "has_previous_default",
        "delinquent_2yrs"
    ]

    CATEGORICAL_FEATURES = [
        "loan_purpose"
    ]

    # Sensitive attributes kept ONLY for post-hoc fairness auditing
    PROTECTED_ATTRIBUTES = [
        "gender",
        "age_group"
    ]

    TARGET_COLUMN = "risk_label"

    def __init__(self, data_path: str = "data/processed/credit_risk_clean.csv"):
        self.data_path = data_path
        self.feature_columns = self.NUMERICAL_FEATURES + self.CATEGORICAL_FEATURES
        self.preprocessor = None
        self.models: Dict[str, Pipeline] = {}
        self.metadata: Dict[str, Any] = {}

    def build_preprocessor(self) -> ColumnTransformer:
        """Construct scikit-learn ColumnTransformer for numerical and categorical features."""
        num_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])

        cat_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ])

        self.preprocessor = ColumnTransformer(
            transformers=[
                ("num", num_transformer, self.NUMERICAL_FEATURES),
                ("cat", cat_transformer, self.CATEGORICAL_FEATURES)
            ]
        )
        return self.preprocessor

    def prepare_data(self, test_size: float = 0.25, random_state: int = 42) -> Dict[str, Any]:
        """Load dataset, partition features and protected columns, and perform stratified split."""
        logger.info(f"Loading credit dataset from {self.data_path}")
        df = pd.read_csv(self.data_path)
        
        # Ensure target column exists
        if self.TARGET_COLUMN not in df.columns:
            raise ValueError(f"Target column '{self.TARGET_COLUMN}' not found in dataset.")

        X = df[self.feature_columns]
        y = df[self.TARGET_COLUMN]
        protected_df = df[self.PROTECTED_ATTRIBUTES]

        # Stratified train-test split
        X_train, X_test, y_train, y_test, p_train, p_test = train_test_split(
            X, y, protected_df,
            test_size=test_size,
            stratify=y,
            random_state=random_state
        )

        logger.info(f"Data split completed: Train={len(X_train)} rows, Test={len(X_test)} rows")
        return {
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test,
            "protected_train": p_train,
            "protected_test": p_test,
            "full_test_df": pd.concat([X_test, p_test, y_test], axis=1)
        }

    def train_models(self, data_splits: Dict[str, Any]) -> Dict[str, Pipeline]:
        """Train 5 distinct ML algorithms across linear, ensemble bagging, boosting, and neural architectures."""
        X_train = data_splits["X_train"]
        y_train = data_splits["y_train"]

        # 1. Baseline: Logistic Regression (Interpretable Linear)
        logger.info("Training Logistic Regression (Baseline)...")
        lr_pipeline = Pipeline(steps=[
            ("preprocessor", self.build_preprocessor()),
            ("classifier", LogisticRegression(max_iter=1000, C=1.0, random_state=42))
        ])
        lr_pipeline.fit(X_train, y_train)
        self.models["baseline_logistic"] = lr_pipeline

        # 2. Ensemble Bagging: Random Forest
        logger.info("Training Random Forest Classifier (Bagging Ensemble)...")
        rf_pipeline = Pipeline(steps=[
            ("preprocessor", self.build_preprocessor()),
            ("classifier", RandomForestClassifier(
                n_estimators=150, max_depth=8, min_samples_split=5, random_state=42, n_jobs=-1
            ))
        ])
        rf_pipeline.fit(X_train, y_train)
        self.models["random_forest"] = rf_pipeline

        # 3. Champion: XGBoost Classifier (Gradient Boosted Trees)
        logger.info("Training XGBoost Classifier (Champion)...")
        xgb_pipeline = Pipeline(steps=[
            ("preprocessor", self.build_preprocessor()),
            ("classifier", XGBClassifier(
                n_estimators=150,
                max_depth=4,
                learning_rate=0.08,
                subsample=0.85,
                colsample_bytree=0.85,
                eval_metric="logloss",
                random_state=42
            ))
        ])
        xgb_pipeline.fit(X_train, y_train)
        self.models["champion_xgboost"] = xgb_pipeline

        # 4. Gradient Boosting: LightGBM (Leaf-wise Optimization)
        logger.info("Training LightGBM Classifier (High-Throughput Boosting)...")
        lgb_pipeline = Pipeline(steps=[
            ("preprocessor", self.build_preprocessor()),
            ("classifier", LGBMClassifier(
                n_estimators=150,
                max_depth=5,
                learning_rate=0.08,
                num_leaves=31,
                random_state=42,
                verbose=-1
            ))
        ])
        lgb_pipeline.fit(X_train, y_train)
        self.models["lightgbm"] = lgb_pipeline

        # 5. Deep Learning / Neural Network: Multi-Layer Perceptron
        logger.info("Training Multi-Layer Perceptron (Neural Network)...")
        mlp_pipeline = Pipeline(steps=[
            ("preprocessor", self.build_preprocessor()),
            ("classifier", MLPClassifier(
                hidden_layer_sizes=(64, 32),
                max_iter=350,
                activation="relu",
                alpha=0.01,
                random_state=42
            ))
        ])
        mlp_pipeline.fit(X_train, y_train)
        self.models["neural_net"] = mlp_pipeline

        return self.models

    def save_artifacts(self, models_dir: str = "models", data_splits: Optional[Dict[str, Any]] = None):
        """Persist all 5 trained pipelines, metadata, and holdout test set."""
        os.makedirs(models_dir, exist_ok=True)
        
        # Save champion model (XGBoost)
        champion_path = os.path.join(models_dir, "credit_model.joblib")
        joblib.dump(self.models["champion_xgboost"], champion_path)
        logger.info(f"Saved Champion XGBoost model to {champion_path}")

        # Save baseline model (Logistic Regression)
        baseline_path = os.path.join(models_dir, "baseline_model.joblib")
        joblib.dump(self.models["baseline_logistic"], baseline_path)
        logger.info(f"Saved Baseline Logistic model to {baseline_path}")

        # Save Random Forest
        rf_path = os.path.join(models_dir, "random_forest.joblib")
        joblib.dump(self.models["random_forest"], rf_path)
        logger.info(f"Saved Random Forest model to {rf_path}")

        # Save LightGBM
        lgb_path = os.path.join(models_dir, "lightgbm.joblib")
        joblib.dump(self.models["lightgbm"], lgb_path)
        logger.info(f"Saved LightGBM model to {lgb_path}")

        # Save Neural Net
        mlp_path = os.path.join(models_dir, "neural_net.joblib")
        joblib.dump(self.models["neural_net"], mlp_path)
        logger.info(f"Saved Neural Network model to {mlp_path}")

        # Save holdout test split for audit & monitoring
        if data_splits is not None:
            test_path = "data/processed/test_split.csv"
            data_splits["full_test_df"].to_csv(test_path, index=False)
            logger.info(f"Saved test split to {test_path} ({len(data_splits['full_test_df'])} rows)")

        # Save model metadata
        metadata = {
            "champion_type": "XGBClassifier",
            "models_available": [
                "Logistic Regression", "Random Forest", "XGBoost", "LightGBM", "Multi-Layer Perceptron"
            ],
            "features": {
                "numerical": self.NUMERICAL_FEATURES,
                "categorical": self.CATEGORICAL_FEATURES,
                "protected_audited": self.PROTECTED_ATTRIBUTES
            },
            "training_samples": len(data_splits["X_train"]) if data_splits else None,
            "test_samples": len(data_splits["X_test"]) if data_splits else None,
            "target": self.TARGET_COLUMN
        }
        meta_path = os.path.join(models_dir, "model_metadata.json")
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2, cls=NpEncoder)
        logger.info(f"Saved metadata to {meta_path}")


if __name__ == "__main__":
    trainer = CreditModelTrainer()
    splits = trainer.prepare_data()
    trainer.train_models(splits)
    trainer.save_artifacts(data_splits=splits)
    print("Model training execution finished successfully!")
