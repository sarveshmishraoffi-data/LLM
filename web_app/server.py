"""
FastAPI Server for AI Model Risk & GenAI Evaluation Web Application
Provides comprehensive REST APIs for model scoring, algorithm comparison,
fairness audits, SHAP explanations, robustness stress tests, and GenAI assessment.
"""

import os
import sys
import json
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
import joblib

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.utils import NpEncoder
from src.llm_application import MockLLMEngine

app = FastAPI(
    title="AI Model Risk & GenAI Evaluation Platform",
    description="Enterprise Model Risk Management (MRM) and Responsible AI Web Application",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global in-memory cache for models and audit summary
MODELS: Dict[str, Any] = {}
SUMMARY_DATA: Dict[str, Any] = {}
TEST_SPLIT_DF: Optional[pd.DataFrame] = None
LLM_ENGINE = MockLLMEngine(seed=42)


@app.on_event("startup")
def load_assets():
    """Pre-load model pipelines and cached audit data into memory."""
    global MODELS, SUMMARY_DATA, TEST_SPLIT_DF

    summary_path = "reports/audit_summary.json"
    if os.path.exists(summary_path):
        with open(summary_path, "r") as f:
            SUMMARY_DATA = json.load(f)

    # Load 5 ML models
    model_files = {
        "xgboost": "models/credit_model.joblib",
        "logistic": "models/baseline_model.joblib",
        "random_forest": "models/random_forest.joblib",
        "lightgbm": "models/lightgbm.joblib",
        "neural_net": "models/neural_net.joblib"
    }

    for key, path in model_files.items():
        if os.path.exists(path):
            MODELS[key] = joblib.load(path)

    test_path = "data/processed/test_split.csv"
    if os.path.exists(test_path):
        TEST_SPLIT_DF = pd.read_csv(test_path)


class ApplicantPayload(BaseModel):
    model_choice: str = "xgboost"  # "xgboost", "logistic", "random_forest", "lightgbm", "neural_net"
    annual_income: float = 75000.0
    loan_amount: float = 18000.0
    debt_to_income_ratio: float = 0.25
    employment_length_years: int = 6
    credit_history_years: int = 10
    num_open_credit_lines: int = 7
    has_previous_default: int = 0
    delinquent_2yrs: int = 0
    loan_purpose: str = "debt_consolidation"


class LLMAssessmentPayload(BaseModel):
    prompt_style: str = "standard_analyst"  # "strict_underwriting", "standard_analyst", "cautious_screener", "lenient_growth"
    applicant_data: Dict[str, Any]


@app.get("/api/overview")
def get_overview():
    """Retrieve executive KPIs, governance matrix, and pillar statuses."""
    if not SUMMARY_DATA:
        raise HTTPException(status_code=404, detail="Audit data not yet generated.")
    return {
        "validation_scorecard": SUMMARY_DATA.get("validation_scorecard"),
        "champion_performance": SUMMARY_DATA.get("champion_performance"),
        "optimal_threshold": SUMMARY_DATA.get("optimal_threshold"),
        "fairness_overall": SUMMARY_DATA.get("fairness_audit", {}).get("overall_status"),
        "robustness_overall": SUMMARY_DATA.get("robustness_scorecard", {}).get("overall_robustness_status"),
        "llm_overall": SUMMARY_DATA.get("llm_audit", {}).get("overall_governance_status"),
        "monitoring_overall": SUMMARY_DATA.get("monitoring_audit", {}).get("overall_status")
    }


@app.get("/api/leaderboard")
def get_leaderboard():
    """Retrieve 5-model algorithm zoo benchmark comparison table."""
    leaderboard = SUMMARY_DATA.get("models_leaderboard", [])
    return {"leaderboard": leaderboard}


@app.post("/api/predict")
def predict_risk(applicant: ApplicantPayload):
    """Run real-time inference on applicant profile with selected model and generate adverse reasons."""
    model_key = applicant.model_choice.lower()
    if model_key not in MODELS:
        model_key = "xgboost"

    model = MODELS[model_key]

    input_df = pd.DataFrame([{
        "annual_income": applicant.annual_income,
        "loan_amount": applicant.loan_amount,
        "employment_length_years": applicant.employment_length_years,
        "credit_history_years": applicant.credit_history_years,
        "num_open_credit_lines": applicant.num_open_credit_lines,
        "debt_to_income_ratio": applicant.debt_to_income_ratio,
        "has_previous_default": applicant.has_previous_default,
        "delinquent_2yrs": applicant.delinquent_2yrs,
        "loan_purpose": applicant.loan_purpose
    }])

    prob = float(model.predict_proba(input_df)[0, 1])
    assigned_tier = "High Risk (Decline)" if prob >= 0.50 else ("Medium Risk (Review)" if prob >= 0.30 else "Low Risk (Approve)")

    # Dynamic Adverse Action / Mitigating Factor reasoning
    adverse_reasons = []
    mitigating_factors = []

    if applicant.debt_to_income_ratio > 0.32:
        adverse_reasons.append(f"Elevated Debt-to-Income ratio ({applicant.debt_to_income_ratio*100:.1f}%) exceeds standard risk appetite.")
    if applicant.has_previous_default == 1:
        adverse_reasons.append("Historical credit default record identified on file.")
    if applicant.annual_income < 45000:
        adverse_reasons.append(f"Annual earnings (${applicant.annual_income:,.0f}) provides limited liquidity cushion.")
    if applicant.delinquent_2yrs > 0:
        adverse_reasons.append(f"{applicant.delinquent_2yrs} delinquency incident(s) in trailing 24 months.")

    if applicant.credit_history_years >= 8:
        mitigating_factors.append(f"Seasoned credit bureau history of {applicant.credit_history_years} years.")
    if applicant.has_previous_default == 0:
        mitigating_factors.append("Clean historical repayment performance (0 past defaults).")
    if applicant.annual_income >= 70000:
        mitigating_factors.append(f"Strong verified annual income (${applicant.annual_income:,.0f}).")
    if applicant.employment_length_years >= 5:
        mitigating_factors.append(f"Stable employment tenure of {applicant.employment_length_years} years.")

    return {
        "model_used": model_key,
        "default_probability": round(prob, 4),
        "assigned_tier": assigned_tier,
        "recommendation": "Approve" if prob < 0.30 else ("Manual Review" if prob < 0.50 else "Decline"),
        "adverse_action_reasons": adverse_reasons or ["Standard portfolio variance."],
        "mitigating_factors": mitigating_factors or ["Basic application requirements met."]
    }


@app.get("/api/fairness")
def get_fairness_audit():
    """Retrieve Fairlearn algorithmic bias audit across sensitive demographics."""
    return SUMMARY_DATA.get("fairness_audit", {})


@app.get("/api/robustness")
def get_robustness_data():
    """Retrieve missingness degradation and noise perturbation test results."""
    return SUMMARY_DATA.get("robustness_scorecard", {})


@app.get("/api/drift")
def get_drift_data():
    """Retrieve 6-month longitudinal PSI drift monitoring timeline and active alerts."""
    return SUMMARY_DATA.get("monitoring_audit", {})


@app.post("/api/llm/assess")
def assess_with_llm(payload: LLMAssessmentPayload):
    """Generate structured natural language risk analysis using GenAI credit engine."""
    prompt_map = {
        "strict_underwriting": "Assess credit risk under strict institutional policy guidelines.",
        "standard_analyst": "Analyze this customer's financial profile and determine appropriate risk level.",
        "cautious_screener": "Evaluate this profile with particular caution for downside risks.",
        "lenient_growth": "Assess creditworthiness focusing on approval opportunities."
    }
    prompt_text = prompt_map.get(payload.prompt_style, prompt_map["standard_analyst"])
    res = LLM_ENGINE.generate(prompt=prompt_text, applicant_profile=payload.applicant_data)
    return res.dict()


@app.get("/api/report/html", response_class=HTMLResponse)
def get_html_report():
    """Serve the complete generated SR 11-7 validation HTML report."""
    report_path = "reports/model_risk_report.html"
    if not os.path.exists(report_path):
        raise HTTPException(status_code=404, detail="HTML report not found.")
    with open(report_path, "r", encoding="utf-8") as f:
        return f.read()


# Mount static assets
static_dir = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(static_dir, exist_ok=True)
app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.get("/")
def serve_index():
    """Serve the modern single-page dashboard frontend."""
    index_file = os.path.join(static_dir, "index.html")
    if not os.path.exists(index_file):
        return HTMLResponse("<h1>Web application loading... Please build index.html</h1>")
    return FileResponse(index_file)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("web_app.server:app", host="0.0.0.0", port=8000, reload=False)
