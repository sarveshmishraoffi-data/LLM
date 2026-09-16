"""
GenAI Credit Risk Assessment Application
Provides modular credit analysis engine supporting:
1. MockLLMEngine (zero-cost, reproducible offline testing with controllable noise & hallucinations)
2. OpenAILLMEngine (OpenAI API integration)
3. GeminiLLMEngine (Google GenAI integration)
"""

import os
import sys
import json
import time
import re
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.utils import logger


class CreditAssessmentResponse(BaseModel):
    """Structured response schema for credit risk assessment narratives."""
    applicant_id: str
    risk_level: str  # "Low", "Medium", "High"
    confidence: str  # "Low", "Medium", "High"
    recommendation: str  # "Approve", "Manual Review", "Decline"
    summary: str
    key_reasons: List[str]
    mitigating_factors: List[str]
    raw_text: str = ""
    metadata: Dict[str, Any] = Field(default_factory=dict)


class BaseLLMEngine:
    """Abstract interface for GenAI model backends."""
    def generate(self, prompt: str, applicant_profile: Dict[str, Any], temperature: float = 0.2) -> CreditAssessmentResponse:
        raise NotImplementedError


class MockLLMEngine(BaseLLMEngine):
    """
    Offline heuristic GenAI engine simulating production LLM behavior.
    Enables reproducible prompt sensitivity testing, consistency scoring,
    and deliberate hallucination simulation without incurring API billing.
    """

    def __init__(self, seed: int = 42, inject_hallucination_rate: float = 0.08):
        self.seed = seed
        self.inject_hallucination_rate = inject_hallucination_rate

    def generate(
        self,
        prompt: str,
        applicant_profile: Dict[str, Any],
        temperature: float = 0.2
    ) -> CreditAssessmentResponse:
        time.sleep(0.01)  # Simulate API latency
        app_id = str(applicant_profile.get("customer_id", "APP_UNKNOWN"))
        income = float(applicant_profile.get("annual_income", 50000))
        dti = float(applicant_profile.get("debt_to_income_ratio", 0.30))
        credit_history = float(applicant_profile.get("credit_history_years", 5))
        has_default = int(applicant_profile.get("has_previous_default", 0))
        delinq = int(applicant_profile.get("delinquent_2yrs", 0))

        # Base financial score (0 = ultra safe, 100 = severe risk)
        base_score = (dti * 60) + (has_default * 35) + (delinq * 15) - (income / 10000) - (credit_history * 1.2)

        # Prompt styling impact (simulates prompt engineering sensitivity)
        prompt_lower = prompt.lower()
        if "strict" in prompt_lower or "conservative" in prompt_lower or "stringent" in prompt_lower:
            base_score += 12  # Strict prompt tends more towards High Risk / Decline
            prompt_style = "Strict Underwriter"
        elif "lenient" in prompt_lower or "opportunity" in prompt_lower or "growth" in prompt_lower:
            base_score -= 10  # Lenient prompt tends more towards Approve
            prompt_style = "Lenient Advisor"
        elif "adversarial" in prompt_lower or "persuasive" in prompt_lower:
            base_score += 18  # Adversarial prompt induces caution
            prompt_style = "Adversarial Screener"
        else:
            prompt_style = "Standard Analyst"

        # Temperature disturbance (stochastic consistency check)
        import random
        rng = random.Random(self.seed + int(income) + int(temperature * 100))
        temp_jitter = (rng.random() - 0.5) * 16 * temperature
        final_score = base_score + temp_jitter

        # Assign Risk Tier
        if final_score >= 38:
            risk_level = "High"
            recommendation = "Decline"
            confidence = "High" if final_score > 50 else "Medium"
        elif final_score >= 20:
            risk_level = "Medium"
            recommendation = "Manual Review"
            confidence = "Medium"
        else:
            risk_level = "Low"
            recommendation = "Approve"
            confidence = "High"

        # Generate realistic reasons based on ground truth facts
        reasons = []
        mitigating = []

        if dti > 0.35:
            reasons.append(f"Elevated debt-to-income ratio ({dti * 100:.1f}%) exceeds standard risk threshold.")
        if has_default == 1:
            reasons.append("Applicant has an acknowledged historical default record on file.")
        if delinq > 0:
            reasons.append(f"Recorded {delinq} delinquency event(s) within the trailing 24 months.")

        if income >= 75000:
            mitigating.append(f"Substantial annual earnings (${income:,.0f}) provides repayment buffer.")
        if credit_history >= 8:
            mitigating.append(f"Established credit bureau tenure of {credit_history:.0f} years demonstrates seasoning.")
        if has_default == 0:
            mitigating.append("Clean historical repayment performance with no recorded prior default events.")

        # Simulate Hallucination (inventing unsupported claims)
        # e.g., claiming applicant has a prior bankruptcy or 3 defaults when ground truth shows clean
        hallucination_flag = False
        if rng.random() < self.inject_hallucination_rate:
            hallucination_flag = True
            if has_default == 0:
                reasons.append("Applicant has a recorded Chapter 7 bankruptcy filing from 3 years prior.")
            else:
                reasons.append("Applicant has four distinct commercial charge-offs with external collections.")

        if not reasons:
            reasons.append("Modest overall debt profile with manageable obligations.")
        if not mitigating:
            mitigating.append("Basic application requirements satisfied.")

        summary = (
            f"Assessment for {app_id}: Assigned {risk_level} Risk with recommendation to {recommendation}. "
            f"Key driver: {reasons[0]} Mitigating factor: {mitigating[0]}"
        )

        raw_text = (
            f"### Credit Risk Assessment Report\n"
            f"**Applicant ID**: {app_id}\n"
            f"**Assigned Risk Level**: {risk_level}\n"
            f"**Recommendation**: {recommendation}\n"
            f"**Confidence**: {confidence}\n"
            f"**Analysis Summary**: {summary}\n\n"
            f"**Primary Risk Factors**:\n" + "\n".join(f"- {r}" for r in reasons) + "\n\n"
            f"**Mitigating Financial Factors**:\n" + "\n".join(f"- {m}" for m in mitigating)
        )

        return CreditAssessmentResponse(
            applicant_id=app_id,
            risk_level=risk_level,
            confidence=confidence,
            recommendation=recommendation,
            summary=summary,
            key_reasons=reasons,
            mitigating_factors=mitigating,
            raw_text=raw_text,
            metadata={
                "engine": "MockLLMEngine",
                "prompt_style": prompt_style,
                "simulated_score": round(final_score, 2),
                "temperature": temperature,
                "simulated_hallucination": hallucination_flag
            }
        )


class OpenAILLMEngine(BaseLLMEngine):
    """OpenAI API client implementation using standard chat completions."""
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o-mini"):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set.")
        from openai import OpenAI
        self.client = OpenAI(api_key=self.api_key)

    def generate(self, prompt: str, applicant_profile: Dict[str, Any], temperature: float = 0.2) -> CreditAssessmentResponse:
        app_id = str(applicant_profile.get("customer_id", "APP_UNKNOWN"))
        system_instruction = (
            "You are an expert Credit Risk Underwriter at a Tier-1 financial institution. "
            "Analyze the applicant profile and respond ONLY with a JSON object containing keys: "
            "'risk_level' ('Low', 'Medium', 'High'), 'confidence' ('Low', 'Medium', 'High'), "
            "'recommendation' ('Approve', 'Manual Review', 'Decline'), 'summary', 'key_reasons' (list), "
            "and 'mitigating_factors' (list)."
        )
        user_message = f"{prompt}\n\nApplicant Financial Profile:\n{json.dumps(applicant_profile, indent=2)}"

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": user_message}
            ],
            temperature=temperature,
            response_format={"type": "json_object"}
        )
        content = response.choices[0].message.content
        parsed = json.loads(content)

        return CreditAssessmentResponse(
            applicant_id=app_id,
            risk_level=parsed.get("risk_level", "Medium"),
            confidence=parsed.get("confidence", "Medium"),
            recommendation=parsed.get("recommendation", "Manual Review"),
            summary=parsed.get("summary", ""),
            key_reasons=parsed.get("key_reasons", []),
            mitigating_factors=parsed.get("mitigating_factors", []),
            raw_text=content,
            metadata={"engine": f"OpenAI:{self.model}", "temperature": temperature}
        )


class GeminiLLMEngine(BaseLLMEngine):
    """Google Gemini client implementation using google-genai SDK."""
    def __init__(self, api_key: Optional[str] = None, model: str = "gemini-2.0-flash"):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.model = model
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY environment variable is not set.")
        from google import genai
        self.client = genai.Client(api_key=self.api_key)

    def generate(self, prompt: str, applicant_profile: Dict[str, Any], temperature: float = 0.2) -> CreditAssessmentResponse:
        app_id = str(applicant_profile.get("customer_id", "APP_UNKNOWN"))
        full_prompt = (
            f"You are a Senior Credit Risk Officer. Analyze this applicant profile and provide a JSON response:\n"
            f"Keys required: 'risk_level' ('Low'/'Medium'/'High'), 'confidence' ('Low'/'Medium'/'High'), "
            f"'recommendation' ('Approve'/'Manual Review'/'Decline'), 'summary', 'key_reasons', 'mitigating_factors'.\n\n"
            f"Directive: {prompt}\n\n"
            f"Profile:\n{json.dumps(applicant_profile, indent=2)}"
        )
        response = self.client.models.generate_content(
            model=self.model,
            contents=full_prompt,
            config={"temperature": temperature, "response_mime_type": "application/json"}
        )
        parsed = json.loads(response.text)
        return CreditAssessmentResponse(
            applicant_id=app_id,
            risk_level=parsed.get("risk_level", "Medium"),
            confidence=parsed.get("confidence", "Medium"),
            recommendation=parsed.get("recommendation", "Manual Review"),
            summary=parsed.get("summary", ""),
            key_reasons=parsed.get("key_reasons", []),
            mitigating_factors=parsed.get("mitigating_factors", []),
            raw_text=response.text,
            metadata={"engine": f"Gemini:{self.model}", "temperature": temperature}
        )


def get_llm_engine(engine_type: str = "mock", **kwargs) -> BaseLLMEngine:
    """Factory helper to retrieve desired LLM engine."""
    engine_type_lower = engine_type.lower()
    if engine_type_lower == "openai":
        return OpenAILLMEngine(**kwargs)
    elif engine_type_lower == "gemini":
        return GeminiLLMEngine(**kwargs)
    else:
        return MockLLMEngine(**kwargs)
