"""
Dataset Generation Script for Credit Risk & Model Evaluation
Generates:
1. Baseline credit risk dataset (clean & raw with injected anomalies for validation testing)
2. Longitudinal monthly batches (Month 1 to Month 6) for data & concept drift monitoring
"""

import os
import numpy as np
import pandas as pd


def generate_credit_dataset(n_samples: int = 5000, seed: int = 42) -> pd.DataFrame:
    """Generate realistic synthetic credit applicant dataset with authentic risk dynamics."""
    np.random.seed(seed)
    
    # Demographics
    age = np.random.normal(loc=40, scale=12, size=n_samples).astype(int)
    age = np.clip(age, 19, 75)
    
    # Sensitive attributes for fairness auditing
    gender_prob = 0.48  # 48% Female, 52% Male
    gender = np.random.choice(['Female', 'Male'], size=n_samples, p=[gender_prob, 1 - gender_prob])
    
    age_group = []
    for a in age:
        if a < 25:
            age_group.append('Young (<25)')
        elif a <= 55:
            age_group.append('Working Age (25-55)')
        else:
            age_group.append('Senior (>55)')
            
    # Financial Profiles (log-normal income)
    income_base = np.random.lognormal(mean=10.9, sigma=0.55, size=n_samples) # ~$55k median
    annual_income = np.round(np.clip(income_base, 18000, 260000), -2)
    
    # Employment and Credit History
    employment_length = np.random.exponential(scale=5.0, size=n_samples).astype(int)
    employment_length = np.clip(employment_length, 0, 30)
    # Employment cannot exceed age - 18
    employment_length = np.minimum(employment_length, np.maximum(age - 18, 0))
    
    credit_history_years = (np.random.normal(loc=0.35 * age, scale=3.0)).astype(int)
    credit_history_years = np.clip(credit_history_years, 1, 40)
    
    # Loan parameters
    loan_purpose_choices = [
        'debt_consolidation', 'credit_card', 'home_improvement', 
        'small_business', 'major_purchase', 'medical'
    ]
    loan_purpose = np.random.choice(
        loan_purpose_choices, size=n_samples, 
        p=[0.45, 0.25, 0.12, 0.08, 0.06, 0.04]
    )
    
    # Loan amount proportional to income
    loan_amount = (annual_income * np.random.uniform(0.1, 0.45, size=n_samples)).astype(int)
    loan_amount = np.round(np.clip(loan_amount, 2000, 50000), -2)
    
    num_open_credit_lines = np.random.poisson(lam=7, size=n_samples)
    num_open_credit_lines = np.clip(num_open_credit_lines, 1, 25)
    
    # Debt-to-Income (DTI)
    dti = np.random.beta(a=2.5, b=6.0, size=n_samples) * 0.75  # Range ~0.05 to ~0.60
    debt_to_income_ratio = np.round(np.clip(dti, 0.04, 0.65), 4)
    
    # Delinquencies & Defaults
    prev_default_prob = 0.12 + 0.15 * (debt_to_income_ratio > 0.35)
    has_previous_default = (np.random.uniform(0, 1, size=n_samples) < prev_default_prob).astype(int)
    
    delinquent_2yrs = np.random.choice([0, 1, 2, 3], size=n_samples, p=[0.75, 0.15, 0.07, 0.03])
    
    # Standardized risk drivers for clear statistical signal
    inc_norm = (annual_income - 60000) / 35000
    dti_norm = (debt_to_income_ratio - 0.28) / 0.14
    loan_norm = (loan_amount - 18000) / 10000
    hist_norm = (credit_history_years - 12) / 7
    emp_norm = (employment_length - 6) / 5
    
    logit = (
        -0.55
        + 1.95 * dti_norm
        + 2.30 * has_previous_default
        + 1.15 * (delinquent_2yrs > 0)
        - 1.35 * inc_norm
        + 1.05 * loan_norm
        - 0.75 * hist_norm
        - 0.55 * emp_norm
    )
    
    default_prob = 1 / (1 + np.exp(-logit))
    default_prob = np.clip(default_prob, 0.02, 0.98)
    risk_label = (default_prob >= 0.50).astype(int)
    # Add 4% label noise to simulate real-world unobserved variance
    flip_noise = np.random.uniform(0, 1, size=n_samples) < 0.04
    risk_label = np.where(flip_noise, 1 - risk_label, risk_label)
    
    customer_ids = [f"CUST_{100000 + i}" for i in range(n_samples)]
    
    df = pd.DataFrame({
        'customer_id': customer_ids,
        'age': age,
        'age_group': age_group,
        'gender': gender,
        'annual_income': annual_income,
        'loan_amount': loan_amount,
        'loan_purpose': loan_purpose,
        'employment_length_years': employment_length,
        'credit_history_years': credit_history_years,
        'num_open_credit_lines': num_open_credit_lines,
        'debt_to_income_ratio': debt_to_income_ratio,
        'has_previous_default': has_previous_default,
        'delinquent_2yrs': delinquent_2yrs,
        'risk_label': risk_label
    })
    
    return df


def inject_anomalies_for_validation(clean_df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """Inject dirty data (missing values, invalid bounds, duplicates) to test validation engine."""
    np.random.seed(seed)
    dirty_df = clean_df.copy()
    n = len(dirty_df)
    
    # 1. Missing values
    missing_idx_inc = np.random.choice(n, size=int(0.02 * n), replace=False)
    dirty_df.loc[missing_idx_inc, 'annual_income'] = np.nan
    
    missing_idx_emp = np.random.choice(n, size=int(0.015 * n), replace=False)
    dirty_df.loc[missing_idx_emp, 'employment_length_years'] = np.nan
    
    # 2. Out of range / Impossible values
    bad_age_idx = np.random.choice(n, size=6, replace=False)
    dirty_df.loc[bad_age_idx[:3], 'age'] = -5  # Negative age
    dirty_df.loc[bad_age_idx[3:], 'age'] = 142 # Impossible age
    
    bad_inc_idx = np.random.choice(n, size=4, replace=False)
    dirty_df.loc[bad_inc_idx, 'annual_income'] = -15000  # Negative income
    
    bad_dti_idx = np.random.choice(n, size=5, replace=False)
    dirty_df.loc[bad_dti_idx, 'debt_to_income_ratio'] = 9.99  # Extreme invalid DTI
    
    # 3. Duplicate records
    dup_rows = dirty_df.iloc[:8].copy()
    dirty_df = pd.concat([dirty_df, dup_rows], ignore_index=True)
    
    return dirty_df


def generate_longitudinal_batches(base_df: pd.DataFrame, n_per_month: int = 1000, seed: int = 101) -> pd.DataFrame:
    """Generate 6 monthly batches demonstrating macroeconomic drift (rising DTI, falling income)."""
    np.random.seed(seed)
    batches = []
    
    for month in range(1, 7):
        # Gradual macroeconomic stress shift
        # DTI increases by 0.02 each month, income shrinks slightly, default risk creeps up
        batch = generate_credit_dataset(n_samples=n_per_month, seed=seed + month * 17)
        batch['batch_month'] = f"Month_{month}"
        
        # Drift factors
        income_shift = 1.0 - (month - 1) * 0.035  # up to 17.5% drop in median income
        dti_shift = (month - 1) * 0.025            # up to +0.125 higher DTI
        
        batch['annual_income'] = np.round(np.clip(batch['annual_income'] * income_shift, 15000, 300000), -2)
        batch['debt_to_income_ratio'] = np.round(np.clip(batch['debt_to_income_ratio'] + dti_shift, 0.04, 0.85), 4)
        
        # Higher risk due to macro stress
        stress_bump = (batch['debt_to_income_ratio'] > 0.40) & (np.random.uniform(0, 1, n_per_month) < 0.20)
        batch.loc[stress_bump, 'risk_label'] = 1
        
        batches.append(batch)
        
    return pd.concat(batches, ignore_index=True)


if __name__ == '__main__':
    data_dir = os.path.dirname(os.path.abspath(__file__))
    raw_dir = os.path.join(data_dir, 'raw')
    processed_dir = os.path.join(data_dir, 'processed')
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)
    
    print("Generating baseline credit risk dataset (5,000 records)...")
    clean_df = generate_credit_dataset(n_samples=5000, seed=42)
    clean_path = os.path.join(processed_dir, 'credit_risk_clean.csv')
    clean_df.to_csv(clean_path, index=False)
    print(f" Saved clean dataset: {clean_path} (Shape: {clean_df.shape})")
    
    print("Generating raw unvalidated dataset with test anomalies...")
    raw_df = inject_anomalies_for_validation(clean_df, seed=42)
    raw_path = os.path.join(raw_dir, 'credit_risk_raw.csv')
    raw_df.to_csv(raw_path, index=False)
    print(f" Saved raw dataset: {raw_path} (Shape: {raw_df.shape})")
    
    print("Generating longitudinal drift monitoring batches (6 months x 1,000 records)...")
    drift_df = generate_longitudinal_batches(clean_df, n_per_month=1000, seed=101)
    drift_path = os.path.join(processed_dir, 'monthly_monitoring_batches.csv')
    drift_df.to_csv(drift_path, index=False)
    print(f" Saved monthly drift batches: {drift_path} (Shape: {drift_df.shape})")
    
    print("\nDataset generation completed successfully!")
