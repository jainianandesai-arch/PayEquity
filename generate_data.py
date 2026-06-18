# ══════════════════════════════════════════════════════════════════════════
# TOTAL REWARDS INTELLIGENCE PLATFORM
# generate_data.py
#
# WHO RUNS THIS: Data team — once before anything else
#
# WHAT IT DOES:
#   Generates 1,400 synthetic employees for CAN North Financial.
#   This data powers the Demo tab in the Streamlit app.
#   It is NOT training data — pay equity is measurement not prediction.
#   It exists solely to demonstrate the tool working.
#
# WHAT IT SAVES:
#   data/can_north_pay_equity_historical.csv
#
# COLUMNS:
#   employee_id    masked code — CNF0001
#   gender         Male / Female
#   job_class      human readable title
#                  Financial Analyst, Finance Manager etc.
#   job_grade      pay grade — Grade 3 through Grade 7
#   role_level     kept for internal reference
#                  Analyst, Senior Analyst etc.
#   division       Pension Administration, Wealth Management,
#                  Retail Banking
#   location       Toronto, Calgary, Vancouver
#   tenure_years   0 to 25 years
#   performance    0 to 100 score
#   salary         annual base salary in CAD
#
# HOW TO RUN:
#   python generate_data.py
#
# ⚠️  DECLARATION:
#   All data is synthetic.
#   CAN North Financial is fictional.
#   No real employees. No real salaries.
# ══════════════════════════════════════════════════════════════════════════

import os
import numpy as np
import pandas as pd
from datetime import datetime


# ══════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════

CONFIG = {
    'n_employees':      1400,
    'random_seed':      42,
    'output_dir':       'data',
    'output_file':      'data/can_north_pay_equity_historical.csv',

    # Gender pay gap embedded deliberately
    # Our analysis must find this — if it cannot it is not trustworthy
    'gender_gap_base':   6500,   # base gap for all female employees
    'gender_gap_senior': 4000,   # additional gap for Grade 6 and 7

    # Replacement cost per employee — used in business case
    'replacement_cost':  78000,
}


# ══════════════════════════════════════════════════════════════════════════
# JOB STRUCTURE
#
# CAN North Financial uses both job_class and job_grade.
# job_class = human readable title employees see
# job_grade = internal pay grade HR uses
#
# Multiple job classes can sit in the same grade.
# Pay equity compares within grade — not just within title.
# This mirrors real Canadian financial services organizations.
# ══════════════════════════════════════════════════════════════════════════

# Maps role_level to job_grade
ROLE_TO_GRADE = {
    'Analyst':        'Grade 3',
    'Senior Analyst': 'Grade 4',
    'Manager':        'Grade 5',
    'Director':       'Grade 6',
    'VP':             'Grade 7',
}

# Maps role_level to job_class by division
# Different divisions have different job titles
# but the same grade structure
JOB_CLASS_MAP = {
    'Pension Administration': {
        'Analyst':        'Pension Analyst',
        'Senior Analyst': 'Senior Pension Analyst',
        'Manager':        'Pension Manager',
        'Director':       'Pension Director',
        'VP':             'Vice President — Pension',
    },
    'Wealth Management': {
        'Analyst':        'Wealth Analyst',
        'Senior Analyst': 'Senior Wealth Analyst',
        'Manager':        'Wealth Manager',
        'Director':       'Wealth Director',
        'VP':             'Vice President — Wealth',
    },
    'Retail Banking': {
        'Analyst':        'Banking Analyst',
        'Senior Analyst': 'Senior Banking Analyst',
        'Manager':        'Branch Manager',
        'Director':       'Regional Director',
        'VP':             'Vice President — Banking',
    },
}

# Base salary by role level — Canadian financial services 2025
ROLE_BASE_SALARY = {
    'Analyst':        65000,
    'Senior Analyst': 85000,
    'Manager':        110000,
    'Director':       145000,
    'VP':             190000,
}

# Division salary premium
DIVISION_PREMIUM = {
    'Pension Administration':  0,
    'Wealth Management':       8000,
    'Retail Banking':         -5000,
}

# Location salary premium
LOCATION_PREMIUM = {
    'Toronto':   5000,
    'Calgary':   2000,
    'Vancouver': 3000,
}


# ══════════════════════════════════════════════════════════════════════════
# STEP 1 — BUILD WORKFORCE STRUCTURE
# ══════════════════════════════════════════════════════════════════════════

def build_workforce(N, seed):
    """
    Creates the workforce structure for CAN North Financial.

    Organizational pyramid — more juniors than seniors.
    Same proportions as real Canadian financial services.
    Gender split reflects industry — slight male skew at senior levels.
    """
    np.random.seed(seed)

    DIVISIONS   = [
        'Pension Administration',
        'Wealth Management',
        'Retail Banking',
    ]
    ROLE_LEVELS = [
        'Analyst', 'Senior Analyst', 'Manager',
        'Director', 'VP',
    ]
    LOCATIONS   = ['Toronto', 'Calgary', 'Vancouver']

    # Organizational pyramid
    role_levels = np.random.choice(
        ROLE_LEVELS, N,
        p=[0.30, 0.28, 0.22, 0.13, 0.07],
    )

    # Division distribution
    divisions = np.random.choice(
        DIVISIONS, N,
        p=[0.35, 0.40, 0.25],
    )

    # Location — Toronto is HQ
    locations = np.random.choice(
        LOCATIONS, N,
        p=[0.60, 0.25, 0.15],
    )

    # Gender — slight male skew reflects industry reality
    genders = np.random.choice(
        ['Male', 'Female'], N,
        p=[0.52, 0.48],
    )

    # Employee IDs — masked codes
    emp_ids = [
        f'CNF{str(i).zfill(4)}'
        for i in range(1, N + 1)
    ]

    # Tenure — exponential distribution
    # Most employees are relatively new
    # Fewer have 15+ years
    tenure_years = np.clip(
        np.random.exponential(5, N).astype(int),
        0, 25,
    )

    # Performance — normally distributed around 70
    performance = np.clip(
        np.random.normal(70, 14, N),
        20, 100,
    ).round(1)

    # Job class — from division + role level mapping
    job_classes = [
        JOB_CLASS_MAP[div][role]
        for div, role in zip(divisions, role_levels)
    ]

    # Job grade — from role level mapping
    job_grades = [
        ROLE_TO_GRADE[role]
        for role in role_levels
    ]

    # Role level encoded — for internal use
    role_order = {
        'Analyst': 1, 'Senior Analyst': 2, 'Manager': 3,
        'Director': 4, 'VP': 5,
    }
    role_encoded = np.array([
        role_order[r] for r in role_levels
    ])

    return pd.DataFrame({
        'employee_id':        emp_ids,
        'gender':             genders,
        'job_class':          job_classes,
        'job_grade':          job_grades,
        'role_level':         role_levels,
        'role_level_encoded': role_encoded,
        'division':           divisions,
        'location':           locations,
        'tenure_years':       tenure_years,
        'performance':        performance,
    })


# ══════════════════════════════════════════════════════════════════════════
# STEP 2 — CALCULATE SALARIES
# ══════════════════════════════════════════════════════════════════════════

def calculate_salaries(df, config):
    """
    Calculates salary for each employee.

    SALARY FORMULA:
    salary = base(role)
           + division_premium
           + location_premium
           + tenure_bonus       (diminishing returns)
           + performance_bonus
           + random_noise       (negotiation, market timing)
           + gender_gap         ← THE INEQUITY WE EMBED

    WHY embed the gap deliberately?
    The analysis must find what we planted.
    If it cannot — the tool is not trustworthy.

    GAP STRUCTURE:
    Base gap of $6,500 for all female employees.
    Additional $4,000 for Grade 6 (Director) and Grade 7 (VP).
    This mirrors real-world patterns where the gap widens
    at senior levels.
    """
    salaries = []

    for i, row in df.iterrows():
        base     = ROLE_BASE_SALARY[row['role_level']]
        div_adj  = DIVISION_PREMIUM[row['division']]
        loc_adj  = LOCATION_PREMIUM[row['location']]

        # Tenure bonus — diminishing returns
        # np.sqrt slows growth — realistic salary curves
        tenure_bonus = (
            row['tenure_years'] * 800
            + np.sqrt(row['tenure_years']) * 1200
        )

        # Performance bonus — centered at 70 (average)
        perf_bonus = (row['performance'] - 70) * 350

        # Random noise — negotiation, market timing
        noise = np.random.normal(0, 6000)

        # Gender gap — embedded deliberately
        # Grade 6 and 7 have larger gaps
        # reflecting real-world senior-level inequity
        if row['gender'] == 'Female':
            gender_gap = -config['gender_gap_base']
            if row['job_grade'] in ['Grade 6', 'Grade 7']:
                gender_gap -= config['gender_gap_senior']
        else:
            gender_gap = 0

        salary = (
            base + div_adj + loc_adj
            + tenure_bonus + perf_bonus
            + noise + gender_gap
        )

        # Salary floor — no one earns less than $45K
        salary = max(salary, 45000)

        # Round to nearest $100
        salaries.append(round(salary, -2))

    return salaries


# ══════════════════════════════════════════════════════════════════════════
# STEP 3 — PRINT AND SAVE
# ══════════════════════════════════════════════════════════════════════════

def print_and_save(df, config):
    """
    Prints summary of generated dataset and saves to CSV.
    Same reporting style as the rest of the platform.
    """

    print(f"\n{'='*65}")
    print(f"DATA GENERATION COMPLETE")
    print(f"Run date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"{'='*65}")
    print(f"\n  File: {config['output_file']}")
    print(f"  Rows: {len(df):,} employees")
    print(f"  Cols: {df.shape[1]}")

    print(f"\n  WORKFORCE COMPOSITION:")
    print(f"  {'Gender':<15} {'Count':>6}  {'%':>6}")
    print(f"  {'─'*15} {'─'*6}  {'─'*6}")
    for gender, count in df['gender'].value_counts().items():
        print(
            f"  {gender:<15} {count:>6}  "
            f"{count/len(df)*100:>5.1f}%"
        )

    print(f"\n  JOB STRUCTURE:")
    print(f"  {'Job Grade':<12} {'Job Class':<35} "
          f"{'Count':>6}")
    print(f"  {'─'*12} {'─'*35} {'─'*6}")
    for grade in [
        'Grade 3', 'Grade 4', 'Grade 5',
        'Grade 6', 'Grade 7',
    ]:
        grp = df[df['job_grade'] == grade]
        for jc in grp['job_class'].unique():
            count = len(grp[grp['job_class'] == jc])
            print(
                f"  {grade:<12} {jc:<35} {count:>6}"
            )

    print(f"\n  SALARY OVERVIEW:")
    print(
        f"  Average salary:   "
        f"${df['salary'].mean():>10,.0f}"
    )
    print(
        f"  Median salary:    "
        f"${df['salary'].median():>10,.0f}"
    )
    print(
        f"  Min salary:       "
        f"${df['salary'].min():>10,.0f}"
    )
    print(
        f"  Max salary:       "
        f"${df['salary'].max():>10,.0f}"
    )

    print(f"\n  RAW GENDER GAP BY JOB GRADE:")
    print(f"  {'Grade':<12} {'Male Avg':>12} "
          f"{'Female Avg':>12} {'Gap':>10}")
    print(f"  {'─'*12} {'─'*12} {'─'*12} {'─'*10}")
    for grade in [
        'Grade 3', 'Grade 4', 'Grade 5',
        'Grade 6', 'Grade 7',
    ]:
        grp    = df[df['job_grade'] == grade]
        male   = grp[grp['gender'] == 'Male']['salary'].mean()
        female = grp[grp['gender'] == 'Female']['salary'].mean()
        gap    = male - female
        print(
            f"  {grade:<12} ${male:>11,.0f} "
            f"${female:>11,.0f} ${gap:>9,.0f}"
        )

    print(f"\n  ⚠️  DECLARATION:")
    print(f"  All data is synthetic.")
    print(f"  CAN North Financial is fictional.")
    print(f"  No real employees. No real salaries.")
    print(f"\n  Next step: python analyse.py")
    print(f"{'='*65}\n")

    os.makedirs(config['output_dir'], exist_ok=True)
    df.to_csv(
        config['output_file'],
        index=False,
        encoding='utf-8',
    )
    print(f"  ✅ Saved: {config['output_file']}")


# ══════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════

def main():
    print(f"\n{'='*65}")
    print(f"CAN NORTH FINANCIAL — DATA GENERATOR")
    print(f"Total Rewards Intelligence Platform")
    print(f"Run date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"{'='*65}")
    print(f"\n  ⚠️  Synthetic data. Fictional organization.")
    print(f"  No real employees. No personal information.")

    N    = CONFIG['n_employees']
    seed = CONFIG['random_seed']

    # Step 1 — Build workforce
    print(f"\n[1/3] Building workforce structure...")
    print(f"      {N:,} employees | seed={seed}")
    df = build_workforce(N, seed)
    print(f"      ✅ Workforce built")
    print(f"      ✅ job_class and job_grade assigned")

    # Step 2 — Calculate salaries
    print(f"\n[2/3] Calculating salaries...")
    print(
        f"      Embedding "
        f"${CONFIG['gender_gap_base']:,} base gender gap"
    )
    print(
        f"      Additional "
        f"${CONFIG['gender_gap_senior']:,} gap "
        f"for Grade 6 and Grade 7"
    )
    np.random.seed(seed)
    df['salary'] = calculate_salaries(df, CONFIG)
    print(f"      ✅ Salaries calculated")

    # Step 3 — Save
    print(f"\n[3/3] Saving dataset...")
    print_and_save(df, CONFIG)


if __name__ == '__main__':
    main()