# ══════════════════════════════════════════════════════════════════════════
# TOTAL REWARDS INTELLIGENCE PLATFORM
# create_sample_data.py
#
# WHO RUNS THIS: Data team — once, after generate_data.py
#
# WHAT IT DOES:
#   Creates three files for the Streamlit app:
#
#   1. can_north_sample_employees.csv
#      200 employees from the CAN North synthetic dataset
#      This is the demo file users download and explore
#      They can tweak values in Excel and re-upload
#      to see how the analysis changes
#
#   2. data_template.csv
#      Empty template showing required columns
#      Users fill this with their own data
#      No personal information required
#      Masked employee codes only
#
#   3. salary_bands_template.csv
#      Empty template for user's own salary bands
#      If they upload this — we use their bands
#      If they do not — we use ESDC government data
#
# PRIVACY:
#   Sample data uses masked employee IDs only
#   No real names, SIN, DOB, or identifiers
#   CAN North Financial is fictional
#
# HOW TO RUN:
#   python create_sample_data.py
#
# ⚠️  DECLARATION:
#   All data is synthetic.
#   CAN North Financial is fictional.
#   No real employees. No real salaries.
# ══════════════════════════════════════════════════════════════════════════

import os
import pandas as pd
import numpy as np
from datetime import datetime


# ── CONFIGURATION ─────────────────────────────────────────────────────────

CONFIG = {
    'historical_file':  'data/can_north_pay_equity_historical.csv',
    'outputs_dir':      'data',
    'sample_file':      'data/can_north_sample_employees.csv',
    'template_file':    'data/data_template.csv',
    'bands_file':       'data/salary_bands_template.csv',
    'sample_size':      200,
    'random_seed':      99,
}


# ══════════════════════════════════════════════════════════════════════════
# FILE 1 — SAMPLE EMPLOYEE DATA
# ══════════════════════════════════════════════════════════════════════════

def create_sample_employees():
    """
    Creates a 200-employee sample from the historical dataset.

    WHY 200 employees?
    Large enough to show meaningful gaps.
    Small enough to open comfortably in Excel.
    Same as the Flight Risk demo file convention.

    New employee IDs are assigned so they look like
    a fresh quarterly upload — not the training data.

    Columns kept:
    Only what is needed for the pay equity analysis.
    No columns that could look like personal data.
    """

    if not os.path.exists(CONFIG['historical_file']):
        print(f"\n  ERROR: {CONFIG['historical_file']} not found.")
        print(f"  Run: python generate_data.py first.")
        return False

    df = pd.read_csv(CONFIG['historical_file'])

    # Take a fresh random sample
    sample = df.sample(
        n=CONFIG['sample_size'],
        random_state=CONFIG['random_seed'],
    ).copy()

    # Assign new masked IDs
    # Format: CNF followed by 4-digit number
    # Starting from 1401 so they do not overlap training data
    sample['employee_id'] = [
        f'CNF{str(i).zfill(4)}'
        for i in range(1401, 1401 + CONFIG['sample_size'])
    ]

    # Keep only the columns needed for pay equity analysis
    # Drop anything that could be personal or redundant
    keep_cols = [
        'employee_id',
        'gender',
        'role_level',
        'division',
        'location',
        'tenure_years',
        'performance',
        'salary',
    ]

    # Only keep columns that exist
    keep_cols = [c for c in keep_cols if c in sample.columns]
    sample    = sample[keep_cols].reset_index(drop=True)

    # Save
    os.makedirs(CONFIG['outputs_dir'], exist_ok=True)
    sample.to_csv(CONFIG['sample_file'], index=False,
                  encoding='utf-8')

    return sample


# ══════════════════════════════════════════════════════════════════════════
# FILE 2 — DATA TEMPLATE
# ══════════════════════════════════════════════════════════════════════════

def create_data_template():
    """
    Creates an empty data template with example rows.

    Users download this, fill it with their own data,
    and upload it to the app.

    WHAT IS REQUIRED:
    employee_id   masked code — e.g. EMP001
                  NO real names. NO SIN numbers.
    gender        Male / Female / Non-binary /
                  Prefer not to say
    role_level    your job levels
                  e.g. Analyst, Manager, Director
                  or Level 1, Level 2, Level 3
    division      your business units
                  can be masked e.g. DIV_A, DIV_B
    location      city or region
                  can be masked e.g. LOC_1, LOC_2
    tenure_years  years at the organization
    performance   performance score on your scale
                  convert to 0-100 if needed
    salary        annual base salary only
                  no bonuses, no benefits

    WHAT IS NOT REQUIRED:
    Full name, SIN, date of birth, address,
    email, phone, or any personal identifier.
    """

    # Three example rows showing the format
    template = pd.DataFrame([
        {
            'employee_id':  'EMP001',
            'gender':       'Female',
            'role_level':   'Analyst',
            'division':     'Finance',
            'location':     'Toronto',
            'tenure_years': 3,
            'performance':  72,
            'salary':       68000,
        },
        {
            'employee_id':  'EMP002',
            'gender':       'Male',
            'role_level':   'Manager',
            'division':     'Operations',
            'location':     'Calgary',
            'tenure_years': 8,
            'performance':  85,
            'salary':       115000,
        },
        {
            'employee_id':  'EMP003',
            'gender':       'Non-binary',
            'role_level':   'Director',
            'division':     'Finance',
            'location':     'Vancouver',
            'tenure_years': 12,
            'performance':  91,
            'salary':       148000,
        },
    ])

    template.to_csv(CONFIG['template_file'], index=False,
                    encoding='utf-8')

    return template


# ══════════════════════════════════════════════════════════════════════════
# FILE 3 — SALARY BANDS TEMPLATE
# ══════════════════════════════════════════════════════════════════════════

def create_bands_template():
    """
    Creates a salary bands template for users to fill in.

    If they upload this — we use their bands.
    If they do not — we use ESDC government data.

    Columns:
    role_level    must match the role_level values
                  in their employee data exactly
    min_salary    minimum of the salary band
    mid_salary    midpoint of the salary band
                  this is the benchmark for compa-ratio
    max_salary    maximum of the salary band

    Source of default values shown as example:
    Government of Canada ESDC wage data
    open.canada.ca
    These are illustrative — replace with your own.
    """

    bands = pd.DataFrame([
        {
            'role_level':  'Analyst',
            'min_salary':  52000,
            'mid_salary':  67000,
            'max_salary':  82000,
            'source':      'Replace with your internal bands',
        },
        {
            'role_level':  'Senior Analyst',
            'min_salary':  72000,
            'mid_salary':  87000,
            'max_salary':  102000,
            'source':      'Replace with your internal bands',
        },
        {
            'role_level':  'Manager',
            'min_salary':  95000,
            'mid_salary':  112000,
            'max_salary':  130000,
            'source':      'Replace with your internal bands',
        },
        {
            'role_level':  'Director',
            'min_salary':  120000,
            'mid_salary':  145000,
            'max_salary':  165000,
            'source':      'Replace with your internal bands',
        },
        {
            'role_level':  'VP',
            'min_salary':  165000,
            'mid_salary':  192000,
            'max_salary':  220000,
            'source':      'Replace with your internal bands',
        },
    ])

    bands.to_csv(CONFIG['bands_file'], index=False,
                 encoding='utf-8')

    return bands


# ══════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════

def main():
    print(f"\n{'='*65}")
    print(f"CREATE SAMPLE DATA")
    print(f"Total Rewards Intelligence Platform")
    print(f"Run date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"{'='*65}")
    print(f"\n  ⚠️  Synthetic data. Fictional organization.")
    print(f"  No real employees. No personal information.")

    # File 1 — Sample employees
    print(f"\n[1/3] Creating sample employee file...")
    sample = create_sample_employees()
    if sample is not None:
        print(f"      ✅ {len(sample):,} employees")
        print(f"      ✅ Columns: {list(sample.columns)}")
        print(f"      ✅ Saved: {CONFIG['sample_file']}")
        print(f"\n      SAMPLE OVERVIEW:")
        print(f"      {'Gender':<15} {'Count':>6}  {'%':>6}")
        print(f"      {'─'*15} {'─'*6}  {'─'*6}")
        for gender, count in sample['gender'].value_counts().items():
            print(f"      {gender:<15} {count:>6}  "
                  f"{count/len(sample)*100:>5.1f}%")
        print(f"\n      {'Role':<20} {'Count':>6}")
        print(f"      {'─'*20} {'─'*6}")
        for role in [
            'Analyst', 'Senior Analyst', 'Manager',
            'Director', 'VP'
        ]:
            count = (sample['role_level'] == role).sum()
            if count > 0:
                print(f"      {role:<20} {count:>6}")
        print(f"\n      Avg salary: "
              f"${sample['salary'].mean():,.0f}")
        male_avg   = sample[
            sample['gender'] == 'Male'
        ]['salary'].mean()
        female_avg = sample[
            sample['gender'] == 'Female'
        ]['salary'].mean()
        print(f"      Male avg:   ${male_avg:,.0f}")
        print(f"      Female avg: ${female_avg:,.0f}")
        print(f"      Raw gap:    ${male_avg - female_avg:,.0f}")

    # File 2 — Data template
    print(f"\n[2/3] Creating data template...")
    template = create_data_template()
    print(f"      ✅ {len(template)} example rows")
    print(f"      ✅ Saved: {CONFIG['template_file']}")
    print(f"\n      Users fill this with their own data.")
    print(f"      Masked employee codes only.")
    print(f"      No names. No SIN. No personal data.")

    # File 3 — Salary bands template
    print(f"\n[3/3] Creating salary bands template...")
    bands = create_bands_template()
    print(f"      ✅ {len(bands)} role bands")
    print(f"      ✅ Saved: {CONFIG['bands_file']}")
    print(f"\n      Users replace these with their own bands.")
    print(f"      If not uploaded — ESDC data is used.")

    print(f"\n{'='*65}")
    print(f"THREE FILES READY FOR THE APP:")
    print(f"{'='*65}")
    print(f"\n  1. {CONFIG['sample_file']}")
    print(f"     Demo data — download, tweak, re-upload")
    print(f"\n  2. {CONFIG['template_file']}")
    print(f"     Empty template — fill with your own data")
    print(f"\n  3. {CONFIG['bands_file']}")
    print(f"     Salary bands — optional upload")
    print(f"\n  Next: streamlit run app.py")
    print(f"{'='*65}\n")


if __name__ == '__main__':
    main()
    