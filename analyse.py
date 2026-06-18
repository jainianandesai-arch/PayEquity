# ══════════════════════════════════════════════════════════════════════════
# TOTAL REWARDS INTELLIGENCE PLATFORM
# analyse.py — Pay Equity Engine
#
# PILLAR 1 OF 3: PAY EQUITY ANALYSIS
#
# WHAT THIS FILE DOES:
#   Answers one question: "Do we have a gender pay gap?"
#   Pure automation. No machine learning.
#   pandas + scipy only.
#
# PRIMARY GROUPING: job_grade
#   Most defensible under the Pay Equity Act.
#   Same grade = same compensation schedule = same job class.
#   If job_grade is not present — falls back to job_class.
#
# SECONDARY GROUPING: job_class
#   Human readable title comparison.
#   Shown alongside grade analysis.
#
# HOW IT WORKS:
#   1. Load and validate employee data
#   2. Detect grouping columns
#   3. Build internal salary bands
#   4. Calculate compa-ratios
#   5. Measure gap by job grade and job class
#   6. Test statistical significance
#   7. Build remediation plan
#   8. Optimise budget allocation
#   9. Compile results for agent
#  10. Save outputs and print report
#
# LEGISLATIVE BASIS:
#   Pay Equity Act, S.C. 2018, c. 27, s. 416
#   laws-lois.justice.gc.ca/eng/acts/P-4.2/
#   Ontario Pay Equity Act, R.S.O. 1990, c. P.7
#   ontario.ca/laws/statute/90p07
#
# PRIVACY:
#   No personal information used or stored.
#   Employee IDs are masked codes only.
#   No names, SIN, DOB, or identifiers.
#
# HOW TO RUN:
#   python analyse.py
#   python analyse.py --data your_file.csv
#   python analyse.py --budget 800000
#   python analyse.py --budget full
#
# OUTPUT:
#   outputs/pay_equity_results.json
#   outputs/pay_equity_report.txt
#   outputs/remediation_list.csv
#
# ⚠️  DECLARATION:
#   Default data is synthetic.
#   CAN North Financial is fictional.
#   No real employees. No real salaries.
# ══════════════════════════════════════════════════════════════════════════

import os
import sys
import json
import argparse
import warnings
from datetime import date, datetime

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings('ignore')


# ══════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════

CONFIG = {
    # File paths
    'data_file':        'data/can_north_pay_equity_historical.csv',
    'outputs_dir':      'outputs',
    'results_file':     'outputs/pay_equity_results.json',
    'report_file':      'outputs/pay_equity_report.txt',
    'remediation_file': 'outputs/remediation_list.csv',

    # Pay equity threshold
    # Below 0.95 = underpaid — industry standard
    'underpaid_threshold': 0.95,

    # Statistical significance threshold
    'significance_level': 0.05,

    # Replacement cost per employee
    'replacement_cost': 78000,

    # Job grade display order
    'grade_order': [
        'Grade 3', 'Grade 4', 'Grade 5',
        'Grade 6', 'Grade 7',
    ],

    # Risk weights for remediation prioritization
    'risk_weights': {
        'gap_size':    0.50,
        'compa_ratio': 0.30,
        'seniority':   0.20,
    },

    # Seniority risk by job grade
    # Senior grades carry highest reputational risk
    'grade_risk': {
        'Grade 7': 1.0,
        'Grade 6': 0.8,
        'Grade 5': 0.5,
        'Grade 4': 0.3,
        'Grade 3': 0.1,
    },

    # Approved legislation sources only
    'legislation': {
        'federal': {
            'name':    'Pay Equity Act, S.C. 2018, c. 27, s. 416',
            'url':     'laws-lois.justice.gc.ca/eng/acts/P-4.2/',
            'section': 'Section 61(2)',
            'rule': (
                'Employers with 100+ employees may phase in '
                'adjustments over maximum 3 years. '
                'Minimum annual spend: 1% of previous year payroll.'
            ),
        },
        'ontario': {
            'name':    'Pay Equity Act, R.S.O. 1990, c. P.7',
            'url':     'ontario.ca/laws/statute/90p07',
            'section': 'Section 13(4)',
            'rule': (
                'Same 1% minimum annual rule. '
                'All incumbents in a female job class '
                'must receive the same dollar adjustment.'
            ),
        },
    },
}


# ══════════════════════════════════════════════════════════════════════════
# STEP 1 — LOAD AND VALIDATE
# ══════════════════════════════════════════════════════════════════════════

def load_and_validate(data_file):
    """
    Loads employee data and validates required columns.

    REQUIRED COLUMNS:
    employee_id   masked code — no real names
    gender        Male / Female / Non-binary / Prefer not to say
    job_class     human readable job title
    division      business unit
    location      city or region
    tenure_years  years at the organization
    performance   performance score 0-100
    salary        annual base salary

    OPTIONAL BUT RECOMMENDED:
    job_grade     pay grade e.g. Grade 5 or Band B
                  if present — used as primary grouping
                  most defensible under Pay Equity Act

    PRIVACY CHECK:
    Warns if column names suggest personal identifiers.
    """

    if not os.path.exists(data_file):
        print(f"\n  ERROR: {data_file} not found.")
        print(f"  Run: python generate_data.py")
        sys.exit(1)

    df = pd.read_csv(data_file)

    # Must have at least one grouping column
    required = [
        'employee_id', 'gender',
        'division', 'location',
        'tenure_years', 'performance', 'salary',
    ]

    # Need job_class OR job_grade OR role_level
    has_grouping = any(
        c in df.columns
        for c in ['job_grade', 'job_class', 'role_level']
    )

    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"\n  ERROR: Missing required columns: {missing}")
        sys.exit(1)

    if not has_grouping:
        print(
            f"\n  ERROR: Need at least one of: "
            f"job_grade, job_class, role_level"
        )
        sys.exit(1)

    # Privacy guardrail
    sensitive = [
        'name', 'sin', 'ssn', 'birth', 'dob',
        'address', 'email', 'phone',
    ]
    for col in df.columns:
        if any(kw in col.lower() for kw in sensitive):
            print(
                f"\n  ⚠️  WARNING: Column '{col}' may contain "
                f"personal information. Use masked codes only."
            )

    # Clean missing values
    if df['salary'].isna().sum() > 0:
        print(
            f"\n  ⚠️  Removing "
            f"{df['salary'].isna().sum()} rows "
            f"with missing salary."
        )
        df = df[df['salary'].notna()]

    if df['gender'].isna().sum() > 0:
        print(
            f"\n  ⚠️  Removing "
            f"{df['gender'].isna().sum()} rows "
            f"with missing gender."
        )
        df = df[df['gender'].notna()]

    return df.reset_index(drop=True)


# ══════════════════════════════════════════════════════════════════════════
# STEP 2 — DETECT GROUPING COLUMNS
# ══════════════════════════════════════════════════════════════════════════

def detect_grouping(df):
    """
    Detects which grouping columns are available.

    Priority:
    1. job_grade — most defensible under Pay Equity Act
    2. job_class — human readable title
    3. role_level — backward compatibility

    Returns dict of available grouping columns
    and which to use as primary.
    """

    available = {}

    if 'job_grade' in df.columns:
        available['job_grade'] = df['job_grade'].nunique()

    if 'job_class' in df.columns:
        available['job_class'] = df['job_class'].nunique()

    if 'role_level' in df.columns:
        available['role_level'] = df['role_level'].nunique()

    # Primary grouping — job_grade preferred
    if 'job_grade' in available:
        primary = 'job_grade'
    elif 'job_class' in available:
        primary = 'job_class'
    else:
        primary = 'role_level'

    return {
        'primary':   primary,
        'available': available,
        'has_grade': 'job_grade' in available,
        'has_class': 'job_class' in available,
    }


# ══════════════════════════════════════════════════════════════════════════
# STEP 3 — BUILD INTERNAL SALARY BANDS
# ══════════════════════════════════════════════════════════════════════════

def build_internal_bands(df, grouping):
    """
    Derives salary bands from the internal data.

    The Pay Equity Act compares job classes WITHIN
    the organization — not against external market data.

    Band structure per group:
    Min = 25th percentile of actual salaries
    Mid = median of actual salaries
    Max = 75th percentile of actual salaries

    Uses primary grouping column.
    """

    primary = grouping['primary']
    bands   = {}

    for group in df[primary].unique():
        salaries    = df[df[primary] == group]['salary']
        bands[group] = {
            'min':   round(float(salaries.quantile(0.25)), -2),
            'mid':   round(float(salaries.quantile(0.50)), -2),
            'max':   round(float(salaries.quantile(0.75)), -2),
            'count': int(len(salaries)),
        }

    return bands


# ══════════════════════════════════════════════════════════════════════════
# STEP 4 — CALCULATE COMPA-RATIOS
# ══════════════════════════════════════════════════════════════════════════

def calculate_compa_ratios(df, bands, grouping):
    """
    Calculates compa-ratio for every employee.

    compa_ratio = actual_salary / tenure_adjusted_midpoint

    Tenure adjustment:
    0 years  → 20% through the band (near minimum)
    25 years → 85% through the band (near maximum)

    Equity flags:
    Critical   = below 0.90
    Underpaid  = 0.90 to 0.95
    Fair       = 0.95 to 1.05
    Above Band = above 1.05
    """

    primary = grouping['primary']

    def tenure_adjusted_midpoint(group, tenure):
        if group not in bands:
            return float(
                np.median([b['mid'] for b in bands.values()])
            )
        b             = bands[group]
        tenure_factor = np.clip(
            0.20 + (tenure / 25) * 0.65, 0.20, 0.85
        )
        return round(
            b['min'] + (b['max'] - b['min']) * tenure_factor,
            -2,
        )

    df                    = df.copy()
    df['expected_salary'] = df.apply(
        lambda row: tenure_adjusted_midpoint(
            row[primary], row['tenure_years'],
        ),
        axis=1,
    )
    df['compa_ratio']     = (
        df['salary'] / df['expected_salary']
    ).round(3)
    df['equity_flag']     = df['compa_ratio'].apply(
        lambda cr:
        'Critical'   if cr < 0.90 else
        'Underpaid'  if cr < 0.95 else
        'Fair'       if cr < 1.05 else
        'Above Band'
    )

    return df


# ══════════════════════════════════════════════════════════════════════════
# STEP 5 — GAP ANALYSIS
# ══════════════════════════════════════════════════════════════════════════

def analyse_gaps(df, grouping):
    """
    Calculates the pay gap by job grade, job class, and division.

    For each segment:
    → Average salary for male and female employees
    → Gap in dollars and percentage
    → T-test: is the gap statistically significant?

    p-value < 0.05 = less than 5% chance the gap
    is due to random variation. The gap is real.

    PRIMARY: by job_grade (most defensible)
    SECONDARY: by job_class (human readable)
    ALSO: by division
    """

    def gap_stats(male_sal, female_sal, group_name):
        if len(male_sal) < 2 or len(female_sal) < 2:
            return None
        gap     = float(male_sal.mean() - female_sal.mean())
        gap_pct = round(gap / male_sal.mean() * 100, 1) \
                  if male_sal.mean() != 0 else 0.0
        _, pval = stats.ttest_ind(male_sal, female_sal)
        return {
            'group':           group_name,
            'male_avg':        round(float(male_sal.mean()), 2),
            'female_avg':      round(float(female_sal.mean()), 2),
            'male_count':      int(len(male_sal)),
            'female_count':    int(len(female_sal)),
            'gap_dollars':     round(gap, 2),
            'gap_pct':         gap_pct,
            'p_value':         round(float(pval), 6),
            'significant':     bool(pval < CONFIG['significance_level']),
            'action_required': bool(
                gap > 0 and
                pval < CONFIG['significance_level']
            ),
        }

    # Overall gap
    all_male   = df[df['gender'] == 'Male']['salary']
    all_female = df[df['gender'] == 'Female']['salary']
    overall    = gap_stats(all_male, all_female, 'ALL')
    overall['group'] = 'ALL EMPLOYEES'

    # By job grade
    by_grade = []
    grade_order = CONFIG['grade_order']
    if grouping['has_grade']:
        for grade in grade_order:
            grp = df[df['job_grade'] == grade]
            s   = gap_stats(
                grp[grp['gender'] == 'Male']['salary'],
                grp[grp['gender'] == 'Female']['salary'],
                grade,
            )
            if s:
                by_grade.append(s)

    # By job class
    by_class = []
    if grouping['has_class']:
        for jc in sorted(df['job_class'].unique()):
            grp = df[df['job_class'] == jc]
            s   = gap_stats(
                grp[grp['gender'] == 'Male']['salary'],
                grp[grp['gender'] == 'Female']['salary'],
                jc,
            )
            if s:
                by_class.append(s)

    # By division
    by_division = []
    for div in sorted(df['division'].unique()):
        grp = df[df['division'] == div]
        s   = gap_stats(
            grp[grp['gender'] == 'Male']['salary'],
            grp[grp['gender'] == 'Female']['salary'],
            div,
        )
        if s:
            by_division.append(s)

    return {
        'overall':     overall,
        'by_grade':    by_grade,
        'by_class':    by_class,
        'by_division': by_division,
    }


# ══════════════════════════════════════════════════════════════════════════
# STEP 6 — REMEDIATION PLAN
# ══════════════════════════════════════════════════════════════════════════

def build_remediation(df, grouping):
    """
    Identifies underpaid employees and builds
    a risk-prioritized remediation plan.

    WHO IS INCLUDED:
    Female employees with compa_ratio below 0.95.

    ADJUSTMENT:
    Bring each to 95% of their tenure-adjusted
    expected salary.

    LEGISLATIVE BUDGET RULE:
    Pay Equity Act, S.C. 2018 — Section 61(2)
    If total cost > 1% of annual payroll:
    → Phase in over maximum 3 years
    → Year 1 and 2: minimum 1% of payroll
    → Year 3: close everything remaining

    RISK SCORE:
    gap_size (50%) + flight_risk (30%)
    + seniority (20%)

    PRIVACY:
    Masked employee_id only. No personal data.
    """

    underpaid = df[
        (df['gender'] == 'Female') &
        (df['compa_ratio'] < CONFIG['underpaid_threshold'])
    ].copy()

    underpaid['adjustment'] = (
        underpaid['expected_salary'] * 0.95
        - underpaid['salary']
    ).clip(lower=0).round(-2)

    underpaid = underpaid[
        underpaid['adjustment'] > 0
    ].copy().reset_index(drop=True)

    total_payroll     = float(df['salary'].sum())
    total_remediation = float(underpaid['adjustment'].sum())
    legal_min         = round(total_payroll * 0.01, 2)
    can_phase         = total_remediation > legal_min

    if len(underpaid) == 0:
        return {
            'underpaid':         pd.DataFrame(),
            'total_employees':   0,
            'total_remediation': 0.0,
            'total_payroll':     total_payroll,
            'legal_min_budget':  legal_min,
            'can_phase':         False,
            'phase_years':       1,
            'yearly_plans':      [],
        }

    # Risk scoring
    def normalize(series):
        mn, mx = series.min(), series.max()
        if mx == mn:
            return pd.Series(
                [0.5] * len(series), index=series.index
            )
        return (series - mn) / (mx - mn)

    w = CONFIG['risk_weights']

    # Seniority risk — use job_grade if available
    primary = grouping['primary']
    if primary == 'job_grade':
        seniority_map = CONFIG['grade_risk']
    else:
        # Build seniority map from unique values
        unique_groups = underpaid[primary].unique()
        n             = len(unique_groups)
        seniority_map = {
            g: (i + 1) / n
            for i, g in enumerate(sorted(unique_groups))
        }

    underpaid['risk_score'] = (
        normalize(underpaid['adjustment'])
        * w['gap_size'] +
        (1 - normalize(underpaid['compa_ratio']))
        * w['compa_ratio'] +
        underpaid[primary]
        .map(seniority_map)
        .fillna(0.3)
        * w['seniority']
    ).round(4)

    underpaid['risk_per_dollar'] = (
        underpaid['risk_score'] /
        underpaid['adjustment'].clip(lower=1)
    ).round(8)

    underpaid = underpaid.sort_values(
        'risk_per_dollar', ascending=False
    ).reset_index(drop=True)

    # Three year phased plan
    yearly_plans  = []
    remaining_ids = set(underpaid['employee_id'].tolist())

    for year in range(1, 4):
        if not remaining_ids:
            break

        remaining_df = underpaid[
            underpaid['employee_id'].isin(remaining_ids)
        ].copy()

        year_budget = (
            float(remaining_df['adjustment'].sum())
            if year == 3
            else legal_min
        )

        selected    = []
        budget_left = year_budget

        for _, row in remaining_df.iterrows():
            if budget_left >= row['adjustment']:
                selected.append(row['employee_id'])
                budget_left -= row['adjustment']

        if not selected:
            break

        year_spent = float(
            underpaid[
                underpaid['employee_id'].isin(selected)
            ]['adjustment'].sum()
        )

        yearly_plans.append({
            'year':     year,
            'budget':   round(year_budget, 2),
            'count':    len(selected),
            'spent':    round(year_spent, 2),
            'selected': selected,
        })

        remaining_ids -= set(selected)

    return {
        'underpaid':         underpaid,
        'total_employees':   len(underpaid),
        'total_remediation': round(total_remediation, 2),
        'total_payroll':     round(total_payroll, 2),
        'legal_min_budget':  legal_min,
        'can_phase':         can_phase,
        'phase_years':       3 if can_phase else 1,
        'yearly_plans':      yearly_plans,
    }


# ══════════════════════════════════════════════════════════════════════════
# STEP 7 — BUDGET OPTIMISATION
# ══════════════════════════════════════════════════════════════════════════

def optimise_budget(remediation, budget_input='legal_min'):
    """
    Optimises remediation allocation given a fixed budget.

    Answers: "Given $X — who do we adjust first to close
    the maximum legal and business risk?"

    BUDGET OPTIONS:
    'legal_min'  1% of payroll — legal minimum (default)
    'full'       close everything now
    any number   custom budget e.g. 800000

    METHOD: greedy knapsack — highest risk per dollar first.
    Auditable and explainable to a Commissioner.
    """

    underpaid         = remediation['underpaid']
    total_remediation = remediation['total_remediation']
    total_payroll     = remediation['total_payroll']
    legal_min         = remediation['legal_min_budget']

    if len(underpaid) == 0:
        return {
            'budget_input':    budget_input,
            'budget_used':     0.0,
            'budget_label':    'No adjustment needed',
            'selected_count':  0,
            'deferred_count':  0,
            'risk_closed_pct': 100.0,
            'gap_closed_pct':  100.0,
            'spent':           0.0,
            'scenarios':       [],
        }

    total_risk = float(underpaid['risk_score'].sum())

    # Resolve budget
    if budget_input == 'legal_min':
        budget       = legal_min
        budget_label = (
            f"Legal minimum — 1% of payroll "
            f"(${legal_min:,.0f}) "
            f"[Pay Equity Act, S.C. 2018 — s.61(2)]"
        )
    elif budget_input == 'full':
        budget       = total_remediation
        budget_label = (
            f"Full remediation — "
            f"close all gaps (${total_remediation:,.0f})"
        )
    else:
        try:
            budget       = float(
                str(budget_input).replace(',', '')
            )
            budget_label = f"Custom budget — ${budget:,.0f}"
        except ValueError:
            budget       = legal_min
            budget_label = (
                f"Legal minimum — ${legal_min:,.0f}"
            )

    def run_allocation(alloc_budget):
        selected    = []
        spent       = 0.0
        budget_left = alloc_budget

        for _, row in underpaid.iterrows():
            if budget_left >= row['adjustment']:
                selected.append(row['employee_id'])
                spent       += row['adjustment']
                budget_left -= row['adjustment']

        selected_df   = underpaid[
            underpaid['employee_id'].isin(selected)
        ]
        risk_closed   = float(selected_df['risk_score'].sum())
        gap_closed    = float(selected_df['adjustment'].sum())

        return {
            'count':           len(selected),
            'spent':           round(spent, 2),
            'selected':        selected,
            'risk_closed_pct': round(
                risk_closed / total_risk * 100, 1
            ) if total_risk > 0 else 0.0,
            'gap_closed_pct':  round(
                gap_closed / total_remediation * 100, 1
            ) if total_remediation > 0 else 0.0,
        }

    chosen = run_allocation(budget)

    # Three scenario comparison
    scenario_defs = [
        (legal_min,
         f"Legal minimum — ${legal_min:,.0f} (1% payroll)"),
        (legal_min * 1.5,
         f"Recommended   — ${legal_min*1.5:,.0f} (1.5% payroll)"),
        (total_remediation,
         f"Full closure  — ${total_remediation:,.0f}"),
    ]

    if budget_input not in ['legal_min', 'full']:
        scenario_defs.insert(
            1, (budget, f"Your budget    — ${budget:,.0f}")
        )

    scenarios = []
    for s_budget, s_label in scenario_defs:
        result = run_allocation(s_budget)
        scenarios.append({
            'label':           s_label,
            'budget':          round(s_budget, 2),
            'count':           result['count'],
            'spent':           result['spent'],
            'risk_closed_pct': result['risk_closed_pct'],
            'gap_closed_pct':  result['gap_closed_pct'],
        })

    return {
        'budget_input':    budget_input,
        'budget_used':     round(budget, 2),
        'budget_label':    budget_label,
        'selected_count':  chosen['count'],
        'deferred_count':  len(underpaid) - chosen['count'],
        'risk_closed_pct': chosen['risk_closed_pct'],
        'gap_closed_pct':  chosen['gap_closed_pct'],
        'spent':           chosen['spent'],
        'selected_ids':    chosen['selected'],
        'scenarios':       scenarios,
    }


# ══════════════════════════════════════════════════════════════════════════
# STEP 8 — COMPILE RESULTS FOR AGENT
# ══════════════════════════════════════════════════════════════════════════

def compile_results(
    df, bands, gaps, remediation,
    optimisation, grouping
):
    """
    Compiles all analysis results into one structured dict.
    This is what the agent reads to answer questions.
    """

    og          = gaps['overall']
    rem         = remediation
    sig_grades  = sum(
        1 for r in gaps['by_grade'] if r['significant']
    )
    total_grades = len(gaps['by_grade'])

    # Business case
    primary      = grouping['primary']
    senior_groups = (
        ['Grade 6', 'Grade 7']
        if primary == 'job_grade'
        else []
    )
    senior_count = len(rem['underpaid'][
        rem['underpaid'][primary].isin(senior_groups)
    ]) if len(rem['underpaid']) > 0 and senior_groups else 0

    retained         = int(senior_count * 0.35)
    retention_value  = retained * CONFIG['replacement_cost']
    legal_risk       = rem['total_remediation'] * 3
    net_benefit      = (
        retention_value + legal_risk - rem['total_remediation']
    )

    # Whizlink
    action = (
        f"{rem['total_employees']} employees require "
        f"salary adjustments totalling "
        f"${rem['total_remediation']:,.0f}."
        if rem['total_employees'] > 0
        else "No employees currently require adjustment."
    )

    grade_note = (
        f"It exists across {sig_grades} of "
        f"{total_grades} job grades."
        if total_grades > 0
        else "It exists across all job classes."
    )

    whizlink = {
        'what': (
            f"Female employees earn "
            f"${og['gap_dollars']:,.0f} less than male "
            f"employees on average — a gap of {og['gap_pct']}%."
        ),
        'why': (
            f"This gap is "
            f"{'statistically significant' if og['significant'] else 'not statistically significant'} "
            f"(p={og['p_value']}). "
            f"{grade_note} "
            f"Under the Pay Equity Act, S.C. 2018, "
            f"this requires a written remediation plan."
        ),
        'so_what': (
            f"{action} Every month without a plan "
            f"increases legal and reputational exposure."
        ),
        'how': (
            f"Phase in adjustments over "
            f"{rem['phase_years']} year(s) spending at least "
            f"${rem['legal_min_budget']:,.0f} per year "
            f"(1% of payroll) as required by the "
            f"Pay Equity Act, S.C. 2018 — Section 61(2)."
        ),
    }

    return {
        'metadata': {
            'run_date':    date.today().strftime('%Y-%m-%d'),
            'run_time':    datetime.now().strftime('%H:%M'),
            'data_file':   CONFIG['data_file'],
            'grouping':    grouping,
            'declaration': (
                'Analysis based on synthetic data. '
                'CAN North Financial is fictional. '
                'No real employees. No real salaries. '
                'No data saved or cached.'
            ),
        },
        'workforce': {
            'total':      int(len(df)),
            'male':       int((df['gender'] == 'Male').sum()),
            'female':     int((df['gender'] == 'Female').sum()),
            'avg_salary': round(float(df['salary'].mean()), 2),
            'salary_min': round(float(df['salary'].min()), 2),
            'salary_max': round(float(df['salary'].max()), 2),
        },
        'bands':       bands,
        'gaps':        gaps,
        'remediation': {
            'total_employees':   rem['total_employees'],
            'total_remediation': rem['total_remediation'],
            'total_payroll':     rem['total_payroll'],
            'legal_min_budget':  rem['legal_min_budget'],
            'can_phase':         rem['can_phase'],
            'phase_years':       rem['phase_years'],
            'yearly_plans':      rem['yearly_plans'],
        },
        'optimisation': optimisation,
        'business_case': {
            'senior_underpaid':   senior_count,
            'retained':           retained,
            'retention_value':    retention_value,
            'legal_risk_avoided': legal_risk,
            'net_benefit':        net_benefit,
            'replacement_cost':   CONFIG['replacement_cost'],
        },
        'legislation': CONFIG['legislation'],
        'whizlink':    whizlink,
    }


# ══════════════════════════════════════════════════════════════════════════
# STEP 9 — SAVE OUTPUTS
# ══════════════════════════════════════════════════════════════════════════

def save_outputs(results, remediation):
    """
    Saves all outputs to the outputs/ folder.

    pay_equity_results.json   agent reads this
    pay_equity_report.txt     human readable
    remediation_list.csv      masked IDs only
    """

    os.makedirs(CONFIG['outputs_dir'], exist_ok=True)

    with open(
        CONFIG['results_file'], 'w', encoding='utf-8'
    ) as f:
        json.dump(results, f, indent=2, default=str)

    if len(remediation['underpaid']) > 0:
        save_cols = [
            c for c in [
                'employee_id', 'job_grade', 'job_class',
                'role_level', 'division', 'location',
                'compa_ratio', 'equity_flag',
                'adjustment', 'risk_score',
            ]
            if c in remediation['underpaid'].columns
        ]
        remediation['underpaid'][save_cols].to_csv(
            CONFIG['remediation_file'], index=False
        )


# ══════════════════════════════════════════════════════════════════════════
# PRINT REPORT
# ══════════════════════════════════════════════════════════════════════════

def print_report(results):
    """
    Prints the pay equity report.
    Whizlink format — What / Why / So What / How.
    Plain English. CHRO reads in 30 seconds.
    Saved to outputs/pay_equity_report.txt.
    """

    lines = []

    def log(line=''):
        print(line)
        lines.append(line)

    today    = datetime.now().strftime('%Y-%m-%d %H:%M')
    w        = results['whizlink']
    gaps     = results['gaps']
    rem      = results['remediation']
    opt      = results['optimisation']
    biz      = results['business_case']
    grouping = results['metadata']['grouping']

    log(f"{'='*65}")
    log(f"PAY EQUITY REPORT")
    log(f"CAN North Financial — Fiscal Year 2025")
    log(f"Run date: {today}")
    log(f"{'='*65}")
    log()
    log(f"  ⚠️  DECLARATION:")
    log(f"  {results['metadata']['declaration']}")

    # 30-second summary
    log()
    log(f"{'─'*65}")
    log(f"  THE FINDING — READ THIS FIRST")
    log(f"{'─'*65}")
    log()
    log(f"  WHAT:     {w['what']}")
    log()
    log(f"  WHY:      {w['why']}")
    log()
    log(f"  SO WHAT:  {w['so_what']}")
    log()
    log(f"  HOW:      {w['how']}")

    # Overall gap
    og = gaps['overall']
    log()
    log(f"{'─'*65}")
    log(f"  OVERALL PAY GAP")
    log(f"{'─'*65}")
    log(f"  Male average salary:    ${og['male_avg']:>10,.0f}")
    log(f"  Female average salary:  ${og['female_avg']:>10,.0f}")
    log(f"  Gap:                    ${og['gap_dollars']:>10,.0f}"
        f"  ({og['gap_pct']}%)")
    log(f"  Statistically real:     "
        f"{'Yes' if og['significant'] else 'No'}"
        f"  (p={og['p_value']})")

    # Gap by job grade
    if gaps['by_grade']:
        log()
        log(f"{'─'*65}")
        log(f"  GAP BY JOB GRADE")
        log(f"  (Primary comparison — most defensible under the Act)")
        log(f"{'─'*65}")
        log(f"  {'Grade':<12} {'Men':>10} {'Women':>10} "
            f"{'Gap':>10} {'%':>7} {'Real?':>8}")
        log(f"  {'─'*12} {'─'*10} {'─'*10} "
            f"{'─'*10} {'─'*7} {'─'*8}")
        for r in gaps['by_grade']:
            sig = '✅ Yes' if r['significant'] else '— No'
            log(f"  {r['group']:<12} "
                f"${r['male_avg']:>9,.0f} "
                f"${r['female_avg']:>9,.0f} "
                f"${r['gap_dollars']:>9,.0f} "
                f"{r['gap_pct']:>6.1f}% "
                f"{sig:>8}")

    # Gap by job class
    if gaps['by_class']:
        log()
        log(f"{'─'*65}")
        log(f"  GAP BY JOB CLASS")
        log(f"{'─'*65}")
        log(f"  {'Job Class':<30} {'Men':>10} {'Women':>10} "
            f"{'Gap':>10} {'Real?':>8}")
        log(f"  {'─'*30} {'─'*10} {'─'*10} "
            f"{'─'*10} {'─'*8}")
        for r in gaps['by_class']:
            sig = '✅ Yes' if r['significant'] else '— No'
            log(f"  {r['group']:<30} "
                f"${r['male_avg']:>9,.0f} "
                f"${r['female_avg']:>9,.0f} "
                f"${r['gap_dollars']:>9,.0f} "
                f"{sig:>8}")

    # Gap by division
    log()
    log(f"{'─'*65}")
    log(f"  GAP BY DIVISION")
    log(f"{'─'*65}")
    log(f"  {'Division':<25} {'Men':>10} {'Women':>10} "
        f"{'Gap':>10} {'Real?':>8}")
    log(f"  {'─'*25} {'─'*10} {'─'*10} {'─'*10} {'─'*8}")
    for d in gaps['by_division']:
        sig = '✅ Yes' if d['significant'] else '— No'
        log(f"  {d['group']:<25} "
            f"${d['male_avg']:>9,.0f} "
            f"${d['female_avg']:>9,.0f} "
            f"${d['gap_dollars']:>9,.0f} "
            f"{sig:>8}")

    # Budget optimisation
    log()
    log(f"{'─'*65}")
    log(f"  BUDGET OPTIMISATION")
    log(f"{'─'*65}")
    log(f"  Budget:              {opt['budget_label']}")
    log(f"  Employees adjusted:  {opt['selected_count']}")
    log(f"  Employees deferred:  {opt['deferred_count']}")
    log(f"  Risk closed:         {opt['risk_closed_pct']}%")
    log(f"  Gap closed:          {opt['gap_closed_pct']}%")
    log()
    log(f"  SCENARIO COMPARISON:")
    log(f"  {'Scenario':<38} {'Employees':>10} "
        f"{'Risk Closed':>12} {'Gap Closed':>12}")
    log(f"  {'─'*38} {'─'*10} {'─'*12} {'─'*12}")
    for s in opt['scenarios']:
        log(f"  {s['label'][:37]:<38} "
            f"{s['count']:>10} "
            f"{s['risk_closed_pct']:>11.1f}% "
            f"{s['gap_closed_pct']:>11.1f}%")

    # Remediation plan
    log()
    log(f"{'─'*65}")
    log(f"  REMEDIATION PLAN")
    log(f"  Pay Equity Act, S.C. 2018 — Section 61(2)")
    log(f"{'─'*65}")
    log(f"  Employees needing adjustment: {rem['total_employees']}")
    log(f"  Total adjustment cost:        "
        f"${rem['total_remediation']:,.0f}")
    log(f"  Annual payroll:               "
        f"${rem['total_payroll']:,.0f}")
    log(f"  Legal minimum (1%):           "
        f"${rem['legal_min_budget']:,.0f}")
    log(f"  Phase in permitted:           "
        f"{'Yes — 3 years' if rem['can_phase'] else 'No'}")

    if rem.get('yearly_plans'):
        log()
        log(f"  THREE YEAR PLAN:")
        log(f"  {'Year':<8} {'Budget':>14} "
            f"{'Employees':>12} {'Spent':>14}")
        log(f"  {'─'*8} {'─'*14} {'─'*12} {'─'*14}")
        for plan in rem['yearly_plans']:
            log(f"  Year {plan['year']:<3} "
                f"${plan['budget']:>13,.0f} "
                f"{plan['count']:>12} "
                f"${plan['spent']:>13,.0f}")

    # Business case
    log()
    log(f"{'─'*65}")
    log(f"  BUSINESS CASE")
    log(f"{'─'*65}")
    log(f"  Total remediation cost:       "
        f"${rem['total_remediation']:,.0f}")
    log(f"  Retention value:              "
        f"${biz['retention_value']:,.0f}")
    log(f"  Legal risk avoided:           "
        f"${biz['legal_risk_avoided']:,.0f}")
    log(f"  Net benefit:                  "
        f"${biz['net_benefit']:,.0f}")

    # Legal references
    log()
    log(f"{'─'*65}")
    log(f"  LEGAL REFERENCES")
    log(f"  All sourced from official government websites only.")
    log(f"{'─'*65}")
    for key, leg in results['legislation'].items():
        log(f"  {leg['name']}")
        log(f"  {leg['url']}")
        log(f"  {leg['rule']}")
        log()

    log(f"{'─'*65}")
    log(f"  FILES SAVED:")
    log(f"  ✅ {CONFIG['results_file']}")
    log(f"  ✅ {CONFIG['report_file']}")
    if rem['total_employees'] > 0:
        log(f"  ✅ {CONFIG['remediation_file']}")
    log()
    log(f"  Next: agent reads these results")
    log(f"  and answers your questions in plain English.")
    log(f"{'='*65}")

    with open(
        CONFIG['report_file'], 'w', encoding='utf-8'
    ) as f:
        f.write('\n'.join(lines))


# ══════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='Pay Equity Analysis Engine'
    )
    parser.add_argument(
        '--data',
        type=str,
        default=CONFIG['data_file'],
        help='Path to employee data CSV',
    )
    parser.add_argument(
        '--budget',
        type=str,
        default='legal_min',
        help=(
            'Annual remediation budget. '
            'Options: legal_min (default), '
            'full, or a dollar amount e.g. 800000'
        ),
    )
    args = parser.parse_args()

    CONFIG['data_file'] = args.data

    print(f"\n{'='*65}")
    print(f"PAY EQUITY ANALYSIS ENGINE")
    print(f"Total Rewards Intelligence Platform — Pillar 1")
    print(
        f"Run date: "
        f"{datetime.now().strftime('%Y-%m-%d %H:%M')}"
    )
    print(f"{'='*65}")
    print(f"\n  ⚠️  No personal information used or stored.")

    # 1 — Load
    print(f"\n[1/9] Loading and validating data...")
    df = load_and_validate(CONFIG['data_file'])
    print(f"      ✅ {len(df):,} employees loaded")
    print(
        f"      ✅ Male:   "
        f"{(df['gender']=='Male').sum():,}"
    )
    print(
        f"      ✅ Female: "
        f"{(df['gender']=='Female').sum():,}"
    )

    # 2 — Detect grouping
    print(f"\n[2/9] Detecting job structure...")
    grouping = detect_grouping(df)
    print(
        f"      ✅ Primary grouping: {grouping['primary']}"
    )
    print(
        f"      ✅ Job grade available: "
        f"{grouping['has_grade']}"
    )
    print(
        f"      ✅ Job class available: "
        f"{grouping['has_class']}"
    )

    # 3 — Internal bands
    print(f"\n[3/9] Building internal salary bands...")
    bands = build_internal_bands(df, grouping)
    print(f"      ✅ {len(bands)} bands built")
    print(f"      ✅ Source: internal salary distribution")

    # 4 — Compa-ratios
    print(f"\n[4/9] Calculating compa-ratios...")
    df      = calculate_compa_ratios(df, bands, grouping)
    n_under = (
        df['compa_ratio'] < CONFIG['underpaid_threshold']
    ).sum()
    print(f"      ✅ {n_under} employees below 0.95")

    # 5 — Gap analysis
    print(f"\n[5/9] Analysing pay gaps...")
    gaps      = analyse_gaps(df, grouping)
    og        = gaps['overall']
    sig_grades = sum(
        1 for r in gaps['by_grade'] if r['significant']
    )
    print(
        f"      ✅ Overall gap: "
        f"${og['gap_dollars']:,.0f} ({og['gap_pct']}%)"
    )
    print(
        f"      ✅ Significant: "
        f"{'Yes' if og['significant'] else 'No'} "
        f"(p={og['p_value']})"
    )
    if gaps['by_grade']:
        print(
            f"      ✅ {sig_grades} of "
            f"{len(gaps['by_grade'])} job grades "
            f"show significant gaps"
        )

    # 6 — Remediation
    print(f"\n[6/9] Building remediation plan...")
    remediation = build_remediation(df, grouping)
    print(
        f"      ✅ {remediation['total_employees']} "
        f"employees require adjustment"
    )
    print(
        f"      ✅ Total cost: "
        f"${remediation['total_remediation']:,.0f}"
    )
    print(
        f"      ✅ Legal min (1%): "
        f"${remediation['legal_min_budget']:,.0f}"
    )

    # 7 — Budget optimisation
    print(f"\n[7/9] Running budget optimisation...")
    optimisation = optimise_budget(remediation, args.budget)
    print(
        f"      ✅ Budget: {optimisation['budget_label']}"
    )
    print(
        f"      ✅ Employees adjusted: "
        f"{optimisation['selected_count']}"
    )
    print(
        f"      ✅ Risk closed: "
        f"{optimisation['risk_closed_pct']}%"
    )
    print(f"\n      SCENARIO COMPARISON:")
    print(
        f"      {'Scenario':<38} {'Employees':>10} "
        f"{'Risk Closed':>12} {'Gap Closed':>12}"
    )
    print(
        f"      {'─'*38} {'─'*10} {'─'*12} {'─'*12}"
    )
    for s in optimisation['scenarios']:
        print(
            f"      {s['label'][:37]:<38} "
            f"{s['count']:>10} "
            f"{s['risk_closed_pct']:>11.1f}% "
            f"{s['gap_closed_pct']:>11.1f}%"
        )

    # 8 — Compile
    print(f"\n[8/9] Compiling results for agent...")
    results = compile_results(
        df, bands, gaps, remediation,
        optimisation, grouping,
    )
    print(f"      ✅ Results compiled")

    # 9 — Save and report
    print(f"\n[9/9] Saving outputs and generating report...")
    save_outputs(results, remediation)
    print(f"      ✅ {CONFIG['results_file']}")
    print(f"      ✅ {CONFIG['report_file']}")
    if remediation['total_employees'] > 0:
        print(f"      ✅ {CONFIG['remediation_file']}")
    print_report(results)


if __name__ == '__main__':
    main()