# ══════════════════════════════════════════════════════════════════════════
# TOTAL REWARDS INTELLIGENCE PLATFORM
# app.py — Pay Equity Intelligence
#
# PILLAR 1 OF 3: PAY EQUITY ANALYSIS
#
# OPENING SCREEN — user chooses their path:
#   Option 1: See the Demo
#             Run on CAN North synthetic data
#   Option 2: Tweak the Demo Data
#             Download synthetic data, change values,
#             re-upload and see changes
#   Option 3: Upload Your Own Data
#             Download template, fill with your data,
#             upload and run your analysis
#
# ALL PATHS LEAD TO:
#   → Full pay equity report
#   → AI Agent answers questions
#   → Results are from whatever data is loaded
#   → Approved legislation URLs only
#   → Declarations on everything
#
# PRIVACY:
#   No personal information collected or stored.
#   No data saved between sessions.
#   Session ends — everything gone.
#
# HOW TO RUN:
#   python -m streamlit run app.py
#
# ⚠️  DECLARATION:
#   Demo data is synthetic. CAN North Financial is fictional.
#   Legislation sourced from official government URLs only.
#   This does not constitute legal advice.
# ══════════════════════════════════════════════════════════════════════════

import io
import os
import json
import warnings
from datetime import date, datetime

import numpy as np
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from analyse import (
    load_and_validate,
    build_internal_bands,
    calculate_compa_ratios,
    analyse_gaps,
    build_remediation,
    optimise_budget,
    compile_results,
    detect_grouping,
    CONFIG as ANALYSE_CONFIG,
)
from agent_pay_equity import (
    fetch_legislation,
    build_context,
    build_system_prompt,
    ask_agent,
)

load_dotenv()
warnings.filterwarnings('ignore')


# ══════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title='Pay Equity Intelligence',
    page_icon='⚖️',
    layout='wide',
    initial_sidebar_state='expanded',
)


# ══════════════════════════════════════════════════════════════════════════
# STYLING
# ══════════════════════════════════════════════════════════════════════════

st.markdown("""
<style>
    .main-header {
        background: linear-gradient(
            135deg, #1F3864 0%, #2E5FA3 100%
        );
        padding: 2rem; border-radius: 10px;
        color: white; margin-bottom: 1.5rem;
    }
    .main-header h1 {
        color: white; margin: 0; font-size: 2rem;
    }
    .main-header p {
        color: #cce0f5;
        margin: 0.5rem 0 0 0; font-size: 1rem;
    }
    .wl-card {
        background: #f8fafc;
        border-left: 4px solid #2E5FA3;
        padding: 1rem 1.2rem; margin: 0.5rem 0;
        border-radius: 0 8px 8px 0;
    }
    .wl-label {
        font-weight: 700; color: #1F3864;
        font-size: 0.8rem; text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .wl-text {
        color: #2d3748;
        margin-top: 0.2rem; font-size: 0.95rem;
    }
    .privacy-notice {
        background: #e8f4f8;
        border: 1px solid #bee3f8;
        border-radius: 8px; padding: 0.8rem 1rem;
        font-size: 0.85rem; color: #2c5f7a;
    }
    .declaration {
        background: #fff8e1;
        border: 1px solid #ffe082;
        border-radius: 8px; padding: 0.8rem 1rem;
        font-size: 0.85rem; color: #5d4037;
    }
    .section-header {
        font-size: 1.05rem; font-weight: 700;
        color: #1F3864;
        border-bottom: 2px solid #C8973A;
        padding-bottom: 0.3rem;
        margin: 1.5rem 0 1rem 0;
    }
    .path-card {
        background: #f8fafc;
        border: 2px solid #e2e8f0;
        border-radius: 12px; padding: 1.5rem;
        text-align: center; cursor: pointer;
        transition: all 0.2s;
    }
    .path-card:hover {
        border-color: #2E5FA3;
        background: #f0f7ff;
    }
    .path-icon {
        font-size: 2.5rem; margin-bottom: 0.5rem;
    }
    .path-title {
        font-size: 1rem; font-weight: 700;
        color: #1F3864; margin-bottom: 0.3rem;
    }
    .path-desc {
        font-size: 0.85rem; color: #64748b;
    }
    .agent-msg {
        background: #f0f7ff;
        border-left: 3px solid #2E5FA3;
        padding: 1rem; border-radius: 0 8px 8px 0;
        margin: 0.5rem 0;
    }
    .user-msg {
        background: #f7f7f7;
        border-left: 3px solid #718096;
        padding: 1rem; border-radius: 0 8px 8px 0;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════

def run_full_analysis(df, budget_input='legal_min'):
    """
    Runs complete pay equity analysis on a dataframe.
    Returns compiled results dict.
    """
    grouping     = detect_grouping(df)
    bands        = build_internal_bands(df, grouping)
    df           = calculate_compa_ratios(df, bands, grouping)
    gaps         = analyse_gaps(df, grouping)
    remediation  = build_remediation(df, grouping)
    optimisation = optimise_budget(remediation, budget_input)
    results      = compile_results(
        df, bands, gaps, remediation,
        optimisation, grouping,
    )
    return results


def generate_template_csv():
    """
    Generates the data template CSV for users to download.
    Shows required columns with example rows.
    No personal information in examples.
    """
    template = pd.DataFrame([
        {
            'employee_id':  'EMP001',
            'gender':       'Female',
            'job_class':    'Financial Analyst',
            'job_grade':    'Grade 4',
            'division':     'Finance',
            'location':     'Toronto',
            'tenure_years': 3,
            'performance':  72,
            'salary':       68000,
        },
        {
            'employee_id':  'EMP002',
            'gender':       'Male',
            'job_class':    'Senior Financial Analyst',
            'job_grade':    'Grade 5',
            'division':     'Finance',
            'location':     'Calgary',
            'tenure_years': 7,
            'performance':  81,
            'salary':       96000,
        },
        {
            'employee_id':  'EMP003',
            'gender':       'Female',
            'job_class':    'Finance Manager',
            'job_grade':    'Grade 6',
            'division':     'Operations',
            'location':     'Vancouver',
            'tenure_years': 11,
            'performance':  88,
            'salary':       112000,
        },
        {
            'employee_id':  'EMP004',
            'gender':       'Male',
            'job_class':    'Finance Manager',
            'job_grade':    'Grade 6',
            'division':     'Operations',
            'location':     'Vancouver',
            'tenure_years': 9,
            'performance':  84,
            'salary':       121000,
        },
    ])
    return template.to_csv(index=False).encode('utf-8')


def prepare_agent(results):
    """
    Initialises the agent with current results.
    Fetches legislation from approved URLs.
    Stores system prompt in session state.
    """
    with st.spinner(
        'Preparing agent — fetching legislation '
        'from approved government sources...'
    ):
        try:
            legislation   = fetch_legislation()
            context       = build_context(
                results, legislation
            )
            system_prompt = build_system_prompt(context)
            st.session_state['system_prompt'] = system_prompt
            st.session_state['agent_ready']   = True
            st.session_state['conversation']  = []
            st.session_state['agent_history'] = []
        except Exception as e:
            st.error(f"Agent setup error: {str(e)}")


def display_whizlink(whizlink):
    """Displays the Whizlink executive summary."""
    for label, key in [
        ('WHAT',    'what'),
        ('WHY',     'why'),
        ('SO WHAT', 'so_what'),
        ('HOW',     'how'),
    ]:
        st.markdown(f"""
        <div class="wl-card">
            <div class="wl-label">{label}</div>
            <div class="wl-text">{whizlink[key]}</div>
        </div>
        """, unsafe_allow_html=True)


def display_full_report(results):
    """
    Displays the complete pay equity report.
    Used for all three paths — demo, tweaked, own data.
    """
    wl  = results['whizlink']
    og  = results['gaps']['overall']
    rem = results['remediation']
    opt = results['optimisation']
    biz = results['business_case']

    # Declaration
    st.markdown(f"""
    <div class="declaration">
    ⚠️  {results['metadata']['declaration']}
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Executive summary
    st.markdown(
        '<div class="section-header">'
        '📋 Executive Summary — Read This First'
        '</div>',
        unsafe_allow_html=True,
    )
    display_whizlink(wl)

    st.markdown("---")

    # Key metrics
    st.markdown(
        '<div class="section-header">'
        '📊 Key Findings'
        '</div>',
        unsafe_allow_html=True,
    )
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(
            'Overall Pay Gap',
            f"${og['gap_dollars']:,.0f}",
            delta=f"{og['gap_pct']}% gap",
            delta_color='inverse',
        )
    with col2:
        st.metric(
            'Statistically Real',
            'Yes' if og['significant'] else 'No',
            delta=f"p = {og['p_value']}",
            delta_color=(
                'inverse'
                if og['significant'] else 'normal'
            ),
        )
    with col3:
        st.metric(
            'Employees Affected',
            f"{rem['total_employees']:,}",
            delta='Require adjustment',
            delta_color='inverse',
        )
    with col4:
        st.metric(
            'Total Remediation',
            f"${rem['total_remediation']:,.0f}",
            delta=f"Over {rem['phase_years']} year(s)",
        )

    st.markdown("---")

    # Gap by job grade
    if results['gaps'].get('by_grade'):
        st.markdown(
            '<div class="section-header">'
            '📊 Pay Gap by Job Grade'
            '<span style="font-size:0.75rem;'
            'font-weight:400;color:#64748b;'
            'margin-left:8px;">'
            'Primary — most defensible under the Act'
            '</span>'
            '</div>',
            unsafe_allow_html=True,
        )
        grade_rows = []
        for r in results['gaps']['by_grade']:
            grade_rows.append({
                'Job Grade':   r['group'],
                'Men (avg)':   f"${r['male_avg']:,.0f}",
                'Women (avg)': f"${r['female_avg']:,.0f}",
                'Gap ($)':     f"${r['gap_dollars']:,.0f}",
                'Gap (%)':     f"{r['gap_pct']}%",
                'Significant': (
                    '✅ Yes'
                    if r['significant'] else '— No'
                ),
            })
        st.dataframe(
            pd.DataFrame(grade_rows),
            use_container_width=True,
            hide_index=True,
        )

    # Gap by job class
    if results['gaps'].get('by_class'):
        st.markdown(
            '<div class="section-header">'
            '👥 Pay Gap by Job Class'
            '</div>',
            unsafe_allow_html=True,
        )
        class_rows = []
        for r in results['gaps']['by_class']:
            class_rows.append({
                'Job Class':   r['group'],
                'Men (avg)':   f"${r['male_avg']:,.0f}",
                'Women (avg)': f"${r['female_avg']:,.0f}",
                'Gap ($)':     f"${r['gap_dollars']:,.0f}",
                'Significant': (
                    '✅ Yes'
                    if r['significant'] else '— No'
                ),
            })
        st.dataframe(
            pd.DataFrame(class_rows),
            use_container_width=True,
            hide_index=True,
        )

    st.markdown("---")

    # Gap by division
    st.markdown(
        '<div class="section-header">'
        '🏢 Pay Gap by Division'
        '</div>',
        unsafe_allow_html=True,
    )
    div_rows = []
    for d in results['gaps'].get('by_division', []):
        div_rows.append({
            'Division':    d['group'],
            'Men (avg)':   f"${d['male_avg']:,.0f}",
            'Women (avg)': f"${d['female_avg']:,.0f}",
            'Gap ($)':     f"${d['gap_dollars']:,.0f}",
            'Significant': (
                '✅ Yes'
                if d['significant'] else '— No'
            ),
        })
    if div_rows:
        st.dataframe(
            pd.DataFrame(div_rows),
            use_container_width=True,
            hide_index=True,
        )

    st.markdown("---")

    # Budget optimisation
    st.markdown(
        '<div class="section-header">'
        '💰 Budget Optimisation'
        '</div>',
        unsafe_allow_html=True,
    )
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(
            'Budget Used',
            f"${opt['budget_used']:,.0f}",
        )
    with col2:
        st.metric(
            'Risk Closed',
            f"{opt['risk_closed_pct']}%",
        )
    with col3:
        st.metric(
            'Gap Closed',
            f"{opt['gap_closed_pct']}%",
        )

    st.markdown("**Scenario Comparison:**")
    scenario_rows = []
    for s in opt.get('scenarios', []):
        scenario_rows.append({
            'Scenario':    s['label'],
            'Budget':      f"${s['budget']:,.0f}",
            'Employees':   s['count'],
            'Risk Closed': f"{s['risk_closed_pct']}%",
            'Gap Closed':  f"{s['gap_closed_pct']}%",
        })
    if scenario_rows:
        st.dataframe(
            pd.DataFrame(scenario_rows),
            use_container_width=True,
            hide_index=True,
        )

    st.markdown("---")

    # Remediation plan
    st.markdown(
        '<div class="section-header">'
        '📅 Remediation Plan'
        '</div>',
        unsafe_allow_html=True,
    )
    col1, col2 = st.columns(2)
    with col1:
        st.metric(
            'Annual Payroll',
            f"${rem['total_payroll']:,.0f}",
        )
        st.metric(
            'Legal Minimum (1%)',
            f"${rem['legal_min_budget']:,.0f}",
            delta='Minimum annual spend required',
        )
    with col2:
        st.metric(
            'Total Remediation Cost',
            f"${rem['total_remediation']:,.0f}",
        )
        st.metric(
            'Phase In',
            'Yes — 3 years'
            if rem['can_phase'] else 'No',
        )

    if rem.get('yearly_plans'):
        st.markdown("**Three Year Plan:**")
        plan_rows = []
        for p in rem['yearly_plans']:
            plan_rows.append({
                'Year':      f"Year {p['year']}",
                'Budget':    f"${p['budget']:,.0f}",
                'Employees': p['count'],
                'Spent':     f"${p['spent']:,.0f}",
            })
        st.dataframe(
            pd.DataFrame(plan_rows),
            use_container_width=True,
            hide_index=True,
        )
    st.caption(
        "Pay Equity Act, S.C. 2018, c. 27, s. 416 — "
        "Section 61(2). "
        "laws-lois.justice.gc.ca/eng/acts/P-4.2/ — "
        f"Retrieved: {date.today().strftime('%B %d, %Y')}"
    )

    st.markdown("---")

    # Business case
    st.markdown(
        '<div class="section-header">'
        '💼 Business Case'
        '</div>',
        unsafe_allow_html=True,
    )
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(
            'Remediation Cost',
            f"${rem['total_remediation']:,.0f}",
            delta='Investment',
        )
    with col2:
        st.metric(
            'Retention Value',
            f"${biz['retention_value']:,.0f}",
            delta='Saved from attrition',
        )
    with col3:
        st.metric(
            'Net Benefit',
            f"${biz['net_benefit']:,.0f}",
            delta='Return on investment',
        )

    st.markdown("---")

    # Legal references
    st.markdown(
        '<div class="section-header">'
        '⚖️ Legal References'
        '</div>',
        unsafe_allow_html=True,
    )
    st.caption(
        "All legislation sourced from official government "
        "websites only. This does not constitute legal advice. "
        "Consult qualified legal counsel for "
        "jurisdiction-specific guidance."
    )
    col1, col2 = st.columns(2)
    with col1:
        st.info(
            "**Pay Equity Act, S.C. 2018, c. 27, s. 416**\n\n"
            "laws-lois.justice.gc.ca/eng/acts/P-4.2/\n\n"
            f"Retrieved: {date.today().strftime('%B %d, %Y')}"
        )
    with col2:
        st.info(
            "**Pay Equity Act, R.S.O. 1990, c. P.7**\n\n"
            "ontario.ca/laws/statute/90p07\n\n"
            f"Retrieved: {date.today().strftime('%B %d, %Y')}"
        )


def render_agent_chat(key_prefix=''):
    """
    Renders the agent chat interface.
    Works with whatever results are in session state.
    key_prefix avoids duplicate widget keys across tabs.
    """

    if not st.session_state.get('agent_ready'):
        st.warning(
            "Agent is not ready. "
            "Run the analysis first."
        )
        return

    # Suggested questions
    st.markdown("**Suggested questions:**")
    col1, col2, col3 = st.columns(3)

    suggestions = [
        ("Do we have a pay gap?",
         f"suggest_1_{key_prefix}"),
        ("Which job grade has the largest gap?",
         f"suggest_2_{key_prefix}"),
        ("What does the law require?",
         f"suggest_3_{key_prefix}"),
        ("How much will remediation cost?",
         f"suggest_4_{key_prefix}"),
        ("What is the business case?",
         f"suggest_5_{key_prefix}"),
        ("Where do we start?",
         f"suggest_6_{key_prefix}"),
    ]

    for i, (label, key) in enumerate(suggestions):
        col = [col1, col2, col3][i % 3]
        with col:
            if st.button(
                label, key=key,
                use_container_width=True,
            ):
                st.session_state[
                    'pending_question'
                ] = label

    # Chat input
    question = st.chat_input(
        "Ask a question about your pay equity analysis...",
        key=f"chat_input_{key_prefix}",
    )

    if 'pending_question' in st.session_state:
        question = st.session_state.pop('pending_question')

    if question:
        if 'conversation' not in st.session_state:
            st.session_state['conversation']  = []
        if 'agent_history' not in st.session_state:
            st.session_state['agent_history'] = []

        st.session_state['conversation'].append({
            'role': 'user', 'content': question,
        })

        with st.spinner('Thinking...'):
            try:
                response, updated = ask_agent(
                    question,
                    st.session_state['system_prompt'],
                    st.session_state['agent_history'],
                )
                st.session_state['agent_history'] = updated
                st.session_state['conversation'].append({
                    'role':    'assistant',
                    'content': response,
                })
            except Exception as e:
                st.error(f"Agent error: {str(e)}")

    # Display conversation
    for msg in st.session_state.get('conversation', []):
        if msg['role'] == 'user':
            st.markdown(f"""
            <div class="user-msg">
            <strong>You:</strong><br>{msg['content']}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="agent-msg">
            <strong>Agent:</strong><br>{msg['content']}
            </div>
            """, unsafe_allow_html=True)

    if st.session_state.get('conversation'):
        if st.button(
            "🗑 Clear conversation",
            key=f"clear_{key_prefix}",
        ):
            st.session_state['conversation']  = []
            st.session_state['agent_history'] = []
            st.rerun()

    st.caption(
        "⚠️  Agent answers only from your analysis results "
        "and approved government legislation. Not legal advice."
    )


# ══════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════

def render_sidebar():
    with st.sidebar:
        st.markdown("## ⚖️ Pay Equity Intelligence")
        st.markdown(
            "*Total Rewards Intelligence Platform*\n\n"
            "*Pillar 1 of 3 — Pay Equity*"
        )
        st.markdown("---")

        st.markdown("### 🔒 Privacy")
        st.markdown("""
        <div class="privacy-notice">
        No personal data collected, stored,
        or transmitted.<br><br>
        Your data lives only in this session.
        Close the app — it is gone.<br><br>
        <strong>Use masked codes only.
        No names. No SIN. No DOB.</strong>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("### ⚖️ Legislation")
        st.markdown(
            "**Federal:**\n\n"
            "Pay Equity Act, S.C. 2018\n\n"
            "**Ontario:**\n\n"
            "Pay Equity Act, R.S.O. 1990\n\n"
            f"*Retrieved: "
            f"{date.today().strftime('%B %d, %Y')}*\n\n"
            "*Official government URLs only*"
        )
        st.markdown("---")
        st.markdown("### ℹ️ About")
        st.markdown(
            "Built by **Jaini Desai**\n\n"
            "People Analytics Portfolio\n\n"
            "Total Rewards Intelligence\n\n"
            "*Synthetic demo data.*\n\n"
            "*Not legal advice.*"
        )


# ══════════════════════════════════════════════════════════════════════════
# OPENING SCREEN — CHOOSE YOUR PATH
# ══════════════════════════════════════════════════════════════════════════

def render_opening_screen():
    """
    First thing users see.
    They choose their path before anything runs.
    Three clear options.
    """

    st.markdown("""
    <div class="main-header">
        <h1>⚖️ Pay Equity Intelligence</h1>
        <p>
        Identify gender pay gaps. Build your remediation plan.
        Meet your legal obligations under the Pay Equity Act.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="declaration">
    ⚠️ <strong>Demo data is synthetic.</strong>
    CAN North Financial is a fictional organization.
    No real employees. No real salaries.
    Legislation references are real — sourced from
    official government websites only.
    This does not constitute legal advice.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    st.markdown(
        "### What would you like to do?"
    )
    st.markdown(
        "Choose your path below. "
        "You can switch between them at any time."
    )

    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="path-card">
            <div class="path-icon">🏦</div>
            <div class="path-title">See the Demo</div>
            <div class="path-desc">
            Run the full pay equity analysis on
            CAN North Financial — 1,400 synthetic
            employees. See exactly what this tool does.
            No upload needed.
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button(
            '▶ Run Demo',
            key='btn_demo',
            type='primary',
            use_container_width=True,
        ):
            st.session_state['path'] = 'demo'
            st.rerun()

    with col2:
        st.markdown("""
        <div class="path-card">
            <div class="path-icon">🔧</div>
            <div class="path-title">Tweak the Demo Data</div>
            <div class="path-desc">
            Download our synthetic data. Change salary
            or tenure values in Excel. Re-upload and
            see how the analysis changes in real time.
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button(
            '⬇ Download Demo Data',
            key='btn_tweak',
            use_container_width=True,
        ):
            st.session_state['path'] = 'tweak'
            st.rerun()

    with col3:
        st.markdown("""
        <div class="path-card">
            <div class="path-icon">📊</div>
            <div class="path-title">Upload Your Own Data</div>
            <div class="path-desc">
            Download our data template. Fill it with
            your employee data. Upload and get your
            own pay equity report instantly.
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button(
            '⬇ Download Template',
            key='btn_own',
            use_container_width=True,
        ):
            st.session_state['path'] = 'own'
            st.rerun()

    st.markdown("---")

    # What is pay equity
    st.markdown("### What is pay equity?")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        Pay equity means **equal pay for work of equal value**
        — regardless of gender.

        It is not about paying everyone the same.
        It is about ensuring that when you control for
        job grade, tenure, and performance — women are
        not systematically paid less than men doing
        equivalent work.
        """)
    with col2:
        st.markdown("""
        In Canada, the **Pay Equity Act (S.C. 2018)**
        makes this a legal obligation for federally
        regulated employers with 10 or more employees.

        Non-compliance can result in Commissioner orders,
        financial penalties, and public disclosure.

        *Source: laws-lois.justice.gc.ca/eng/acts/P-4.2/*
        """)


# ══════════════════════════════════════════════════════════════════════════
# PATH 1 — DEMO
# ══════════════════════════════════════════════════════════════════════════

def render_demo_path():
    """
    Runs analysis on CAN North synthetic data.
    Shows full report and agent.
    """

    st.markdown("""
    <div class="main-header">
        <h1>🏦 CAN North Financial — Demo</h1>
        <p>
        Pay equity analysis on 1,400 synthetic employees.
        Fictional organization. Real methodology.
        </p>
    </div>
    """, unsafe_allow_html=True)

    if st.button('← Back to start', key='back_demo'):
        for key in [
            'path', 'results', 'agent_ready',
            'conversation', 'agent_history',
        ]:
            st.session_state.pop(key, None)
        st.rerun()

    st.markdown("---")

    demo_file = 'data/can_north_pay_equity_historical.csv'

    if not os.path.exists(demo_file):
        st.error(
            "Demo data not found. "
            "Run: python generate_data.py"
        )
        return

    # Run analysis if not already done
    if 'results' not in st.session_state or \
       st.session_state.get('results_source') != 'demo':

        with st.spinner(
            'Running pay equity analysis on '
            'CAN North Financial...'
        ):
            try:
                df      = pd.read_csv(demo_file)
                results = run_full_analysis(df)
                st.session_state['results']        = results
                st.session_state['results_source'] = 'demo'
                st.session_state['data_label']     = (
                    'CAN North Financial '
                    '(Demo — 1,400 synthetic employees)'
                )
                st.session_state['agent_ready']    = False
            except Exception as e:
                st.error(f"Analysis error: {str(e)}")
                return

    results = st.session_state['results']
    og      = results['gaps']['overall']
    rem     = results['remediation']

    st.success(
        f"✅ Analysis complete — "
        f"{st.session_state['data_label']}"
    )

    if og['significant']:
        st.error(
            f"⚠️  Pay gap detected. "
            f"Female employees earn "
            f"**${og['gap_dollars']:,.0f}** less "
            f"({og['gap_pct']}%). "
            f"Significant (p={og['p_value']}). "
            f"**{rem['total_employees']} employees** "
            f"require adjustment."
        )
    else:
        st.success(
            "✅ No statistically significant "
            "pay gap detected."
        )

    # Tabs — report and agent
    tab1, tab2 = st.tabs([
        '📋 Pay Equity Report',
        '🤖 Ask the Agent',
    ])

    with tab1:
        display_full_report(results)

        st.markdown("---")
        st.markdown(
            '<div class="section-header">'
            '📥 Downloads'
            '</div>',
            unsafe_allow_html=True,
        )
        col1, col2, col3 = st.columns(3)
        with col1:
            if os.path.exists(demo_file):
                st.download_button(
                    label='⬇ Demo Data (CSV)',
                    data=open(demo_file, 'rb').read(),
                    file_name='can_north_demo_data.csv',
                    mime='text/csv',
                    help=(
                        'Download, tweak in Excel, '
                        'go back and choose '
                        '"Tweak the Demo Data"'
                    ),
                )
        with col2:
            if os.path.exists(
                'outputs/pay_equity_report.txt'
            ):
                st.download_button(
                    label='⬇ Report (TXT)',
                    data=open(
                        'outputs/pay_equity_report.txt',
                        'rb',
                    ).read(),
                    file_name='pay_equity_report.txt',
                    mime='text/plain',
                )
        with col3:
            if os.path.exists(
                'outputs/remediation_list.csv'
            ):
                st.download_button(
                    label='⬇ Remediation List (CSV)',
                    data=open(
                        'outputs/remediation_list.csv',
                        'rb',
                    ).read(),
                    file_name='remediation_list.csv',
                    mime='text/csv',
                )

    with tab2:
        st.markdown("""
        <div class="declaration">
        ⚠️ The agent answers <strong>only</strong> from
        the CAN North demo analysis results and approved
        government legislation sources.
        No other sources. Not legal advice.
        </div>
        """, unsafe_allow_html=True)
        st.markdown("---")

        if not st.session_state.get('agent_ready'):
            prepare_agent(results)

        if st.session_state.get('agent_ready'):
            render_agent_chat(key_prefix='demo')


# ══════════════════════════════════════════════════════════════════════════
# PATH 2 — TWEAK DEMO DATA
# ══════════════════════════════════════════════════════════════════════════

def render_tweak_path():
    """
    User downloads synthetic data, tweaks it,
    re-uploads and sees how analysis changes.
    """

    st.markdown("""
    <div class="main-header">
        <h1>🔧 Tweak the Demo Data</h1>
        <p>
        Download our synthetic data. Change values in Excel.
        Re-upload and see how the pay equity analysis changes.
        </p>
    </div>
    """, unsafe_allow_html=True)

    if st.button('← Back to start', key='back_tweak'):
        for key in [
            'path', 'results', 'agent_ready',
            'conversation', 'agent_history',
        ]:
            st.session_state.pop(key, None)
        st.rerun()

    st.markdown("---")

    demo_file = 'data/can_north_pay_equity_historical.csv'

    # Step 1 — Download
    st.markdown(
        '<div class="section-header">'
        'Step 1 — Download the Demo Data'
        '</div>',
        unsafe_allow_html=True,
    )
    st.markdown("""
    Download the CAN North synthetic dataset.
    Open it in Excel. Try changing some values:

    - Lower a female employee's salary in Grade 6 or 7
    - Increase tenure for male employees
    - Change division assignments

    Save as CSV and upload in Step 2 to see
    how the analysis results change.
    """)

    if os.path.exists(demo_file):
        st.download_button(
            label='⬇ Download CAN North Demo Data (CSV)',
            data=open(demo_file, 'rb').read(),
            file_name='can_north_demo_data.csv',
            mime='text/csv',
        )
    else:
        st.error(
            "Demo data not found. "
            "Run: python generate_data.py"
        )
        return

    st.markdown("---")

    # Step 2 — Upload tweaked file
    st.markdown(
        '<div class="section-header">'
        'Step 2 — Upload Your Tweaked File'
        '</div>',
        unsafe_allow_html=True,
    )

    uploaded = st.file_uploader(
        'Upload your tweaked CSV file',
        type='csv',
        key='tweak_upload',
    )

    if uploaded is None:
        st.info(
            "Download the data above, tweak values "
            "in Excel, save as CSV, and upload here."
        )
        return

    try:
        df = pd.read_csv(uploaded)
        st.success(f"✅ {len(df):,} employees loaded")
    except Exception as e:
        st.error(f"Could not read file: {str(e)}")
        return

    st.markdown("---")

    # Step 3 — Budget choice
    st.markdown(
        '<div class="section-header">'
        'Step 3 — Choose Your Budget'
        '</div>',
        unsafe_allow_html=True,
    )
    budget_option = st.radio(
        'Annual remediation budget:',
        options=[
            'Legal minimum — 1% of payroll',
            'Custom amount',
            'Full remediation',
        ],
        index=0,
        key='tweak_budget',
    )
    budget_input = 'legal_min'
    if 'Custom' in budget_option:
        custom = st.number_input(
            'Enter budget ($):',
            min_value=0, value=500000,
            step=50000, format='%d',
            key='tweak_custom',
        )
        budget_input = str(custom)
    elif 'Full' in budget_option:
        budget_input = 'full'

    st.markdown("---")

    # Step 4 — Run
    st.markdown(
        '<div class="section-header">'
        'Step 4 — Run Analysis'
        '</div>',
        unsafe_allow_html=True,
    )

    if not st.button(
        '▶ Run Pay Equity Analysis',
        key='run_tweak',
        type='primary',
        use_container_width=True,
    ):
        st.info("Click above to run the analysis.")
        return

    with st.spinner('Running analysis...'):
        try:
            results = run_full_analysis(df, budget_input)
            st.session_state['results']        = results
            st.session_state['results_source'] = 'tweak'
            st.session_state['data_label']     = (
                f'Tweaked demo data — {uploaded.name}'
            )
            st.session_state['agent_ready']    = False
        except Exception as e:
            st.error(f"Analysis error: {str(e)}")
            return

    st.success("✅ Analysis complete")

    og  = results['gaps']['overall']
    rem = results['remediation']

    if og['significant']:
        st.error(
            f"⚠️  Pay gap detected. "
            f"${og['gap_dollars']:,.0f} gap ({og['gap_pct']}%). "
            f"Significant (p={og['p_value']}). "
            f"{rem['total_employees']} employees affected."
        )
    else:
        st.success(
            "✅ No statistically significant gap detected."
        )

    # Tabs
    tab1, tab2 = st.tabs([
        '📋 Pay Equity Report',
        '🤖 Ask the Agent',
    ])

    with tab1:
        display_full_report(results)

    with tab2:
        st.markdown("""
        <div class="declaration">
        ⚠️ The agent answers <strong>only</strong> from
        your tweaked data analysis results and approved
        government legislation. Not legal advice.
        </div>
        """, unsafe_allow_html=True)
        st.markdown("---")

        if not st.session_state.get('agent_ready'):
            prepare_agent(results)

        if st.session_state.get('agent_ready'):
            render_agent_chat(key_prefix='tweak')


# ══════════════════════════════════════════════════════════════════════════
# PATH 3 — YOUR OWN DATA
# ══════════════════════════════════════════════════════════════════════════

def render_own_data_path():
    """
    User downloads template, fills it,
    uploads and runs their own analysis.
    """

    st.markdown("""
    <div class="main-header">
        <h1>📊 Your Pay Equity Analysis</h1>
        <p>
        Upload your employee data and get an instant
        pay equity report. Your data never leaves this session.
        </p>
    </div>
    """, unsafe_allow_html=True)

    if st.button('← Back to start', key='back_own'):
        for key in [
            'path', 'results', 'agent_ready',
            'conversation', 'agent_history',
        ]:
            st.session_state.pop(key, None)
        st.rerun()

    st.markdown("""
    <div class="privacy-notice">
    🔒 <strong>Privacy:</strong>
    Your data is processed only within this session.
    When you close the app — everything is gone.
    No data saved, cached, or transmitted.<br><br>
    <strong>Use masked employee codes only.
    No names. No SIN. No dates of birth.
    No personal identifiers.</strong>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Step 1 — Download template
    st.markdown(
        '<div class="section-header">'
        'Step 1 — Download the Data Template'
        '</div>',
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns([3, 1])

    with col1:
        st.markdown("""
        Download the template. Open in Excel or
        Google Sheets. Replace the example rows
        with your employee data. Save as CSV and
        upload in Step 2.

        **Required columns:**

        | Column | What to enter |
        |---|---|
        | `employee_id` | Masked code — EMP001. No real names. |
        | `gender` | Male / Female / Non-binary / Prefer not to say |
        | `job_class` | Your job title — e.g. Financial Analyst |
        | `job_grade` | Your pay grade — e.g. Grade 5 *(recommended)* |
        | `division` | Your business unit |
        | `location` | City or region |
        | `tenure_years` | Years at your organization |
        | `performance` | Score on 0-100 scale |
        | `salary` | Annual base salary only |

        **Note on job_grade:**
        If your organization has pay grades — include them.
        Grade-based comparison is most defensible under
        the Pay Equity Act. If you do not have grades —
        job_class is used instead.

        **Do not include:** Full name, SIN, date of birth,
        address, email, or any personal identifier.
        """)

    with col2:
        st.markdown("&nbsp;")
        st.download_button(
            label='⬇ Download Template',
            data=generate_template_csv(),
            file_name='pay_equity_data_template.csv',
            mime='text/csv',
            help=(
                'Fill this with your data '
                'and upload in Step 2'
            ),
        )

    st.markdown("---")

    # Step 2 — Upload
    st.markdown(
        '<div class="section-header">'
        'Step 2 — Upload Your Completed File'
        '</div>',
        unsafe_allow_html=True,
    )

    uploaded = st.file_uploader(
        'Upload your completed employee CSV',
        type='csv',
        key='own_upload',
        help='Fill the template from Step 1 and upload here',
    )

    if uploaded is None:
        st.info(
            "Download the template in Step 1, "
            "fill it with your data, and upload above."
        )
        return

    try:
        df = pd.read_csv(uploaded)
    except Exception as e:
        st.error(f"Could not read file: {str(e)}")
        return

    # Validate
    required = [
        'employee_id', 'gender',
        'division', 'location',
        'tenure_years', 'salary',
    ]
    has_grouping = any(
        c in df.columns
        for c in ['job_grade', 'job_class', 'role_level']
    )
    missing = [
        c for c in required if c not in df.columns
    ]

    if missing:
        st.error(
            f"Missing required columns: {missing}. "
            f"Please use the template from Step 1."
        )
        return

    if not has_grouping:
        st.error(
            "File needs at least one of: "
            "job_grade, job_class, role_level. "
            "Please use the template from Step 1."
        )
        return

    st.success(f"✅ {len(df):,} employees loaded")

    with st.expander("Preview your data (first 5 rows)"):
        st.dataframe(df.head(5), use_container_width=True)

    st.markdown("---")

    # Step 3 — Budget
    st.markdown(
        '<div class="section-header">'
        'Step 3 — Choose Your Remediation Budget'
        '</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        "The Pay Equity Act requires spending at least "
        "**1% of annual payroll per year**."
    )

    budget_option = st.radio(
        'Annual remediation budget:',
        options=[
            'Legal minimum — 1% of payroll (recommended)',
            'Custom amount',
            'Full remediation — close all gaps now',
        ],
        index=0,
        key='own_budget',
    )
    budget_input = 'legal_min'
    if 'Custom' in budget_option:
        custom = st.number_input(
            'Enter your annual budget ($):',
            min_value=0, value=500000,
            step=50000, format='%d',
            key='own_custom',
        )
        budget_input = str(custom)
    elif 'Full' in budget_option:
        budget_input = 'full'

    st.markdown("---")

    # Step 4 — Run
    st.markdown(
        '<div class="section-header">'
        'Step 4 — Run Your Analysis'
        '</div>',
        unsafe_allow_html=True,
    )

    if not st.button(
        '▶ Run Pay Equity Analysis',
        key='run_own',
        type='primary',
        use_container_width=True,
    ):
        st.info("Click above to run your analysis.")
        return

    with st.spinner('Running pay equity analysis...'):
        try:
            results = run_full_analysis(df, budget_input)
            st.session_state['results']        = results
            st.session_state['results_source'] = 'own'
            st.session_state['data_label']     = uploaded.name
            st.session_state['agent_ready']    = False
        except Exception as e:
            st.error(f"Analysis error: {str(e)}")
            st.caption(
                "Common issues: missing required columns, "
                "non-numeric salaries, fewer than 2 employees "
                "of each gender per job class."
            )
            return

    st.success("✅ Analysis complete")

    og  = results['gaps']['overall']
    rem = results['remediation']

    if og['significant']:
        st.error(
            f"⚠️  Pay gap detected. "
            f"${og['gap_dollars']:,.0f} gap ({og['gap_pct']}%). "
            f"Significant (p={og['p_value']}). "
            f"{rem['total_employees']} employees affected."
        )
    else:
        st.success(
            "✅ No statistically significant gap detected."
        )

    # Tabs — report, agent, downloads
    tab1, tab2, tab3 = st.tabs([
        '📋 Pay Equity Report',
        '🤖 Ask the Agent',
        '📥 Downloads',
    ])

    with tab1:
        display_full_report(results)

    with tab2:
        st.markdown("""
        <div class="declaration">
        ⚠️ The agent answers <strong>only</strong> from
        your uploaded data analysis results and approved
        government legislation sources.
        No other sources. Not legal advice.
        </div>
        """, unsafe_allow_html=True)
        st.markdown("---")

        if not st.session_state.get('agent_ready'):
            prepare_agent(results)

        if st.session_state.get('agent_ready'):
            render_agent_chat(key_prefix='own')

    with tab3:
        st.markdown(
            '<div class="section-header">'
            '📥 Download Your Results'
            '</div>',
            unsafe_allow_html=True,
        )

        col1, col2 = st.columns(2)

        with col1:
            st.download_button(
                label='⬇ Full Results (JSON)',
                data=json.dumps(
                    results, indent=2, default=str
                ).encode('utf-8'),
                file_name='pay_equity_results.json',
                mime='application/json',
            )

        with col2:
            yearly = rem.get('yearly_plans', [])
            if yearly and yearly[0].get('selected'):
                year1 = pd.DataFrame({
                    'employee_id': yearly[0]['selected'],
                    'remediation_year': 1,
                })
                st.download_button(
                    label='⬇ Year 1 Action List (CSV)',
                    data=year1.to_csv(
                        index=False
                    ).encode('utf-8'),
                    file_name='remediation_year1.csv',
                    mime='text/csv',
                )


# ══════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════

def main():
    render_sidebar()

    # Route based on chosen path
    path = st.session_state.get('path', None)

    if path == 'demo':
        render_demo_path()
    elif path == 'tweak':
        render_tweak_path()
    elif path == 'own':
        render_own_data_path()
    else:
        render_opening_screen()


if __name__ == '__main__':
    main()