# ══════════════════════════════════════════════════════════════════════════
# TOTAL REWARDS INTELLIGENCE PLATFORM
# agent_pay_equity.py — Pay Equity Agent
#
# WHAT THIS FILE DOES:
#   Answers questions about the pay equity analysis.
#   Reads analysis results from the current session.
#   Reads legislation from approved government URLs only.
#   Answers in plain English. Whizlink format.
#   No hallucination. Nothing outside approved sources.
#
# KNOWLEDGE SOURCES — TWO ONLY:
#   1. Analysis results passed in from the session
#      The gap analysis, remediation plan,
#      budget optimisation, business case.
#
#   2. Approved legislation URLs — fetched live:
#      laws-lois.justice.gc.ca/eng/acts/P-4.2/
#      ontario.ca/laws/statute/90p07
#      payequity.gov.on.ca
#
#   NOTHING ELSE. EVER.
#
# HOW TO RUN STANDALONE:
#   python agent_pay_equity.py
#   python agent_pay_equity.py --question "Do we have a gap?"
#
# REQUIRES:
#   ANTHROPIC_API_KEY in .env file
#   analyse.py must have been run first
#
# ⚠️  DECLARATION:
#   Responses are based on whatever data is loaded.
#   Demo responses use synthetic CAN North data.
#   Legislation references are real and sourced from
#   official government websites only.
#   This does not constitute legal advice.
# ══════════════════════════════════════════════════════════════════════════

import os
import sys
import json
import argparse
import warnings
import urllib.request
import re
from datetime import date, datetime

import anthropic
from dotenv import load_dotenv

load_dotenv()
warnings.filterwarnings('ignore')


# ══════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════

CONFIG = {
    # Analysis results file — written by analyse.py
    'results_file': 'outputs/pay_equity_results.json',

    # Approved legislation URLs — no other sources permitted
    'approved_urls': [
        'https://laws-lois.justice.gc.ca/eng/acts/P-4.2/',
        'https://www.ontario.ca/laws/statute/90p07',
        'https://www.payequity.gov.on.ca',
    ],

    # Claude model
    'model':      'claude-sonnet-4-6',
    'max_tokens': 1500,
}


# ══════════════════════════════════════════════════════════════════════════
# STEP 1 — LOAD RESULTS
# ══════════════════════════════════════════════════════════════════════════

def load_results(results=None):
    """
    Loads pay equity analysis results.

    If results dict is passed in (from Streamlit session)
    — uses that directly.

    If running standalone — loads from JSON file.
    """
    if results is not None:
        return results

    if not os.path.exists(CONFIG['results_file']):
        print(f"\n  ERROR: {CONFIG['results_file']} not found.")
        print(f"  Run: python analyse.py")
        sys.exit(1)

    with open(
        CONFIG['results_file'], 'r', encoding='utf-8'
    ) as f:
        return json.load(f)


# ══════════════════════════════════════════════════════════════════════════
# STEP 2 — FETCH LEGISLATION
# ══════════════════════════════════════════════════════════════════════════

def fetch_legislation():
    """
    Fetches legislation text from approved government URLs only.

    APPROVED SOURCES:
    1. laws-lois.justice.gc.ca — Federal Pay Equity Act
    2. ontario.ca/laws          — Ontario Pay Equity Act
    3. payequity.gov.on.ca      — Ontario Pay Equity Office

    NO OTHER WEBSITES. NO THIRD-PARTY SOURCES.
    NO TRAINING DATA USED FOR LEGISLATION CONTENT.

    If fetch fails — records the error.
    Agent uses citation text from results as fallback.
    """

    legislation_text = []
    retrieved        = date.today().strftime('%B %d, %Y')

    for url in CONFIG['approved_urls']:
        try:
            req = urllib.request.Request(
                url,
                headers={'User-Agent': 'Mozilla/5.0'}
            )
            with urllib.request.urlopen(
                req, timeout=10
            ) as r:
                html    = r.read().decode(
                    'utf-8', errors='ignore'
                )
                text    = re.sub(r'<[^>]+>', ' ', html)
                text    = re.sub(r'\s+', ' ', text).strip()
                excerpt = text[:3000]
                legislation_text.append(
                    f"SOURCE: {url}\n"
                    f"RETRIEVED: {retrieved}\n"
                    f"CONTENT:\n{excerpt}\n"
                )
        except Exception as e:
            legislation_text.append(
                f"SOURCE: {url}\n"
                f"STATUS: Could not fetch — {str(e)}\n"
                f"NOTE: Use citation text from analysis only.\n"
            )

    return '\n'.join(legislation_text)


# ══════════════════════════════════════════════════════════════════════════
# STEP 3 — BUILD CONTEXT DOCUMENT
# ══════════════════════════════════════════════════════════════════════════

def build_context(results, legislation_text):
    """
    Builds the complete context document for the agent.

    This is EVERYTHING the agent is allowed to know.
    The agent answers ONLY from this document.
    Nothing from training data.
    Nothing from the internet beyond approved URLs.
    Nothing outside these two sources.

    Structure:
    Part 1 — Analysis results
    Part 2 — Legislation from approved URLs
    """

    og   = results['gaps']['overall']
    rem  = results['remediation']
    opt  = results['optimisation']
    biz  = results['business_case']
    wl   = results['whizlink']
    wf   = results['workforce']
    meta = results['metadata']

    # Gap by job grade
    grade_gaps = '\n'.join([
        f"  {r['group']:<12} "
        f"Men: ${r['male_avg']:,.0f}  "
        f"Women: ${r['female_avg']:,.0f}  "
        f"Gap: ${r['gap_dollars']:,.0f} ({r['gap_pct']}%)  "
        f"Significant: {'Yes' if r['significant'] else 'No'}"
        for r in results['gaps'].get('by_grade', [])
    ]) or "  Not available"

    # Gap by job class
    class_gaps = '\n'.join([
        f"  {r['group']:<30} "
        f"Men: ${r['male_avg']:,.0f}  "
        f"Women: ${r['female_avg']:,.0f}  "
        f"Gap: ${r['gap_dollars']:,.0f}  "
        f"Significant: {'Yes' if r['significant'] else 'No'}"
        for r in results['gaps'].get('by_class', [])
    ]) or "  Not available"

    # Gap by division
    div_gaps = '\n'.join([
        f"  {d['group']:<25} "
        f"Men: ${d['male_avg']:,.0f}  "
        f"Women: ${d['female_avg']:,.0f}  "
        f"Gap: ${d['gap_dollars']:,.0f}  "
        f"Significant: {'Yes' if d['significant'] else 'No'}"
        for d in results['gaps'].get('by_division', [])
    ]) or "  Not available"

    # Yearly plan
    yearly = '\n'.join([
        f"  Year {p['year']}: "
        f"{p['count']} employees — "
        f"${p['spent']:,.0f}"
        for p in rem.get('yearly_plans', [])
    ]) or "  Not available"

    # Scenarios
    scenarios = '\n'.join([
        f"  {s['label']}: "
        f"{s['count']} employees — "
        f"risk closed {s['risk_closed_pct']}% — "
        f"gap closed {s['gap_closed_pct']}%"
        for s in opt.get('scenarios', [])
    ]) or "  Not available"

    # Data source label
    data_label = (
        "SYNTHETIC DEMO DATA — CAN North Financial (fictional)"
        if 'fictional' in meta.get('declaration', '').lower()
        else f"USER-UPLOADED DATA — {meta.get('data_file', '')}"
    )

    context = f"""
══════════════════════════════════════════════════════════════════
PAY EQUITY ANALYSIS — AGENT CONTEXT DOCUMENT
Generated: {meta.get('run_date')} {meta.get('run_time')}
Data: {data_label}
══════════════════════════════════════════════════════════════════

DECLARATION:
{meta.get('declaration')}
This analysis does not constitute legal advice.
Consult qualified legal counsel for jurisdiction-specific guidance.

══════════════════════════════════════════════════════════════════
PART 1 — ANALYSIS RESULTS
Source: pay equity analysis run on {meta.get('run_date')}
══════════════════════════════════════════════════════════════════

WORKFORCE:
Total employees:   {wf['total']:,}
Male employees:    {wf['male']:,}
Female employees:  {wf['female']:,}
Average salary:    ${wf['avg_salary']:,.0f}

EXECUTIVE SUMMARY (Whizlink):
WHAT:     {wl['what']}
WHY:      {wl['why']}
SO WHAT:  {wl['so_what']}
HOW:      {wl['how']}

OVERALL PAY GAP:
Male average salary:    ${og['male_avg']:,.0f}
Female average salary:  ${og['female_avg']:,.0f}
Gap:                    ${og['gap_dollars']:,.0f} ({og['gap_pct']}%)
Statistically real:     {'Yes' if og['significant'] else 'No'} (p={og['p_value']})

GAP BY JOB GRADE (primary — most defensible under the Act):
{grade_gaps}

GAP BY JOB CLASS (human readable titles):
{class_gaps}

GAP BY DIVISION:
{div_gaps}

REMEDIATION PLAN:
Employees needing adjustment: {rem['total_employees']}
Total adjustment cost:        ${rem['total_remediation']:,.0f}
Annual payroll:               ${rem['total_payroll']:,.0f}
Legal minimum per year (1%):  ${rem['legal_min_budget']:,.0f}
Phase in permitted:           {'Yes — 3 years' if rem['can_phase'] else 'No'}

THREE YEAR PLAN:
{yearly}

BUDGET OPTIMISATION:
Budget:              {opt['budget_label']}
Employees adjusted:  {opt['selected_count']}
Employees deferred:  {opt['deferred_count']}
Risk closed:         {opt['risk_closed_pct']}%
Gap closed:          {opt['gap_closed_pct']}%

SCENARIO COMPARISON:
{scenarios}

BUSINESS CASE:
Total remediation cost:  ${rem['total_remediation']:,.0f}
Retention value:         ${biz['retention_value']:,.0f}
Legal risk avoided:      ${biz['legal_risk_avoided']:,.0f}
Net benefit:             ${biz['net_benefit']:,.0f}

══════════════════════════════════════════════════════════════════
PART 2 — LEGISLATION
Sources: approved government URLs only
Retrieved: {date.today().strftime('%B %d, %Y')}
══════════════════════════════════════════════════════════════════

{legislation_text}

══════════════════════════════════════════════════════════════════
END OF CONTEXT DOCUMENT
══════════════════════════════════════════════════════════════════
"""
    return context


# ══════════════════════════════════════════════════════════════════════════
# STEP 4 — BUILD SYSTEM PROMPT
# ══════════════════════════════════════════════════════════════════════════

def build_system_prompt(context):
    """
    Builds the system prompt for the pay equity agent.

    Defines:
    → What the agent knows (only the context document)
    → What the agent answers (only from that context)
    → How the agent answers (Whizlink, plain English)
    → What the agent never does (hallucinate, go outside scope)

    These rules cannot be overridden by any user message.
    """

    return f"""You are the Pay Equity Agent for the
Total Rewards Intelligence Platform.

YOUR ROLE:
Answer questions about pay equity analysis results
and pay equity legislation.

YOUR KNOWLEDGE — TWO SOURCES ONLY:
1. The analysis results in the context document below.
2. The legislation text fetched from approved government
   URLs in the context document below.

YOU NEVER:
→ Use information from your training data
→ Reference any website not in the approved list
→ Make up statistics, numbers, or legal provisions
→ Answer questions outside pay equity scope
→ Provide legal advice

IF ASKED SOMETHING OUTSIDE SCOPE — say exactly:
"I can only answer questions about the pay equity
analysis results and the approved legislation sources
provided in this session. For anything outside this
scope please consult a qualified professional."

HOW YOU ANSWER — ALWAYS USE WHIZLINK FORMAT:
WHAT:    State the fact plainly.
WHY:     Explain why it matters.
SO WHAT: What does it mean for the organization?
HOW:     What is the specific action to take?

LANGUAGE RULES:
→ Plain English only. No jargon.
→ A CHRO reads every answer in 30 seconds.
→ Numbers in dollars and percentages.
→ Always cite which source your answer comes from.

CITATIONS — ALWAYS INCLUDE:
→ Full Act name
→ Section number if known
→ Source URL
→ Retrieved date: {date.today().strftime('%B %d, %Y')}

DECLARATION — INCLUDE WHEN RELEVANT:
→ State whether answer is based on demo
  (synthetic) data or user-uploaded data
→ This does not constitute legal advice
→ Consult qualified legal counsel for
  jurisdiction-specific guidance

CONTEXT DOCUMENT:
{context}"""


# ══════════════════════════════════════════════════════════════════════════
# STEP 5 — ASK THE AGENT
# ══════════════════════════════════════════════════════════════════════════

def ask_agent(question, system_prompt, conversation_history):
    """
    Sends a question to the pay equity agent.

    Maintains conversation history for multi-turn chat.
    Every response is grounded in the context only.

    Parameters:
        question             the user's question
        system_prompt        full system prompt with context
        conversation_history list of prior messages

    Returns:
        response_text    the agent's answer
        updated_history  conversation history updated
    """

    from dotenv import load_dotenv
    load_dotenv()
    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

    conversation_history.append({
        'role':    'user',
        'content': question,
    })

    response = client.messages.create(
        model=CONFIG['model'],
        max_tokens=CONFIG['max_tokens'],
        system=system_prompt,
        messages=conversation_history,
    )

    response_text = response.content[0].text

    conversation_history.append({
        'role':    'assistant',
        'content': response_text,
    })

    return response_text, conversation_history


# ══════════════════════════════════════════════════════════════════════════
# MAIN — STANDALONE MODE
# ══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='Pay Equity Agent'
    )
    parser.add_argument(
        '--question',
        type=str,
        default=None,
        help='Single question to ask the agent',
    )
    args = parser.parse_args()

    print(f"\n{'='*65}")
    print(f"PAY EQUITY AGENT")
    print(f"Total Rewards Intelligence Platform")
    print(
        f"Run date: "
        f"{datetime.now().strftime('%Y-%m-%d %H:%M')}"
    )
    print(f"{'='*65}")
    print(f"\n  Knowledge sources:")
    print(f"  1. outputs/pay_equity_results.json")
    print(f"  2. Approved legislation URLs only")
    print(f"\n  ⚠️  Not legal advice.")

    print(f"\n  Loading analysis results...")
    results = load_results()
    print(
        f"  ✅ Results loaded — "
        f"{results['metadata']['run_date']}"
    )

    print(f"\n  Fetching legislation from approved URLs...")
    legislation = fetch_legislation()
    print(f"  ✅ Legislation fetched")

    print(f"\n  Building agent context...")
    context       = build_context(results, legislation)
    system_prompt = build_system_prompt(context)
    print(f"  ✅ Agent ready")

    print(f"\n{'='*65}")
    print(f"  AGENT READY")
    print(f"  Type 'quit' to stop.")
    print(f"  Type 'reset' to clear conversation history.")
    print(f"{'='*65}")

    conversation_history = []

    if args.question:
        response, _ = ask_agent(
            args.question,
            system_prompt,
            conversation_history,
        )
        print(f"\n  Q: {args.question}")
        print(f"\n  A: {response}")
        return

    print(f"\n  SUGGESTED QUESTIONS:")
    print(f"  → Do we have a pay gap?")
    print(f"  → Which job grade has the largest gap?")
    print(f"  → What does the law require us to do?")
    print(f"  → How much will remediation cost?")
    print(f"  → What is the business case?")
    print()

    while True:
        try:
            question = input("  Your question: ").strip()
        except (KeyboardInterrupt, EOFError):
            print(f"\n\n  Session ended.")
            break

        if not question:
            continue

        if question.lower() in ['quit', 'exit']:
            print(f"\n  Session ended.")
            break

        if question.lower() == 'reset':
            conversation_history = []
            print(f"\n  ✅ Conversation cleared.")
            continue

        print(f"\n  Thinking...")

        try:
            response, conversation_history = ask_agent(
                question,
                system_prompt,
                conversation_history,
            )
            print(f"\n{'─'*65}")
            for line in response.split('\n'):
                print(f"  {line}")
            print(f"{'─'*65}")
            print(
                f"\n  ⚠️  Source: analysis results + "
                f"approved legislation URLs only. "
                f"Not legal advice."
            )
            print()

        except anthropic.AuthenticationError:
            print(f"\n  ERROR: Invalid API key.")
            print(
                f"  Check your .env file: "
                f"ANTHROPIC_API_KEY=your_key"
            )
            break

        except Exception as e:
            print(f"\n  ERROR: {str(e)}")


if __name__ == '__main__':
    main()