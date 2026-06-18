# ══════════════════════════════════════════════════════════════════════════
# CAN NORTH FINANCIAL — PAY EQUITY INTELLIGENCE
# wage_cache.py
#
# WHO USES THIS: Every other file in the project
# WHAT IT DOES:  Manages salary band data from the
#                Government of Canada ESDC open data portal
#                Caches locally to avoid repeated downloads
#                Refreshes on a smart schedule
#                Stamps every output with retrieval date
#
# CACHE RULES:
#   Normal months    → refresh weekly (every 7 days)
#   November         → refresh daily (ESDC annual release month)
#   User uploads     → always use user data, skip fetch
#   Fetch fails      → use stale cache + show warning
#
# DATA SOURCE:
#   Employment and Social Development Canada
#   Wage Data by National Occupational Classification
#   open.canada.ca — Government of Canada Open Data
#   Updated annually every November
#
# HOW TO RUN STANDALONE (force refresh):
#   python wage_cache.py --refresh
#   python wage_cache.py --status
#
# ══════════════════════════════════════════════════════════════════════════

import os
import json
import requests
import pandas as pd
import argparse
from datetime import date, datetime, timedelta

# ── CONFIGURATION ─────────────────────────────────────────────────────────

CACHE_CONFIG = {
    # Cache file locations
    'cache_dir':        'data/cache',
    'wages_file':       'data/cache/esdc_wages.csv',
    'metadata_file':    'data/cache/esdc_metadata.json',
    'noc_map_file':     'data/cache/noc_mapping.json',

    # Refresh schedule
    'refresh_days':     7,    # normal — weekly
    'update_month':     11,   # November — ESDC annual release
    'update_window':    14,   # daily for 14 days in November

    # Government of Canada ESDC open data
    # Employment and Social Development Canada
    # Wage data by NOC code — annual release
    # Free. Official. No API key required.
    'esdc_url': (
        'https://open.canada.ca/data/dataset/'
        'adad580f-76b0-4502-bd05-20c125de9116/resource/'
        'd16e10ea-77bf-4db8-bdb5-adc709e6cada/download/'
        '2a71-das-wage2024opendata-esdc-all-11dec2024-vf.csv'
    ),
    'esdc_open_data_page': (
        'https://open.canada.ca/data/en/dataset/'
        'adad580f-76b0-4502-bd05-20c125de9116'
    ),
    'hours_per_year': 2080,  # 40 hrs × 52 weeks
}

# ── NOC CODE REFERENCE ────────────────────────────────────────────────────
# Common NOC codes across industries
# User maps their role titles to these codes
# We fetch official wage data for each code
#
# Full NOC directory: noc.esdc.gc.ca
# Wage data by NOC:   jobbank.gc.ca/wagereport/occupation/{id}

NOC_REFERENCE = {
    # Financial Services
    '11101': 'Financial auditors and accountants',
    '11102': 'Financial and investment analysts',
    '10010': 'Financial managers',
    '10011': 'Human resources managers',
    '12010': 'Administrative officers',
    '12011': 'Administrative assistants',

    # Technology
    '21232': 'Software developers and programmers',
    '21211': 'Data scientists',
    '21220': 'Cybersecurity specialists',
    '21223': 'Database analysts and data administrators',
    '21221': 'Business systems specialists',

    # Healthcare
    '31301': 'Registered nurses',
    '31302': 'Nurse practitioners',
    '32101': 'Licensed practical nurses',
    '31101': 'Specialists in clinical and laboratory',
    '31102': 'General practitioners and family physicians',

    # Operations and Management
    '00011': 'Senior managers — financial, communications',
    '00012': 'Senior managers — health, education',
    '00015': 'Senior managers — goods production',
    '10019': 'Other administrative services managers',
    '60010': 'Retail and wholesale trade managers',

    # Human Resources
    '11200': 'Human resources professionals',
    '11201': 'Employment insurance and HR officers',
    '12101': 'Human resources and recruitment officers',
}


# ══════════════════════════════════════════════════════════════════════════
# CACHE MANAGEMENT
# ══════════════════════════════════════════════════════════════════════════

def load_metadata():
    """
    Loads cache metadata from disk.
    Returns empty dict if no metadata exists yet.
    """
    path = CACHE_CONFIG['metadata_file']
    if not os.path.exists(path):
        return {}
    with open(path, 'r') as f:
        return json.load(f)


def save_metadata(metadata):
    """Saves cache metadata to disk."""
    os.makedirs(CACHE_CONFIG['cache_dir'], exist_ok=True)
    with open(CACHE_CONFIG['metadata_file'], 'w') as f:
        json.dump(metadata, f, indent=2)


def in_esdc_update_window():
    """
    Returns True if today is in the ESDC annual
    update window (November).
    During this period we check daily instead of weekly
    because the government may release new data any day.
    """
    return date.today().month == CACHE_CONFIG['update_month']


def cache_needs_refresh():
    """
    Determines whether to fetch fresh data or use cache.

    Logic:
    1. No cache file → fetch
    2. In November update window → refresh daily
    3. Otherwise → refresh weekly
    4. Fetch failed last time → use stale + warn

    Returns:
        needs_refresh: bool
        reason: human readable explanation
    """
    wages_path = CACHE_CONFIG['wages_file']
    metadata   = load_metadata()

    # No cache at all
    if not os.path.exists(wages_path):
        return True, 'No cache — first run'

    if not metadata:
        return True, 'No metadata — rebuilding'

    # Calculate age
    last_fetched = datetime.strptime(
        metadata.get('last_fetched', '2000-01-01'),
        '%Y-%m-%d'
    ).date()
    age_days = (date.today() - last_fetched).days

    # November — check daily
    if in_esdc_update_window():
        if age_days >= 1:
            return True, f'November update window — {age_days}d old'
        return False, f'November window — checked today'

    # Normal — weekly refresh
    refresh_days = CACHE_CONFIG['refresh_days']
    if age_days >= refresh_days:
        return True, f'Cache is {age_days} days old (>{refresh_days}d)'

    return False, f'Cache is fresh — {age_days} days old'


# ══════════════════════════════════════════════════════════════════════════
# FETCH AND CACHE
# ══════════════════════════════════════════════════════════════════════════

def fetch_esdc_data():
    """
    Downloads ESDC wage CSV from Government of Canada
    open data portal and saves to local cache.

    Source: Employment and Social Development Canada
    URL:    open.canada.ca
    Data:   Wage rates by NOC code, province, region
    Cost:   Free. No API key. Official government data.
    """
    os.makedirs(CACHE_CONFIG['cache_dir'], exist_ok=True)
    today = date.today().strftime('%Y-%m-%d')

    print(f"  Downloading ESDC wage data...")
    print(f"  Source: Government of Canada Open Data")
    print(f"  URL: {CACHE_CONFIG['esdc_open_data_page']}")

    try:
        response = requests.get(
            CACHE_CONFIG['esdc_url'],
            timeout=60
        )
        response.raise_for_status()

        # Save raw CSV
        with open(CACHE_CONFIG['wages_file'], 'wb') as f:
            f.write(response.content)

        df = pd.read_csv(CACHE_CONFIG['wages_file'])

        # Calculate next refresh date
        if in_esdc_update_window():
            next_refresh = (
                date.today() + timedelta(days=1)
            ).strftime('%Y-%m-%d')
            refresh_note = 'Daily — November update window'
        else:
            next_refresh = (
                date.today() + timedelta(
                    days=CACHE_CONFIG['refresh_days']
                )
            ).strftime('%Y-%m-%d')
            refresh_note = f"Weekly — every {CACHE_CONFIG['refresh_days']} days"

        metadata = {
            # Retrieval information
            'last_fetched':        today,
            'next_scheduled':      next_refresh,
            'refresh_schedule':    refresh_note,

            # Source information — shown on every report
            'source_name':         (
                'Employment and Social Development Canada'
            ),
            'source_short':        'ESDC — Government of Canada',
            'source_url':          CACHE_CONFIG['esdc_url'],
            'source_page':         CACHE_CONFIG['esdc_open_data_page'],
            'source_citation':     (
                'Employment and Social Development Canada. '
                'Wage Data by National Occupational '
                'Classification (NOC). '
                'Government of Canada Open Data. '
                f'Retrieved: {date.today().strftime("%B %d, %Y")}. '
                f'Source: {CACHE_CONFIG["esdc_open_data_page"]}'
            ),

            # Data information
            'rows':                len(df),
            'columns':             list(df.columns),
            'esdc_update_month':   'November (annual)',

            # Status
            'fetch_status':        'success',
            'using_stale':         False,
            'fetch_warning':       None,
        }

        save_metadata(metadata)

        print(f"  ✅ {len(df):,} wage records cached")
        print(f"  ✅ Retrieved: {today}")
        print(f"  ✅ Next refresh: {next_refresh} ({refresh_note})")

        return df, metadata, True

    except Exception as e:
        print(f"  ⚠️  Could not fetch: {str(e)}")

        # Use stale cache if available
        if os.path.exists(CACHE_CONFIG['wages_file']):
            df       = pd.read_csv(CACHE_CONFIG['wages_file'])
            metadata = load_metadata()
            metadata['fetch_warning'] = (
                f"Fetch failed {today}: {str(e)}. "
                f"Using cached data from "
                f"{metadata.get('last_fetched', 'unknown date')}."
            )
            metadata['using_stale'] = True
            save_metadata(metadata)
            print(f"  Using stale cache from "
                  f"{metadata.get('last_fetched', 'unknown')}")
            return df, metadata, False

        print(f"  ❌ No cache available")
        return None, {
            'fetch_status':  'failed',
            'fetch_warning': str(e),
            'last_fetched':  None,
        }, False


def get_esdc_wages(force_refresh=False):
    """
    Main entry point for wage data.
    All other files call this — never the fetch directly.

    Parameters:
        force_refresh: if True — bypass cache and fetch fresh

    Returns:
        df:           wage data DataFrame (or None if unavailable)
        metadata:     source, freshness, citation info
        from_cache:   True if served from cache
    """
    needs_refresh, reason = cache_needs_refresh()

    if force_refresh:
        print(f"  Force refresh requested")
        return fetch_esdc_data()

    if needs_refresh:
        print(f"  Refreshing: {reason}")
        return fetch_esdc_data()

    # Serve from cache
    df       = pd.read_csv(CACHE_CONFIG['wages_file'])
    metadata = load_metadata()
    return df, metadata, True


# ══════════════════════════════════════════════════════════════════════════
# BAND BUILDING
# ══════════════════════════════════════════════════════════════════════════

def build_bands_from_noc(role_noc_mapping, province='ON'):
    """
    Builds salary bands for given roles using
    official ESDC NOC wage data.

    Parameters:
        role_noc_mapping: dict of role_title → noc_code
                          e.g. {'Analyst': '11102',
                                'Manager': '10010'}
                          NOC codes WITHOUT the 'NOC_' prefix
        province:         two-letter province code
                          'ON' = Ontario (default)
                          'BC', 'AB', 'QC', 'NS', etc.
                          'NAT' = national average

    Returns:
        bands:      dict of role → (min, mid, max) annual
        citation:   full source citation string
        metadata:   freshness and source details
        unmapped:   list of roles that could not be matched
    """
    df, metadata, from_cache = get_esdc_wages()

    if df is None:
        return None, 'ESDC data unavailable', metadata, []

    bands    = {}
    unmapped = []
    hours    = CACHE_CONFIG['hours_per_year']

    for role_title, noc_code in role_noc_mapping.items():

        # ESDC stores NOC as 'NOC_11102' — add prefix
        noc_formatted = f"NOC_{noc_code}"

        # Filter by NOC code and province
        # Column names from actual ESDC CSV:
        # NOC_CNP = NOC code
        # prov    = province (two-letter)
        row = df[
            (df['NOC_CNP'].astype(str).str.strip() ==
             noc_formatted) &
            (df['prov'].astype(str).str.strip().str.upper() ==
             province.upper().strip())
        ]

        if len(row) == 0:
            # Try national if province not found
            row = df[
                (df['NOC_CNP'].astype(str).str.strip() ==
                 noc_formatted) &
                (df['prov'].astype(str).str.strip() == 'NAT')
            ]

        if len(row) > 0:
            try:
                row = row.iloc[0]

                # Check if wages are already annual
                # Annual_Wage_Flag = 1 → already annual
                # Annual_Wage_Flag = 0 → hourly × 2080
                is_annual = int(
                    row.get(
                        'Annual_Wage_Flag_Salaire_annuel', 0
                    )
                ) == 1
                multiplier = 1 if is_annual else hours

                # Extract wage values
                # Use NaN-safe conversion
                def safe_wage(val):
                    try:
                        v = float(str(val).replace(',', ''))
                        return v if not pd.isna(v) else None
                    except:
                        return None

                low    = safe_wage(
                    row['Low_Wage_Salaire_Minium']
                )
                median = safe_wage(
                    row['Median_Wage_Salaire_Median']
                )
                high   = safe_wage(
                    row['High_Wage_Salaire_Maximal']
                )

                # Need at least median to proceed
                if median is None:
                    unmapped.append(
                        f"{role_title}: No median wage "
                        f"for NOC {noc_code} in {province}"
                    )
                    continue

                # If low or high missing — estimate from median
                if low  is None:
                    low  = median * 0.75
                if high is None:
                    high = median * 1.30

                bands[role_title] = (
                    round(low    * multiplier, -2),
                    round(median * multiplier, -2),
                    round(high   * multiplier, -2),
                )

            except Exception as e:
                unmapped.append(
                    f"{role_title} (NOC {noc_code}): {e}"
                )
        else:
            unmapped.append(
                f"{role_title}: NOC {noc_code} "
                f"not found for province {province} or NAT"
            )

    # Build citation stamp
    retrieved = metadata.get(
        'last_fetched',
        date.today().strftime('%Y-%m-%d')
    )
    next_ref  = metadata.get('next_scheduled', 'unknown')
    warning   = metadata.get('fetch_warning', None)

    citation = (
        f"Salary bands sourced from: "
        f"Employment and Social Development Canada (ESDC) "
        f"— Wage Data by National Occupational "
        f"Classification (NOC). "
        f"Government of Canada Open Data. "
        f"Province: {province}. "
        f"Data retrieved: {retrieved}. "
        f"Next scheduled refresh: {next_ref}. "
        f"Source: {CACHE_CONFIG['esdc_open_data_page']}"
    )

    if warning:
        citation += f" ⚠️  Note: {warning}"

    return bands, citation, metadata, unmapped
# ══════════════════════════════════════════════════════════════════════════
# REFRESH STAMP — goes on every output
# ══════════════════════════════════════════════════════════════════════════

def get_refresh_stamp():
    """
    Returns a formatted stamp for reports, UI, and agent responses.
    Every output that uses wage data shows this stamp.

    Example output:
    ┌─────────────────────────────────────────────────────┐
    │  📊 Wage Data Source                                │
    │  Source:   ESDC — Government of Canada              │
    │  Retrieved: June 17, 2026                           │
    │  Next refresh: June 24, 2026                        │
    │  Status:   ✅ Current                               │
    │  open.canada.ca                                     │
    └─────────────────────────────────────────────────────┘
    """
    metadata = load_metadata()

    if not metadata:
        return {
            'source':    'ESDC — Government of Canada',
            'retrieved': 'Not yet fetched',
            'next':      'Run wage_cache.py --refresh',
            'status':    '⚠️  No data',
            'warning':   None,
            'url':       CACHE_CONFIG['esdc_open_data_page'],
        }

    retrieved = metadata.get('last_fetched', 'unknown')
    next_ref  = metadata.get('next_scheduled', 'unknown')
    warning   = metadata.get('fetch_warning')
    stale     = metadata.get('using_stale', False)

    if stale or warning:
        status = '⚠️  Using cached data — refresh failed'
    elif in_esdc_update_window():
        status = '🔄 November update window — checking daily'
    else:
        status = '✅ Current'

    return {
        'source':    'ESDC — Government of Canada',
        'retrieved': datetime.strptime(
            retrieved, '%Y-%m-%d'
        ).strftime('%B %d, %Y') if retrieved != 'unknown' else 'unknown',
        'next':      datetime.strptime(
            next_ref, '%Y-%m-%d'
        ).strftime('%B %d, %Y') if next_ref != 'unknown' else 'unknown',
        'status':    status,
        'warning':   warning,
        'url':       CACHE_CONFIG['esdc_open_data_page'],
        'citation':  metadata.get('source_citation', ''),
    }


def print_refresh_stamp():
    """Prints the refresh stamp to console — for reports."""
    stamp = get_refresh_stamp()
    print(f"\n  {'─'*50}")
    print(f"  📊 WAGE DATA SOURCE")
    print(f"  {'─'*50}")
    print(f"  Source:        {stamp['source']}")
    print(f"  Retrieved:     {stamp['retrieved']}")
    print(f"  Next refresh:  {stamp['next']}")
    print(f"  Status:        {stamp['status']}")
    if stamp['warning']:
        print(f"  ⚠️  Warning:   {stamp['warning']}")
    print(f"  URL:           {stamp['url']}")
    print(f"  {'─'*50}\n")


# ══════════════════════════════════════════════════════════════════════════
# MAIN — standalone run
# ══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='ESDC wage cache manager'
    )
    parser.add_argument(
        '--refresh',
        action='store_true',
        help='Force refresh from ESDC'
    )
    parser.add_argument(
        '--status',
        action='store_true',
        help='Show cache status'
    )
    args = parser.parse_args()

    print(f"\n{'='*65}")
    print(f"ESDC WAGE CACHE — CAN NORTH PAY EQUITY")
    print(f"Government of Canada Open Data")
    print(f"{'='*65}")

    if args.status:
        needs_refresh, reason = cache_needs_refresh()
        print(f"\n  Cache status: {reason}")
        print_refresh_stamp()
        return

    if args.refresh:
        print(f"\n  Force refreshing ESDC data...")
        df, metadata, success = fetch_esdc_data()
    else:
        needs, reason = cache_needs_refresh()
        print(f"\n  Cache check: {reason}")
        df, metadata, from_cache = get_esdc_wages()

    print_refresh_stamp()

    if df is not None:
        print(f"  Wage records available: {len(df):,}")
        print(f"  Columns: {list(df.columns)[:6]}...")
        print(f"\n  Next step: python train_pipeline.py")

    print(f"{'='*65}\n")


if __name__ == '__main__':
    main()