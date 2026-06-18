# test_cache.py
# Run this once to see what the ESDC data looks like
# So we know the exact column names before we build on them

import pandas as pd
from wage_cache import get_esdc_wages, CACHE_CONFIG

print("Fetching ESDC data...")
df, metadata, from_cache = get_esdc_wages()

if df is not None:
    print(f"\nRows: {len(df):,}")
    print(f"Columns: {list(df.columns)}")
    print(f"\nFirst 3 rows:")
    print(df.head(3).to_string())
    print(f"\nUnique GEO values (provinces):")
    print(df['GEO'].unique() if 'GEO' in df.columns else "GEO column not found")
    print(f"\nSample NOC values:")
    print(df['NOC'].head(10).tolist() if 'NOC' in df.columns else "NOC column not found")
else:
    print("Could not fetch data")
    print(f"Metadata: {metadata}")

    # Add to bottom of test_cache.py

print("\n\nTESTING build_bands_from_noc...")
from wage_cache import build_bands_from_noc

# CAN North default NOC mapping
test_mapping = {
    'Analyst':        '11102',
    'Senior Analyst': '11102',
    'Manager':        '10010',
    'Director':       '10010',
    'VP':             '00011',
}

bands, citation, metadata, unmapped = build_bands_from_noc(
    test_mapping, province='ON'
)

print(f"\nBANDS:")
if bands:
    for role, (lo, mid, hi) in bands.items():
        print(f"  {role:<20} ${lo:>10,} ${mid:>10,} ${hi:>10,}")
else:
    print("  No bands returned")

print(f"\nUNMAPPED: {unmapped}")
print(f"\nCITATION:")
print(f"  {citation[:120]}...")

# Add to bottom of test_cache.py
# Search for financial services NOC codes in the data

print("\n\nSEARCHING FOR FINANCIAL SERVICES NOC CODES...")
from wage_cache import get_esdc_wages

df, metadata, from_cache = get_esdc_wages()

# Show all NOC titles containing 'financial' or 'manager'
# Province = ON, region = provincial level only (ER_Code = prov code)
financial = df[
    (df['prov'] == 'ON') &
    (df['ER_Code_Code_RE'] == 'ON') &  # provincial level only
    (
        df['NOC_Title_eng'].str.lower().str.contains('financial', na=False) |
        df['NOC_Title_eng'].str.lower().str.contains('manager', na=False) |
        df['NOC_Title_eng'].str.lower().str.contains('analyst', na=False) |
        df['NOC_Title_eng'].str.lower().str.contains('senior', na=False) |
        df['NOC_Title_eng'].str.lower().str.contains('director', na=False) |
        df['NOC_Title_eng'].str.lower().str.contains('executive', na=False)
    )
][['NOC_CNP', 'NOC_Title_eng',
   'Low_Wage_Salaire_Minium',
   'Median_Wage_Salaire_Median',
   'High_Wage_Salaire_Maximal',
   'Annual_Wage_Flag_Salaire_annuel']].drop_duplicates()

print(f"\n{'NOC Code':<14} {'Title':<50} "
      f"{'Low':>10} {'Median':>10} {'High':>10} {'Annual':>8}")
print(f"{'─'*14} {'─'*50} "
      f"{'─'*10} {'─'*10} {'─'*10} {'─'*8}")

for _, row in financial.iterrows():
    title = str(row['NOC_Title_eng'])[:48]
    low   = f"${row['Low_Wage_Salaire_Minium']:,.0f}" \
            if pd.notna(row['Low_Wage_Salaire_Minium']) else 'N/A'
    med   = f"${row['Median_Wage_Salaire_Median']:,.0f}" \
            if pd.notna(row['Median_Wage_Salaire_Median']) else 'N/A'
    high  = f"${row['High_Wage_Salaire_Maximal']:,.0f}" \
            if pd.notna(row['High_Wage_Salaire_Maximal']) else 'N/A'
    ann   = str(int(row['Annual_Wage_Flag_Salaire_annuel'])) \
            if pd.notna(row['Annual_Wage_Flag_Salaire_annuel']) else '?'
    print(f"{row['NOC_CNP']:<14} {title:<50} "
          f"{low:>10} {med:>10} {high:>10} {ann:>8}")