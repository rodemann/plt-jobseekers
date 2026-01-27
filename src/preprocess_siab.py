import pandas as pd
from typing import Optional


def filter_to_cohort_people(
    df_raw: pd.DataFrame,
    entry_month_start: str,
    entry_month_end: Optional[str] = None,
) -> pd.DataFrame:
    """
    Pre-filter raw data to only people who enter unemployment in target month range.
    
    This keeps ALL spells for the identified people, not just their
    unemployment spells, because biographical variables need full employment history.
    """
    # If no end month specified, use start month
    if entry_month_end is None:
        entry_month_end = entry_month_start

    if entry_month_start == entry_month_end:
        print(f"PRE-FILTERING TO COHORT: {entry_month_start}")
    else:
        print(f"PRE-FILTERING TO COHORT: {entry_month_start} to {entry_month_end}")

    original_size = len(df_raw)
    print(f"Original dataset: {original_size:,} rows")

    # Step 1: Identify people who entered unemployment in target month range
    # We look for ASU spells (quelle_gr == 5) with job seeking status (erwstat_gr == 21)

    # Convert date column if not already datetime
    if df_raw['begepi'].dtype != 'datetime64[ns]':
        print("Converting episode_start_date to datetime...")
        df_raw['begepi'] = pd.to_datetime(df_raw['begepi'])

    if entry_month_start == entry_month_end:
        print(f"Identifying unemployment entries in {entry_month_start}...")
    else:
        print(f"Identifying unemployment entries from {entry_month_start} to {entry_month_end}...")

    # Find unemployment entries in target month range
    unemployment_entries = df_raw[
        (df_raw['quelle_gr'] == 5) &  # ASU/XASU source
        (df_raw['erwstat_gr'] == 21) &  # Job seeking while unemployed
        (df_raw['begepi'].dt.to_period('M') >= entry_month_start) &
        (df_raw['begepi'].dt.to_period('M') <= entry_month_end)
    ].copy()

    if len(unemployment_entries) == 0:
        if entry_month_start == entry_month_end:
            print(f"WARNING: No unemployment entries found in {entry_month_start}!")
        else:
            print(f"WARNING: No unemployment entries found from {entry_month_start} to {entry_month_end}!")
        return df_raw  # Return original data

    # Get unique person IDs
    target_person_ids = unemployment_entries['persnr_siab_r'].unique()
    n_people = len(target_person_ids)

    if entry_month_start == entry_month_end:
        print(f"Found {n_people:,} people entering unemployment in {entry_month_start}")
    else:
        print(f"Found {n_people:,} people entering unemployment from {entry_month_start} to {entry_month_end}")
    print(f"  (from {len(unemployment_entries):,} unemployment spell entries)")

    # Step 2: Keep ALL spells for these people
    print(f"Filtering to all spells for these {n_people:,} people...")
    df_filtered = df_raw[df_raw['persnr_siab_r'].isin(target_person_ids)].copy()

    filtered_size = len(df_filtered)
    reduction_pct = (1 - filtered_size / original_size) * 100

    print(f"  Original: {original_size:,} rows")
    print(f"  Filtered: {filtered_size:,} rows")
    print(f"  Reduction: {reduction_pct:.1f}%")
    print(f"  Average spells per person: {filtered_size / n_people:.1f}")

    return df_filtered


def preprocess_siab_data(df_raw: pd.DataFrame) -> pd.DataFrame:
   
    from src.feature_engineering_siab import (
        identify_unemployment_episodes,
        generate_biographical_variables,
        index_unemployment_entries
    )
    from src.column_mapping import map_siab_columns

    print("Mapping column names...")
    df = map_siab_columns(df_raw)

    print("Identifying unemployment episodes...")
    df = identify_unemployment_episodes(df)

    print("Generating biographical variables...")
    df = generate_biographical_variables(df)

    print("Indexing unemployment entries...")
    df = index_unemployment_entries(df)

    return df
