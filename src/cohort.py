import pandas as pd
import numpy as np
from typing import Tuple, Optional


def define_cohort(
    df: pd.DataFrame,
    entry_month: Optional[str] = None,
    entry_month_start: Optional[str] = None,
    entry_month_end: Optional[str] = None,
    train_size: float = 0.8,
    random_seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Define cohort and split into train/test sets at the person level.
    """
    # Validate inputs
    if entry_month is not None and (entry_month_start is not None or entry_month_end is not None):
        raise ValueError("Cannot specify both entry_month and entry_month_start/entry_month_end")

    if entry_month is None and (entry_month_start is None or entry_month_end is None):
        raise ValueError("Must specify either entry_month OR both entry_month_start and entry_month_end")

    # Convert date column
    df['beg_ue_date'] = pd.to_datetime(df['beg_ue_date'])

    # Create filter based on input type
    if entry_month is not None:
        # Single month filter
        date_filter = df['beg_ue_date'].dt.to_period('M') == entry_month
        time_label = entry_month
    else:
        # Time range filter
        start_period = pd.Period(entry_month_start, freq='M')
        end_period = pd.Period(entry_month_end, freq='M')

        if start_period > end_period:
            raise ValueError(f"entry_month_start ({entry_month_start}) must be before or equal to entry_month_end ({entry_month_end})")

        periods = df['beg_ue_date'].dt.to_period('M')
        date_filter = (periods >= start_period) & (periods <= end_period)
        time_label = f"{entry_month_start} to {entry_month_end}"


    print(f"DEFINING COHORT: {time_label}")

    # Apply filter - include ue_duration to track the cohort spell
    cohort = df[
        (df['beg_ue'] == 1) &
        date_filter
    ][['person_id', 'beg_ue_date', 'spell_count', 'ue_duration']].copy()

    # Remove duplicates (in case someone has multiple unemployment starts in the time period)
    cohort = cohort.drop_duplicates(subset='person_id', keep='first')

    initial_count = len(cohort)
    print(f"Initial cohort: {initial_count:,} individuals")

    # Filter out unemployment spells with 0 duration
    cohort = cohort[cohort['ue_duration'] > 0].copy()

    filtered_count = len(cohort)
    if filtered_count < initial_count:
        print(f"Filtered out {initial_count - filtered_count} individuals with ue_duration=0")
        print(f"After filtering: {filtered_count:,} individuals")

    # Split into train/test
    n_train = int(len(cohort) * train_size)

    # Shuffle and split
    cohort_shuffled = cohort.sample(frac=1, random_state=random_seed).reset_index(drop=True)
    train_cohort = cohort_shuffled.iloc[:n_train].copy()
    test_cohort = cohort_shuffled.iloc[n_train:].copy()

    print(f"\nSplit into:")
    print(f"  Train cohort: {len(train_cohort):,} individuals ({train_size:.0%})")
    print(f"  Test cohort:  {len(test_cohort):,} individuals ({1-train_size:.0%})")

    return train_cohort, test_cohort


def cohort_stability(
    df_t0: pd.DataFrame,
    df_t1: pd.DataFrame,
    feature_cols: Optional[list] = None,
    outcome_cols: Optional[list] = None,
    verbose: bool = True
) -> dict:
    """
    Analyze feature stability between two time points for the same cohort.
    """
    
    # Ensure both dataframes have person_id
    if 'person_id' not in df_t0.columns or 'person_id' not in df_t1.columns:
        raise ValueError("Both dataframes must have 'person_id' column")

    # Find common person_ids
    person_ids_t0 = set(df_t0['person_id'].unique())
    person_ids_t1 = set(df_t1['person_id'].unique())
    common_person_ids = person_ids_t0.intersection(person_ids_t1)

    if len(common_person_ids) == 0:
        raise ValueError("No common person_ids found between the two dataframes")

    # Filter to common person_ids and sort by person_id
    df_t0_common = df_t0[df_t0['person_id'].isin(common_person_ids)].sort_values('person_id').reset_index(drop=True)
    df_t1_common = df_t1[df_t1['person_id'].isin(common_person_ids)].sort_values('person_id').reset_index(drop=True)

    # Determine feature columns to check
    if feature_cols is None:
        # Use all columns except person_id and outcome columns
        exclude_cols = {'person_id'}
        if outcome_cols:
            exclude_cols.update(outcome_cols)
        feature_cols = [col for col in df_t0_common.columns if col not in exclude_cols]

    # Check that all specified columns exist in both dataframes
    missing_in_t0 = set(feature_cols) - set(df_t0_common.columns)
    missing_in_t1 = set(feature_cols) - set(df_t1_common.columns)
    if missing_in_t0 or missing_in_t1:
        raise ValueError(f"Columns missing in t0: {missing_in_t0}, missing in t1: {missing_in_t1}")

    # Initialize result dictionary
    result = {
        'total_count': len(common_person_ids),
        'feature_change_counts': {},
        'outcome_change_counts': {}
    }

    # Track which person_ids have any feature changes
    person_has_change = pd.Series(False, index=df_t0_common.index)

    # Check each feature column for changes
    for col in feature_cols:
        # Convert categorical to string for comparison (categoricals need same categories to compare)
        if pd.api.types.is_categorical_dtype(df_t0_common[col]):
            col_t0 = df_t0_common[col].astype(str)
            col_t1 = df_t1_common[col].astype(str)
        else:
            col_t0 = df_t0_common[col]
            col_t1 = df_t1_common[col]

        # Compare values (handling NaN as equal)
        changed = ~(
            (col_t0 == col_t1) |
            (col_t0.isna() & col_t1.isna())
        )

        change_count = changed.sum()
        result['feature_change_counts'][col] = change_count

        # Update person-level change tracking
        person_has_change = person_has_change | changed

    # Identify stable vs unstable person_ids
    stable_mask = ~person_has_change
    result['stable_person_ids'] = set(df_t0_common.loc[stable_mask, 'person_id'].values)
    result['unstable_person_ids'] = set(df_t0_common.loc[~stable_mask, 'person_id'].values)
    result['stable_count'] = len(result['stable_person_ids'])
    result['stable_fraction'] = result['stable_count'] / result['total_count'] if result['total_count'] > 0 else 0

    # Check outcome columns separately (if provided)
    if outcome_cols:
        for col in outcome_cols:
            if col in df_t0_common.columns and col in df_t1_common.columns:
                # Convert categorical to string for comparison
                if pd.api.types.is_categorical_dtype(df_t0_common[col]):
                    outcome_t0 = df_t0_common[col].astype(str)
                    outcome_t1 = df_t1_common[col].astype(str)
                else:
                    outcome_t0 = df_t0_common[col]
                    outcome_t1 = df_t1_common[col]

                changed = ~(
                    (outcome_t0 == outcome_t1) |
                    (outcome_t0.isna() & outcome_t1.isna())
                )
                result['outcome_change_counts'][col] = changed.sum()



    print(f"\nTotal individuals: {result['total_count']}")
    print(f"Stable individuals: {result['stable_count']} ({result['stable_fraction']:.1%})")
    print(f"Unstable individuals: {len(result['unstable_person_ids'])} ({1-result['stable_fraction']:.1%})")

    print("FEATURES WITH MOST CHANGES:")
   
    # Sort features by change count
    sorted_features = sorted(result['feature_change_counts'].items(), key=lambda x: x[1], reverse=True)
    for col, count in sorted_features[:20]:  # Show top 20
        pct = count / result['total_count'] * 100
        print(f"{col:40s}: {count:6d} changes ({pct:5.1f}%)")

    if outcome_cols and result['outcome_change_counts']:
        print("OUTCOME CHANGES:")
        for col, count in result['outcome_change_counts'].items():
            pct = count / result['total_count'] * 100
            print(f"{col:40s}: {count:6d} changes ({pct:5.1f}%)")


    return result