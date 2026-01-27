import pandas as pd
import numpy as np

def generate_covariates_for_cohort_at_timepoint(
    df: pd.DataFrame,
    cohort: pd.DataFrame,
    time_offset_days: int = 0,
    outcome_horizon_days: int = 60
) -> pd.DataFrame:
    """
    Generate employment history covariates for a cohort at a specific timepoint.        ... )
    """
    
    # ensure necessary date columns are datetime
    date_columns = [
        "beg_ue_date",
        "episode_start_date",
        "episode_end_date",
        "start_employment1",
        "lm_contact_date",
        "ft_lm_contact_date"
    ]
    for col in date_columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col])

    # Create person-specific reference dates and preserve cohort spell duration
    cohort_work = cohort[['person_id', 'beg_ue_date', 'ue_duration']].copy()
    cohort_work['reference_date'] = pd.to_datetime(cohort_work['beg_ue_date']) + pd.Timedelta(days=time_offset_days)
    cohort_work['reference_date'] = pd.to_datetime(cohort_work['reference_date'])
    cohort_work['cohort_ue_duration'] = cohort_work['ue_duration']  # Rename to avoid confusion with spell-level ue_duration
    cohort_work = cohort_work.drop(columns=['ue_duration'])
    
    
    df_work = df.copy()
    
    # =========================================================================
    # STEP 1: FILTER TO COHORT AND MERGE INDIVIDUAL REFERENCE DATES
    # =========================================================================

    # Filter to cohort people
    cohort_person_ids = cohort_work['person_id'].tolist()
    df_work = df_work[df_work['person_id'].isin(cohort_person_ids)].copy()

    # Merge individual reference dates
    df_work = df_work.merge(cohort_work[['person_id', 'reference_date']], on='person_id', how='inner')

    # Filter to spells starting before each person's individual reference date
    df_work = df_work[df_work['episode_start_date'] <= df_work['reference_date']].copy()

    if len(df_work) == 0:
        # No historical data for this cohort
        print(f"  WARNING: No historical spells found for cohort")
        return pd.DataFrame()

    print(f"  Processing {len(cohort_person_ids):,} people with {len(df_work):,} historical spells")

    # =========================================================================
    # STEP 2: SPELL TYPE INDICATORS
    # =========================================================================

    df_work['lhgspell'] = (df_work['spell_source'] == 3).astype(int)
    df_work['behspell'] = (df_work['spell_source'] == 1).astype(int)
    df_work['lehspell'] = (df_work['spell_source'] == 2).astype(int)
    df_work['asuspell'] = (df_work['spell_source'] == 5).astype(int)
    df_work['mthspell'] = (df_work['spell_source'] == 4).astype(int)

    # =========================================================================
    # STEP 3: STATUS BEFORE (within 42 days before UE spell OR reference date)
    # =========================================================================
    # - For UNEMPLOYED people: "before" = before their current active UE spell started
    # - For NOT UNEMPLOYED people: "before" = before reference date

    ue_spells_all = df[df['person_id'].isin(cohort_person_ids) & (df['beg_ue'] == 1)].copy()

    # Merge reference dates for each person
    ue_spells_all = ue_spells_all.merge(
        cohort_work[['person_id', 'reference_date']],
        on='person_id',
        how='inner'
    )

    ue_spells_all['beg_ue_date'] = pd.to_datetime(ue_spells_all['beg_ue_date'])
    ue_spells_all['end_ue_date'] = pd.to_datetime(ue_spells_all['end_ue_date'])

    # Find active UE spells at reference date
    # Active means: started on/before reference_date AND still ongoing at reference_date
    # Uses end_ue_date from feature_engineering_siab.py for consistency
    ue_spells_all['spell_active'] = (
        (ue_spells_all['beg_ue_date'] <= ue_spells_all['reference_date']) &
        (ue_spells_all['reference_date'] < ue_spells_all['end_ue_date'])
    )

    # Get active UE spell info (one per person)
    # If multiple active spells (rare), keep the one with longest remaining duration
    active_ue_spells = ue_spells_all[ue_spells_all['spell_active']][
        ['person_id', 'beg_ue_date', 'end_ue_date', 'reference_date']
    ].copy()
    active_ue_spells['days_remaining'] = (
        active_ue_spells['end_ue_date'] - active_ue_spells['reference_date']
    ).dt.days
    active_ue_spells = active_ue_spells.sort_values('days_remaining', ascending=False)
    active_ue_spells = active_ue_spells.drop_duplicates(subset='person_id', keep='first')

    # Rename for clarity
    active_ue_spells = active_ue_spells.rename(columns={
        'beg_ue_date': 'active_ue_start_date',
        'end_ue_date': 'active_ue_end_date',
        'days_remaining': 'days_remaining_in_spell'
    })
    active_ue_spells = active_ue_spells[['person_id', 'active_ue_start_date', 'active_ue_end_date', 'days_remaining_in_spell']]

    # Merge active UE spell info back to df_work
    df_work = df_work.merge(active_ue_spells, on='person_id', how='left')

    # Create lookup_date: use active UE start if unemployed, otherwise reference_date
    df_work['lookup_date_for_before'] = df_work['active_ue_start_date'].fillna(df_work['reference_date'])

    # Keep spells that STARTED before each person's lookup date
    before_df = df_work[df_work['episode_start_date'] < df_work['lookup_date_for_before']].copy()

    if len(before_df) > 0:
        # Get most recent spell before lookup date for each person
        before_df = before_df.sort_values(['person_id', 'episode_start_date'])
        before_df['status_beg_date'] = before_df.groupby('person_id')['episode_start_date'].transform('max')
        before_df = before_df[before_df['episode_start_date'] == before_df['status_beg_date']].copy()

        # Cap episode_end_date at lookup_date (can't use future information)
        # If a spell is ongoing (extends past lookup_date), we cap it
        before_df['episode_end_date_capped'] = before_df[['episode_end_date', 'lookup_date_for_before']].min(axis=1)

        # Time since last spell (using lookup dates and capped end dates)
        before_df['t_since_spell'] = (
            before_df['lookup_date_for_before'] - before_df['episode_end_date_capped'] + pd.Timedelta(days=1)
        ).dt.days

        # Status indicators (only if within 42 days)
        before_df['employed'] = np.where(
            before_df['t_since_spell'] <= 42,
            before_df.groupby('person_id')['behspell'].transform('max'),
            0
        )
        before_df['receipt_leh'] = np.where(
            before_df['t_since_spell'] <= 42,
            before_df.groupby('person_id')['lehspell'].transform('max'),
            0
        )
        before_df['receipt_lhg'] = np.where(
            before_df['t_since_spell'] <= 42,
            before_df.groupby('person_id')['lhgspell'].transform('max'),
            0
        )

        # Subsidized employment
        if 'se_ep' in before_df.columns:
            before_df['se'] = np.where(
                before_df['t_since_spell'] <= 42,
                before_df.groupby('person_id')['se_ep'].transform('max'),
                0
            )
        else:
            before_df['se'] = 0

        # ASU status
        before_df['employment_status_group'] = before_df['employment_status_group'].fillna(-1)
        before_df['ASU_notue_seeking_help'] = (before_df['employment_status_group'] == 23).astype(int)
        before_df['ASU_notue_seeking'] = np.where(
            before_df['t_since_spell'] <= 42,
            before_df.groupby('person_id')['ASU_notue_seeking_help'].transform('max'),
            0
        )
        before_df['ASU_other_help'] = np.where(
            before_df['t_since_spell'] <= 42,
            before_df.groupby('person_id')['asuspell'].transform('max'),
            0
        )
        before_df['ASU_other'] = np.where(
            (before_df['ASU_other_help'] == 1) & (before_df['ASU_notue_seeking'] != 1),
            1,
            0
        )

        before_df['break'] = (before_df['t_since_spell'] > 42).astype(int)

        # Aggregate to person level
        before_cols = ['employed', 'receipt_leh', 'receipt_lhg', 'se',
                       'ASU_notue_seeking', 'ASU_other', 'break']
        status_before_vars = before_df.groupby('person_id').first()[before_cols].reset_index()
        status_before_vars.columns = ['person_id'] + [f'{col}_before' for col in before_cols]

        # Merge back to spell-level data (line 1056 in original)
        df_work = df_work.merge(status_before_vars, on='person_id', how='left')

        # Fill missing with defaults (lines 1058-1069)
        for col in ['employed_before', 'receipt_leh_before', 'receipt_lhg_before',
                    'se_before', 'ASU_notue_seeking_before', 'ASU_other_before']:
            if col in df_work.columns:
                df_work[col] = df_work[col].fillna(0)
        if 'break_before' in df_work.columns:
            df_work['break_before'] = df_work['break_before'].fillna(1)
    else:
        # No spells before reference date (lines 1070-1080)
        for col in ['employed_before', 'receipt_leh_before', 'receipt_lhg_before',
                    'se_before', 'ASU_notue_seeking_before', 'ASU_other_before']:
            df_work[col] = 0
        df_work['break_before'] = 1

    # =========================================================================
    # STEP 4: NUMBER OF JOBS AND ESTABLISHMENTS
    # =========================================================================
    # Use lookup_date_for_before for stability (employment history frozen at UE start for unemployed)

    jobs_df = df_work[
        (df_work['episode_start_date'] < df_work['lookup_date_for_before']) &
        (df_work['spell_source'] == 1)
    ].copy()

    if len(jobs_df) > 0 and 'ein_job' in jobs_df.columns:
        jobs_df = jobs_df.sort_values(['person_id', 'ein_job', 'establishment_id', 'episode_start_date'])

        # Count unique jobs (ein_job × establishment combinations)
        jobs_df['nrE'] = jobs_df.groupby(['person_id', 'ein_job', 'establishment_id']).cumcount() + 1
        jobs_df = jobs_df[jobs_df['nrE'] == 1].copy()
        jobs_df['emp_total'] = jobs_df.groupby('person_id')['person_id'].transform('count')

        # IMPORTANT: Save jobs_df before filtering to establishments for emp1_total calculation
        jobs_df_unique_jobs = jobs_df.copy()

        # Count unique establishments (use nunique, not row count)
        if 'establishment_id' in jobs_df.columns:
            jobs_df['est_total'] = jobs_df.groupby('person_id')['establishment_id'].transform('nunique')

        # Jobs without vocational training (use unique jobs, not establishment-filtered data)
        jobs_df_no_voc = (
            jobs_df_unique_jobs[jobs_df_unique_jobs['apprentice'] != 1].copy()
            if 'apprentice' in jobs_df_unique_jobs.columns
            else jobs_df_unique_jobs.copy()
        )
        if len(jobs_df_no_voc) > 0:
            jobs_df_no_voc = jobs_df_no_voc.sort_values(
                ['person_id', 'ein_job', 'establishment_id', 'episode_start_date']
            )
            jobs_df_no_voc['nrE1'] = (
                jobs_df_no_voc.groupby(['person_id', 'ein_job', 'establishment_id']).cumcount() + 1
            )
            jobs_df_no_voc = jobs_df_no_voc[jobs_df_no_voc['nrE1'] == 1].copy()
            jobs_df_no_voc['emp1_total'] = jobs_df_no_voc.groupby('person_id')['person_id'].transform('count')
        else:
            jobs_df_no_voc = pd.DataFrame(columns=['person_id', 'emp1_total'])

        # Aggregate to person level
        jobs_summary = jobs_df.groupby('person_id').first()[
            [col for col in ['est_total', 'emp_total'] if col in jobs_df.columns]
        ].reset_index()

        emp1_summary = jobs_df_no_voc[['person_id', 'emp1_total']].drop_duplicates(
            subset='person_id', keep='first'
        ) if len(jobs_df_no_voc) > 0 else pd.DataFrame(columns=['person_id', 'emp1_total'])

        jobs_vars = jobs_summary.merge(emp1_summary, on='person_id', how='left')

        # Merge back to spell-level data (line 1141 in original)
        df_work = df_work.merge(jobs_vars, on='person_id', how='left')

        for col in ['est_total', 'emp_total', 'emp1_total']:
            if col in df_work.columns:
                df_work[col] = df_work[col].fillna(0)
    else:
        # No jobs (lines 1147-1148)
        for col in ['est_total', 'emp_total', 'emp1_total']:
            df_work[col] = 0

    # =========================================================================
    # STEP 5: CUMULATIVE EMPLOYMENT DURATION
    # =========================================================================
    # Calculate cumulative durations based on spells before lookup_date_for_before
    # Durations are capped at lookup_date_for_before to avoid counting future days
    # Uses lookup_date_for_before for stability (frozen at UE start for unemployed people)

    # Calculate emp1_total_dur: total days in non-apprentice employment
    emp1_dur_df = df_work[
        (df_work["episode_start_date"] < df_work["lookup_date_for_before"]) &
        (df_work["emp1"] == 1)
    ].copy()

    if len(emp1_dur_df) > 0:
        # Cap episode_end_date at lookup_date_for_before (can't count future days)
        emp1_dur_df["episode_end_date_capped"] = emp1_dur_df[["episode_end_date", "lookup_date_for_before"]].min(axis=1)
        emp1_dur_df["spell_duration"] = (
            emp1_dur_df["episode_end_date_capped"] - emp1_dur_df["episode_start_date"] + pd.Timedelta(days=1)
        ).dt.days

        # Sum up durations per person
        emp1_total_dur = emp1_dur_df.groupby("person_id")["spell_duration"].sum().reset_index()
        emp1_total_dur.columns = ["person_id", "emp1_total_dur"]
        df_work = df_work.merge(emp1_total_dur, on="person_id", how="left")

    # Calculate emp2_total_dur: total days in non-apprentice, non-internship employment
    if "emp2" in df_work.columns:
        emp2_dur_df = df_work[
            (df_work["episode_start_date"] < df_work["lookup_date_for_before"]) &
            (df_work["emp2"] == 1)
        ].copy()

        if len(emp2_dur_df) > 0:
            emp2_dur_df["episode_end_date_capped"] = emp2_dur_df[["episode_end_date", "lookup_date_for_before"]].min(axis=1)
            emp2_dur_df["spell_duration"] = (
                emp2_dur_df["episode_end_date_capped"] - emp2_dur_df["episode_start_date"] + pd.Timedelta(days=1)
            ).dt.days
            emp2_total_dur = emp2_dur_df.groupby("person_id")["spell_duration"].sum().reset_index()
            emp2_total_dur.columns = ["person_id", "emp2_total_dur"]
            df_work = df_work.merge(emp2_total_dur, on="person_id", how="left")

    # Calculate emp3_total_dur: total days in regular employment (emp3)
    if "emp3" in df_work.columns:
        emp3_dur_df = df_work[
            (df_work["episode_start_date"] < df_work["lookup_date_for_before"]) &
            (df_work["emp3"] == 1)
        ].copy()

        if len(emp3_dur_df) > 0:
            emp3_dur_df["episode_end_date_capped"] = emp3_dur_df[["episode_end_date", "lookup_date_for_before"]].min(axis=1)
            emp3_dur_df["spell_duration"] = (
                emp3_dur_df["episode_end_date_capped"] - emp3_dur_df["episode_start_date"] + pd.Timedelta(days=1)
            ).dt.days
            emp3_total_dur = emp3_dur_df.groupby("person_id")["spell_duration"].sum().reset_index()
            emp3_total_dur.columns = ["person_id", "emp3_total_dur"]
            df_work = df_work.merge(emp3_total_dur, on="person_id", how="left")

    # Calculate average duration per job (emp1_m_dur)
    if "emp1_total_dur" in df_work.columns and "emp1_total" in df_work.columns:
        df_work["emp1_m_dur"] = df_work["emp1_total_dur"] / df_work["emp1_total"].replace(
            0, np.nan
        )
        df_work["emp1_m_dur"] = df_work["emp1_m_dur"].fillna(0)

    # =========================================================================
    # STEP 5.5: BENEFIT RECEIPT HISTORY (LHG and LEH)
    # =========================================================================
    # Calculate aggregates for benefit receipt spells before lookup_date_for_before

    # LHG benefits (spell_source == 3)
    lhg_df = df_work[
        (df_work['episode_start_date'] < df_work['lookup_date_for_before']) &
        (df_work['spell_source'] == 3)
    ].copy()

    if len(lhg_df) > 0:
        # Cap episode_end_date at lookup_date_for_before (can't use future information)
        lhg_df['episode_end_date_capped'] = lhg_df[['episode_end_date', 'lookup_date_for_before']].min(axis=1)

        # Calculate duration for each spell (capped at lookup_date_for_before)
        lhg_df['lhg_spell_dur'] = (
            lhg_df['episode_end_date_capped'] - lhg_df['episode_start_date']
        ).dt.days + 1

        # Count number of LHG benefit spells
        lhg_df['LHG_total'] = lhg_df.groupby('person_id')['person_id'].transform('count')

        # Total duration across all spells (up to lookup_date_for_before)
        lhg_df['LHG_tot_dur'] = lhg_df.groupby('person_id')['lhg_spell_dur'].transform('sum')

        # Mean duration per spell
        lhg_df['LHG_m_dur'] = lhg_df['LHG_tot_dur'] / lhg_df['LHG_total']

        # Aggregate to person level
        lhg_summary = lhg_df.groupby('person_id').first()[
            ['LHG_total', 'LHG_tot_dur', 'LHG_m_dur']
        ].reset_index()

        df_work = df_work.merge(lhg_summary, on='person_id', how='left')

    # Fill missing with 0
    for col in ['LHG_total', 'LHG_tot_dur', 'LHG_m_dur']:
        if col not in df_work.columns:
            df_work[col] = 0
        else:
            df_work[col] = df_work[col].fillna(0)

    # LEH benefits (spell_source == 2)
    leh_df = df_work[
        (df_work['episode_start_date'] < df_work['lookup_date_for_before']) &
        (df_work['spell_source'] == 2)
    ].copy()

    if len(leh_df) > 0:
        # Cap episode_end_date at lookup_date_for_before (can't use future information)
        leh_df['episode_end_date_capped'] = leh_df[['episode_end_date', 'lookup_date_for_before']].min(axis=1)

        # Calculate duration for each spell (capped at lookup_date_for_before)
        leh_df['leh_spell_dur'] = (
            leh_df['episode_end_date_capped'] - leh_df['episode_start_date']
        ).dt.days + 1

        # Count number of LEH benefit spells
        leh_df['LEH_total'] = leh_df.groupby('person_id')['person_id'].transform('count')

        # Total duration across all spells (up to lookup_date_for_before)
        leh_df['LEH_tot_dur'] = leh_df.groupby('person_id')['leh_spell_dur'].transform('sum')

        # Mean duration per spell
        leh_df['LEH_m_dur'] = leh_df['LEH_tot_dur'] / leh_df['LEH_total']

        # Aggregate to person level
        leh_summary = leh_df.groupby('person_id').first()[
            ['LEH_total', 'LEH_tot_dur', 'LEH_m_dur']
        ].reset_index()

        df_work = df_work.merge(leh_summary, on='person_id', how='left')

    # Fill missing with 0
    for col in ['LEH_total', 'LEH_tot_dur', 'LEH_m_dur']:
        if col not in df_work.columns:
            df_work[col] = 0
        else:
            df_work[col] = df_work[col].fillna(0)

    # =========================================================================
    # STEP 6: TIME SINCE FIRST EMPLOYMENT & LAST LABOR MARKET CONTACT
    # =========================================================================
    # Calculate time-since variables using reference_date to preserve economic meaning
    # (how disconnected from labor market).

    # Find first employment spell before reference_date for each person
    emp1_before_df = df_work[
        (df_work['episode_start_date'] < df_work['reference_date']) &
        (df_work['emp1'] == 1)
    ].copy()

    if len(emp1_before_df) > 0:
        # Calculate first employment date as minimum episode_start_date before reference_date
        first_emp_dates = emp1_before_df.groupby('person_id')['episode_start_date'].min().reset_index()
        first_emp_dates.columns = ['person_id', 'start_employment1_at_ref']

        # Merge back to main df
        df_work = df_work.merge(first_emp_dates, on='person_id', how='left')

        # Calculate time since first employment using reference_date
        df_work["tsince_ein_erw1"] = (
            df_work["reference_date"] - df_work["start_employment1_at_ref"]
        ).dt.days
    else:
        # No employment before reference_date for anyone
        df_work["tsince_ein_erw1"] = np.nan

    # Create categorical version with large bins: <1yr, 1-3yr, 3-5yr, 5+yr
    # Someone unemployed for 180 days stays in <1yr bin throughout
    df_work["tsince_ein_erw1_cat"] = 99999  # Missing/never employed
    df_work.loc[
        (df_work["tsince_ein_erw1"] > 0) & (df_work["tsince_ein_erw1"] <= 365),
        "tsince_ein_erw1_cat",
    ] = 1  # <1 year
    df_work.loc[
        (df_work["tsince_ein_erw1"] > 365) & (df_work["tsince_ein_erw1"] <= 1095),
        "tsince_ein_erw1_cat",
    ] = 2  # 1-3 years
    df_work.loc[
        (df_work["tsince_ein_erw1"] > 1095) & (df_work["tsince_ein_erw1"] <= 1825),
        "tsince_ein_erw1_cat",
    ] = 3  # 3-5 years
    df_work.loc[df_work["tsince_ein_erw1"] > 1825, "tsince_ein_erw1_cat"] = 4  # 5+ years

    # time since last contact with labour market (using reference_date)
    lm_contact_df = df_work[
        (df_work["episode_start_date"] < df_work["reference_date"]) & (df_work["behspell"] == 1)
    ].copy()

    if len(lm_contact_df) > 0:
        lm_contact_df["lm_contact_date"] = lm_contact_df.groupby("person_id")[
            "episode_end_date"
        ].transform("max")
        if "part_time" in lm_contact_df.columns:
            lm_contact_df["ft_lm_contact_date"] = (
                lm_contact_df[lm_contact_df["part_time"] != 1]
                .groupby("person_id")["episode_end_date"]
                .transform("max")
            )

        lm_cols = ["lm_contact_date"]
        if "ft_lm_contact_date" in lm_contact_df.columns:
            lm_cols.append("ft_lm_contact_date")
        lm_contact_summary = (
            lm_contact_df.groupby("person_id").first()[lm_cols].reset_index()
        )
        df_work = df_work.merge(lm_contact_summary, on="person_id", how="left")

        # Cap lm_contact_date at reference_date (can't count employment after reference timepoint)
        df_work["lm_contact_date"] = df_work[["lm_contact_date", "reference_date"]].min(axis=1)
        df_work["tsince_lm_contact"] = (df_work["reference_date"] - df_work["lm_contact_date"]).dt.days

        if "ft_lm_contact_date" in df_work.columns:
            df_work["ft_lm_contact_date"] = df_work[["ft_lm_contact_date", "reference_date"]].min(axis=1)
            df_work["tsince_ft_lm_contact"] = (
                df_work["reference_date"] - df_work["ft_lm_contact_date"]
            ).dt.days

        # Create categorical versions with large bins: <1yr, 1-3yr, 3-5yr, 5+yr
        for var in ["tsince_lm_contact", "tsince_ft_lm_contact"]:
            if var in df_work.columns:
                df_work[f"{var}_cat"] = 99999  # Missing
                df_work.loc[df_work[var] <= 365, f"{var}_cat"] = 1  # <1 year
                df_work.loc[(df_work[var] > 365) & (df_work[var] <= 1095), f"{var}_cat"] = 2  # 1-3 years
                df_work.loc[(df_work[var] > 1095) & (df_work[var] <= 1825), f"{var}_cat"] = 3  # 3-5 years
                df_work.loc[df_work[var] > 1825, f"{var}_cat"] = 4  # 5+ years


    # =========================================================================
    # STEP 6.5: LAST JOB CHARACTERISTICS
    # =========================================================================

    # Determine job duration column to use
    job_duration_col = None
    if 'tage_job' in df_work.columns:
        job_duration_col = 'tage_job'
    elif 'tage_job_kum' in df_work.columns:
        job_duration_col = 'tage_job_kum'

    lastjob_df = df_work[
        (df_work["episode_start_date"] < df_work["lookup_date_for_before"]) & (df_work["behspell"] == 1)
    ].copy()

    if len(lastjob_df) > 0 and job_duration_col is not None:
        # Cap episode_end_date at lookup_date_for_before (can't use future information)
        lastjob_df["episode_end_date_capped"] = lastjob_df[["episode_end_date", "lookup_date_for_before"]].min(axis=1)

        lastjob_df["last_job_end"] = np.where(
            lastjob_df["episode_end_date_capped"]
            == lastjob_df.groupby("person_id")["episode_end_date_capped"].transform("max"),
            1,
            0,
        )
        lastjob_df = lastjob_df[lastjob_df["last_job_end"] == 1].copy()

        # Calculate actual duration up to reference_date
        lastjob_df["lastjob_dur_actual"] = (
            lastjob_df["episode_end_date_capped"] - lastjob_df["episode_start_date"]
        ).dt.days + 1  # +1 to include both start and end day

        lastjob_df = lastjob_df.sort_values(
            ["person_id", "lastjob_dur_actual", "daily_wage"],
            ascending=[True, False, False],
        )
        lastjob_df["nrLEmp"] = lastjob_df.groupby("person_id").cumcount() + 1
        lastjob_df = lastjob_df[lastjob_df["nrLEmp"] == 1].copy()

        lastjob_cols = [
            "lastjob_dur_actual",  # Use calculated duration instead of tage_job
            "pg5",
            "part_time",
            "wage_defl",
            "occ_blo",
            "occupation_skill_level",
            "temporary_employment",
            "fixed_term_employment",
            "education_level_imputed",
            "district_region",
            "industry_destatis",
            "industry_classification",  # Alternative industry variable
        ]
        available_cols = [col for col in lastjob_cols if col in lastjob_df.columns]
        lastjob_summary = lastjob_df[["person_id"] + available_cols].copy()
        lastjob_summary = lastjob_summary.drop_duplicates(
            subset="person_id", keep="first"
        )
        rename_dict = {
            "lastjob_dur_actual": "lastjob_tot_dur",  # Rename calculated duration
            "pg5": "lastjob_type",
            "part_time": "lastjob_pt",
            "wage_defl": "lastjob_wage_defl",
            "occ_blo": "lastjob_occblo",
            "occupation_skill_level": "lastjob_niveau",
            "temporary_employment": "lastjob_leih",
            "fixed_term_employment": "lastjob_befrist",
            "education_level_imputed": "lastjob_educ",
            "district_region": "lastjob_ao_kreis",
            "industry_destatis": "lastjob_industry_destatis",
            "industry_classification": "lastjob_industry",  # Last industry from classification
        }
        lastjob_summary = lastjob_summary.rename(columns=rename_dict)
        df_work = df_work.merge(lastjob_summary, on="person_id", how="left")
        df_work["lastjob_none"] = df_work["lastjob_tot_dur"].isna().astype(int)

        for col in lastjob_summary.columns:
            if col != "person_id":
                if col in ["lastjob_tot_dur", "lastjob_wage_defl"]:
                    df_work.loc[df_work["lastjob_none"] == 1, col] = 0
                else:
                    df_work.loc[df_work["lastjob_none"] == 1, col] = 99999
                    df_work[col] = df_work[col].fillna(99999)
    else:
        # No employment spells or no job duration column - everyone has no last job
        df_work["lastjob_none"] = 1

    # =========================================================================
    # STEP 7: IMPUTE BIOGRAPHICAL VARIABLES (BEFORE COLLAPSING)
    # =========================================================================
    # These handle time-varying biographical variables by taking max or most recent
    # IMPORTANT: Do this BEFORE collapsing to person-level, while we still have all spells

    # Filter to historical spells only (before lookup_date_for_before)
    historical_spells = df_work[df_work['episode_start_date'] < df_work['lookup_date_for_before']].copy()

    # For categorical variables, take max within person from historical spells
    # Max works because 99999 (missing) is recoded to -1, so any real value > -1
    for var in ['german_citizen', 'education_level_imputed', 'school_completed',
                'occupation_skill_level']:
        if var in historical_spells.columns:
            # Recode special missing values
            historical_spells[var] = historical_spells[var].replace({99999: -1, np.nan: -2})
            # Take max across person's historical spells
            var_max = historical_spells.groupby('person_id')[var].max().reset_index()
            var_max.columns = ['person_id', f'{var}_imputed']
            # Recode back
            var_max[f'{var}_imputed'] = var_max[f'{var}_imputed'].replace({-1: 99999, -2: np.nan})
            # Merge back to df_work
            df_work = df_work.merge(var_max, on='person_id', how='left')
            # Use imputed value
            df_work[var] = df_work[f'{var}_imputed']
            df_work = df_work.drop(columns=[f'{var}_imputed'])

    # For commuter and wage_group, take most recent (last) non-missing value from historical spells
    for var in ['commuter', 'daily_wage']:  # daily_wage is wage_group
        if var in historical_spells.columns:
            var_df = historical_spells[historical_spells[var].notna() & (historical_spells[var] != 99999)].copy()
            if len(var_df) > 0:
                var_df = var_df.sort_values(['person_id', 'episode_start_date'])
                var_df = var_df.groupby('person_id').last().reset_index()[['person_id', var]]
                # Use wage_group instead of daily_wage for consistency
                new_col_name = 'wage_group' if var == 'daily_wage' else var
                var_df.columns = ['person_id', new_col_name]
                df_work = df_work.merge(var_df, on='person_id', how='left', suffixes=('_old', ''))
                # Drop the old column if it exists
                if f'{new_col_name}_old' in df_work.columns:
                    df_work = df_work.drop(columns=[f'{new_col_name}_old'])

    # =========================================================================
    # STEP 8: COLLAPSE TO PERSON-LEVEL
    # =========================================================================
    # Based on transform.py line 127
    # Since all employment history covariates are person-level (replicated across spells),
    # we can take any row per person. Let's take the most recent spell.

    df_work = df_work.sort_values(['person_id', 'episode_start_date'], ascending=[True, False])
    person_level = df_work.groupby('person_id').first().reset_index()

    # =========================================================================
    # STEP 9: CREATE DERIVED VARIABLES
    # =========================================================================

    # Create Bundesland (state) variable from district_region if needed
    if 'district_region' in person_level.columns:
        person_level['bula'] = np.nan
        person_level.loc[(person_level['district_region'] < 2000) & person_level['district_region'].notna(), 'bula'] = 1
        person_level.loc[(person_level['district_region'] >= 2000) & (person_level['district_region'] < 3000), 'bula'] = 2
        person_level.loc[(person_level['district_region'] >= 3000) & (person_level['district_region'] < 4000), 'bula'] = 3
        person_level.loc[(person_level['district_region'] >= 4000) & (person_level['district_region'] < 5000), 'bula'] = 4
        person_level.loc[(person_level['district_region'] >= 5000) & (person_level['district_region'] < 6000), 'bula'] = 5
        person_level.loc[(person_level['district_region'] >= 6000) & (person_level['district_region'] < 7000), 'bula'] = 6
        person_level.loc[(person_level['district_region'] >= 7000) & (person_level['district_region'] < 8000), 'bula'] = 7
        person_level.loc[(person_level['district_region'] >= 8000) & (person_level['district_region'] < 9000), 'bula'] = 8
        person_level.loc[(person_level['district_region'] >= 9000) & (person_level['district_region'] < 10000), 'bula'] = 9
        person_level.loc[(person_level['district_region'] >= 10000) & (person_level['district_region'] < 11000), 'bula'] = 10
        person_level.loc[(person_level['district_region'] >= 11000) & (person_level['district_region'] < 12000), 'bula'] = 11
        person_level.loc[(person_level['district_region'] >= 12000) & (person_level['district_region'] < 13000), 'bula'] = 12
        person_level.loc[(person_level['district_region'] >= 13000) & (person_level['district_region'] < 14000), 'bula'] = 13
        person_level.loc[(person_level['district_region'] >= 14000) & (person_level['district_region'] < 15000), 'bula'] = 14
        person_level.loc[(person_level['district_region'] >= 15000) & (person_level['district_region'] < 16000), 'bula'] = 15
        person_level.loc[person_level['district_region'] >= 16000, 'bula'] = 16

    # =========================================================================
    # STEP 10: FILL MISSING VALUES
    # =========================================================================

    # fill missing values for various variable groups
    variable_groups = {
        "status_before": [
            "employed_before",
            "receipt_leh_before",
            "receipt_lhg_before",
            "se_before",
            "ASU_notue_seeking_before",
            "ASU_other_before",
            "break_before",
        ],
        "lastcontact_lm": [
            "tsince_ein_erw1",  # Will be filled with 99999 if no employment
            "tsince_ein_erw1_cat",
            "tsince_ft_lm_contact",
            "tsince_lm_contact",
            "tsince_lm_contact_cat",
            "tsince_ft_lm_contact_cat",
        ],
        "emp_total": [
            "emp_total",
            "est_total",
            "emp1_total",
            "emp1_total_dur",
            "emp2_total_dur",
            "emp3_total_dur",
            "emp1_m_dur",
            "main_industry",
            "secjob_tot_dur",
            "minijob_tot_dur",
            "ft_tot_dur",
            "befrist_tot_dur",
            "leih_tot_dur",
        ],
        "lastjob": [
            "lastjob_none",
            "lastjob_tot_dur",
            "lastjob_parallel",
            "lastjob_type",
            "lastjob_pt",
            "lastjob_wage_defl",
            "lastjob_occblo",
            "lastjob_niveau",
            "lastjob_leih",
            "lastjob_befrist",
            "lastjob_educ",
            "lastjob_ao_kreis",
            "lastjob_industry_destatis",
            "lastjob_industry",  # Last industry from industry_classification
        ],
        "benefit_hist": [
            "LHG_total",
            "LHG_tot_dur",
            "LHG_m_dur",
            "LEH_total",
            "LEH_tot_dur",
            "LEH_m_dur",
        ],
        "seeking_hist": [
            "tsince_lastseeking",
            "tsince_lastseeking_cat",
            "seeking1_total",
            "seeking1_tot_dur",
            "seeking1_m_dur",
        ],
        "bio": [
            "age",
            "female",
            "german",
            "german_citizen",
            "education_level_imputed",
            "school_completed",
            "occupation_skill_level",
            "commuter",
            "wage_group",
            "togerman",
            "moves",
            "moves_noinfo",
            "relocated",
            "wo_east",
            "bula",  # German state/Bundesland
        ],
    }

    # fill missing values with appropriate defaults
    for group, vars_list in variable_groups.items():
        for var in vars_list:
            if var in person_level.columns:
                if var in [
                    "emp_total",
                    "est_total",
                    "emp1_total",
                    "LHG_total",
                    "LEH_total",
                    "seeking1_total",
                    "moves",
                ]:
                    person_level[var] = person_level[var].fillna(0)
                elif var in [
                    "emp1_total_dur",
                    "emp2_total_dur",
                    "emp3_total_dur",
                    "emp1_m_dur",
                    "secjob_tot_dur",
                    "minijob_tot_dur",
                    "ft_tot_dur",
                    "befrist_tot_dur",
                    "leih_tot_dur",
                    "lastjob_tot_dur",
                    "lastjob_wage_defl",
                    "LHG_tot_dur",
                    "LHG_m_dur",
                    "LEH_tot_dur",
                    "LEH_m_dur",
                    "seeking1_tot_dur",
                    "seeking1_m_dur",
                ]:
                    person_level[var] = person_level[var].fillna(0)
                elif var in [
                    "main_industry",
                    "lastjob_type",
                    "lastjob_occblo",
                    "lastjob_niveau",
                    "lastjob_leih",
                    "lastjob_befrist",
                    "lastjob_educ",
                    "lastjob_ao_kreis",
                    "lastjob_industry_destatis",
                    "lastjob_industry",
                    "german",
                    "german_citizen",
                    "education_level_imputed",
                    "school_completed",
                    "occupation_skill_level",
                    "commuter",
                    "wage_group",
                    "bula",
                    "wo_east",
                    "relocated",
                    "tsince_ein_erw1",  # Days since first employment (99999 = no prior employment)
                ]:
                    person_level[var] = person_level[var].fillna(99999)
                elif var in [
                    "lastjob_parallel",
                    "lastjob_pt",
                    "employed_before",
                    "receipt_leh_before",
                    "receipt_lhg_before",
                    "se_before",
                    "ASU_notue_seeking_before",
                    "ASU_other_before",
                    "break_before",
                    "togerman",
                    "moves_noinfo",
                ]:
                    person_level[var] = person_level[var].fillna(0)
                elif var in [
                    "tsince_lastseeking_cat",
                    "tsince_lm_contact_cat",
                    "tsince_ft_lm_contact_cat",
                    "tsince_ein_erw1_cat",
                ]:
                    person_level[var] = person_level[var].fillna(99999)

    # select final variables to keep
    spell_original = [
        "person_id",
        "year",
        "spell_source",
        "episode_start_date",
        "episode_end_date",
    ]
    spell_generated = [ "nrEntry", "beg_ue", "ue_duration"]

    # keep all variables from variable groups plus response and imputed max variables
    keep_vars = spell_original + spell_generated

    for group_vars in variable_groups.values():
        keep_vars.extend([v for v in group_vars if v in person_level.columns])

    # Add outcome-related columns from Step 3's active spell detection
    outcome_vars = ['days_remaining_in_spell', 'active_ue_start_date', 'active_ue_end_date']
    keep_vars.extend([v for v in outcome_vars if v in person_level.columns])

    # Remove duplicates while preserving order
    seen = set()
    keep_vars = [v for v in keep_vars if not (v in seen or seen.add(v))]

    df_final = person_level[keep_vars].copy()

    # =========================================================================
    # STEP 11: OUTCOME VARIABLES
    # =========================================================================


    # 1. Binary indicator: Is person currently unemployed at reference date?
    # Person is unemployed if they have an active UE spell (days_remaining_in_spell is not NaN)
    df_final['still_unemployed'] = (~df_final['days_remaining_in_spell'].isna()).astype(int)

    # 2. Binary outcome: Will remain unemployed for at least outcome_horizon_days?
    # Only applicable if currently unemployed (otherwise automatically 0)
    df_final['remains_ue_horizon_days'] = (
        df_final['days_remaining_in_spell'].fillna(0) >= outcome_horizon_days
    ).astype(int)

    # Drop intermediate columns we don't need in final output
    cols_to_drop = ['active_ue_start_date', 'active_ue_end_date']
    df_final = df_final.drop(columns=[c for c in cols_to_drop if c in df_final.columns])

    # =========================================================================
    # STEP 12: BIN CONTINUOUS VARIABLES
    # =========================================================================
    # Create categorical versions of continuous variables for model stability

    # Bin age: <25, 25-34, 35-44, 45-54, 55+
    if 'age' in df_final.columns:
        df_final['age_cat'] = pd.cut(
            df_final['age'],
            bins=[-0.1, 25, 35, 45, 55, float('inf')],
            labels=[0, 1, 2, 3, 4],
            include_lowest=True
        ).astype('Int64')
        df_final['age_cat'] = df_final['age_cat'].fillna(99999)

    # Bin est_total (number of establishments): 0, 1-5, 6-15, 16-30, 31-50, 51+
    if 'est_total' in df_final.columns:
        df_final['est_total_cat'] = pd.cut(
            df_final['est_total'],
            bins=[-0.1, 0.5, 5.5, 15.5, 30.5, 50.5, float('inf')],
            labels=[0, 1, 2, 3, 4, 5],
            include_lowest=True
        ).astype('Int64')
        df_final['est_total_cat'] = df_final['est_total_cat'].fillna(99999)

    # Bin emp1_total (number of jobs): 0, 1-5, 6-15, 16-30, 31-50, 51+
    if 'emp1_total' in df_final.columns:
        df_final['emp1_total_cat'] = pd.cut(
            df_final['emp1_total'],
            bins=[-0.1, 0.5, 5.5, 15.5, 30.5, 50.5, float('inf')],
            labels=[0, 1, 2, 3, 4, 5],
            include_lowest=True
        ).astype('Int64')
        df_final['emp1_total_cat'] = df_final['emp1_total_cat'].fillna(99999)

    # Bin emp1_total_dur (total days in employment): 0, <1yr, 1-3yr, 3-5yr, 5yr+
    if 'emp1_total_dur' in df_final.columns:
        df_final['emp1_total_dur_cat'] = pd.cut(
            df_final['emp1_total_dur'],
            bins=[-0.1, 0.5, 365, 3*365, 5*365, float('inf')],
            labels=[0, 1, 2, 3, 4],
            include_lowest=True
        ).astype('Int64')
        df_final['emp1_total_dur_cat'] = df_final['emp1_total_dur_cat'].fillna(99999)

    # Bin emp1_m_dur (average job duration): 0, <90d, 90-180d, 180-365d, 1yr+
    if 'emp1_m_dur' in df_final.columns:
        df_final['emp1_m_dur_cat'] = pd.cut(
            df_final['emp1_m_dur'],
            bins=[-0.1, 0.5, 90, 180, 365, float('inf')],
            labels=[0, 1, 2, 3, 4],
            include_lowest=True
        ).astype('Int64')
        df_final['emp1_m_dur_cat'] = df_final['emp1_m_dur_cat'].fillna(99999)

    # Bin lastjob_tot_dur (last job duration): 0, <90d, 90-180d, 180-365d, 1-2yr, 2yr+
    if 'lastjob_tot_dur' in df_final.columns:
        df_final['lastjob_tot_dur_cat'] = pd.cut(
            df_final['lastjob_tot_dur'],
            bins=[-0.1, 0.5, 90, 180, 365, 2*365, float('inf')],
            labels=[0, 1, 2, 3, 4, 5],
            include_lowest=True
        ).astype('Int64')
        df_final['lastjob_tot_dur_cat'] = df_final['lastjob_tot_dur_cat'].fillna(99999)

    # Bin LHG_total and LEH_total (benefit spell counts): 0, 1-5, 6-15, 16-30, 31-50, 51+
    for var in ['LHG_total', 'LEH_total']:
        if var in df_final.columns:
            df_final[f'{var}_cat'] = pd.cut(
                df_final[var],
                bins=[-0.1, 0.5, 5.5, 15.5, 30.5, 50.5, float('inf')],
                labels=[0, 1, 2, 3, 4, 5],
                include_lowest=True
            ).astype('Int64')
            df_final[f'{var}_cat'] = df_final[f'{var}_cat'].fillna(0)

    # Bin LHG_m_dur and LEH_m_dur (mean benefit duration): 0, <100d, 100-200d, 200+d
    for var in ['LHG_m_dur', 'LEH_m_dur']:
        if var in df_final.columns:
            df_final[f'{var}_cat'] = pd.cut(
                df_final[var],
                bins=[-0.1, 0.5, 100, 200, float('inf')],
                labels=[0, 1, 2, 3],
                include_lowest=True
            ).astype('Int64')
            df_final[f'{var}_cat'] = df_final[f'{var}_cat'].fillna(0)

    # Bin LHG_tot_dur and LEH_tot_dur (total benefit duration): 0, <6mo, 6mo-1yr, 1-3yr, 3-5yr, 5yr+
    for var in ['LHG_tot_dur', 'LEH_tot_dur']:
        if var in df_final.columns:
            df_final[f'{var}_cat'] = pd.cut(
                df_final[var],
                bins=[-0.1, 0.5, 180, 365, 3*365, 5*365, float('inf')],
                labels=[0, 1, 2, 3, 4, 5],
                include_lowest=True
            ).astype('Int64')
            df_final[f'{var}_cat'] = df_final[f'{var}_cat'].fillna(0)

    # order variables
    ordered_vars = spell_original + spell_generated + ["ltue"]
    for group_name, group_vars in variable_groups.items():
        ordered_vars.extend([v for v in group_vars if v in df_final.columns])

    # Remove duplicates from ordered_vars while preserving order
    seen = set()
    ordered_vars = [v for v in ordered_vars if not (v in seen or seen.add(v))]

    # reorder columns
    existing_ordered = [v for v in ordered_vars if v in df_final.columns]
    remaining_cols = [v for v in df_final.columns if v not in existing_ordered]
    df_final = df_final[existing_ordered + remaining_cols]

    print(f"  Collapsed to {len(df_final):,} people with {len(df_final.columns)} columns")

    return df_final


def select_model_columns(df):

    # ID column
    id_col = ['person_id']

    # Outcome columns
    outcome_cols = ['still_unemployed', 'days_remaining_in_spell', 'remains_ue_horizon_days']

    # Feature columns
    status_before = [
        'employed_before',
        'receipt_leh_before',
        'receipt_lhg_before',
        'se_before',
        'ASU_notue_seeking_before',
        'ASU_other_before',
        'break_before',
    ]

    time_since = [
        'tsince_ein_erw1_cat',
        'tsince_lm_contact_cat',
        'tsince_ft_lm_contact_cat',
    ]

    employment_history = [
        'est_total_cat',
        'emp1_total_cat',
        'emp1_total_dur_cat',
        'emp1_m_dur_cat',
    ]

    lastjob = [
        'lastjob_none',
        'lastjob_tot_dur_cat',
        'lastjob_type',
        'lastjob_pt',
        'lastjob_niveau',
        'lastjob_leih',
        'lastjob_befrist',
        'lastjob_industry',
    ]

    demographics = [
        'age_cat',
        'female',
        'german_citizen',
        'education_level_imputed',
        'school_completed',
        'occupation_skill_level',
        'commuter',
        'bula'
    ]

    benefits = [
        'LHG_total_cat',
        'LHG_tot_dur_cat',
        'LHG_m_dur_cat',
        'LEH_total_cat',
        'LEH_tot_dur_cat',
        'LEH_m_dur_cat',
    ]

    # Combine all columns
    all_cols = (id_col + outcome_cols + status_before + time_since +
                employment_history + lastjob + demographics + benefits)

    # Select only columns that exist in the dataframe
    available_cols = [col for col in all_cols if col in df.columns]

    # Select columns
    df_selected = df[available_cols].copy()

    # Convert categorical variables to pandas Categorical dtype
    categorical_cols = [
        # Status before (binary)
        'employed_before', 'receipt_leh_before', 'receipt_lhg_before',
        'se_before', 'ASU_notue_seeking_before', 'ASU_other_before', 'break_before',
        # Time since (categorical versions)
        'tsince_ein_erw1_cat', 'tsince_lm_contact_cat', 'tsince_ft_lm_contact_cat',
        # Employment counts (binned categorical versions only)
        'est_total_cat', 'emp1_total_cat', 'emp1_total_dur_cat', 'emp1_m_dur_cat',
        # Last job characteristics
        'lastjob_none', 'lastjob_type', 'lastjob_pt', 'lastjob_niveau',
        'lastjob_leih', 'lastjob_befrist', 'lastjob_industry', 'lastjob_tot_dur_cat',
        # Demographics
        'age_cat', 'female', 'german_citizen', 'education_level_imputed', 'school_completed',
        'occupation_skill_level', 'commuter', 'bula',
        # Benefits (binned categorical versions only)
        'LHG_total_cat', 'LEH_total_cat', 'LHG_m_dur_cat', 'LEH_m_dur_cat',
        'LHG_tot_dur_cat', 'LEH_tot_dur_cat',
        # Outcomes
        'still_unemployed'
    ]

    # Convert to categorical dtype
    for col in categorical_cols:
        if col in df_selected.columns:
            df_selected[col] = df_selected[col].astype('category')

    return df_selected


