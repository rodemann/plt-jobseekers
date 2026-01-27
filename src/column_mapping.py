
import pandas as pd

def map_siab_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Translates the column names of the SIAB data from German to English.
    """
    # German -> English feature name mapping
    # Based on official SIAB documentation
    translation_dict = {
        "persnr_siab_r": "person_id",
        "bnn": "establishment_id",
        "spell": "spell_count",
        "quelle_gr": "spell_source",
        "begorig": "observation_start_date",
        "endorig": "observation_end_date",
        "begepi": "episode_start_date",
        "endepi": "episode_end_date",
        "frau": "female",
        "gebjahr": "birth_year",
        "deutsch": "german_citizen",
        "ausbildung_gr": "education_level",
        "ausbildung_imp": "education_level_imputed",
        "schule": "school_completed",
        "tentgelt_gr": "daily_wage",  # Note: This is wage GROUP, not actual daily wage
        "beruf_gr": "occupation_group_1988",
        "beruf2010_gr": "occupation_group_2010",
        "niveau": "occupation_skill_level",
        "teilzeit": "part_time",
        "stib": "working_hours",
        "erwstat_gr": "employment_status_group",
        "gleitz": "transition_zone",
        "leih": "temporary_employment",
        "befrist": "fixed_term_employment",
        "grund_gr": "termination_reason_group",
        "tage_jung": "days_employed_before_17",
        "tage_alt": "days_employed_after_62",
        "alo_dau": "unemployment_duration_days",
        "ao_region": "district_region",
        "pendler": "commuter",
        "w08_gen_gr": "industry_classification"
    }

    column_mapping = translation_dict

    # Only rename columns that exist
    rename_dict = {k: v for k, v in column_mapping.items() if k in df.columns}

    df_renamed = df.rename(columns=rename_dict)

    # Convert date columns to datetime
    date_cols = [
        'observation_start_date',
        'observation_end_date',
        'episode_start_date',
        'episode_end_date'
    ]

    for col in date_cols:
        if col in df_renamed.columns:
            df_renamed[col] = pd.to_datetime(df_renamed[col])

    print(f"Renamed {len(rename_dict)} columns from German to English")

    return df_renamed