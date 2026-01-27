
import pandas as pd
import numpy as np


def identify_unemployment_episodes(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # calculate year and age
    if "episode_start_date" in df.columns:
        df["year"] = df["episode_start_date"].dt.year
    if "birth_year" in df.columns:
        df["age"] = df["year"] - df["birth_year"]

    # keep only MTH (spell_source == 4) and (X)ASU (spell_source == 5) spells
    ue_df = df[df["spell_source"].isin([4, 5])].copy()

    # step 1: identify job seeking during unemployment
    # add feature `seeking1`: (X)ASU spells with employment status `21` (job seeking while unemployed)
    ue_df["seeking1"] = (ue_df["spell_source"] == 5) & (
        ue_df["employment_status_group"] == 21
    )

    # keep only seeking1 spells
    ue_df = ue_df[ue_df["seeking1"] == 1].copy()

    # -------------------------------------------------------------------------
    # on this subset of spells, we compute:
    # - Time gaps between spells
    # - Whether spells are continuous (≤42 days apart)
    # - Episode boundaries (beg_ue, end_ue)
    # - Episode start/end dates
    # - Episode durations
    # -------------------------------------------------------------------------

    # sort by person and spell
    ue_df = ue_df.sort_values(["person_id", "spell_count"])

    # calculate time since last seeking episode
    # add feature `t_since_seeking`: time since last episode
    ue_df["t_since_seeking"] = ue_df["episode_start_date"] - ue_df.groupby("person_id")[
        "episode_end_date"
    ].shift(1)

    # calculate time since last ASU spell
    # add feature `t_since_asu`: time since last ASU spell
    ue_df["t_since_asu"] = ue_df["episode_start_date"] - ue_df.groupby("person_id")[
        "episode_end_date"
    ].shift(1)

    # tag spells that are part of continuous job seeking episode
    # add feature `continuous`: spells that are part of continuous job seeking episode
    # (gap <= 42 days means continuous)
    ue_df["continuous"] = (ue_df["t_since_asu"] <= pd.Timedelta(days=42)).astype(int)

    # step 2: identify separate unemployment episodes
    # 2.1 - begin of unemployment: every spell that is not continuous
    # add feature `beg_ue`: every spell that is not continuous
    ue_df["beg_ue"] = (ue_df["continuous"] != 1).astype(int)
    ue_df["beg_ue_date"] = np.where(
        ue_df["beg_ue"] == 1,
        ue_df["episode_start_date"],
        # using pd.NaT instead of np.nan for type consistency
        pd.NaT,
    )
    # ensure datetime dtype
    ue_df["beg_ue_date"] = pd.to_datetime(ue_df["beg_ue_date"])

    # copy begin date to all subsequent spells in same episode
    ue_df = ue_df.sort_values(["person_id", "spell_count"])
    ue_df["beg_ue_date"] = ue_df.groupby("person_id")["beg_ue_date"].ffill()

    # 2.2 - end of unemployment: spell before a new beginning
    # add feature `end_ue`: spell before a new beginning
    ue_df = ue_df.sort_values(["person_id", "spell_count"])
    ue_df["end_ue"] = (
        (ue_df.groupby("person_id")["beg_ue"].shift(-1) == 1)
        & (ue_df.groupby("person_id")["person_id"].shift(-1) == ue_df["person_id"])
    ).astype(int)

    # correction: last spell of each person is also an end
    ue_df["nrASU"] = ue_df.groupby("person_id").cumcount() + 1
    ue_df["NrASU"] = ue_df.groupby("person_id")["nrASU"].transform("max")
    ue_df.loc[ue_df["nrASU"] == ue_df["NrASU"], "end_ue"] = 1

    # date of end of unemployment
    ue_df["end_ue_date_help"] = np.where(
        ue_df["end_ue"] == 1,
        ue_df["episode_end_date"],
        # using pd.NaT instead of np.nan for type consistency
        pd.NaT,
    )
    # ensure datetime dtype
    ue_df["end_ue_date_help"] = pd.to_datetime(ue_df["end_ue_date_help"])
    ue_df["end_ue_date"] = ue_df.groupby(["person_id", "beg_ue_date"])[
        "end_ue_date_help"
    ].transform("max")

    # drop auxiliary variables
    ue_df = ue_df.drop(columns=["nrASU", "NrASU", "end_ue_date_help"])

    # step 3: calculate duration of unemployment
    # add feature `ue_duration`: duration of unemployment (in days)
    # calculate duration for all rows first, then filter
    ue_df["duration_help"] = (ue_df["episode_end_date"] - ue_df["beg_ue_date"]).dt.days
    ue_df.loc[ue_df["end_ue"] != 1, "duration_help"] = np.nan
    ue_df["ue_duration"] = ue_df.groupby(["person_id", "beg_ue_date"])[
        "duration_help"
    ].transform("max")
    ue_df = ue_df.drop(columns=["duration_help"])

    # -------------------------------------------------------------------------
    # We are done computing features for this subset and will now merge
    # back to the original dataset.
    # -------------------------------------------------------------------------

    # keep only necessary columns for merge
    ue_cols = [
        "person_id",
        "spell_count",
        "seeking1",
        "t_since_seeking",
        "t_since_asu",
        "beg_ue",
        "end_ue",
        "beg_ue_date",
        "end_ue_date",
        "ue_duration",
    ]
    ue_df = ue_df[ue_cols]

    # merge back to original dataset
    df = df.merge(ue_df, on=["person_id", "spell_count"], how="left")

    # step 4: find exits into employment using BeH spells
    # add feature `next_employed_date`: next employed date
    # add feature `t_till_emp`: time until employment
    # keep only BeH (spell_source == 1) and ASU (spell_source == 5) spells
    beh_asu_df = df[df["spell_source"].isin([1, 5])].copy()

    # drop ASU spells that don't mark end of unemployment
    beh_asu_df = beh_asu_df[
        ~((beh_asu_df["spell_source"] == 5) & (beh_asu_df["end_ue"] != 1))
    ]

    # sort by person, begin date, and source
    beh_asu_df = beh_asu_df.sort_values(
        ["person_id", "episode_start_date", "spell_source"]
    )

    # find next employment date (only standard social security employment, employment_status_group == 1)
    beh_asu_df["next_employed_date"] = np.where(
        (beh_asu_df["spell_source"] == 1)
        & (beh_asu_df["employment_status_group"] == 1),
        beh_asu_df["episode_start_date"],
        pd.NaT,  # using pd.NaT instead of np.nan for type consistency
    )
    # ensure datetime dtype
    beh_asu_df["next_employed_date"] = pd.to_datetime(beh_asu_df["next_employed_date"])

    # forward fill next employment date within person
    beh_asu_df["next_employed_date"] = beh_asu_df.groupby("person_id")[
        "next_employed_date"
    ].ffill()

    # calculate time until employment
    # add feature `t_till_emp`: time until employment
    # calculate for all rows first, then filter
    beh_asu_df["t_till_emp"] = (
        beh_asu_df["next_employed_date"]
        - beh_asu_df["episode_end_date"]
        - pd.Timedelta(days=1)
    )
    beh_asu_df.loc[beh_asu_df["end_ue"] != 1, "t_till_emp"] = pd.NaT

    # keep only end of unemployment spells
    exit_df = beh_asu_df[beh_asu_df["end_ue"] == 1][
        ["person_id", "spell_count", "next_employed_date", "t_till_emp"]
    ].copy()

    # merge back
    df = df.merge(exit_df, on=["person_id", "spell_count"], how="left")

    # copy to all spells of same unemployment episode
    df["next_employed_date_help"] = df["next_employed_date"]
    df["t_till_emp_help"] = df["t_till_emp"]

    df["next_employed_date"] = df.groupby(["person_id", "beg_ue_date"])[
        "next_employed_date_help"
    ].transform("max")
    df["t_till_emp"] = df.groupby(["person_id", "beg_ue_date"])[
        "t_till_emp_help"
    ].transform("max")

    df = df.drop(columns=["next_employed_date_help", "t_till_emp_help"])

    # step 5: flag long-term unemployment (>= 365 days)
    df["ltue"] = (df["ue_duration"] >= 365).astype(int)
    df.loc[df["ue_duration"].isna(), "ltue"] = np.nan

    # flag beginning of long-term unemployment
    df["beg_ltue"] = np.where((df["beg_ue"] == 1) & (df["ltue"] == 1), 1, np.nan)

    # copy unemployment episode dates to parallel spells
    df = df.sort_values(["person_id", "episode_start_date", "spell_source"])

    # forward fill beg_ue_date and end_ue_date within same episode
    df["beg_ue_date"] = df.groupby(["person_id", "episode_start_date"])[
        "beg_ue_date"
    ].ffill()
    df["end_ue_date"] = df.groupby(["person_id", "episode_start_date"])[
        "end_ue_date"
    ].ffill()
    df["ue_duration"] = df.groupby(["person_id", "episode_start_date"])[
        "ue_duration"
    ].ffill()

    # sort by person and spell
    df = df.sort_values(["person_id", "spell_count"])

    return df


def generate_biographical_variables(df: pd.DataFrame) -> pd.DataFrame:

    # work on copy of dataframe
    df = df.copy()

    # observation counters
    df = df.sort_values(
        ["person_id", "episode_start_date", "spell_source", "spell_count"]
    )
    df["level1"] = df.groupby(
        ["person_id", "episode_start_date", "spell_source"]
    ).cumcount()
    df["level2"] = df.groupby(["person_id", "episode_start_date"]).cumcount()

    # tag vocational training, interns, and minijobs
    df["apprentice"] = (df["employment_status_group"] == 2).astype(int)
    df["intern"] = (df["employment_status_group"] == 5).astype(int)
    df["minijob"] = (df["employment_status_group"] == 3).astype(int)

    # employment person groups (5 categories)
    df["pg5"] = 5  # default: other
    df.loc[df["employment_status_group"] == 1, "pg5"] = 1  # standard employment
    df.loc[df["employment_status_group"] == 12, "pg5"] = 2  # apprentices
    df.loc[df["employment_status_group"] == 14, "pg5"] = 3  # part-time retirees
    df.loc[df["employment_status_group"] == 13, "pg5"] = 4  # mini-job

    # version 1: employment without vocational training
    df["emp1"] = ((df["apprentice"] != 1) & (df["spell_source"] == 1)).astype(int)

    # first day in employment (version 1)
    # add feature `start_employment1`: first day in employment (w/o voc. training)
    emp1_df = df[df["emp1"] == 1].copy()
    if len(emp1_df) > 0:
        start_employment1 = emp1_df.groupby("person_id")["observation_start_date"].min()
        df = df.merge(
            start_employment1.to_frame("start_employment1"),
            left_on="person_id",
            right_index=True,
            how="left",
        )
    else:
        df["start_employment1"] = np.nan

    # version 2: without vocational training and internships
    df["emp2"] = (
        (df["apprentice"] != 1) & (df["intern"] != 1) & (df["spell_source"] == 1)
    ).astype(int)

    # add feature `start_employment2`: first day in employment (w/o voc. training, internships)
    emp2_df = df[df["emp2"] == 1].copy()
    if len(emp2_df) > 0:
        start_employment2 = emp2_df.groupby("person_id")["observation_start_date"].min()
        df = df.merge(
            start_employment2.to_frame("start_employment2"),
            left_on="person_id",
            right_index=True,
            how="left",
        )
    else:
        df["start_employment2"] = np.nan

    # version 3: without vocational training, internships, and minijobs
    df["emp3"] = (
        (df["apprentice"] != 1)
        & (df["intern"] != 1)
        & (df["minijob"] != 1)
        & (df["spell_source"] == 1)
    ).astype(int)

    # add feature `start_employment3`: first day in employment (w/o voc. training, internships, minijobs)
    emp3_df = df[df["emp3"] == 1].copy()
    if len(emp3_df) > 0:
        start_employment3 = emp3_df.groupby("person_id")["observation_start_date"].min()
        df = df.merge(
            start_employment3.to_frame("start_employment3"),
            left_on="person_id",
            right_index=True,
            how="left",
        )
    else:
        df["start_employment3"] = np.nan

    # number of days in employment (running totals)
    df = df.sort_values(["person_id", "episode_start_date", "spell_count"])

    # version 1
    # add feature `cum_days_emp1`: cumulative days in employment (w/o voc. training)
    emp1_spells = df[df["emp1"] == 1].copy()
    if len(emp1_spells) > 0:
        emp1_spells["d1"] = (
            emp1_spells["episode_end_date"]
            - emp1_spells["episode_start_date"]
            + pd.Timedelta(days=1)
        ).dt.days
        emp1_spells["cum_days_emp1"] = emp1_spells.groupby("person_id")["d1"].cumsum()
        cum_days_emp1_max = (
            emp1_spells.groupby("person_id")["cum_days_emp1"]
            .max()
            .to_frame("cum_days_emp1")
        )
        df = df.merge(
            cum_days_emp1_max, left_on="person_id", right_index=True, how="left"
        )
    else:
        df["cum_days_emp1"] = 0

    # version 2
    # add feature `cum_days_emp2`: cumulative days in employment (w/o voc. training, internships)
    emp2_spells = df[(df["emp2"] == 1) & (df["intern"] != 1)].copy()
    if len(emp2_spells) > 0:
        emp2_spells["d2"] = (
            emp2_spells["episode_end_date"]
            - emp2_spells["episode_start_date"]
            + pd.Timedelta(days=1)
        ).dt.days
        emp2_spells["cum_days_emp2"] = emp2_spells.groupby("person_id")["d2"].cumsum()
        cum_days_emp2_max = (
            emp2_spells.groupby("person_id")["cum_days_emp2"]
            .max()
            .to_frame("cum_days_emp2")
        )
        df = df.merge(
            cum_days_emp2_max, left_on="person_id", right_index=True, how="left"
        )
    else:
        df["cum_days_emp2"] = 0

    # version 3
    # add feature `cum_days_emp3`: cumulative days in employment (w/o voc. training, internships, minijobs)
    emp3_spells = df[
        (df["emp3"] == 1) & (df["intern"] != 1) & (df["minijob"] != 1)
    ].copy()
    if len(emp3_spells) > 0:
        emp3_spells["d3"] = (
            emp3_spells["episode_end_date"]
            - emp3_spells["episode_start_date"]
            + pd.Timedelta(days=1)
        ).dt.days
        emp3_spells["cum_days_emp3"] = emp3_spells.groupby("person_id")["d3"].cumsum()
        cum_days_emp3_max = (
            emp3_spells.groupby("person_id")["cum_days_emp3"]
            .max()
            .to_frame("cum_days_emp3")
        )
        df = df.merge(
            cum_days_emp3_max, left_on="person_id", right_index=True, how="left"
        )
    else:
        df["cum_days_emp3"] = 0

    # parallel jobs (in different establishments)
    # add feature `morethanonejob`: indicator for parallel jobs in different establishments
    # add feature `ndays_morethanonejob`: cumulative days with more than one job
    beh_df = df[df["spell_source"] == 1].copy()
    if len(beh_df) > 0:
        beh_df["tentgelt_h"] = -beh_df["daily_wage"]
        beh_df = beh_df.sort_values(
            ["person_id", "episode_start_date", "establishment_id", "tentgelt_h"]
        )
        beh_df["nrSE"] = (
            beh_df.groupby(
                ["person_id", "episode_start_date", "establishment_id", "tentgelt_h"]
            ).cumcount()
            + 1
        )

        beh_df["nrE"] = np.where(
            beh_df["nrSE"] == 1,
            beh_df.groupby(["person_id", "episode_start_date"]).cumcount() + 1,
            np.nan,
        )
        beh_df["nr_jobs"] = np.where(
            beh_df["nrSE"] == 1,
            beh_df.groupby(["person_id", "episode_start_date"])["nrSE"].transform(
                "count"
            ),
            np.nan,
        )

        beh_df["morethanonejob"] = (beh_df["nr_jobs"] > 1).astype(float)
        beh_df["morethanonejob"] = beh_df.groupby(["person_id", "episode_start_date"])[
            "morethanonejob"
        ].ffill()

        # duration with more than one job
        beh_df["d"] = np.where(
            (beh_df["nrSE"] == 1)
            & (beh_df["nrE"] == 1)
            & (beh_df["morethanonejob"] == 1),
            (
                beh_df["episode_end_date"]
                - beh_df["episode_start_date"]
                + pd.Timedelta(days=1)
            ).dt.days,
            np.nan,
        )
        beh_df["ndays_morethanonejob"] = beh_df.groupby("person_id")["d"].cumsum()

        morethanonejob_cols = [
            "person_id",
            "spell_count",
            "morethanonejob",
            "ndays_morethanonejob",
        ]
        df = df.merge(
            beh_df[morethanonejob_cols], on=["person_id", "spell_count"], how="left"
        )
    else:
        df["morethanonejob"] = np.nan
        df["ndays_morethanonejob"] = np.nan

    # first day in establishment
    # add feature `first_day_estab`: first day in establishment
    if "establishment_id" in df.columns:
        first_day_estab = (
            df[df["establishment_id"].notna()]
            .groupby(["person_id", "establishment_id"])["episode_start_date"]
            .min()
        )
        first_day_estab = (
            first_day_estab.groupby("person_id").min().to_frame("first_day_estab")
        )
        df = df.merge(
            first_day_estab, left_on="person_id", right_index=True, how="left"
        )
    else:
        df["first_day_estab"] = np.nan

    # number of days in establishment
    # add feature `ndays_estab`: cumulative days in establishment
    if "establishment_id" in df.columns:
        bet_df = df[df["establishment_id"].notna()].copy()
        bet_df = bet_df.sort_values(
            [
                "person_id",
                "establishment_id",
                "episode_start_date",
                "episode_end_date",
                "spell_count",
            ]
        )
        bet_df["nrB"] = (
            bet_df.groupby(
                [
                    "person_id",
                    "establishment_id",
                    "episode_start_date",
                    "episode_end_date",
                ]
            ).cumcount()
            + 1
        )
        bet_df["dauer"] = np.where(
            bet_df["nrB"] == 1,
            (
                bet_df["episode_end_date"]
                - bet_df["episode_start_date"]
                + pd.Timedelta(days=1)
            ).dt.days,
            np.nan,
        )
        bet_df["ndays_estab"] = bet_df.groupby(["person_id", "establishment_id"])[
            "dauer"
        ].cumsum()
        ndays_estab_max = (
            bet_df.groupby(["person_id", "establishment_id"])["ndays_estab"]
            .max()
            .groupby("person_id")
            .max()
            .to_frame("ndays_estab")
        )
        df = df.merge(
            ndays_estab_max, left_on="person_id", right_index=True, how="left"
        )
    else:
        df["ndays_estab"] = np.nan

    # first day in job and number of days in job
    # add feature `first_day_job`: first day in job
    # add feature `tage_job_kum`: cumulative days in job
    # add feature `beg_job_sp`: indicator for start of new job
    if "establishment_id" in df.columns:
        job_df = df[df["establishment_id"].notna()].copy()
        job_df = job_df.sort_values(
            ["person_id", "apprentice", "establishment_id", "spell_count"]
        )

        # mark subsequent episodes of same job
        job_df["job"] = (
            (job_df["person_id"] == job_df["person_id"].shift(1))
            & (job_df["establishment_id"] == job_df["establishment_id"].shift(1))
            & (job_df["apprentice"] == job_df["apprentice"].shift(1))
        ).astype(int)

        # consider ending notification and gaps
        # check for termination_reason_group or grund
        termination_col = None
        for col in ["termination_reason_group", "grund_gr", "grund"]:
            if col in job_df.columns:
                termination_col = col
                break

        if termination_col is not None:
            job_df["end"] = job_df.groupby(
                ["person_id", "apprentice", "establishment_id", "episode_start_date"]
            )[termination_col].transform(lambda x: x.iloc[0] if len(x) > 0 else np.nan)
            job_df["end"] = job_df["end"].isin([0, 5]).astype(int)
        else:
            job_df["end"] = 0

        job_df["gap"] = np.where(
            job_df["job"] == 1,
            (
                job_df["episode_start_date"]
                - job_df["episode_end_date"].shift(1)
                - pd.Timedelta(days=1)
            ).dt.days,
            np.nan,
        )

        ### JUDGMENT CALL: thresholds for counting as new job (92 days, 366 days)
        # count as new job if gap > 92 days with end notification, or gap > 366 days
        job_df.loc[(job_df["end"].shift(1) == 1) & (job_df["gap"] > 92), "job"] = np.nan
        job_df.loc[job_df["gap"] > 366, "job"] = np.nan

        # generate start date of job
        job_df["ein_job"] = job_df["episode_start_date"]
        job_df.loc[job_df["job"] == 1, "ein_job"] = job_df.loc[
            job_df["job"] == 1, "ein_job"
        ].shift(1)

        # number of days in job
        job_df["nrA"] = (
            job_df.groupby(
                ["person_id", "apprentice", "establishment_id", "episode_start_date"]
            ).cumcount()
            + 1
        )
        job_df["jobdauer"] = (
            job_df["episode_end_date"]
            - job_df["episode_start_date"]
            + pd.Timedelta(days=1)
        ).dt.days
        job_df["jobdauer_kum"] = job_df["jobdauer"]
        job_df["jobdauer_dup"] = np.where(job_df["nrA"] != 1, job_df["jobdauer"], 0)

        job_df.loc[job_df["job"] == 1, "jobdauer_kum"] = (
            job_df.loc[job_df["job"] == 1, "jobdauer_kum"].shift(1)
            + job_df.loc[job_df["job"] == 1, "jobdauer"]
            - job_df.loc[job_df["job"] == 1, "jobdauer_dup"]
        )

        job_df["job_start"] = np.where(
            (job_df["job"].isna()) & (job_df["spell_source"] == 1), 1, 0
        )
        job_df["tage_job"] = job_df["jobdauer_kum"]

        job_cols = ["person_id", "spell_count", "ein_job", "tage_job", "job_start"]
        df = df.merge(job_df[job_cols], on=["person_id", "spell_count"], how="left")
    else:
        df["ein_job"] = np.nan
        df["tage_job"] = np.nan
        df["job_start"] = 0

    # benefit receipts (LeH and LHG)
    # LeH benefits (spell_source == 2)
    # add feature `anz_lst_leh_kum`: cumulative count of LeH benefit receipts
    # add feature `tage_lst_leh_kum`: cumulative days receiving LeH benefits
    df["quelleLeH"] = (df["spell_source"] == 2).astype(int)
    leh_df = df[df["quelleLeH"] == 1].copy()
    if len(leh_df) > 0:
        leh_df = leh_df.sort_values(
            ["person_id", "episode_start_date", "spell_source", "spell_count"]
        )
        leh_df["nrLeH"] = (
            leh_df.groupby(["person_id", "episode_start_date", "quelleLeH"]).cumcount()
            + 1
        )

        leh_df["ende_vor_leh"] = leh_df.groupby("person_id")["episode_end_date"].shift(
            1
        )
        leh_df["ende_vor_leh"] = leh_df.groupby("person_id")["ende_vor_leh"].ffill()


        leh_df["lst_leh"] = (
            (leh_df["quelleLeH"] == 1)
            & (leh_df["nrLeH"] == 1)
            & (
                (leh_df["episode_start_date"] - leh_df["ende_vor_leh"])
                > pd.Timedelta(days=10)
            )
        ).astype(int)

        leh_df = leh_df.sort_values(
            ["person_id", "episode_start_date", "lst_leh"],
            ascending=[True, True, False],
        )
        leh_df["anz_lst_leh"] = leh_df.groupby("person_id")["lst_leh"].cumsum()

        leh_df["breceipt_leh"] = 1
        leh_df = leh_df.sort_values(
            ["person_id", "episode_start_date", "quelleLeH"],
            ascending=[True, True, False],
        )
        leh_df["breceipt_leh"] = leh_df.groupby(["person_id", "episode_start_date"])[
            "breceipt_leh"
        ].ffill()

        # duration of LeH benefits
        leh_df["lstdauer_leh"] = np.where(
            (leh_df["quelleLeH"] == 1) & (leh_df["nrLeH"] == 1),
            (
                leh_df["episode_end_date"]
                - leh_df["episode_start_date"]
                + pd.Timedelta(days=1)
            ).dt.days,
            np.nan,
        )
        leh_df["tage_lst_leh"] = leh_df.groupby("person_id")["lstdauer_leh"].cumsum()
        leh_df["tage_lst_leh"] = leh_df.groupby(["person_id", "episode_start_date"])[
            "tage_lst_leh"
        ].transform("first")

        leh_cols = [
            "person_id",
            "spell_count",
            "anz_lst_leh",
            "beg_lst_leh_sp",
            "lst_leh",
            "tage_lst_leh",
            "tage_lst_leh_sp",
            "tage_lst_leh_ep",
        ]
        leh_df["beg_lst_leh_sp"] = leh_df["lst_leh"]
        leh_df["tage_lst_leh_sp"] = leh_df["lstdauer_leh"]
        leh_df["tage_lst_leh_ep"] = leh_df.groupby(["person_id", "anz_lst_leh"])[
            "lstdauer_leh"
        ].transform("sum")

        df = df.merge(leh_df[leh_cols], on=["person_id", "spell_count"], how="left")
    else:
        for col in [
            "anz_lst_leh",
            "beg_lst_leh_sp",
            "lst_leh",
            "tage_lst_leh",
            "tage_lst_leh_sp",
            "tage_lst_leh_ep",
        ]:
            df[col] = 0

    # similar process for LHG benefits (spell_source == 3)
    # add feature `anz_lst_lhg_kum`: cumulative count of LHG benefit receipts
    # add feature `tage_lst_lhg_kum`: cumulative days receiving LHG benefits
    df["quelleLHG"] = (df["spell_source"] == 3).astype(int)
    lhg_df = df[df["quelleLHG"] == 1].copy()
    if len(lhg_df) > 0:
        lhg_df = lhg_df.sort_values(
            ["person_id", "episode_start_date", "spell_source", "spell_count"]
        )
        lhg_df["nrLHG"] = (
            lhg_df.groupby(["person_id", "episode_start_date", "quelleLHG"]).cumcount()
            + 1
        )

        lhg_df["ende_vor_lhg"] = lhg_df.groupby("person_id")["episode_end_date"].shift(
            1
        )
        lhg_df["ende_vor_lhg"] = lhg_df.groupby("person_id")["ende_vor_lhg"].ffill()

        ### JUDGMENT CALL: gap threshold for counting as separate benefit receipt (10 days)
        lhg_df["lst_lhg"] = (
            (lhg_df["quelleLHG"] == 1)
            & (lhg_df["nrLHG"] == 1)
            & (
                (lhg_df["episode_start_date"] - lhg_df["ende_vor_lhg"])
                > pd.Timedelta(days=10)
            )
        ).astype(int)

        lhg_df = lhg_df.sort_values(
            ["person_id", "episode_start_date", "lst_lhg"],
            ascending=[True, True, False],
        )
        lhg_df["anz_lst_lhg"] = lhg_df.groupby("person_id")["lst_lhg"].cumsum()

        lhg_df["breceipt_lhg"] = 1
        lhg_df = lhg_df.sort_values(
            ["person_id", "episode_start_date", "quelleLHG"],
            ascending=[True, True, False],
        )
        lhg_df["breceipt_lhg"] = lhg_df.groupby(["person_id", "episode_start_date"])[
            "breceipt_lhg"
        ].ffill()

        lhg_df["lstdauer_lhg"] = np.where(
            (lhg_df["quelleLHG"] == 1) & (lhg_df["nrLHG"] == 1),
            (
                lhg_df["episode_end_date"]
                - lhg_df["episode_start_date"]
                + pd.Timedelta(days=1)
            ).dt.days,
            np.nan,
        )
        lhg_df["tage_lst_lhg"] = lhg_df.groupby("person_id")["lstdauer_lhg"].cumsum()
        lhg_df["tage_lst_lhg"] = lhg_df.groupby(["person_id", "episode_start_date"])[
            "tage_lst_lhg"
        ].transform("first")

        lhg_cols = [
            "person_id",
            "spell_count",
            "anz_lst_lhg",
            "beg_lst_lhg_sp",
            "lst_lhg",
            "tage_lst_lhg",
            "tage_lst_lhg_sp",
            "tage_lst_lhg_ep",
        ]
        lhg_df["beg_lst_lhg_sp"] = lhg_df["lst_lhg"]
        lhg_df["tage_lst_lhg_sp"] = lhg_df["lstdauer_lhg"]
        lhg_df["tage_lst_lhg_ep"] = lhg_df.groupby(["person_id", "anz_lst_lhg"])[
            "lstdauer_lhg"
        ].transform("sum")

        df = df.merge(lhg_df[lhg_cols], on=["person_id", "spell_count"], how="left")
    else:
        for col in [
            "anz_lst_lhg",
            "beg_lst_lhg_sp",
            "lst_lhg",
            "tage_lst_lhg",
            "tage_lst_lhg_sp",
            "tage_lst_lhg_ep",
        ]:
            df[col] = 0

    # subsidized employment duration
    # add feature `se_ep`: indicator for subsidized employment episode
    # add feature `tage_se_kum`: cumulative days in subsidized employment
    df["se"] = (
        (df["spell_source"] == 4) & (df["employment_status_group"].isin([41, 44]))
    ).astype(int)
    df = df.sort_values(
        ["person_id", "episode_start_date", "spell_source"],
        ascending=[True, True, False],
    )
    df["se"] = df.groupby(["person_id", "episode_start_date"])["se"].ffill()

    se_df = df[
        (df["spell_source"] == 4) & (df["employment_status_group"].isin([41, 44]))
    ].copy()
    if len(se_df) > 0:
        se_df["quelleMSE"] = 1
        se_df["nrMSE"] = (
            se_df.groupby(["person_id", "episode_start_date", "quelleMSE"]).cumcount()
            + 1
        )
        se_df["sedauer"] = np.where(
            se_df["nrMSE"] == 1,
            (
                se_df["episode_end_date"]
                - se_df["episode_start_date"]
                + pd.Timedelta(days=1)
            ).dt.days,
            np.nan,
        )
        se_df["tage_se"] = se_df.groupby("person_id")["sedauer"].cumsum()
        se_df["tage_se"] = se_df.groupby(["person_id", "episode_start_date"])[
            "tage_se"
        ].ffill()

        tage_se_max = se_df.groupby("person_id")["tage_se"].max().to_frame("tage_se")
        df = df.merge(tage_se_max, left_on="person_id", right_index=True, how="left")
    else:
        df["tage_se"] = 0

    # adjust missing values for establishment/job variables
    for var in ["first_day_estab", "ndays_estab", "ein_job", "tage_job"]:
        if var in df.columns:
            df.loc[df["spell_source"] != 1, var] = np.nan

    # sort by person and spell
    df = df.sort_values(["person_id", "spell_count"])

    return df