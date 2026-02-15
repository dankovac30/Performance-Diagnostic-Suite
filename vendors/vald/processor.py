from typing import Any

import numpy as np
import pandas as pd

from config import Config
from vendors.vald.auth import get_vald_token
from vendors.vald.mappings import VALD_CMJ_METRICS, VALD_DJ_METRICS, VALD_IMTP_METRICS
from vendors.vald.sync import fetch_force_decks_trials_data, fetch_force_frame_nord_bord_trials_data, fetch_profiles


class ValdProcessor:
    """
    Processor class for handling VALD performance data (ForceDecks, ForceFrame, NordBord).

    This class manages authentication, data fetching, transformation, cleaning,
    and aggregation of trial data into a unified structure for analysis and database storage.
    """

    def __init__(self, params: dict[str, Any] | None = None):
        """
        Initialize the ValdProcessor with configuration and API tokens.

        Args:
            params (dict[str, Any] | None): Optional dictionary of parameters for API requests
                                            (e.g., date ranges, specific test IDs).
        """
        self.params = params
        self.vald_token = get_vald_token(Config.VALD_TOKEN_URL, Config.VALD_CLIENT_ID, Config.VALD_CLIENT_SECRET)
        self.tenant_id = Config.VALD_TENANT_ID
        self.team_uid = Config.VALD_TEAM_UID
        self.profiles_url = Config.VALD_CLIENTS_URL
        self.fd_base_url = Config.VALD_FORCEDECKS_BASE_URL
        self.ff_base_url = Config.VALD_FORCE_FRAME_BASE_URL
        self.nb_base_url = Config.VALD_NORDBORD_BASE_URL
        self.tests_endpoint = Config.VALD_TESTS_ENDPOINT
        self.tests_endpoint_v2 = Config.VALD_TESTS_ENDPOINT_V2
        self.trials_endpoint = Config.VALD_TRIALS_ENDPOINT

    def calculate_asymmetry(self, l_val: pd.Series | float, r_val: pd.Series | float) -> pd.Series | float:
        """
        Calculate asymmetry between left and right values using the standard formula.

        Args:
            l_val (pd.Series | float): Left side value(s).
            r_val (pd.Series | float): Right side value(s).

        Returns:
            pd.Series | float: Calculated asymmetry ratio.
        """
        return (r_val - l_val) / ((r_val + l_val) / 2)

    def create_asymmetry_cols(self, df: pd.DataFrame, l_ending: str) -> pd.DataFrame:
        """
        Generate asymmetry columns for all matching Left/Right metric pairs in the DataFrame.

        Args:
            df (pd.DataFrame): The input DataFrame containing metrics.
            l_ending (str): The suffix identifying left-side metrics (e.g., "_L" or " (L)").

        Returns:
            pd.DataFrame: A DataFrame containing only the new asymmetry columns.
        """
        l_cols = [col for col in df.columns if str(col).endswith(l_ending)]

        r_ending = l_ending.replace("L", "R")
        a_ending = l_ending.replace("L", "A")
        ending_length = len(l_ending)

        asym_columns = {}

        for l_col in l_cols:
            base_id = l_col[:-ending_length]
            r_col = f"{base_id}{r_ending}"
            l_col = f"{base_id}{l_ending}"
            a_col = f"{base_id}{a_ending}"

            # Calculate only if the corresponding Right column exists
            if r_col in df.columns:
                asym_columns[a_col] = self.calculate_asymmetry(df[l_col], df[r_col])

        asym_df = pd.DataFrame(asym_columns, index=df.index)

        return asym_df

    def process_fd_data(self) -> pd.DataFrame:
        """
        Fetch and process ForceDecks data.

        Steps:
        1. Fetch raw trial data from VALD API.
        2. Calculate initial asymmetries.
        3. Map specific metrics based on test type (CMJ, DJ, IMTP).
        4. Derive additional metrics (e.g., ratios, relative forces).
        5. Clean and structure the DataFrame.

        Returns:
            pd.DataFrame: Processed ForceDecks data or empty DataFrame if no data found.
        """
        df = pd.DataFrame(
            fetch_force_decks_trials_data(
                self.vald_token,
                self.tenant_id,
                self.team_uid,
                self.fd_base_url,
                self.tests_endpoint,
                self.trials_endpoint,
                self.params,
            )
        )

        if df.empty:
            return pd.DataFrame()

        # Create unique trial ID from time and test ID
        df["trial_id"] = df["recorded_at"].astype(str) + "_" + df["session_id"].astype(str)

        # Convert created time to datetime object
        df["recorded_at"] = pd.to_datetime(df["recorded_at"], utc=True)

        # Normalize timestamps to date-only format
        df["test_date"] = df["recorded_at"].dt.date

        # Calculate initial asymmetries based on raw columns ending in "_L"
        asym_df = self.create_asymmetry_cols(df, l_ending="_L")

        df = pd.concat([df, asym_df], axis=1)

        variable_mapping = {
            "CMJ": VALD_CMJ_METRICS,
            "SLJ": VALD_CMJ_METRICS,
            "DJ": VALD_DJ_METRICS,
            "SLDJ": VALD_DJ_METRICS,
            "IMTP": VALD_IMTP_METRICS,
        }

        processed_dfs = []

        # Process each test category separately to apply specific logic
        for exercise, group_df in df.groupby(df["exercise"]):
            if exercise not in variable_mapping.keys():
                continue

            rename_map = {}

            # Create renaming map for metrics including L, R, and asymmetry variants
            for metric_id, metric_name in variable_mapping[exercise].items():
                rename_map[metric_id] = metric_name
                rename_map[f"{metric_id}_L"] = f"{metric_name} (L)"
                rename_map[f"{metric_id}_R"] = f"{metric_name} (R)"
                rename_map[f"{metric_id}_A"] = f"{metric_name} (A)"

            group_df = group_df.rename(columns=rename_map)

            keep_columns = ["vald_id", "trial_id", "session_id", "exercise", "recorded_at", "test_date", "laterality"]
            final_columns = keep_columns + list(rename_map.values())
            existing_columns = [col for col in final_columns if col in group_df.columns]

            cleaned_df = group_df[existing_columns]
            cleaned_df = cleaned_df.dropna(axis=1, how="all").copy()

            # Derived metrics calculation
            if exercise == "CMJ" or exercise == "SLJ":
                cleaned_df["Eccentric Unloading Duration [s]"] = (
                    cleaned_df["Eccentric Duration [s]"] - cleaned_df["Eccentric Braking Duration [s]"]
                )
                cleaned_df["Peak Force [N]"] = cleaned_df[
                    ["Concentric Peak Force [N]", "Eccentric Peak Force [N]", "Force at Zero Velocity [N]"]
                ].max(axis=1)
                cleaned_df["Concentric Peak Force [N/kg]"] = (
                    cleaned_df["Concentric Peak Force [N]"] / cleaned_df["Body Weight [kg]"]
                )
                cleaned_df["Eccentric Peak Force [N/kg]"] = (
                    cleaned_df["Eccentric Peak Force [N]"] / cleaned_df["Body Weight [kg]"]
                )
                cleaned_df["Force at Zero Velocity [N/kg]"] = (
                    cleaned_df["Force at Zero Velocity [N]"] / cleaned_df["Body Weight [kg]"]
                )
                cleaned_df["Peak Force [N/kg]"] = cleaned_df["Peak Force [N]"] / cleaned_df["Body Weight [kg]"]
                cleaned_df["Min Eccentric Force [N/kg]"] = (
                    cleaned_df["Min Eccentric Force [N]"] / cleaned_df["Body Weight [kg]"]
                )
                cleaned_df["Force at Peak Power [N/kg]"] = (
                    cleaned_df["Force at Peak Power [N]"] / cleaned_df["Body Weight [kg]"]
                )
                cleaned_df["Concentric Mean Force [N/kg]"] = (
                    cleaned_df["Concentric Mean Force [N]"] / cleaned_df["Body Weight [kg]"]
                )
                cleaned_df["Eccentric Mean Force [N/kg]"] = (
                    cleaned_df["Eccentric Mean Force [N]"] / cleaned_df["Body Weight [kg]"]
                )
                cleaned_df["Eccentric Mean Braking Force [N/kg]"] = (
                    cleaned_df["Eccentric Mean Braking Force [N]"] / cleaned_df["Body Weight [kg]"]
                )
                cleaned_df["RSI Modified [m/s]"] = (cleaned_df["Jump Height (Imp-Mom) [cm]"] / 100) / cleaned_df[
                    "Contraction Time [s]"
                ]
                cleaned_df["Eccentric:Concentric Mean Force Ratio"] = (
                    cleaned_df["Eccentric Mean Force [N]"] / cleaned_df["Concentric Mean Force [N]"]
                )
                cleaned_df["Eccentric:Concentric Duration Ratio"] = (
                    cleaned_df["Eccentric Duration [s]"] / cleaned_df["Concentric Duration [s]"]
                )
                cleaned_df["Eccentric Braking:Concentric Duration Ratio"] = (
                    cleaned_df["Eccentric Braking Duration [s]"] / cleaned_df["Concentric Duration [s]"]
                )
                cleaned_df["Eccentric:Concentric Peak Power Ratio"] = (
                    cleaned_df["Eccentric Peak Power [W]"] / cleaned_df["Peak Power [W]"]
                )
                cleaned_df["Eccentric Braking:Concentric Impulse Ratio"] = (
                    cleaned_df["Eccentric Braking Impulse [N·s]"] / cleaned_df["Concentric Impulse [N·s]"]
                )
                cleaned_df["Peak Landing Force [N/kg]"] = (
                    cleaned_df["Peak Landing Force [N]"] / cleaned_df["Body Weight [kg]"]
                )
                cleaned_df["Eccentric Deceleration RFD [N/(s·kg)]"] = (
                    cleaned_df["Eccentric Deceleration RFD [N/s]"] / cleaned_df["Body Weight [kg]"]
                )

                if exercise == "CMJ":
                    cmj_df = cleaned_df.copy()
                    processed_dfs.append(cmj_df)
                elif exercise == "SLJ":
                    slj_df = cleaned_df.copy()
                    processed_dfs.append(slj_df)

            elif exercise == "DJ" or exercise == "SLDJ":
                cleaned_df["Takeoff Velocity (Flight Time) [m/s]"] = (
                    (2 * cleaned_df["Jump Height (Flight Time) [cm]"] / 100) * 9.80665
                ) ** 0.5
                cleaned_df["Takeoff Velocity (Imp-Mom) [m/s]"] = (
                    (2 * cleaned_df["Jump Height (Imp-Mom) [cm]"] / 100) * 9.80665
                ) ** 0.5
                cleaned_df["RSI (FT) [m/s]"] = (cleaned_df["Jump Height (Flight Time) [cm]"] / 100) / cleaned_df[
                    "Contact Time [s]"
                ]
                cleaned_df["RSI (IM) [m/s]"] = (cleaned_df["Jump Height (Imp-Mom) [cm]"] / 100) / cleaned_df[
                    "Contact Time [s]"
                ]
                cleaned_df["RSR (ft/ct)"] = cleaned_df["Flight Time [s]"] / cleaned_df["Contact Time [s]"]
                cleaned_df["DRI"] = (
                    (cleaned_df["Jump Height (Flight Time) [cm]"] + cleaned_df["Drop Height [cm]"]) / 100
                ) / (9.80665 * (cleaned_df["Contact Time [s]"] ** 2))
                cleaned_df["Net Impulse [N·s]"] = (
                    cleaned_df["Eccentric Net Impulse [N·s]"] + cleaned_df["Concentric Net Impulse [N·s]"]
                )
                cleaned_df["COM Δv [m/s]"] = cleaned_df["Net Impulse [N·s]"] / cleaned_df["Body Weight [kg]"]
                cleaned_df["Gross Impulse [N·s]"] = cleaned_df["Net Impulse [N·s]"] + (
                    cleaned_df["Contact Time [s]"] * cleaned_df["Body Weight [kg]"] * 9.80665
                )
                cleaned_df["Peak Landing Force [N/kg]"] = (
                    cleaned_df["Peak Landing Force [N]"] / cleaned_df["Body Weight [kg]"]
                )

                # Remove intermediate impulse variables to avoid retaining methodologically uncertain metrics
                cleaned_df.drop(
                    [
                        "Eccentric Net Impulse [N·s]",
                        "Eccentric Net Impulse [N·s] (A)",
                        "Eccentric Net Impulse [N·s] (L)",
                        "Eccentric Net Impulse [N·s] (R)",
                        "Concentric Net Impulse [N·s]",
                        "Concentric Net Impulse [N·s] (A)",
                        "Concentric Net Impulse [N·s] (L)",
                        "Concentric Net Impulse [N·s] (R)",
                    ],
                    axis=1,
                    inplace=True,
                    errors="ignore",
                )

                if exercise == "DJ":
                    dj_df = cleaned_df.copy()
                    processed_dfs.append(dj_df)
                elif exercise == "SLDJ":
                    sldj_df = cleaned_df.copy()
                    processed_dfs.append(sldj_df)

            elif exercise == "IMTP":
                cleaned_df["Body Weight [kg]"] = cleaned_df["Peak Force [N]"] / cleaned_df["Peak Force [N/kg]"]
                cleaned_df["Peak Net Force [N/kg]"] = cleaned_df["Peak Net Force [N]"] / cleaned_df["Body Weight [kg]"]

                imtp_df = cleaned_df.copy()
                processed_dfs.append(imtp_df)

        if not processed_dfs:
            return pd.DataFrame()

        processed_trials_df = pd.concat(processed_dfs, ignore_index=True)

        # Defragment dataframe
        processed_trials_df = processed_trials_df.copy()

        return processed_trials_df

    def process_ff_nb_test(self, test: dict) -> list:
        """
        Extract and normalize metrics from a single ForceFrame or NordBord test response.

        Args:
            test (dict): Raw JSON dictionary of a single test from VALD API.

        Returns:
            list: A list of normalized trial dictionaries (one test can yield multiple trials).
        """
        # ForceFrame: Ankle & Hip Flexion
        if test.get("testPositionName") in ("Ankle Plantar Flexion - Seated", "Hip Flexion - Prone"):
            trial = {
                "vald_id": test["profileId"],
                "recorded_at": test["testDateUtc"],
                "session_id": test["testId"],
                "trial_id": f"{test['testDateUtc']}_{test['testId']}",
                "exercise": test["testTypeName"],
                "Peak Force [N] (L)": test["outerLeftMaxForce"],
                "Peak Force [N] (R)": test["outerRightMaxForce"],
            }

            return [trial]

        # ForceFrame: Hip Adduction / Abduction
        elif test.get("testPositionName") == "Hip AD/AB - Supine (Knee)":
            adduction = {
                "vald_id": test["profileId"],
                "recorded_at": test["testDateUtc"],
                "session_id": test["testId"],
                "trial_id": f"{test['testDateUtc']}{test['testId']}_ad",
                "exercise": "Hip Adduction",
                "Peak Force [N] (L)": test["innerLeftMaxForce"],
                "Peak Force [N] (R)": test["innerRightMaxForce"],
            }

            abduction = {
                "vald_id": test["profileId"],
                "recorded_at": test["testDateUtc"],
                "session_id": test["testId"],
                "trial_id": f"{test['testDateUtc']}{test['testId']}_ab",
                "exercise": "Hip Abduction",
                "Peak Force [N] (L)": test["outerLeftMaxForce"],
                "Peak Force [N] (R)": test["outerRightMaxForce"],
            }

            return [adduction, abduction]

        # NordBord: Nordic & ISO Prone
        elif test.get("testTypeName") in ("Nordic", "ISO Prone"):
            trial = {
                "vald_id": test["profileId"],
                "recorded_at": test["testDateUtc"],
                "session_id": test["testId"],
                "trial_id": f"{test['testDateUtc']}_{test['testId']}",
                "exercise": test["testTypeName"],
                "Peak Force [N] (L)": test["leftMaxForce"],
                "Peak Force [N] (R)": test["rightMaxForce"],
            }

            return [trial]
        else:
            return []

    def process_ff_nb_data(self) -> pd.DataFrame:
        """
        Fetch and process ForceFrame and NordBord data. Aggregates multiple
        tests from the same day into a single 'best' record per test type.

        Returns:
            pd.DataFrame: Processed aggregated data or empty DataFrame.
        """
        tests_list = []

        # Fetch ForceFrame data
        tests_list.extend(
            fetch_force_frame_nord_bord_trials_data(
                self.vald_token, self.tenant_id, self.ff_base_url, self.tests_endpoint_v2, self.params
            )
        )
        # Fetch NordBord data
        tests_list.extend(
            fetch_force_frame_nord_bord_trials_data(
                self.vald_token, self.tenant_id, self.nb_base_url, self.tests_endpoint_v2, self.params
            )
        )

        if not tests_list:
            return pd.DataFrame()

        all_trials = []

        # Process each test
        for test in tests_list:
            trial = self.process_ff_nb_test(test)
            all_trials.extend(trial)

        df = pd.DataFrame(all_trials)

        # Convert created time to datetime object
        df["recorded_at"] = pd.to_datetime(df["recorded_at"], utc=True)

        # Normalize timestamps to date-only format
        df["test_date"] = df["recorded_at"].dt.date

        # Take max force, keep first occurrence for metadata
        agg_rules = {
            "Peak Force [N] (L)": "max",
            "Peak Force [N] (R)": "max",
            "session_id": "first",
            "trial_id": "first",
            "recorded_at": "first",
        }
        # Group by Athlete + Date + Test type to consolidate daily session
        final_df = df.groupby(["vald_id", "test_date", "exercise"], as_index=False).agg(agg_rules)

        # Calculate asymmetry for the max values
        final_df["Peak Force [N] (A)"] = self.calculate_asymmetry(
            final_df["Peak Force [N] (L)"], final_df["Peak Force [N] (R)"]
        )

        # Flag these as best reps
        final_df["is_best_rep"] = True

        bilateral_tests = ["Nordic", "ISO Prone"]
        final_df["laterality"] = np.where(final_df["exercise"].isin(bilateral_tests), "Bilateral", "Unilateral")

        return final_df

    def flag_best_rep(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Identify and flag the 'best' repetition for each athlete's daily session based on key performance metrics.

        Deciding Metrics:
            - CMJ/SLJ: Jump Height (Impulse-Momentum)
            - DJ/SLDJ: RSI (Flight Time)
            - IMTP: Peak Force [N/kg]

        Args:
            df (pd.DataFrame): DataFrame containing all trial reps.

        Returns:
            pd.DataFrame: DataFrame with an added 'is_best_rep' boolean column.
        """
        deciding_metrics = {
            "CMJ": "Jump Height (Imp-Mom) [cm]",
            "SLJ": "Jump Height (Imp-Mom) [cm]",
            "DJ": "RSI (FT) [m/s]",
            "SLDJ": "RSI (FT) [m/s]",
            "IMTP": "Peak Force [N/kg]",
        }

        # Initialize the flag column with False for all rows
        df["is_best_rep"] = False

        for exercise, metric in deciding_metrics.items():
            if metric not in df.columns:
                continue

            mask_type = df["exercise"] == exercise

            # Group by athlete, date AND laterality (to handle L/R bests separately for unilateral jumps)
            idx_best = df.loc[mask_type].groupby(["vald_id", "test_date", "laterality"])[metric].idxmax()
            valid_indices = idx_best.dropna()
            df.loc[valid_indices, "is_best_rep"] = True

        return df

    def process_unilateral_fd(self, unilateral_fd: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms unilateral ForceDecks data into a unified single-row structure. Pairs the best
        Left rep and best Right rep of the day into one row with (L) and (R) suffixed columns.

        Args:
            unilateral_fd (pd.DataFrame): DataFrame containing only unilateral trials (Left/Right).

        Returns:
            pd.DataFrame: Unified DataFrame with one row per session (L+R merged).
        """
        unilateral_fd = unilateral_fd.dropna(axis=1, how="all")

        # Split into Left and Right dataframes, filtering only for best reps
        left_df = unilateral_fd[(unilateral_fd["laterality"] == "Left") & (unilateral_fd["is_best_rep"])].copy()
        right_df = unilateral_fd[(unilateral_fd["laterality"] == "Right") & (unilateral_fd["is_best_rep"])].copy()

        # Define metadata columns that should NOT receive L/R suffixes
        meta_columns = [
            "vald_id",
            "recorded_at",
            "session_id",
            "trial_id",
            "exercise",
            "laterality",
            "Body Weight [kg]",
            "is_best_rep",
            "test_date",
        ]

        # Rename value columns to include (L) and (R) suffixes
        left_df.columns = [f"{col} (L)" if col not in meta_columns else col for col in left_df.columns]
        right_df.columns = [f"{col} (R)" if col not in meta_columns else col for col in right_df.columns]

        # Concatenate and then merge
        combined_df = pd.concat([left_df, right_df])
        df = combined_df.groupby(["vald_id", "test_date", "exercise"], as_index=False).first()

        # Calculate asymmetry on the newly joined columns
        asym_df = self.create_asymmetry_cols(df, l_ending="(L)")
        df = pd.concat([df, asym_df], axis=1)

        # Mark as processed unilateral pair
        df["laterality"] = "Unilateral"

        return df

    def calculate_dsi(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Dynamic Strength Index (DSI) for athletes.
        DSI = CMJ Peak Force / IMTP Peak Force

        Args:
            df (pd.DataFrame): The complete dataset containing both CMJ and IMTP tests.

        Returns:
            pd.DataFrame: A DataFrame containing DSI calculations per athlete per session.
        """
        # Filter for relevant tests that are marked as best reps
        dsi_raw = df[(df["exercise"].isin(["CMJ", "IMTP"])) & df["is_best_rep"]]

        if dsi_raw.empty:
            return pd.DataFrame()

        dsi_list = []

        # Split into CMJ and IMTP dfs
        cmj_df = dsi_raw[dsi_raw["exercise"] == "CMJ"]
        imtp_df = dsi_raw[dsi_raw["exercise"] == "IMTP"]

        # Merge them on Athlete + Date
        dsi_merged = pd.merge(cmj_df, imtp_df, on=["athlete_id", "test_date"], suffixes=("", "_IMTP"))

        # Calculate DSI
        dsi_merged["DSI"] = dsi_merged["Peak Force [N]"] / dsi_merged["Peak Force [N]_IMTP"]

        result = {
            "athlete_id": dsi_merged["athlete_id"],
            "athlete_name": dsi_merged["athlete_name"],
            "test_date": dsi_merged["test_date"],
            "recorded_at": dsi_merged["recorded_at"],
            "trial_id": dsi_merged["trial_id"] + "_" + dsi_merged["trial_id_IMTP"],
            "CMJ Peak Force [N]": dsi_merged["Peak Force [N]"],
            "IMTP Peak Force [N]": dsi_merged["Peak Force [N]_IMTP"],
            "CMJ Peak Force [N/kg]": dsi_merged["Peak Force [N/kg]"],
            "IMTP Peak Force [N/kg]": dsi_merged["Peak Force [N/kg]_IMTP"],
            "DSI": dsi_merged["DSI"],
        }

        return pd.DataFrame(result)

    def download_vald_data(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Main execution method to download, process, and unify all VALD data.

        Flow:
        1. Fetch Users.
        2. Fetch & Process ForceDecks Trials.
        3. Fetch & Process ForceFrame/NordBord Trials.
        4. Filter for valid users.
        5. Flag best reps and handle unilateral merging.
        6. Combine all datasets.
        7. Calculate DSI.
        8. Final cleanup and formatting.

        Returns:
            tuple[pd.DataFrame, pd.DataFrame]: (Complete Processed Trials DataFrame, DSI DataFrame)
        """
        print("Syncing VALD data")
        print("Downloading users")
        users_df = fetch_profiles(self.vald_token, self.tenant_id, self.profiles_url)
        print(f"Downloaded {len(users_df)} users")

        fd_trials_df = self.process_fd_data()
        ff_nb_trials_df = self.process_ff_nb_data()

        # Filter trials to keep only valid athletes (ones with external ID)
        valid_ids = users_df["vald_id"].unique()

        if not fd_trials_df.empty:
            fd_trials_df = fd_trials_df[fd_trials_df["vald_id"].isin(valid_ids)]

        if not ff_nb_trials_df.empty:
            ff_nb_trials_df = ff_nb_trials_df[ff_nb_trials_df["vald_id"].isin(valid_ids)]

        if fd_trials_df.empty and ff_nb_trials_df.empty:
            print("No trials found for the selected timeframe")
            print("Vald processing completed!")
            return pd.DataFrame(), pd.DataFrame()

        print(f"Processing {len(fd_trials_df) + len(ff_nb_trials_df)} trials")

        # Uni/bilateral processing
        if not fd_trials_df.empty:
            # Mark best reps per day
            fd_trials_df = self.flag_best_rep(fd_trials_df)

            # Separate Bilateral and Unilateral tests
            bilateral_fd = fd_trials_df[fd_trials_df["laterality"].isin(["Bilateral"])]
            unilateral_fd = fd_trials_df[fd_trials_df["laterality"].isin(["Left", "Right"])]

            # Process unilateral tests
            if not unilateral_fd.empty:
                unilateral_processed = self.process_unilateral_fd(unilateral_fd)
            else:
                unilateral_processed = pd.DataFrame()
        else:
            bilateral_fd = pd.DataFrame()
            unilateral_processed = pd.DataFrame()

        # Combine: Bilateral FD + Unilateral FD + ForceFrame/NordBord
        fetched_trials = pd.concat([bilateral_fd, unilateral_processed, ff_nb_trials_df], ignore_index=True)

        # Merge with User profiles
        merged_trials_df = pd.merge(users_df, fetched_trials, on="vald_id", how="inner")

        # Deduplicate to ensure one record per trial ID
        merged_trials_df = merged_trials_df.drop_duplicates(subset=["trial_id"], keep="first")

        # Calculate Dynamic Strength Index
        dsi_df = self.calculate_dsi(merged_trials_df)

        # Calculate decimal age at the time of recording
        age_series = (
            merged_trials_df["recorded_at"].dt.tz_localize(None) - merged_trials_df["birth_date"].dt.tz_localize(None)
        ).dt.days / 365.25
        merged_trials_df["age_at_test"] = age_series.round(1)

        # Convert time to local Prague time
        merged_trials_df["recorded_at"] = merged_trials_df["recorded_at"].dt.tz_convert("Europe/Prague")

        # Drop temporary/redundant columns
        merged_trials_df.drop(columns=["birth_date", "vald_id"], inplace=True)

        # Rename Body Weight to weight
        merged_trials_df.rename(columns={"Body Weight [kg]": "weight"}, inplace=True)

        # Reorder columns: Meta first, then metrics alphabetically
        meta_cols = [
            "athlete_id",
            "athlete_name",
            "test_date",
            "recorded_at",
            "age_at_test",
            "session_id",
            "trial_id",
            "is_best_rep",
            "weight",
            "exercise",
            "laterality",
        ]

        meta_cols = [c for c in meta_cols if c in merged_trials_df.columns]
        data_cols = sorted([c for c in merged_trials_df.columns if c not in meta_cols])

        merged_trials_df = merged_trials_df[meta_cols + data_cols]

        print("Vald processing completed!")

        return users_df, merged_trials_df, dsi_df
