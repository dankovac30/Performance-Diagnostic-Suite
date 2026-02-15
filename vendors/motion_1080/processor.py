from typing import Any

import numpy as np
import pandas as pd

from config import Config
from core.signal_processing import apply_butterworth_filter, find_speed_plateau
from sprint_science.profiler import SprintSpeedTimeProfiler
from sprint_science.step_analysis import StepAnalyzer
from vendors.motion_1080.decoder import decode_1080_samples
from vendors.motion_1080.sync import fetch_profiles, fetch_training_data


class Motion1080Processor:
    """
    Orchestrates the processing of data retrieved from the 1080 Motion API.

    This class handles the ETL pipeline:
    1. Fetches raw data (runs, athletes, measurements).
    2. Merges and synchronizes data sources (especially time-variant metrics like weight).
    3. Filters data for valid testing sessions.
    4. Applies biomechanical analysis (F-V profiling, step analysis).
    5. Aggregates Load-Velocity profiles across sessions.

    Attributes:
        params (dict): Optional parameters for the API request.
    """

    def __init__(self, params: dict[str, Any] | None = None):
        self.params = params
        self.api_key = Config.MOTION_API_KEY
        self.base_api_url = Config.MOTION_BASE_URL
        self.profiles_endpoint = Config.MOTION_CLIENT_ENDPOINT
        self.sessions_endpoint = Config.MOTION_SESSIONS_ENDPOINT
        self.training_endpoint = Config.MOTION_TRAINING_DATA_ENDPOINT

    def extract_raw_metrics(self, row: pd.Series) -> dict[str, float]:
        """
        Calculates basic kinematic variables from the raw time-series data.

        Args:
            row (pd.Series): Metadata of the specific run.
            sample (pd.DataFrame): High-frequency sampled data (time, speed, position).

        Returns:
            dict[str, float]: Basic metrics including 'top_speed', 'total_distance', 'total_time'.
        """
        # Decode sample
        sample = decode_1080_samples(row["sample_data"])

        total_time = sample["time"].max()

        # Invert values, because API returns negative speed/position values for assisted runs
        if row["mode"] == "assisted":
            sample["speed"] = -sample["speed"]
            sample["position"] = -sample["position"]

        # Normalize position to start at 0
        zero_position = sample.iloc[0]["position"]
        sample["distance"] = sample["position"] - zero_position

        total_distance = round(sample["distance"].max(), 0)

        # Apply signal processing to find true top speed
        smooth_speed_array = apply_butterworth_filter(
            raw_spatiometric_data=sample, data_to_smooth="speed", padding="symmetric"
        )
        top_speed = smooth_speed_array[find_speed_plateau(smooth_speed_array)]

        result = {
            "top_speed": top_speed,
            "total_distance": total_distance,
            "total_time": total_time,
            "sample_df": sample,
        }
        return pd.Series(result)

    def calculate_profile(self, row: pd.Series, sample: pd.DataFrame) -> dict[str, float]:
        """
        Computes the Force-Velocity (F-V) profile using the Morin method.

        Args:
            row (pd.Series): Metadata of the specific run.
            sample (pd.DataFrame): High-frequency sampled data (time, speed, position).

        Returns:
            dict[str, float]: F-V metrics (F0, V0, Pmax, Tau, etc.).
        """
        spatiotemporal_data = sample
        height = row["height"]
        weight = row["weight"]

        # Object instantiation and profile calculation
        profiler = SprintSpeedTimeProfiler(raw_spatiotemporal_data=spatiotemporal_data, height=height, weight=weight)
        profile_dict = profiler.calculate_profile()

        keys_to_export = ["F0", "F0_abs", "V0", "Pmax", "F_V_slope", "Rf_max", "DRF", "Model_Adherence", "Tau", "t0"]

        return {key: profile_dict[key] for key in keys_to_export}

    def analyze_steps(self, row: pd.Series, sample: pd.DataFrame) -> dict[str, float | None]:
        """
        Analyzes step asymmetry (Step Length and Frequency) between left and right leg
        and technical efficiency (Ratio of Forces) during acceleration and max velocity.

        Args:
            row (pd.Series): Metadata of the specific run.
            sample (pd.DataFrame): High-frequency sampled data (time, speed, position).

        Returns:
            dict[str, float]: Asymmetry for length and frequency, efficiency ratios (propulsion vs braking).
        """
        spatiotemporal_data = sample
        starting_leg = row["starting_leg"]
        height = row["height"]
        weight = row["weight"]

        # Object instantiation
        analyzer = StepAnalyzer(
            starting_leg=starting_leg, raw_spatiotemporal_data=spatiotemporal_data, height=height, weight=weight
        )

        # Analyze steps
        steps_df = analyzer.analyze_steps()

        # Calculate means for steps > 4
        left_length = steps_df.loc[(steps_df["leg"] == "left") & (steps_df["step_number"] > 4), "step_length"].mean()
        right_length = steps_df.loc[(steps_df["leg"] == "right") & (steps_df["step_number"] > 4), "step_length"].mean()
        left_freq = steps_df.loc[(steps_df["leg"] == "left") & (steps_df["step_number"] > 4), "step_freq"].mean()
        right_freq = steps_df.loc[(steps_df["leg"] == "right") & (steps_df["step_number"] > 4), "step_freq"].mean()

        # Calculate Asymmetry Index: (R - L) / Average
        length_asym = None
        if pd.notna(left_length) and pd.notna(right_length):
            length_asym = (right_length - left_length) / ((right_length + left_length) / 2)

        freq_asym = None
        if pd.notna(left_freq) and pd.notna(right_freq):
            freq_asym = (right_freq - left_freq) / ((right_freq + left_freq) / 2)

        # Analyze technique
        acc_propulsion, acc_braking = analyzer.analyze_acc_technical_efficiency()
        maxv_propulsion, maxv_braking = analyzer.analyze_maxv_technical_efficiency()

        # Calculate efficiency ratio (braking / propulsion)
        acc_eff = -acc_braking / acc_propulsion
        maxv_eff = -maxv_braking / maxv_propulsion

        return {
            "step_length_asym": length_asym,
            "step_freq_asym": freq_asym,
            "acc_efficiency": acc_eff,
            "maxv_efficiency": maxv_eff,
        }

    def analyze_sprint(self, row: pd.Series) -> pd.Series:
        """
        This function aggregates the outputs of multiple analysis functions.

        Args:
            row (pd.Series): Metadata of the specific run.

        Returns:
            pd.Series: The processed row with new columns.
        """
        sample = row["sample_df"]
        results = {}

        # Only perform biomechanical calculations on testing sessions
        if row["session_id"] in self.testing_sessions:
            results.update(self.calculate_profile(row, sample))
            results.update(self.analyze_steps(row, sample))

        return pd.Series(results).reindex(self.target_columns)

    def calculate_load_velocity(self) -> pd.DataFrame:
        """
        Calculates Load-Velocity (L-V) profiles by aggregating runs within testing sets.

        Returns:
            pd.DataFrame: A dataframe where each row is a L-V profile for a specific session.
        """
        lv_results = []

        for _, sesh_df in self.all_testing_runs.groupby("session_id"):
            # Minimum 3 data points (loads)
            if sesh_df["load"].nunique() < 3:
                continue

            meta = sesh_df.iloc[0]

            # Get max speed for each load to build the profile
            load_v_data = sesh_df.groupby("load")["top_speed"].max().reset_index()

            # Requirement: Must include light load (<5kg) and heavy load (>20% BW)
            if not ((load_v_data["load"].min() < 5) & (load_v_data["load"].max() > (meta["weight"] * 0.2))):
                continue

            # Linear regression: Top speed vs Load
            slope, intercept = np.polyfit(load_v_data["top_speed"], load_v_data["load"], 1)

            # Calculate physical parameters from regression
            l0 = intercept  # Theoretical load at 0 velocity
            v0 = -intercept / slope  # Theoretical velocity at 0 load
            p_max_l = l0 / 2  # Max Power (Load based)
            p_max_rel_l = p_max_l / meta["weight"]
            r2 = np.corrcoef(load_v_data["top_speed"], load_v_data["load"])[0, 1] ** 2

            res = {
                "athlete_id": meta["athlete_id"],
                "lv_created": meta["recorded_at"],
                "L0": l0,
                "v0_lv": v0,
                "p_max_l": p_max_l,
                "p_max_rel_l": p_max_rel_l,
                "R2": r2,
                "lv_slope": slope,
            }

            lv_results.append(res)

        return pd.DataFrame(lv_results)

    def flag_indoor_sessions(self) -> None:
        """
        Identifies and flags sessions held in controlled indoor enviroment.

        Updates:
            self.merged_df: Adds a boolean column 'is_indoor' (True/False).
        """
        # Create a working copy to perform intermediate aggregations without affecting the main DF
        working_df = self.merged_df.copy()

        # Identify indoor (i) runs
        flagged_runs = working_df[working_df["comment"] == "i"]
        flagged_dates = flagged_runs["recorded_at"].dt.date.unique()

        testing_sessions = []

        # Iterate through dates where at least one valid flag was found
        for date in flagged_dates:
            day_data = working_df[working_df["test_date"] == date]

            # Create a "fingerprint" for each session on that day:
            sesh_distance_dict = day_data.groupby("session_id")["total_distance"].apply(set).to_dict()

            # Get IDs of the manually verified sessions
            flagged_ids = day_data[day_data["comment"] == "i"]["session_id"].unique()

            # Extract the protocols used in the verified sessions
            testing_distances = [sesh_distance_dict[f_id] for f_id in flagged_ids]

            # Include any other session, that used the same protocol that day
            for session_id, running_distances in sesh_distance_dict.items():
                if running_distances in testing_distances:
                    testing_sessions.append(session_id)

        # Apply the final mask to the main dataframe
        final_mask = self.merged_df["session_id"].isin(testing_sessions)
        self.merged_df["is_indoor"] = False
        self.merged_df.loc[final_mask, "is_indoor"] = True

    def flag_best_rep(self) -> None:
        """
        Identify and flag the 'best' repetition for each athlete's daily session based on key performance metrics.

        Deciding Metrics:
            - FV run (45m 2kg): Max Power
            - Any other run: Top Speed

        Updates:
            self.merged_df: Adds 'is_best_rep' boolean column.
        """
        # Initialize the flag column with False for all rows
        self.merged_df["is_best_rep"] = False

        # Split dataset into FV profiling runs and other training runs
        fv_runs = self.merged_df[self.testing_mask]
        rest_runs = self.merged_df[~self.testing_mask]

        # Process FV runs: Group by Day and Athlete to find one best rep per session
        if not fv_runs.empty:
            best_fv_reps = fv_runs.groupby(["test_date", "athlete_id"])["Pmax"].idxmax()
            self.merged_df.loc[best_fv_reps, "is_best_rep"] = True

        # Process other runs: We group by 'load', flag the best rep for each load.
        if not rest_runs.empty:
            best_rest_reps = rest_runs.groupby(["test_date", "athlete_id", "load"])["top_speed"].idxmax()
            self.merged_df.loc[best_rest_reps, "is_best_rep"] = True

    def download_1080_data(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Primary entry point for downloading and processing 1080 data.

        1. Fetches data via API.
        2. Merges user data and runs measurements.
        3. Filters data for data quality.
        4. Identifies F-V testing sessions and L-V training sessions.
        5. Computes all metrics.

        Returns:
            tuple[pd.DataFrame, pd.DataFrame]:
                - final_df: Detailed run-by-run data.
                - load_velocity_df: Session-aggregated Load-Velocity profiles.
        """

        print("Syncing 1080 Motion data")
        print("Downloading users")
        users_df, biometrics_df = fetch_profiles(self.api_key, self.base_api_url, self.profiles_endpoint)
        print(f"Downloaded {len(users_df)} users")
        raw_runs_df = pd.DataFrame(
            fetch_training_data(
                self.api_key, self.base_api_url, self.sessions_endpoint, self.training_endpoint, self.params
            )
        )

        if raw_runs_df.empty:
            print("No trials found for the selected timeframe")
            print("1080 Motion processing completed")
            return pd.DataFrame(), pd.DataFrame()

        print(f"Processing {len(raw_runs_df)} trials")

        # Merge runs with user data
        users_runs = pd.merge(
            raw_runs_df,
            users_df,
            on="motion_id",
            how="left",
        )

        # Standardize timezones
        users_runs["recorded_at"] = pd.to_datetime(users_runs["recorded_at"], utc=True)
        biometrics_df["measured_at"] = pd.to_datetime(biometrics_df["measured_at"], utc=True, format="ISO8601")

        users_runs = users_runs.sort_values("recorded_at")
        biometrics_df = biometrics_df.sort_values("measured_at")

        # Merge user measurements
        self.merged_df = pd.merge_asof(
            users_runs,
            biometrics_df,
            left_on="recorded_at",
            right_on="measured_at",
            by="athlete_id",
            direction="nearest",
        )

        # Calculate time diff between run and measurement to check validity
        self.merged_df["measurement_diff"] = (
            self.merged_df["recorded_at"] - self.merged_df["measured_at"]
        ).abs().dt.total_seconds() / (24 * 3600)

        # Calculate basic kinematic variables
        self.merged_df[["top_speed", "total_distance", "total_time", "sample_df"]] = self.merged_df.apply(
            self.extract_raw_metrics, axis=1
        )

        # Basic quality filtering
        self.merged_df = self.merged_df[
            (self.merged_df["total_distance"] >= 2)
            & (self.merged_df["total_time"] > 1)
            & (self.merged_df["top_speed"] > 1)
        ].copy()

        # Identify standardized F-V testing sessions
        self.testing_mask = (
            (self.merged_df["total_distance"].between(44, 46))
            & (self.merged_df["mode"] == "resisted")
            & (self.merged_df["measurement_diff"] < 1.5)
            & (self.merged_df["load"] == 2)
        )

        # Split dataset into Testing and General training
        fv_testing_runs = self.merged_df[self.testing_mask]
        self.testing_sessions = set(fv_testing_runs["session_id"].dropna().unique())
        training_mask = ~self.merged_df["session_id"].isin(self.testing_sessions)
        self.all_testing_runs = self.merged_df[~training_mask]

        # Apply advanced analysis
        self.target_columns = [
            "F0",
            "F0_abs",
            "V0",
            "Pmax",
            "F_V_slope",
            "Rf_max",
            "DRF",
            "Model_Adherence",
            "Tau",
            "t0",
            "step_length_asym",
            "step_freq_asym",
            "acc_efficiency",
            "maxv_efficiency",
        ]

        # Apply row-wise processing
        self.merged_df[self.target_columns] = self.merged_df.apply(self.analyze_sprint, axis=1)

        # Drop sample dataframes from memory
        self.merged_df.drop(columns=["sample_df"], inplace=True)

        # Clear testing metadata for non-testing runs
        testing_metadata = ["weight", "height", "starting_leg"]
        self.merged_df.loc[training_mask, testing_metadata] = pd.NA

        # Normalize timestamps to date-only format
        self.merged_df["test_date"] = self.merged_df["recorded_at"].dt.date

        # Flag indoor sessions
        self.flag_indoor_sessions()

        # Mark best reps per day
        self.flag_best_rep()

        # Calculate Load-Velocity profiles
        load_velocity_df = self.calculate_load_velocity()

        # Calculate decimal age at the time of recording
        age_series = (
            self.merged_df["recorded_at"].dt.tz_localize(None) - self.merged_df["birth_date"].dt.tz_localize(None)
        ).dt.days / 365.25
        self.merged_df["age_at_test"] = age_series.round(1)

        # Convert time to local Prague time
        self.merged_df["recorded_at"] = self.merged_df["recorded_at"].dt.tz_convert("Europe/Prague")

        # Define final schema
        meta_cols = [
            "athlete_id",
            "athlete_name",
            "test_date",
            "recorded_at",
            "birth_date",  # delete later
            "age_at_test",
            "session_id",
            "trial_id",
            "is_best_rep",
            "weight",
            "height",
            "exercise",
            "mode",
            "load",
            "total_distance",
            "total_time",
            "top_speed",
            "starting_leg",
            "is_indoor",
            "sample_data",
        ]

        data_cols = [
            "F0",
            "F0_abs",
            "V0",
            "Pmax",
            "F_V_slope",
            "Rf_max",
            "DRF",
            "Model_Adherence",
            "Tau",
            "t0",
            "step_length_asym",
            "step_freq_asym",
            "acc_efficiency",
            "maxv_efficiency",
        ]

        processed_runs_df = self.merged_df[meta_cols + data_cols].copy()

        print("1080 Motion processing completed!")

        from dev.excel_export import export_to_excel

        export_to_excel(processed_runs_df, "processed_runs_df")

        return users_df, processed_runs_df, load_velocity_df
