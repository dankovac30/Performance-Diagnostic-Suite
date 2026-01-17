from typing import Any

import numpy as np
import pandas as pd

from config import Config
from core.signal_processing import apply_butterworth_filter, find_speed_plateau
from sprint_science.profiler import SprintSpeedTimeProfiler
from sprint_science.step_analysis import StepAnalyzer
from vendors.motion_1080.decoder import decode_1080_samples
from vendors.motion_1080.sync import fetch_profiles, fetch_training_data


class SprintProcessor:
    """
    Orchestrates the processing of data retrieved from the 1080 Motion API.

    This class handles the ETL pipeline:
    1. Fetches raw data (runs, athletes, measurements).
    2. Merges and synchronizes data sources (especially time-variant metrics like weight).
    3. Filters data for valid testing sessions.
    4. Applies biomechanical analysis (F-V profiling, step analysis).
    5. Aggregates Load-Velocity profiles across sessions.

    Attributes:
        api_key (str): Authentication key for the 1080 Motion API.
        params (dict): Optional parameters for the API request.
    """

    def __init__(self, params: dict[str, Any] | None = None):
        self.api_key = Config.MOTION_API_KEY
        self.params = params

    def extract_raw_metrics(self, row: pd.Series, sample: pd.DataFrame) -> dict[str, float]:
        """
        Calculates basic kinematic variables from the raw time-series data.

        Args:
            row (pd.Series): Metadata of the specific run.
            sample (pd.DataFrame): High-frequency sampled data (time, speed, position).

        Returns:
            dict[str, float]: Basic metrics including 'top_speed', 'total_distance', 'total_time'.
        """
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
        }
        return result

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
        Analyzes step asymmetry (Step Length and Frequency) between left and right leg.

        Args:
            row (pd.Series): Metadata of the specific run.
            sample (pd.DataFrame): High-frequency sampled data (time, speed, position).

        Returns:
            dict[str, float]: Asymmetry indices for length and frequency.
        """
        spatiotemporal_data = sample
        starting_leg = row["side"]
        height = row["height"]
        weight = row["weight"]

        # Object instantiation
        analyzer = StepAnalyzer(
            starting_leg=starting_leg, raw_spatiotemporal_data=spatiotemporal_data, height=height, weight=weight
        )

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

        return {"length_asym": length_asym, "freq_asym": freq_asym}

    def analyze_technique(self, row: pd.Series, sample: pd.DataFrame) -> dict[str, float]:
        """
        Analyzes technical efficiency (Ratio of Forces) during acceleration and max velocity.

        Args:
            row (pd.Series): Metadata of the specific run.
            sample (pd.DataFrame): High-frequency sampled data (time, speed, position).

        Returns:
            dict[str, float]: Efficiency ratios (propulsion vs braking).
        """
        spatiotemporal_data = sample
        starting_leg = row["side"]
        height = row["height"]
        weight = row["weight"]

        # Object instantiation
        analyzer = StepAnalyzer(
            starting_leg=starting_leg, raw_spatiotemporal_data=spatiotemporal_data, height=height, weight=weight
        )

        acc_propulsion, acc_braking = analyzer.analyze_acc_technical_efficiency()
        maxv_propulsion, maxv_braking = analyzer.analyze_maxv_technical_efficiency()

        # Calculate efficiency ratio (braking / propulsion)
        acc_eff = -acc_braking / acc_propulsion
        maxv_eff = -maxv_braking / maxv_propulsion

        return {"acc_efficiency": acc_eff, "maxv_efficiency": maxv_eff}

    def analyze_sprint(self, row: pd.Series) -> pd.Series:
        """
        This function aggregates the outputs of multiple analysis functions.

        Args:
            row (pd.Series): Metadata of the specific run.

        Returns:
            pd.Series: The processed row with new columns.
        """
        # Decode sample
        sample = decode_1080_samples(row["sample_data"])

        # Calculates kinematic variables
        results = self.extract_raw_metrics(row, sample)

        # Only perform expensive biomechanical calculations on testing sessions
        if row["set_id"] in self.testing_sessions:
            results.update(self.calculate_profile(row, sample))
            results.update(self.analyze_steps(row, sample))
            results.update(self.analyze_technique(row, sample))

        return pd.Series(results).reindex(self.target_columns)

    def calculate_load_velocity(self) -> pd.DataFrame:
        """
        Calculates Load-Velocity (L-V) profiles by aggregating runs within testing sets.

        Returns:
            pd.DataFrame: A dataframe where each row is a L-V profile for a specific session.
        """
        lv_results = []

        for _, sesh_df in self.all_testing_runs.groupby("set_id"):
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
                "externalId": meta["externalId"],
                "lv_created": meta["run_created"],
                "L0": l0,
                "v0_lv": v0,
                "p_max_l": p_max_l,
                "p_max_rel_l": p_max_rel_l,
                "R2": r2,
                "lv_slope": slope,
            }

            lv_results.append(res)

        return pd.DataFrame(lv_results)

    def flag_publication_quality_sessions(self):
        """
        Identifies and flags sessions meeting full methodological standards for publication

        Updates:
            self.complete_df: Adds a boolean column 'publication_valid' (True/False).
        """
        # Create a working copy to perform intermediate aggregations without affecting the main DF
        working_df = self.complete_df.copy()

        # Identify valid ("v") and explicitly invalid ("i") runs
        flagged_runs = working_df[working_df["comment"] == "v"]
        flagged_dates = flagged_runs["run_created"].dt.date.unique()
        invalid_sessions = set(working_df[working_df["comment"] == "i"]["set_id"])

        working_df["date_created"] = working_df["run_created"].dt.date

        testing_sessions = []

        # Iterate through dates where at least one valid flag was found
        for date in flagged_dates:
            day_data = working_df[working_df["date_created"] == date]

            # Create a "fingerprint" for each session on that day:
            sesh_distance_dict = day_data.groupby("set_id")["total_distance"].apply(set).to_dict()

            # Get IDs of the manually verified sessions
            flagged_ids = day_data[day_data["comment"] == "v"]["set_id"].unique()

            # Extract the protocols used in the verified sessions
            testing_distances = [sesh_distance_dict[f_id] for f_id in flagged_ids]

            # Include any other session, that used the same protocol that day
            for session_id, running_distances in sesh_distance_dict.items():
                if running_distances in testing_distances:
                    testing_sessions.append(session_id)

        # Remove sessions explicitly marked as invalid
        valid_testing_sessions = set(testing_sessions) - invalid_sessions

        # Apply the final mask to the main dataframe
        final_mask = self.complete_df["set_id"].isin(valid_testing_sessions)
        self.complete_df["publication_valid"] = False
        self.complete_df.loc[final_mask, "publication_valid"] = True

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

        # Fetch Data
        users, measurements = fetch_profiles(self.api_key)
        fetched_runs = pd.DataFrame(fetch_training_data(self.api_key, self.params))

        # Loading from local cache for development
        # users = pd.read_pickle("dev/vendors/motion_1080/users.pkl")
        # measurements = pd.read_pickle("dev/vendors/motion_1080/measurements.pkl")
        # fetched_runs = pd.read_pickle("dev/vendors/motion_1080/runs.pkl")

        # Merge runs with user data
        users_runs = pd.merge(
            fetched_runs,
            users,
            left_on="client",
            right_on="id",
            how="left",
        )

        # Standardize timezones
        users_runs["run_created"] = pd.to_datetime(users_runs["run_created"], utc=True)
        measurements["date"] = pd.to_datetime(measurements["date"], utc=True, format="ISO8601")

        users_runs = users_runs.sort_values("run_created")
        measurements = measurements.sort_values("date")

        # Merge user measurements
        self.complete_df = pd.merge_asof(
            users_runs,
            measurements,
            left_on="run_created",
            right_on="date",
            by="externalId",
            direction="nearest",
        )

        # Calculate time diff between run and measurement to check validity
        self.complete_df["measurement_diff"] = (
            self.complete_df["run_created"] - self.complete_df["date"]
        ).abs().dt.total_seconds() / (24 * 3600)

        # Calculate Decimal Age
        self.complete_df["dateOfBirth"] = pd.to_datetime(self.complete_df["dateOfBirth"], utc=True)
        age_delta = self.complete_df["run_created"] - self.complete_df["dateOfBirth"]
        self.complete_df["age"] = age_delta.dt.days / 365.25

        # Basic quality filtering
        self.complete_df = self.complete_df[
            (self.complete_df["total_distance"] >= 2)
            & (self.complete_df["total_time"] > 1)
            & (self.complete_df["top_speed"] > 1)
        ].copy()

        # Identify standardized F-V testing sessions
        testing_mask = (
            (self.complete_df["total_distance"].between(44, 46))
            & (self.complete_df["mode"] == "resisted")
            & (self.complete_df["measurement_diff"] < 1.5)
            & (self.complete_df["load"] == 2)
        )

        # Split dataset into Testing and General training
        fv_testing_runs = self.complete_df[testing_mask]
        self.testing_sessions = list(fv_testing_runs["set_id"].dropna().unique())
        training_mask = ~self.complete_df["set_id"].isin(self.testing_sessions)
        self.all_testing_runs = self.complete_df[~training_mask]

        # Apply advanced analysis
        self.target_columns = [
            "top_speed",
            "total_distance",
            "total_time",
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
            "length_asym",
            "freq_asym",
            "acc_efficiency",
            "maxv_efficiency",
        ]

        # Apply row-wise processing
        self.complete_df[self.target_columns] = self.complete_df.apply(self.analyze_sprint, axis=1)

        # Clear measurement data for non-testing runs
        measurement_cols = ["weight", "height"]
        self.complete_df.loc[training_mask, measurement_cols] = pd.NA

        # Flag sessions with full methodological integrity
        self.flag_publication_quality_sessions()

        # Calculate Load-Velocity profiles
        load_velocity_df = self.calculate_load_velocity()

        # Final cleanup
        final_df = self.complete_df.drop(  # edit later
            columns=["client", "set_id", "id", "group", "date", "measurement_diff"]  # "displayName", "dateOfBirth",]
        )

        return final_df, load_velocity_df
