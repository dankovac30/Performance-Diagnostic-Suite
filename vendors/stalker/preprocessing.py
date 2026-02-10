import numpy as np
import pandas as pd

from core.signal_processing import apply_butterworth_filter, find_speed_plateau
from sprint_science.profiler import SprintSpeedTimeProfiler


def crop_start(spatiotemporal_data: pd.DataFrame, time_shift: float = 2.0) -> pd.DataFrame:
    """
    Detects the main acceleration phase based on maximum velocity gain and crops
    pre-movement noise or slow onset.

    Args:
        spatiotemporal_data (pd.DataFrame): DataFrame containing 'time' and speed columns.
        time_shift (float, optional): The duration of the sliding window used to detect the surge in velocity.

    Returns:
        pd.DataFrame: The cropped DataFrame starting from the onset of significant acceleration, with time reset to 0.0s.
    """
    # Calculate the sampling interval
    dt = spatiotemporal_data["time"].diff().mean()

    # Convert time_shift into number of rows
    row_shift = int(round(time_shift / dt))

    # Apply filter (2Hz) to determine the trend without noise
    butter_filter = apply_butterworth_filter(
        spatiotemporal_data, data_to_smooth="median_speed", padding="symmetric", cutoff_freq=2
    )
    spatiotemporal_data["smooth_speed"] = butter_filter

    # Calculate velocity gain during time_shift time window
    delta_v = spatiotemporal_data["smooth_speed"].shift(-row_shift) - spatiotemporal_data["smooth_speed"]

    # Find the index where the velocity gain is highest
    max_delta_v_idx = delta_v.idxmax()

    # Crop data starting from that index
    cropped_spatiotemporal_data = spatiotemporal_data.loc[max_delta_v_idx:].copy()

    # Reset the index and reset time
    cropped_spatiotemporal_data = cropped_spatiotemporal_data.reset_index(drop=True)
    cropped_spatiotemporal_data["time"] = (
        cropped_spatiotemporal_data["time"] - cropped_spatiotemporal_data["time"].iloc[0]
    )

    # Delete temporary smooth column
    cropped_spatiotemporal_data = cropped_spatiotemporal_data.drop("smooth_speed", axis=1)

    return cropped_spatiotemporal_data


def reconstruct_start(spatiotemporal_data):
    """
    Reconstructs the missing initial acceleration phase using the mono-exponential Morin sprint Model.

    Args:
        spatiotemporal_data (pd.DataFrame): The input data, starting at a non-zero velocity.

    Returns:
        pd.DataFrame: A complete dataset with synthetic start data prepended, starting at 0.0s and 0.0 m/s.
    """
    # Calculate the sampling interval
    dt = spatiotemporal_data["time"].diff().mean()

    # Cutoff post peak speed on filtered array
    smooth_speed_data = apply_butterworth_filter(spatiotemporal_data, data_to_smooth="median_speed", cutoff_freq=1.3)
    speed_plateau_idx = find_speed_plateau(smooth_speed_data)

    # Acceleration phase only
    acceleratin_data = spatiotemporal_data.iloc[:speed_plateau_idx].copy()

    # Initialize Profiler (using dummy weight/height as fit_morin only needs kinematics)
    profiler = SprintSpeedTimeProfiler(acceleratin_data, height=85, weight=190)

    # Fit Morin's model to extract macroscopic kinematic parameters
    _, _, params = profiler.fit_velocity_model(acceleratin_data["median_speed"], acceleratin_data["time"])

    tau = params["Tau"]
    v_max = params["V_max"]

    # Get the velocity of the first available frame
    init_speed = spatiotemporal_data["median_speed"].iloc[0]

    # Calculate 'missing time' using the inverse Morin equation
    missing_time = -tau * np.log(1 - (init_speed / v_max))

    # If missing time is less than one frame just re-zero and return
    if missing_time < dt:
        spatiotemporal_data["time"] = spatiotemporal_data["time"] - spatiotemporal_data["time"].iloc[0]
        return spatiotemporal_data

    # Generate time steps for the missing past
    synthetic_time = np.arange(-missing_time, -dt / 2, dt)

    # Calculate synthetic velocities for these time steps
    real_time = synthetic_time + missing_time
    synthetic_speed = v_max * (1 - np.exp(-real_time / tau))

    # Create DataFrame for the synthetic data
    synthetic_data = pd.DataFrame({"time": synthetic_time, "median_speed": synthetic_speed})

    # Stitch synthetic history with real measurements and re-zero
    full_data = pd.concat([synthetic_data, spatiotemporal_data], ignore_index=True)
    full_data["time"] = full_data["time"] - full_data["time"].iloc[0]

    return full_data


def crop_end(spatiotemporal_data: pd.DataFrame) -> pd.DataFrame:
    """
    Trims the end of the dataset based on significant velocity drop-off.

    Args:
        spatiotemporal_data (pd.DataFrame): The full run data.

    Returns:
        pd.DataFrame: The dataset trimmed at the point of significant deceleration.
    """
    # Smooth data to find the top speed
    smooth_speed_data = apply_butterworth_filter(spatiotemporal_data, data_to_smooth="median_speed", cutoff_freq=1.3)
    top_speed_idx = find_speed_plateau(smooth_speed_data)
    top_speed = smooth_speed_data[top_speed_idx]

    # Analyze only the data agter the top speed
    post_top_speed_data = smooth_speed_data[top_speed_idx:]

    # Create a mask where speed drops below 80% of top speed
    mask = post_top_speed_data < top_speed * 0.8

    # Find the first index where this condition is met
    if np.any(mask):
        fatigue_idx = np.argmax(mask) + top_speed_idx
    else:
        fatigue_idx = None

    # Slice the dataframe up to the detected end point
    spatiotemporal_data = spatiotemporal_data.loc[:fatigue_idx]

    return spatiotemporal_data
