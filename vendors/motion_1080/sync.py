"""
1080 Motion Data Synchronization Module.

This module handles the extraction and transformation of data from the 1080 Motion API.
It includes functions to retrieve athlete profiles, historical anthropometric measurements
and training data. It creates a bridge between the nested JSON structure
of the API and flat structures suitable for Pandas DataFrames.
"""

from typing import Any

import pandas as pd
from tqdm import tqdm

from vendors.motion_1080.client import fetch_data


def fetch_profiles(api_key: str, base_url: str, endpoint: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Fetches athlete profiles and their historical anthropometric measurements. This
    function retrieves the full client list, validates the existence and uniqueness
    of 'externalId' (used for linking with other systems), and extracts historical
    weight and height data into a separate structure.

    Args:
        api_key (str): API key for authentication.
        base_url (str): The root URL of the 1080 Motion API.
        endpoint (str): Specific path to the profiles resource.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: A tuple containing two DataFrames:
            1. df_user_info: Basic athlete metadata (id, name, DOB, externalId).
            2. df_user_measurements: Time-series of height and weight measurements.

    Raises:
        ValueError: If any user is missing an 'externalId' or if duplicate
                    'externalId's are found in the system.
    """

    # Fetch data from the API
    data = fetch_data(api_key, base_url, endpoint)

    if not data:
        raise ValueError("No profiles to fetch from 1080 Motion")

    # Process user information
    users_df = pd.DataFrame(data)
    users_df = users_df[["id", "displayName", "dateOfBirth", "group", "externalId"]]

    # Check for missing External IDs
    if not users_df["externalId"].notna().all():
        missing_df = users_df[users_df["externalId"].isna()]
        na_id_username = missing_df["displayName"].tolist()
        error_message = f"Users {na_id_username} missing externalId"
        raise ValueError(error_message)

    # Check for duplicate External IDs
    if users_df["externalId"].duplicated().any():
        duplicate_df = users_df[users_df["externalId"].duplicated(keep=False)]
        duplicate_username = duplicate_df["displayName"].tolist()
        error_message = f"Users {duplicate_username} have duplicate external IDs"
        raise ValueError(error_message)

    # Map API response fields to standardized internal schema
    rename_map = {
        "id": "motion_id",
        "displayName": "athlete_name",
        "dateOfBirth": "birth_date",
        "externalId": "athlete_id",
    }
    users_df = users_df.rename(columns=rename_map)

    # Convert birth dates to datetime objects
    users_df["birth_date"] = pd.to_datetime(users_df["birth_date"]).dt.tz_localize(None)

    # Unpack the nested list of historical measurements for each user
    user_biometrics = []

    for user in data:
        athlete_id = user["externalId"]
        measurements = user["historicalMeasurements"]

        for record in measurements:
            date = record["entryDate"]
            height = record["height"]
            weight = record["weight"]

            measurement = {"athlete_id": athlete_id, "measured_at": date, "height": height, "weight": weight}
            user_biometrics.append(measurement)

    biometrics_df = pd.DataFrame(user_biometrics)

    # Standardize timestamps to UTC and convert to local Prague time
    biometrics_df["measured_at"] = pd.to_datetime(biometrics_df["measured_at"], format="ISO8601", utc=True)
    biometrics_df["measured_at"] = biometrics_df["measured_at"].dt.tz_convert("Europe/Prague")

    return users_df, biometrics_df


def fetch_sessions(api_key: str, base_url: str, endpoint: str, params: dict[str, Any] | None = None) -> list[str]:
    """
    Searches for and retrieves a list of session IDs based on filter parameters.

    Args:
        api_key (str): API key for authentication.
        base_url (str): The root URL of the 1080 Motion API.
        endpoint (str): Specific path to the sessions resource.
        params dict[str, Any] | None: Query parameters for filtering sessions

    Returns:
        list[str]: A list of unique session IDs (UUID strings).
    """

    # Fetch data from the API
    sessions = fetch_data(api_key, base_url, endpoint, params)

    # Extract just the IDs from the session
    sessions_list = [s.get("id") for s in sessions]

    return sessions_list


def fetch_training_data(
    api_key: str,
    base_url: str,
    sessions_endpoint: str,
    training_data_endpoint: str,
    params: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """
    Fetches detailed training data (runs) for all available sessions.

    Args:
        api_key (str): API key for authentication.
        base_url (str): The root URL of the 1080 Motion API.
        endpoint (str): Specific path to the training data resource.
        params dict[str, Any] | None: Parameters passed to `fetch_sessions`
            to limit the scope of data (e.g., date range).

    Returns:
        list[dict[str, Any]]: A list of dictionaries, where each dictionary represents
        a single run/sprint with flattened metadata and signal data.
    """

    # Get the list of session IDs to process
    sessions_list = fetch_sessions(api_key, base_url, sessions_endpoint, params)

    if not sessions_list:
        return []

    fetched_trials = []

    # Iterate through each session and fetch detailed data
    for session in tqdm(sessions_list, desc="Downloading sessions", unit="session"):
        # Construct the specific URL for this session
        session_endpoint = training_data_endpoint + session

        # Fetch specific session data
        training = fetch_data(
            api_key, base_url, session_endpoint, params={"includeSamples": "true", "filterMode": "None"}
        )

        # Flatten the JSON structure
        for training_set in training:
            motion_id = training_set["clientId"]
            exercise = training_set["exerciseName"]
            session_id = training_set["setId"]

            for motion_group in training_set["motionGroups"]:
                trial_id = motion_group["id"]
                side = motion_group["side"]
                comment = motion_group.get("comment", "")

                for run in motion_group["motions"]:
                    # Extract run metrics
                    created = run["created"]
                    distance = run["totalDistance"]
                    time = run["totalTime"]
                    top_speed = run["topSpeed"]
                    is_eccentric = run["isEccentric"]
                    sample_data = run["sampleData"]

                    resistance_values = run["resistanceValues"]

                    # Determine mode (resisted or assisted)
                    if is_eccentric == False:
                        mode = "resisted"
                        load = resistance_values["concentricLoad"]
                    else:
                        mode = "assisted"
                        load = resistance_values["eccentricLoad"]

                    # Construct the flat dictionary
                    run_data = {
                        "motion_id": motion_id,
                        "session_id": session_id,
                        "exercise": exercise,
                        "trial_id": trial_id,
                        "recorded_at": created,
                        "starting_leg": side,
                        "comment": comment,
                        "total_distance": distance,
                        "total_time": time,
                        "top_speed": top_speed,
                        "mode": mode,
                        "load": load,
                        "sample_data": sample_data,
                    }

                    fetched_trials.append(run_data)

    return fetched_trials
