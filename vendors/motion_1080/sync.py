"""
1080 Motion Data Synchronization Module.

This module handles the extraction and transformation of data from the 1080 Motion API.
It includes functions to retrieve athlete profiles, historical anthropometric measurements
and training data. It creates a bridge between the nested JSON structure
of the API and flat structures suitable for Pandas DataFrames.
"""

from typing import Any

import pandas as pd

from vendors.motion_1080.client import fetch_data


def fetch_profiles(api_key: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Fetches athlete profiles and their historical anthropometric measurements. This
    function retrieves the full client list, validates the existence and uniqueness
    of 'externalId' (used for linking with other systems), and extracts historical
    weight and height data into a separate structure.

    Args:
        api_key (str): API key for authentication.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: A tuple containing two DataFrames:
            1. df_user_info: Basic athlete metadata (id, name, DOB, externalId).
            2. df_user_measurements: Time-series of height and weight measurements.

    Raises:
        ValueError: If any user is missing an 'externalId' or if duplicate
                    'externalId's are found in the system.
    """
    CLIENTS_URL = "/Client"

    # Fetch data from the API
    data = fetch_data(api_key, CLIENTS_URL)

    # Process user information
    df_user_info = pd.DataFrame(data)
    df_user_info = df_user_info[["id", "displayName", "dateOfBirth", "group", "externalId"]]

    # Check for missing External IDs
    if not df_user_info["externalId"].notna().all():
        missing_df = df_user_info[df_user_info["externalId"].isna()]
        na_id_username = missing_df["displayName"].tolist()
        error_message = f"Users {na_id_username} missing externalId"
        raise ValueError(error_message)

    # Check for duplicate External IDs
    if df_user_info["externalId"].duplicated().any():
        duplicate_df = df_user_info[df_user_info["externalId"].duplicated(keep=False)]
        duplicate_username = duplicate_df["displayName"].tolist()
        error_message = f"Users {duplicate_username} have duplicate external IDs"
        raise ValueError(error_message)

    # Unpack the nested list of historical measurements for each user
    user_measurements = []

    for user in data:
        external_id = user["externalId"]
        measurements = user["historicalMeasurements"]

        for record in measurements:
            date = record["entryDate"]
            height = record["height"]
            weight = record["weight"]

            measurement = {"externalId": external_id, "date": date, "height": height, "weight": weight}
            user_measurements.append(measurement)

    df_user_measurements = pd.DataFrame(user_measurements)

    return df_user_info, df_user_measurements


def fetch_sessions(api_key: str, params: dict[str, Any] | None = None) -> list[str]:
    """
    Searches for and retrieves a list of session IDs based on filter parameters.

    Args:
        api_key (str): API key for authentication.
        params dict[str, Any] | None: Query parameters for filtering sessions

    Returns:
        list[str]: A list of unique session IDs (UUID strings).
    """
    SESSIONS_URL = "/Session/Search"

    # Fetch data from the API
    sessions = fetch_data(api_key, SESSIONS_URL, params)

    # Extract just the IDs from the session
    sessions_list = [s["id"] for s in sessions]

    return sessions_list


def fetch_training_data(api_key: str, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """
    Fetches detailed training data (runs) for all available sessions.

    Args:
        api_key (str): API key for authentication.
        params dict[str, Any] | None: Parameters passed to `fetch_sessions`
            to limit the scope of data (e.g., date range).

    Returns:
        list[dict[str, Any]]: A list of dictionaries, where each dictionary represents
        a single run/sprint with flattened metadata and signal data.
    """

    TRAINING_DATA_URL = "/TrainingData/Session/"

    # Get the list of session IDs to process
    session_list = fetch_sessions(api_key, params)

    fetched_runs = []

    # Iterate through each session and fetch detailed data
    for session in session_list:
        # Construct the specific URL for this session
        session_url = TRAINING_DATA_URL + session

        # Fetch specific session data
        training = fetch_data(api_key, session_url, params={"includeSamples": "true", "filterMode": "None"})

        # Flatten the JSON structure
        for training_set in training:
            client = training_set["clientId"]
            exercise = training_set["exerciseName"]
            session_id = training_set["setId"]

            for motion_group in training_set["motionGroups"]:
                run_id = motion_group["id"]
                side = motion_group["side"]

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
                        "client": client,
                        "set_id": session_id,
                        "exercise": exercise,
                        "run_id": run_id,
                        "run_created": created,
                        "side": side,
                        "total_distance": distance,
                        "total_time": time,
                        "top_speed": top_speed,
                        "mode": mode,
                        "load": load,
                        "sample_data": sample_data,
                    }

                    fetched_runs.append(run_data)

    return fetched_runs
