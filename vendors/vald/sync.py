"""
VALD Motion Data Synchronization Module.

This module handles the extraction and transformation of data from the VALD API.
It includes functions to retrieve athlete profiles and training data.
"""

import time
from typing import Any

import pandas as pd
import requests
from tqdm import tqdm

from vendors.vald.client import fetch_data


def fetch_profiles(vald_token: str, tenant_id: str, url: str) -> pd.DataFrame:
    """
    Fetches athlete profiles from the VALD API.

    Args:
        vald_token (str): The OAuth2 access token.
        tenant_Id (str): The unique identifier for your VALD account.
        url (str): The specific API url path.

    Returns:
        pd.DataFrame: Basic athlete metadata (id, name, date of birth, externalId).

    Raises:
        ValueError: If the API returns no data.
        ValueError: If duplicate 'externalId's are found in the system.
    """

    # Fetch data from the API
    data = fetch_data(vald_token, tenant_id, url)

    if not data:
        raise ValueError("No profiles to fetch from Vald")

    # Process user information
    users_info = data["profiles"]
    users_df = pd.DataFrame(users_info)
    users_df["athlete_name"] = users_df["givenName"].str.strip() + " " + users_df["familyName"].str.strip()
    users_df = users_df[["profileId", "athlete_name", "dateOfBirth", "externalId"]]
    users_df.dropna(subset=["externalId"], inplace=True)

    # Check for duplicate External IDs
    if users_df["externalId"].duplicated().any():
        duplicate_df = users_df[users_df["externalId"].duplicated(keep=False)]
        duplicate_username = duplicate_df["athlete_name"].tolist()
        error_message = f"Users {duplicate_username} have duplicate external IDs"
        raise ValueError(error_message)

    # Map API response fields to standardized internal schema
    rename_map = {
        "profileId": "vald_id",
        "dateOfBirth": "birth_date",
        "externalId": "athlete_id",
    }
    users_df = users_df.rename(columns=rename_map)

    # Convert birth dates to datetime objects
    users_df["birth_date"] = pd.to_datetime(users_df["birth_date"]).dt.tz_localize(None)

    return users_df


def fetch_tests(
    vald_token: str, tenant_id: str, base_url: str, endpoint: str, params: dict = None
) -> list[dict[str, Any]]:
    """
    Fetches a list of tests from the VALD API, iterates through API pages to retrieve
    all available tests matching the parameters.

    Args:
        vald_token (str): The OAuth2 access token.
        tenant_Id (str): The unique identifier for your VALD account.
        base_url (str): The base URL of the API.
        endpoint (str): The specific endpoint (e.g., '/tests').
        params (dict, optional): Query parameters for the API call (e.g., date range).

    Returns:
        list[dict[str, Any]]: A list of dictionaries, where each dictionary represents a test.
    """
    url = base_url + endpoint

    # Create a copy of params to avoid modifying the original dictionary reference
    current_params = params.copy() if params else {}

    tests_list = []

    # Safety limit: max 100 pages to prevent infinite loops
    for _ in range(100):
        # Fetch data from the API
        tests = fetch_data(vald_token, tenant_id, url, current_params)

        batch = tests.get("tests", [])

        if not batch:
            break

        tests_list.extend(batch)

        # Find the latest modification date in the current batch.
        last_test_date = max([t["modifiedDateUtc"] for t in batch])

        # Add 1 millisecond to the last date to start the next batch *after* this record
        new_start_date = (pd.to_datetime(last_test_date) + pd.Timedelta(milliseconds=1)).strftime(
            "%Y-%m-%dT%H:%M:%S.%f"
        )[:-3] + "Z"

        # Update params for the next iteration
        current_params["ModifiedFromUtc"] = new_start_date

    return tests_list


def fetch_force_decks_trials_data(
    vald_token: str,
    tenant_id: str,
    team_uid: str,
    base_url: str,
    tests_endpoint: str,
    trials_endpoint: str,
    params: dict = None,
) -> list[dict[str, Any]]:
    """
    Fetches and processes trial data specifically for ForceDecks.

    Args:
        vald_token (str): The OAuth2 access token.
        tenant_id (str): The unique identifier for your VALD account.
        team_uid (str): The specific Team ID.
        base_url (str): The base URL of the API.
        tests_endpoint (str): Endpoint to fetch the list of tests.
        trials_endpoint (str): Endpoint format string to fetch trials for a specific test.
                               Expected format: ".../tests/{test_id}/trials"
        params (dict, optional): Query parameters.

    Returns:
        list[dict[str, Any]]: A list of dictionaries, where each dictionary represents single rep.
    """
    # Get the list of test sessions
    tests_list = fetch_tests(vald_token, tenant_id, base_url, tests_endpoint, params)

    if not tests_list:
        return []

    fetched_trials = []

    # Get into each test to get trial data
    for test in tqdm(tests_list, desc="Downloading tests", unit="test"):
        # Extract Test Metadata
        vald_id = test["profileId"]
        session_id = test["testId"]
        exercise = test["testType"]
        test_created = test["recordedDateUtc"]
        base_time = pd.to_datetime(test_created)

        # Construct URL for this test's trials
        current_endpoint = trials_endpoint.format(team_uid=team_uid, test_id=session_id)
        url = f"{base_url}{current_endpoint}"

        # Fetch trial data with retry logic
        try:
            trials = fetch_data(vald_token, tenant_id, url)

        except requests.exceptions.ReadTimeout:
            time.sleep(5)
            trials = fetch_data(vald_token, tenant_id, url)

        # Process each individual trial (rep) within the test
        for trial in trials:
            laterality = trial["limb"]

            if laterality == "Both":
                laterality = "Bilateral"

            # Calculate precise timestamp for the specific rep
            start_time = trial["startTime"]
            offset = pd.to_timedelta(start_time, unit="s")
            rep_time = base_time + offset

            # Format timestamp
            rep_time_str = rep_time.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"

            # Initialize the result row with metadata
            result_dict = {
                "vald_id": vald_id,
                "session_id": session_id,
                "exercise": exercise,
                "recorded_at": rep_time_str,
                "laterality": laterality,
            }

            results = trial["results"]

            # Flatten the nested "results" list
            for result in results:
                result_id = result["resultId"]
                result_value = result["value"]
                limb = result.get("limb", "Trial")

                # Map metrics to columns based on limb context
                if limb == "Trial":
                    result_dict[result_id] = result_value

                if limb == "Left":
                    result_dict[f"{result_id}_L"] = result_value

                if limb == "Right":
                    result_dict[f"{result_id}_R"] = result_value

            fetched_trials.append(result_dict)

    return fetched_trials


def fetch_force_frame_nord_bord_trials_data(
    vald_token: str,
    tenant_id: str,
    base_url: str,
    tests_endpoint: str,
    params: dict = None,
) -> list[dict[str, Any]]:
    """
    Fetches trial data for ForceFrame and NordBord devices.

    Args:
        vald_token (str): The OAuth2 access token.
        tenant_id (str): The unique identifier for your VALD account.
        base_url (str): The base URL of the API.
        tests_endpoint (str): Endpoint to fetch the list of tests.
        params (dict, optional): Query parameters.

    Returns:
        list[dict[str, Any]]: A list of test dictionaries containing test results.
    """
    # Retrieve tests data
    tests = fetch_tests(vald_token, tenant_id, base_url, tests_endpoint, params)

    return tests
