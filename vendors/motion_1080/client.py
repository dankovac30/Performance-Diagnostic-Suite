"""
1080 Motion API Client Module.

This module provides the low-level HTTP client functionality for interacting
with the 1080 Motion Public API. It handles URL construction, header management
(including authentication), and basic error handling for network requests.
"""

import requests


def fetch_data(api_key: str, base_url: str, endpoint: str, params: dict = None) -> dict | list:
    """
    Executes a GET request against the 1080 Motion API.

    Args:
        api_key (str): The valid X-1080-API-Key provided by the vendor.
        base_url (str): The root URL of the API service.
        endpoint (str): The specific API endpoint path (e.g., "/TrainingData/Session/").
        params (dict, optional): A dictionary of query parameters to append to the URL

    Returns:
        dict | list: The parsed JSON response from the API.
    """
    # Construct the full URL
    url = f"{base_url}{endpoint}"

    headers = {"X-1080-API-Key": api_key, "Content-Type": "application/json"}

    try:
        response = requests.get(url, headers=headers, params=params)

        if response.status_code == 200:
            data = response.json()

        else:
            error_message = f"API Error {response.status_code} loading {endpoint}: {response.text}"
            raise ValueError(error_message)

    except Exception as e:
        print(f"Error loading {endpoint}: {e}")
        raise

    return data
