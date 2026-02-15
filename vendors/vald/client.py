"""
VALD API Client Module.

This module provides the low-level HTTP client functionality for interacting
with the VALD Public API. It handles header management and error handling
for network requests.
"""

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

SESSION = requests.Session()

RETRY_STRATEGY = Retry(
    total=3,
    backoff_factor=1,
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=["GET"],
)
SESSION.mount("https://", HTTPAdapter(max_retries=RETRY_STRATEGY))


def fetch_data(vald_token: str, tenant_id: str, url: str, params: dict = None) -> dict | list:
    """
    Executes a GET request against the VALD API.

    Args:
        vald_token (str): The OAuth2 access token.
        tenant_Id (str): The unique identifier for your VALD account.
        url (str): The specific API url path.
        params (dict, optional): A dictionary of query parameters to append to the URL

    Returns:
        dict[str, Any] | list[Any]: The parsed JSON response from the API.
            - If status is 200, returns the JSON body.
            - If status is 204 (No Content), returns a default empty structure {"tests": []}.
    Raises:
        ValueError: If the API returns a non-success status code (e.g., 400, 401, 500).
        requests.RequestException: If a network connectivity error occurs.
    """
    if params is None:
        params = {}

    params["tenantId"] = tenant_id

    headers = {"Authorization": f"Bearer {vald_token}", "Accept": "application/json"}

    try:
        response = SESSION.get(url, headers=headers, params=params, timeout=60)

        if response.status_code == 200:
            data = response.json()

        elif response.status_code == 204:
            data = {"tests": []}

        else:
            response.raise_for_status()

    except requests.exceptions.HTTPError as e:
        error_message = f"API Error {response.status_code} loading {url}: {response.text}"
        print(error_message)
        raise ValueError(error_message) from e

    except Exception as e:
        print(f"Error loading {url}: {e}")
        raise

    return data
