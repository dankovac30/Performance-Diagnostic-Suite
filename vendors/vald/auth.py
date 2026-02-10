import json
import os
import time

import requests


def get_vald_token(url: str, client_id: str, client_secret: str) -> str:
    """
    Retrieves an OAuth2 access token for the VALD API using the Client Credentials flow.

    Args:
        url (str): The token endpoint URL.
        client_id (str): The client ID provided by VALD.
        client_secret (str): The client secret provided by VALD.

    Returns:
        str: The valid access token.

    Raises:
        ValueError: If the API returns a non-200 status code.
        Exception: For network issues or other unexpected errors.
    """
    # Determine the absolute path to the cache file relative to this script.
    abs_path = os.path.abspath(__file__)
    dir_name = os.path.dirname(abs_path)
    cache_file = os.path.join(dir_name, "vald_token.json")

    # Check if a cached token exists and is valid
    if os.path.exists(cache_file):
        with open(cache_file) as f:
            cache = json.load(f)

        # Check if the token is still valid.
        if time.time() < cache["exp"] - 600:
            return cache["token"]

    # If invalid cache, request a new token
    payload = {"grant_type": "client_credentials"}

    try:
        response = requests.post(
            url=url,
            data=payload,
            auth=(client_id, client_secret),
            timeout=30,
        )

        if response.status_code == 200:
            data = response.json()
            token = data["access_token"]

            # Calculate expiration time
            exp = time.time() + data["expires_in"]

            # Save the new token and expiration to the cache file
            with open(cache_file, "w") as f:
                json.dump({"token": token, "exp": exp}, f)

            return token

        else:
            error_message = f"API Authorication Error {response.status_code}: {response.text}"
            raise ValueError(error_message)

    except Exception as e:
        print(f"Authorication Error: {e}")
        raise
