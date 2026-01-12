"""
1080 Motion Raw Data Decoder

This module handles the low-level decoding of binary sample data retrieved from the
1080 Motion API. This utility converts raw strings into structured Pandas DataFrames containing
time-series metrics (Time, Position, Speed, Acceleration, Force), enabling
downstream biomechanical analysis.
"""

import base64
import struct

import pandas as pd


def decode_1080_samples(base64_string: str) -> pd.DataFrame:
    """
    Decodes a base64 encoded string from the 1080 Motion API into a structured DataFrame.

    The 1080 Motion API packages sample data into a binary blob encoded in base64.
    Each sample consists of 5 measurements, each represented as a 4-byte float (float32).
    Therefore, one complete data point occupies 20 bytes (5 metrics * 4 bytes).

    Args:
        base64_string (str): The raw base64 string extracted from the 'sampleData'
                             field of the API response.

    Returns:
        pd.DataFrame: A DataFrame with columns ['time', 'position', 'speed', 'acceleration', 'force'].
    """
    try:
        # Decode the Base64 string back to raw bytes
        decoded_bytes = base64.b64decode(base64_string)

        # Calculate the number of samples
        n_samples = len(decoded_bytes) // 20

        # Create the struct format string
        fmt = f"<{n_samples * 5}f"

        # Unpack the binary data into a flat tuple of floats
        unpacked_data = struct.unpack(fmt, decoded_bytes)

        # Transform the flat data into separate columns
        data = {
            "time": unpacked_data[0::5],
            "position": unpacked_data[1::5],
            "speed": unpacked_data[2::5],
            "acceleration": unpacked_data[3::5],
            "force": unpacked_data[4::5],
        }

        return pd.DataFrame(data)

    except Exception as e:
        print(f"Decoding Error: {e}")
        raise
