import pandas as pd


def load_stalker_rda(file_path: str, sampling_rate: float = 46.875) -> pd.DataFrame:
    """
    Loads raw velocity data from a Stalker radar file.

    Args:
        file_path (str): The absolute or relative path to the source file.
        sampling_rate (float, optional): The radar's data acquisition frequency in Hz.
                                         Defaults to 46.875 Hz (standard Stalker ATS setting).

    Returns:
        pd.DataFrame: A pandas DataFrame containing 'time' and 'raw_speed'
    """
    raw_values = []

    # Open the file in read mode
    with open(file_path, encoding="utf-8") as f:
        for line in f:
            # Clean the line
            line_clean = line.strip().replace(",", ".")

            # Skip empty lines
            if not line_clean:
                continue

            # Convert the string to a float.
            try:
                value = float(line_clean)
                raw_values.append(value)

            # Skip error line
            except ValueError:
                continue

    # Dataframe construction
    df = pd.DataFrame(raw_values, columns=["raw_speed"])

    dt = 1.0 / sampling_rate
    df["time"] = df.index * dt

    df = df[["time", "raw_speed"]]

    return df
