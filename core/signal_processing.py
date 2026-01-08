import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, medfilt


def apply_butterworth_filter(
    raw_spatiometric_data: pd.DataFrame,
    data_to_smooth: str,
    padding: str = None,
    cutoff_freq: float = 1.3,
    order: int = 4,
) -> np.ndarray:
    """
    Applies a low-pass Butterworth filter with padding to handle transient edges.

    Padding Methods:
    - 'edge': Extends the first value constantly (flat line). Good for static starts.
    - 'symmetric': Applies odd extension (point symmetry) around the first point.

    Args:
        raw_spatiometric_data (pd.DataFrame): DataFrame containing 'time' and value columns.
        data_to_smooth (str): Name of the column to filter.
        padding (str, optional): Padding method ('edge' or 'symmetric').
        cutoff_freq (float): Filter cutoff frequency in Hz (e.g., 1.3 for velocity trend).
        order (int): Order of the filter (steepness). Defaults to 4.

    Returns:
    np.ndarray: The smoothed data array with padding removed.
    """
    dt = np.mean(np.diff(raw_spatiometric_data["time"]))
    sample_rate = 1 / dt

    raw_data = raw_spatiometric_data[data_to_smooth].values

    # Padding length calculation
    if padding:
        # Calculate padding duration based on the filter's time constant.
        cutoff_period = 1.0 / cutoff_freq
        pad_duration = 3.0 * cutoff_period
        pad_samples = int(pad_duration * sample_rate)

        first_value = raw_data[0]

        # Constant padding: Extends the first value as a flat line.
        if padding == "edge":
            padding_start = np.full(pad_samples, first_value)

        # Symmetric padding: Projects the signal backwards in time by mirroring it across the first point.
        elif padding == "symmetric":
            padding_start = 2 * first_value - raw_data[1 : pad_samples + 1][::-1]

        # Prepend the calculated padding to the raw data
        data = np.concatenate((padding_start, raw_data))

    # No padding
    else:
        data = raw_data

    # Filter configuration
    nyquist = 0.5 * sample_rate
    normal_cutoff = cutoff_freq / nyquist

    b, a = butter(order, normal_cutoff, btype="low", analog=False)
    smooth_series = filtfilt(b, a, data)

    # Remove padding
    if padding:
        smooth_series = smooth_series[pad_samples:]

    return smooth_series


def find_speed_plateau(smooth_speed_array: np.ndarray) -> int:
    """Identifies the index where the athlete reaches maximum velocity."""
    idx_peak_speed = np.argmax(smooth_speed_array)

    return idx_peak_speed


def apply_corridor_filter(data_series: pd.Series, window_size: int = 15, threshold: float = 1.0) -> pd.Series:
    """
    Applies a 'corridor' filter to remove gross outliers and interference.

    Args:
        data_series (pd.Series): The input raw velocity data time-series.
        window_size (int, optional): The number of samples for the rolling median window.
        threshold (float, optional): The maximum allowed deviation from the guide curve in the data's unit.

    Returns:
        pd.Series: The cleaned data series with outliers interpolated.
    """
    # Create the guide curve
    guide_curve = data_series.rolling(window=window_size, center=True, min_periods=1).median()

    # Define the Corridor
    upper_bound = guide_curve + threshold
    lower_bound = guide_curve - threshold

    # Create validity mask
    valid_mask = (data_series >= lower_bound) & (data_series <= upper_bound)

    # Outlier removal and gap filling
    clean_series = data_series.copy()
    clean_series[~valid_mask] = np.nan
    clean_series = clean_series.interpolate(method="linear", limit_direction="both")

    return clean_series


def apply_median_filter(data_series: pd.Series | np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """
    Applies a standard median filter to despike the signal.

    Args:
        data_series (Union[pd.Series, np.ndarray]): The input velocity data.
        kernel_size (int, optional): The size of the sliding window.

    Returns:
        np.ndarray: The despiked data array.
    """
    # Kernel size validation
    if kernel_size % 2 == 0:
        kernel_size += 1

    # Apply filter
    median_series = medfilt(data_series, kernel_size)

    return median_series
