import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt, medfilt
from typing import Union


def apply_butterworth_filter(raw_spatiometric_data: pd.DataFrame,
                             data_to_smooth: str,
                             padding: str = None,
                             cutoff_freq: float = 1.3,
                             order: int = 4) -> np.ndarray:
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
    dt = np.mean(np.diff(raw_spatiometric_data['time']))
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
        if padding == 'edge':
            padding_start = np.full(pad_samples, first_value)

        # Symmetric padding: Projects the signal backwards in time by mirroring it across the first point.
        elif padding == 'symmetric':
            padding_start = 2 * first_value - raw_data[1:pad_samples+1][::-1]

        # Prepend the calculated padding to the raw data
        data = np.concatenate((padding_start, raw_data))

    # No padding
    else:
        data = raw_data 

    # Filter configuration
    nyquist = 0.5 * sample_rate
    normal_cutoff = cutoff_freq / nyquist

    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    smooth_series = filtfilt(b, a, data)

    # Remove padding
    if padding:
        smooth_series = smooth_series[pad_samples:]

    return smooth_series


def find_speed_plateau(smooth_speed_array: np.ndarray) -> int:
    """Identifies the index where the athlete reaches maximum velocity."""
    idx_peak_speed = np.argmax(smooth_speed_array)

    return idx_peak_speed