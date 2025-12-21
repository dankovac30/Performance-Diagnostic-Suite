import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt


def apply_butterworth_filter(raw_spatiometric_data: pd.DataFrame,
                             data_to_smooth: str,
                             cutoff_freq: float = 1.3,
                             order: int = 4) -> np.ndarray:
    """
    Applies a low-pass Butterworth filter with padding to handle transient edges.
    """
    dt = np.mean(np.diff(raw_spatiometric_data['time']))
    sample_rate = 1 / dt

    # Padding length calculation
    cutoff_period = 1.0 / cutoff_freq
    pad_duration = 3.0 * cutoff_period
    pad_samples = int(pad_duration * sample_rate)

    first_value = raw_spatiometric_data[data_to_smooth].iloc[0]
    padding_start = np.full(pad_samples, first_value)

    raw_data = raw_spatiometric_data[data_to_smooth].values
    padded_data = np.concatenate((padding_start, raw_data))

    # Filter configuration
    nyquist = 0.5 * sample_rate
    normal_cutoff = cutoff_freq / nyquist

    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y_padded = filtfilt(b, a, padded_data)

    # Remove padding
    y = y_padded[pad_samples:]

    return y


def find_speed_plateau(smooth_speed_array: np.ndarray) -> int:
    """Identifies the index where the athlete reaches maximum velocity."""
    idx_peak_speed = np.argmax(smooth_speed_array)

    return idx_peak_speed