import numpy as np
import pandas as pd

from core.signal_processing import apply_butterworth_filter, apply_corridor_filter, apply_median_filter
from vendors.stalker.importer import load_stalker_rda
from vendors.stalker.preprocessing import crop_end, crop_start, reconstruct_start


def process_rda_file(file_path: str, sampling_rate: float = 46.875) -> pd.DataFrame:
    """
    Drives the signal processing pipeline for raw Stalker radar data (.rda).

    This function transforms raw, noisy Doppler radar data into a clean, biomechanically
    valid dataset. It handles file ingestion, multi-stage noise filtering, temporal
    segmentation (cropping start/end), and model-based reconstruction of the start
    phase to correct for trigger delays.

    The pipeline follows this specific order:
    1. **Ingest:** Loads raw velocity data.
    2. **Pre-filter:** Applies corridor and median filters to remove "ghost" signals and spikes.
    3. **Segmentation:** Detects the true movement onset and takeoff/finish, cutting irrelevant data.
    4. **Reconstruction:** Uses Morin's macroscopic model to mathematically regenerate the
       missing start phase if the operator triggered the radar late.
    5. **Smoothing:** Applies a symmetric Butterworth filter to the stitched dataset.
    6. **Derivation:** Calculates acceleration and cumulative distance.

    Args:
        file_path (str): The absolute or relative path to the .rda file.
        sampling_rate (float, optional): The radar's frequency in Hz. Defaults to 46.875.

    Returns:
        pd.DataFrame: A DataFrame containing the fully processed sprint data.
        Key columns include:
            - 'time': Corrected time axis starting at 0.00s.
            - 'smooth_speed': The final filtered velocity curve.
            - 'smooth_acceleration': Derived acceleration.
            - 'smooth_distance': Derived distance.
    """
    # Ingest data
    raw_df = load_stalker_rda(file_path, sampling_rate)

    # Coarse noise removal
    corridor_filtered = apply_corridor_filter(raw_df["raw_speed"])
    raw_df["corridor_speed"] = corridor_filtered

    # Spike Removal
    median_filtered = apply_median_filter(raw_df["corridor_speed"])
    raw_df["median_speed"] = median_filtered

    # Segmentation: Identify the main acceleration phase
    cropped_df = crop_end(crop_start(raw_df))

    # Model-based reconstruction
    reconstructed_df = reconstruct_start(cropped_df)

    # Final smoothing
    smooth_speed = apply_butterworth_filter(reconstructed_df, "median_speed", padding="symmetric", cutoff_freq=1.3)
    reconstructed_df["smooth_speed"] = smooth_speed

    # Kinematic derivations
    dt = 1 / sampling_rate
    reconstructed_df["smooth_acceleration"] = np.gradient(reconstructed_df["smooth_speed"], dt)
    reconstructed_df["smooth_distance"] = (
        (reconstructed_df["smooth_speed"].rolling(window=2).mean() * dt).fillna(0).cumsum()
    )

    return reconstructed_df
