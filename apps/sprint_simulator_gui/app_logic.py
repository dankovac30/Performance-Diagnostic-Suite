from typing import Any

import pandas as pd

from sprint_science.simulator import SprintSimulation


def run_simulation_logic(F0: float, V0: float, weight: float, height: float) -> tuple[dict[str, Any], pd.DataFrame]:
    """
    Controller logic bridging the GUI inputs and the Physics Engine.

    Returns:
        Tuple[Dict[str, Any], pd.DataFrame]:
            - results: Dictionary containing calculated metrics (Time 100m, Fly 30m, Top Speed, etc.).
            - report: Full pandas DataFrame with frame-by-frame simulation data.
    """
    # Standard testing parameters
    running_distance = 100.0
    external_force_N = 0.0
    fly_length = 30.0

    # Input sanitization
    f0_clean = F0.replace(",", ".")
    v0_clean = V0.replace(",", ".")
    weight_clean = weight.replace(",", ".")
    height_clean = height.replace(",", ".")

    # Type conversion and validation
    try:
        f0_f = float(f0_clean)
        v0_f = float(v0_clean)
        weight_f = float(weight_clean)
        height_f = float(height_clean)

    except ValueError:
        raise ValueError("Invalid input. Please enter numbers only.")

    if f0_f <= 0:
        raise ValueError("F0 must be a positive number (greater than 0).")

    if v0_f <= 0:
        raise ValueError("V0 mmust be a positive number (greater than 0).")

    if height_f <= 0:
        raise ValueError("Height must be a positive number (greater than 0).")

    if weight_f <= 0:
        raise ValueError("Weight must be a positive number (greater than 0).")

    # Simulation
    try:
        athlete = SprintSimulation(
            F0=f0_f,
            V0=v0_f,
            weight=weight_f,
            height=height_f,
            running_distance=running_distance,
            external_force_N=external_force_N,
            fly_length=fly_length,
        )
        # Run simulation
        data = athlete.run_sprint()

        # Metrics extraction
        running_time = data["time"].iloc[-1]
        top_speed = athlete.top_speed()
        fly_segment = athlete.flying_sections()
        fly_time = str(f"{fly_segment['first_fast']['time']:.2f} s")
        fly_start = str(f"{fly_segment['first_fast']['start']:.0f} m")
        fly_finish = str(f"{fly_segment['first_fast']['finish']:.0f} m")
        time_30_m = data[data["distance"] > 30]["time"].iloc[0]

        results = {
            "running_time_100m": running_time,
            "time_30m": time_30_m,
            "time_30m_fly": fly_time,
            "fly_start": fly_start,
            "fly_finish": fly_finish,
            "top_speed": top_speed["top_speed"],
            "top_speed_distance": top_speed["distance_top_speed"],
        }

        return results, data

    except Exception as e:
        raise Exception(f"An error occurred during simulation: {e}")
