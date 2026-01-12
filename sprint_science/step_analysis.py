import numpy as np
import pandas as pd

from core.signal_processing import apply_butterworth_filter, find_speed_plateau
from sprint_science.physics import calculate_air_density, calculate_air_resistance_force, calculate_frontal_area


class StepAnalyzer:
    """
    A class responsible for step-by-step analysis of sprint data.

    This class processes raw spatiotemporal data from 1080 Sprint to calculate
    kinetic (force, impulse) and kinematic (length, frequency) parameters
    for each individual step, distinguishing between left and right limbs.
    """

    def __init__(
        self,
        starting_leg: str,
        raw_spatiotemporal_data: pd.DataFrame,
        height: float,
        weight: float,
        wind_speed: float = 0.0,
        temperature_c: float = 20.0,
        barometric_pressure_hpa: float = 1013.25,
    ):
        """
        Initialize the StepAnalyzer with athlete data and environmental conditions.

        Args:
            starting_leg (str): The front leg on the start ('left' or 'right').
            raw_spatiotemporal_data (pd.DataFrame): Raw data export containing time, speed, force, etc.
            weight (float): Athlete's body mass (kg).
            height (float): Athlete's height (cm or m).
            wind_speed (float, optional): Headwind (-) or tailwind (+) in m/s.
            temperature_c (float, optional): For air density calculation.
            barometric_pressure_hpa (float, optional): For air density calculation.
        """
        self.starting_leg = starting_leg.lower()
        self.raw_spatiotemporal_data = raw_spatiotemporal_data.copy()
        self.spatiotemporal_data = None
        self.height = height
        self.weight = weight
        self.wind_speed = wind_speed

        # Calculate sampling interval (dt) for integration
        self.dt = self.raw_spatiotemporal_data["time"].diff().mean()

        # Calculate aerodynamic constants based on athlete's body metrics
        self.frontal_area = calculate_frontal_area(height, weight)
        self.air_density = calculate_air_density(temperature_c, barometric_pressure_hpa)

    def prepare_dataframe(self) -> pd.DataFrame:
        """
        Prepares, cleans, and enriches the raw data with physics calculations.

        Performs the following operations:
        1. Calculates current athletes's distance from 1080's position.
        2. Calculates Total Force (Mass*Acc + Cable Force + Aero Drag).
        3. Trims the start (speed < 0.5 m/s).
        4. Trims the end based on fatigue (speed drops below 90% of Vmax).

        Returns:
            pd.DataFrame: The processed dataframe ready for step detection.
        """
        if self.spatiotemporal_data is None:
            # Create a working copy and reset index
            self.spatiotemporal_data = self.raw_spatiotemporal_data.copy().reset_index(drop=True)

            # Normalize distance to start at 0
            zero_position = self.raw_spatiotemporal_data.iloc[0]["position"]
            self.spatiotemporal_data["distance"] = self.raw_spatiotemporal_data["position"] - zero_position

            # Rename 'force' from device to 'external_force' (cable tension)
            self.spatiotemporal_data = self.spatiotemporal_data.rename(columns={"force": "external_force"})

            # Calculate net force required to accelerate the body mass (F = m*a)
            self.spatiotemporal_data["force"] = self.spatiotemporal_data["acceleration"] * self.weight

            # Calculate aerodynamic drag
            self.spatiotemporal_data["air_drag"] = calculate_air_resistance_force(
                self.spatiotemporal_data["speed"], self.air_density, self.frontal_area, self.wind_speed
            )

            # Total force = Inertial force + machine resistance + air resistance
            self.spatiotemporal_data["total_force"] = (
                self.spatiotemporal_data["force"]
                + self.spatiotemporal_data["external_force"]
                + self.spatiotemporal_data["air_drag"]
            )

            # TRIMMING
            # Cut data before athlete reaches 0.5 m/s
            speed_threshold_indices = self.spatiotemporal_data.index[self.spatiotemporal_data["speed"] > 0.5]
            first_idx = speed_threshold_indices[0]

            # Detect fatigue/deceleration
            # Find top speed index
            smooth_speed = apply_butterworth_filter(
                self.spatiotemporal_data, data_to_smooth="speed", cutoff_freq=1.3, order=4
            )
            top_speed_idx = find_speed_plateau(smooth_speed)
            top_speed = smooth_speed[top_speed_idx]

            # Find index where speed drops below 90% of Vmax, add to top speed index
            post_top_speed_data = smooth_speed[top_speed_idx:]
            mask = post_top_speed_data < top_speed * 0.9
            if np.any(mask):
                fatigue_idx = np.argmax(mask) + top_speed_idx
            else:
                fatigue_idx = None

            # Slice the dataframe to keep only valid sprint effort
            self.spatiotemporal_data = self.spatiotemporal_data.loc[first_idx:fatigue_idx].reset_index(drop=True)

        return self.spatiotemporal_data

    def detect_steps(self, data) -> list[int]:
        """
        Detects step boundaries using Zero-Crossing method on Total Force.

        A step boundary (touch-down) is defined as the moment Total Force transitions
        from positive (Propulsion) to negative (Braking).

        Returns:
            List[int]: A list of indices representing the start of each step (touch-down).
        """
        # Find where Force is < 0 NOW, but was >= 0 BEFORE
        mark_crossing = (data["total_force"] < 0) & (data["total_force"].shift(1) >= 0)

        indices_raw = data.index[mark_crossing].tolist()

        # Prevent detecting multiple crossings due to signal noise
        idx_time_gap = 0.05
        min_idx_gap = idx_time_gap / self.dt

        indices = [indices_raw[0]]
        last_idx = indices_raw[0]

        for idx in indices_raw:
            if idx - last_idx > min_idx_gap:
                indices.append(idx)
                last_idx = idx

        return indices

    def analyze_steps(self) -> pd.DataFrame:
        """
        Main analysis loop. Iterates through detected steps and calculates
        detailed metrics for each limb.

        Returns:
            pd.DataFrame: A table where each row is a step with kinetic and kinematic data.
        """
        spatiotemporal_data = self.prepare_dataframe()

        # Get step touch-down indices
        cross_indices = self.detect_steps(spatiotemporal_data)

        # Determine step 2 leg
        current_leg = "left" if self.starting_leg == "right" else "right"

        # Analysis starts from step 2 unaffected by the standing start
        current_step = 2
        previous_step_time = None
        result = []

        # Loop through intervals between touch-downs
        for start, end in zip(cross_indices, cross_indices[1:]):
            step_df = spatiotemporal_data.iloc[start:end]

            # Split cycle into braking and propulsion
            braking_df = step_df[step_df["total_force"] < 0]
            propulsive_df = step_df[step_df["total_force"] > 0]

            # Kinetics
            peak_propulsive_force = propulsive_df["total_force"].max()
            peak_braking_force = propulsive_df["total_force"].min()
            braking_impulse = (braking_df["total_force"] * self.dt).sum()
            propulsive_impulse = (propulsive_df["total_force"] * self.dt).sum()
            net_impulse = braking_impulse + propulsive_impulse
            aerodynamic_impulse = (step_df["air_drag"] * self.dt).sum()
            if propulsive_impulse != 0:
                braking_propulsive_ratio = abs(braking_impulse / propulsive_impulse)
            else:
                braking_propulsive_ratio = 0

            # Kinematics
            v_start = step_df.iloc[0]["speed"]
            v_end = step_df.iloc[-1]["speed"]
            delta_v = v_end - v_start
            time_start = step_df.iloc[0]["time"]
            time_end = step_df.iloc[-1]["time"]
            step_time = time_end - time_start
            step_freq = 1 / step_time
            step_length = step_df.iloc[-1]["distance"] - step_df.iloc[0]["distance"]

            # Scraps if a step duration varies by >30% compared to previous
            if current_step > 5:
                if not (previous_step_time * 0.7) < step_time < (previous_step_time * 1.3):
                    break

            res = {
                "step_number": current_step,
                "leg": current_leg,
                "peak_propulsive_force": peak_propulsive_force,
                "peak_braking_force": peak_braking_force,
                "braking_impulse": braking_impulse,
                "propulsive_impulse": propulsive_impulse,
                "net_impulse": net_impulse,
                "aerodynamic_impulse": aerodynamic_impulse,
                "braking_propulsive_ratio": braking_propulsive_ratio,
                "v_start": v_start,
                "v_end": v_end,
                "delta_v": delta_v,
                "time_start": time_start,
                "time_end": time_end,
                "step_time": step_time,
                "step_freq": step_freq,
                "step_length": step_length,
            }

            result.append(res)

            # Update for next iteration
            current_step += 1
            current_leg = "left" if current_leg == "right" else "right"
            previous_step_time = step_time

        return pd.DataFrame(result)

    def calculate_propulsive_braking_imp(self, spatiotemporal_data: pd.DataFrame) -> tuple[float, float]:
        """
        Helper method to calculate total Propulsive and Braking impulse for a given dataframe slice.

        Returns:
            Tuple[float, float]: (Propulsive Impulse, Braking Impulse)
        """
        df_propulsion = spatiotemporal_data["total_force"][spatiotemporal_data["total_force"] > 0]
        df_braking = spatiotemporal_data["total_force"][spatiotemporal_data["total_force"] < 0]

        # Integrate force over time
        propulsion_impulse = df_propulsion.sum() * self.dt
        braking_impulse = df_braking.sum() * self.dt

        return propulsion_impulse, braking_impulse

    def analyze_acc_technical_efficiency(self) -> tuple[float, float]:
        """
        Analyzes the technical efficiency during the initial acceleration phase.
        Targets steps 2 to 5 to evaluate the 'Drive Phase'.

        Returns:
            Tuple[float, float]: (Total Propulsive Impulse, Total Braking Impulse) for Accel Phase.
        """
        spatiotemporal_data = self.prepare_dataframe()

        # Get step touch-down indices for steps 2, 3, 4, 5
        cross_indices = self.detect_steps(spatiotemporal_data)
        acc_phase = cross_indices[:5]

        # Slice data from start of step 2 to end of step 5
        acc_spatiotemporal = spatiotemporal_data.loc[acc_phase[0] : acc_phase[-1]]

        propulsion, braking = self.calculate_propulsive_braking_imp(acc_spatiotemporal)

        return propulsion, braking

    def analyze_maxv_technical_efficiency(self):
        """
        Analyzes technical efficiency during the Maximum Velocity phase.

        Logic:
        1. Identifies the point where speed > 90% of Vmax.
        2. Snaps this point to the nearest actual Step touch-down to ensure data integrity.
        3. Analyzes all complete steps from that point until the last touch-down.

        Returns:
            Tuple[float, float]: (Total Propulsive Impulse, Total Braking Impulse) for MaxV Phase.
        """
        spatiotemporal_data = self.prepare_dataframe()

        # Find vmax and the 90% threshold index
        smooth_speed = apply_butterworth_filter(spatiotemporal_data, data_to_smooth="speed", cutoff_freq=1.3, order=4)
        top_speed_idx = find_speed_plateau(smooth_speed)
        top_speed = smooth_speed[top_speed_idx]

        # Find first index crossing 90% vmax
        top_speed_zone_idx = np.where(smooth_speed > (top_speed * 0.9))[0][0]
        cross_indices = self.detect_steps(spatiotemporal_data)
        differences = np.abs(cross_indices - top_speed_zone_idx)

        # Find the step touch-down closest to the 90% vmax
        zone_start_idx = cross_indices[np.argmin(differences)]

        # End at the last detected touch down
        zone_end_idx = cross_indices[-1]

        # Slice data containing only maxv cycles
        maxv_spatiotemporal = spatiotemporal_data.iloc[zone_start_idx:zone_end_idx]

        propulsion, braking = self.calculate_propulsive_braking_imp(maxv_spatiotemporal)

        return propulsion, braking
