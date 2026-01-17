import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import linregress

from core.signal_processing import apply_butterworth_filter, find_speed_plateau
from sprint_science.physics import (
    calculate_air_density,
    calculate_air_resistance_force,
    calculate_frontal_area,
)
from sprint_science.simulator import SprintSimulation


class BaseSprintProfiler:
    """
    Base class containing shared physics and dynamics calculations for Sprint Profiling.
    https://doi.org/10.1111/sms.12490

    This class handles the initialization of environmental constants and the
    calculation of force-velocity-power relationships once the kinematic
    parameters (Vmax, Tau) are determined by child classes.
    """

    def __init__(
        self,
        height: float,
        weight: float,
        wind_speed: float = 0.0,
        temperature_c: float = 20.0,
        barometric_pressure_hpa: float = 1013.25,
    ):
        """
        Initialize the base profiler with athlete anthropometrics and environmental conditions.

        Args:
            height (float): Athlete's height in centimeters or meters.
            weight (float): Athlete's body mass in kg.
            wind_speed (float, optional): Headwind (+) or tailwind (-) in m/s. Defaults to 0.0.
            temperature_c (float, optional): Air temperature in Celsius. Defaults to 20.0.
            barometric_pressure_hpa (float, optional): Pressure in hPa. Defaults to 1013.25.
        """
        self.height = height
        self.weight = weight
        self.wind_speed = wind_speed

        # Calculate constants for air resistance
        self.frontal_area = calculate_frontal_area(height, weight)
        self.air_density = calculate_air_density(temperature_c, barometric_pressure_hpa)

    def calculate_Rfmax_DRF(self, F0: float, V0: float) -> tuple[float, float]:
        """
        Calculates Rf_max and DRF using an iterative simulation.

        Methodology:
        1. Simulate an ideal sprint with the given F0/V0 profile.
        2. Extract metrics from the smooth simulated curve.
        3. Cutoff at acceleration < 0.05 m/s^2 to ensure DRF represents only the acceleration phase.

        Args:
            F0 (float): Theoretical maximal horizontal force (N/kg).
            V0 (float): Theoretical maximal velocity (m/s).

        Returns:
            tuple[float, float]: (Rf_max in %, DRF slope).
        """
        sprinter = SprintSimulation(F0, V0, self.weight, self.height, running_distance=100, fatigue_toggle="OFF")
        df = sprinter.run_sprint()

        vertical_force = self.weight * 9.80665

        # Filter: Start after 0.3s
        df_cropped = df[(df["time"] >= 0.3) & (df["acceleration"] > 0.05)].copy()

        # Calculate Ratio of Force (RF)
        df_cropped["RF"] = (
            df_cropped["propulsion_force"] / np.sqrt(df_cropped["propulsion_force"] ** 2 + vertical_force**2)
        ) * 100

        # Linear regression to find DRF
        reg_result = linregress(x=df_cropped["speed"], y=df_cropped["RF"])

        DRF = reg_result.slope
        Rf_max = df_cropped.iloc[0]["RF"]

        return Rf_max, DRF

    def calculate_dynamics(self, spatiotemporal_data: pd.DataFrame) -> dict:
        """
        Computes the macroscopic Force-Velocity-Power profile from kinematic data.

        This method calculates horizontal force (mass * acc + air resistance),
        power, and performs linear regression to find F0 and V0.

        Args:
            spatiotemporal_data (pd.DataFrame): DataFrame containing at least:
                - 'Model speed (m/s)'
                - 'Acceleration (m/s2)'

        Returns:
            dict: Dictionary containing key metrics:
                - F0 (N/kg), V0 (m/s), Pmax (W/kg)
                - F_V_slope, Rf_max, DRF
                - source_data (DataFrame with calculated forces)
                - regress_mask (Boolean mask used for the F-V regression)
        """

        # Create external load collumn for inputs without external resistance
        if "F external load (N)" not in spatiotemporal_data:
            spatiotemporal_data["F external load (N)"] = 0

        # F = m * a
        spatiotemporal_data["F Hzt (N)"] = spatiotemporal_data["Acceleration (m/s2)"] * self.weight

        # F_air = 0.5 * rho * A * Cd * v^2
        spatiotemporal_data["F air (N)"] = calculate_air_resistance_force(
            spatiotemporal_data["Model speed (m/s)"],
            self.air_density,
            self.frontal_area,
            self.wind_speed,
        )

        # Total Horizontal Force
        spatiotemporal_data["F Hzt total (N)"] = (
            spatiotemporal_data["F Hzt (N)"]
            + spatiotemporal_data["F air (N)"]
            + spatiotemporal_data["F external load (N)"]
        )

        # Relative Force and Power
        spatiotemporal_data["F Hzt total (N/kg)"] = spatiotemporal_data["F Hzt total (N)"] / self.weight
        spatiotemporal_data["Power Hzt (W/kg)"] = (
            spatiotemporal_data["Model speed (m/s)"] * spatiotemporal_data["F Hzt total (N/kg)"]
        )

        # Resultant Force (Vector sum of horizontal and vertical)
        spatiotemporal_data["F Resultant (N)"] = np.sqrt(
            spatiotemporal_data["F Hzt total (N)"] ** 2 + (self.weight * 9.80665) ** 2
        )
        spatiotemporal_data["RF (%)"] = (
            spatiotemporal_data["F Hzt total (N)"] / spatiotemporal_data["F Resultant (N)"]
        ) * 100

        # Mask: Only include lines where athlete is producing force
        mask_force_produced_raw = spatiotemporal_data["F Hzt total (N)"] > spatiotemporal_data["F external load (N)"]
        mask_force_produced = mask_force_produced_raw.cumsum().astype(bool)

        res_f_v = linregress(
            x=spatiotemporal_data.loc[mask_force_produced, "Model speed (m/s)"],
            y=spatiotemporal_data.loc[mask_force_produced, "F Hzt total (N)"],
        )

        # Final parameters
        slope = -res_f_v.slope
        F0_abs = res_f_v.intercept
        F0 = F0_abs / self.weight
        V0 = F0_abs / slope
        F_V_slope = F0 / V0
        Pmax = (F0 * V0) / 4

        # Calculate derived metrics via simulation
        Rf_max, DRF = self.calculate_Rfmax_DRF(F0, V0)

        results = {
            "F0": F0,
            "F0_abs": F0_abs,
            "V0": V0,
            "Pmax": Pmax,
            "F_V_slope": F_V_slope,
            "Rf_max": Rf_max,
            "DRF": DRF,
            "source_data": spatiotemporal_data,
            "regress_mask": mask_force_produced,
        }

        return results


class SprintSpeedTimeProfiler(BaseSprintProfiler):
    """
    A class to analyze sprint performance using Stalker ATS radar or 1080 Motion Sprint data.
    """

    def __init__(
        self,
        raw_spatiotemporal_data: pd.DataFrame,
        height: float,
        weight: float,
        wind_speed: float = 0.0,
        temperature_c: float = 20.0,
        barometric_pressure_hpa: float = 1013.25,
    ):
        """
        Initialize the profiler with athlete antropometrics, environmental conditions and speed-time data.

        Args:
            raw_spatiotemporal_data (pd.DataFrame): Raw data from 1080 Motion (time, speed, force).
            height (float): Athlete's height in centimeters or meters.
            weight (float): Athlete's body mass in kg.
            wind_speed (float, optional): Headwind (+) or tailwind (-) in m/s. Defaults to 0.0.
            temperature_c (float, optional): Air temperature in Celsius. Defaults to 20.0.
            barometric_pressure_hpa (float, optional): Pressure in hPa. Defaults to 1013.25.
        """
        super().__init__(
            weight=weight,
            height=height,
            temperature_c=temperature_c,
            barometric_pressure_hpa=barometric_pressure_hpa,
            wind_speed=wind_speed,
        )
        self.raw_spatiotemporal_data = raw_spatiotemporal_data.copy()

    @staticmethod
    def morin_velocity_model(t: np.ndarray, vmax: float, tau: float, t0: float) -> np.ndarray:
        """
        Mono-exponential velocity function with a time delay parameter.
        Formula: v(t) = Vmax * (1 - e^(-(t-t0)/tau))
        """
        return np.where(t > t0, vmax * (1 - np.exp(-(t - t0) / tau)), 0)

    def fit_velocity_model(
        self, v_data: np.ndarray, t_data: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
        """
        Fits the Morin velocity model to the measured velocity data using non-linear least squares.

        Returns:
            Tuple: (model_speed_array, model_acceleration_array, params_dict)
        """
        # Initial guess & bounds [Vmax, Tau, t0]
        p0 = [np.max(v_data), 1.3, 0]
        bounds = ([0, 0.01, -3], [15, 5, 3])

        popt, _ = curve_fit(self.morin_velocity_model, t_data, v_data, p0=p0, bounds=bounds)
        vmax, tau, t0 = popt

        # Generate model curve based on fitted parameters
        model_speed = np.where(t_data > t0, vmax * (1 - np.exp(-(t_data - t0) / tau)), 0)

        # Analytic derivation of acceleration: a(t) = (Vmax/Tau) * e^(-(t-t0)/Tau)
        raw_acc = (vmax / tau) * np.exp(-(t_data - t0) / tau)
        model_acc = np.where(t_data > t0, raw_acc, 0)

        params = {"Tau": tau, "t0": t0, "V_max": vmax}

        return model_speed, model_acc, params

    def calculate_profile(self) -> dict[str, float]:
        """
        Main pipeline for Speed Time data.

        Steps:
        1. Pre-process and smooth raw data.
        2. Fit Morin's model to determine kinematic parameters (Vmax, Tau, t0).
        3. Validate model adherence (Goodness of fit).
        4. Calculate dynamics (Force, Power) including air resistance.

        Returns:
            Dict containing F0, V0, Pmax, F-V Slope, RFmax, DRF, etc.
        """
        raw_spatiotemporal_data = self.raw_spatiotemporal_data.copy()

        # Smoothing: Smooth external force and speed
        if "force" in raw_spatiotemporal_data:
            raw_spatiotemporal_data["F external load (N)"] = apply_butterworth_filter(
                raw_spatiotemporal_data, "force", padding="symmetric", cutoff_freq=1.3, order=4
            )

        smooth_speed_array = apply_butterworth_filter(raw_spatiotemporal_data, "speed", padding="symmetric")

        # Cutoff post peak speed on filtered array
        idx_peak_speed = find_speed_plateau(smooth_speed_array)
        raw_spatiotemporal_data["smooth_speed"] = smooth_speed_array

        # Acceleration phase only
        spatiotemporal_data = raw_spatiotemporal_data.iloc[: idx_peak_speed + 1].copy()

        v_data = spatiotemporal_data["speed"].to_numpy()
        t_data = spatiotemporal_data["time"].to_numpy()

        # Model fitting
        model_speed, model_acc, params = self.fit_velocity_model(v_data, t_data)

        spatiotemporal_data["Model speed (m/s)"] = model_speed
        spatiotemporal_data["Acceleration (m/s2)"] = model_acc
        dt = spatiotemporal_data["time"].diff().fillna(0)
        delta_distance = spatiotemporal_data["Model speed (m/s)"] * dt
        spatiotemporal_data["Position (m)"] = delta_distance.cumsum()

        # Model Adherence: Compares filtered real speed with model
        spatiotemporal_data["square_differences"] = (
            spatiotemporal_data["smooth_speed"] - spatiotemporal_data["Model speed (m/s)"]
        ) ** 2

        ss_res = spatiotemporal_data["square_differences"].sum()
        mean_speed = spatiotemporal_data["smooth_speed"].mean()
        ss_tot = ((spatiotemporal_data["smooth_speed"] - mean_speed) ** 2).sum()

        if ss_tot > 0:
            model_adherence = 1 - (ss_res / ss_tot)
        else:
            model_adherence = 0

        tau = params.get("Tau")
        t0 = params.get("t0")
        vmax = params.get("V_max")

        params = {
            "Model_Adherence": model_adherence,
            "Tau": tau,
            "V_max": vmax,
            "t0": t0,
        }

        results = self.calculate_dynamics(spatiotemporal_data)
        results.update(params)

        return results


class SprintSplitTimeProfiler(BaseSprintProfiler):
    """
    A class to analyze sprint performance using discrete split times.
    (e.g., Timing Gates, Video Analysis)
    """

    def __init__(
        self,
        distances: np.ndarray,
        times: np.ndarray,
        height: float,
        weight: float,
        wind_speed: float = 0.0,
        temperature_c: float = 20.0,
        barometric_pressure_hpa: float = 1013.25,
    ):
        """
        Initialize the profiler with athlete antropometrics, environmental conditions and split times.

        Args:
            distances (np.ndarray): Cumulative distances [d1, d2, d3] (e.g., [5, 10, 15]).
            times (np.ndarray): Cumulative times [t1, t2, t3] corresponding to distances.
            height (float): Athlete's height in centimeters or meters.
            weight (float): Athlete's body mass in kg.
            wind_speed (float, optional): Headwind (+) or tailwind (-) in m/s. Defaults to 0.0.
            temperature_c (float, optional): Air temperature in Celsius. Defaults to 20.0.
            barometric_pressure_hpa (float, optional): Pressure in hPa. Defaults to 1013.25.
        """
        super().__init__(
            weight=weight,
            height=height,
            temperature_c=temperature_c,
            barometric_pressure_hpa=barometric_pressure_hpa,
            wind_speed=wind_speed,
        )

        self.distances = distances
        self.times = times

    @staticmethod
    def morin_distance_model(t: np.ndarray, vmax: float, tau: float) -> np.ndarray:
        """
        Distance-time relationship derived from the mono-exponential velocity model.
        Formula: d(t) = Vmax * (t - tau * (1 - e^(-t/tau)))
        """
        t = np.maximum(0, t)
        model_distance = vmax * (t - tau * (1 - np.exp(-t / tau)))
        return model_distance

    def fit_distance_model(
        self, distances: np.ndarray, times: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
        """
        Fits the distance model to discrete timing points.

        Returns:
            tuple: (model_speed, model_acc, simulated_time, params_dict)
        """
        # Initial guess & bounds [Vmax, Tau]
        top_avg_speed = (distances[-1] - distances[-2]) / (times[-1] - times[-2])
        p0 = [top_avg_speed, 1.3]
        bounds = ([0, 0.01], [15, 5])

        popt, _ = curve_fit(self.morin_distance_model, times, distances, p0=p0, bounds=bounds)
        vmax, tau = popt

        # Generate synthetic time data (extended to 6s or past last split)
        simulation_end = max(times[-1] + 1, 6)
        simulated_time = np.arange(0, simulation_end, 0.01)

        # Generate model curve based on fitted parameters
        model_speed = vmax * (1 - np.exp(-simulated_time / tau))

        # Analytic derivation of acceleration: a(t) = (Vmax/Tau) * e^(-(t-t0)/Tau)
        model_acc = (vmax / tau) * np.exp(-simulated_time / tau)

        params = {"Tau": tau, "V_max": vmax}

        return model_speed, model_acc, simulated_time, params

    def calculate_profile(self) -> dict[str, float]:
        """
        Main pipeline for Split Times.

        Steps:
        1. Validates data to ensure increasing speed (acceleration phase).
        2. Fit Morin's model to determine kinematic parameters (Vmax, Tau, t0).
        3. Calculates adherence (Goodness of fit on distances).
        4. Computes dynamics using the Base class.

        Returns:
            Dict containing F0, V0, Pmax, F-V Slope, RFmax, DRF, etc.
        """

        distances = self.distances
        times = self.times

        # Calculate average speed per segment to detect end of acceleration
        delta_d = np.diff(distances)
        delta_t = np.diff(times)

        first_segment_speed = distances[0] / times[0]
        segment_speeds = delta_d / delta_t

        speeds = np.concatenate(([first_segment_speed], segment_speeds))

        acc_end = 0

        # Iterate to find where speed stops increasing
        for i in range(1, len(speeds)):
            if speeds[i] > speeds[i - 1]:
                acc_end = i

            else:
                break

        # Cut data to include the acceleration phase only
        distances = distances[: acc_end + 1]
        times = times[: acc_end + 1]

        # Need at least 4 points for a valid fit
        if len(distances) < 4:
            raise ValueError("Insufficient data points, include at least 4 data points with increasing speed")

        # Model fitting
        model_speed, model_acc, simulated_time, params = self.fit_distance_model(distances, times)

        tau = params["Tau"]
        vmax = params["V_max"]

        # Model Adherence: Compare actual measured distances vs model predicted distances
        model_distances_at_times = self.morin_distance_model(times, vmax, tau)

        ss_res = np.sum((distances - model_distances_at_times) ** 2)
        ss_tot = np.sum((distances - np.mean(distances)) ** 2)

        if ss_tot > 0:
            model_adherence = 1 - (ss_res / ss_tot)
        else:
            model_adherence = 0

        params = {
            "Model_Adherence": model_adherence,
            "Tau": tau,
            "V_max": vmax,
            "distances": distances,
        }

        # Prepare DataFrame for the parent class
        spatiotemporal_data = pd.DataFrame(
            {"Model speed (m/s)": model_speed, "Acceleration (m/s2)": model_acc, "time": simulated_time}
        )

        results = self.calculate_dynamics(spatiotemporal_data)
        results.update(params)

        return results
