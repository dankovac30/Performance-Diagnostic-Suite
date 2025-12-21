from typing import Tuple, Dict, Any, Optional
from scipy.optimize import curve_fit
import numpy as np
import math
import pandas as pd
from scipy.stats import linregress
from sprint_science.physics import calculate_frontal_area, calculate_air_density, calculate_air_resistance_force
from core.signal_processing import apply_butterworth_filter, find_speed_plateau
from sprint_science.simulator import SprintSimulation


class SprintProfilation:
    """
    A class to analyze sprint performance using 1080 Motion data based on 
    Morin's macroscopic force-velocity model.
    https://doi.org/10.1007/978-3-319-05633-3_11
    """

    def __init__(self,
                 raw_spatiotemporal_data: pd.DataFrame,
                 height: float,
                 weight: float,
                 wind_speed: float = 0.0,
                 temperature_c: float=20.0,
                 barometric_pressure_hpa: float = 1013.25):
        """
        Initialize the profiler with athlete data and environmental conditions.

        Args:
            raw_spatiotemporal_data (pd.DataFrame): Raw data from 1080 Motion (time, speed, force).
            height (float): Athlete's height in centimeters or meters.
            weight (float): Athlete's body mass in kg.
            wind_speed (float, optional): Headwind (+) or tailwind (-) in m/s. Defaults to 0.0.
            temperature_c (float, optional): Air temperature in Celsius. Defaults to 20.0.
            barometric_pressure_hpa (float, optional): Pressure in hPa. Defaults to 1013.25.
        """

        self.raw_spatiotemporal_data = raw_spatiotemporal_data.copy()
        self.height = height
        self.weight = weight
        self.wind_speed = wind_speed
        
        # Calculate constants for air resistance
        self.frontal_area = calculate_frontal_area(height, weight)
        self.air_density = calculate_air_density(temperature_c, barometric_pressure_hpa)


    @staticmethod    
    def morin_velocity_model(t: np.ndarray, vmax: float, tau: float, t0: float) -> np.ndarray:
        """
        Mono-exponential velocity function with a time delay parameter.
        Formula: v(t) = Vmax * (1 - e^(-(t-t0)/tau))
        """
        return np.where(t > t0, vmax * (1 - np.exp(-(t - t0) / tau)), 0)
    

    def fit_morin(self, v_data: np.ndarray, t_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
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
        raw_acc = (vmax/tau) * np.exp(-(t_data - t0) / tau)
        model_acc = np.where(t_data > t0, raw_acc, 0)

        params = {'Tau': tau, 't0': t0, 'V_max': vmax}

        return model_speed, model_acc, params


    def calculate_Rfmax_DRF(self, F0: float, V0: float) -> Tuple[float, float]:
        """
        Calculates Rf_max and DRF using an iterative simulation.
        
        Methodology:
        1. Simulate an ideal sprint with the given F0/V0 profile.
        2. Extract metrics from the smooth simulated curve.
        3. Cutoff at acceleration < 0.05 m/s^2 to ensure DRF represents only the acceleration phase.
        """
        sprinter = SprintSimulation(F0, V0, self.weight, self.height, running_distance=100, fatigue_toggle='OFF')
        df = sprinter.run_sprint()

        vertical_force = self.weight * 9.80665

        # Filter: Start after 0.3s
        df_cropped = df[(df['time'] >= 0.3) & (df['acceleration'] > 0.05)].copy()
        
        # Calculate Ratio of Force (RF)
        df_cropped['RF'] = (df_cropped['propulsion_force'] / np.sqrt(df_cropped['propulsion_force']**2 + vertical_force**2)) * 100

        # Linear regression to find DRF
        reg_result = linregress(x=df_cropped['speed'], y=df_cropped['RF'])

        DRF = reg_result.slope
        Rf_max = df_cropped.iloc[0]['RF']

        return Rf_max, DRF
        

    def calculate_profile(self) -> Dict[str, float]:
        """
        Main pipeline to compute the complete Force-Velocity profile.
        
        Steps:
        1. Pre-process and smooth raw data.
        2. Fit Morin's model to determine kinematic parameters (Vmax, Tau, t0).
        3. Calculate dynamics (Force, Power) including air resistance.
        4. Validate model adherence (Goodness of fit).
        
        Returns:
            Dict containing F0, V0, Pmax, F-V Slope, RFmax, DRF, etc.
        """
        raw_spatiotemporal_data = self.raw_spatiotemporal_data.copy()
        raw_spatiotemporal_data = raw_spatiotemporal_data.drop(columns=['position', 'acceleration'])

        # Smoothing: Smooth external force and speed
        raw_spatiotemporal_data['force'] = apply_butterworth_filter(raw_spatiotemporal_data, 'force', cutoff_freq=1.3, order=4)
        smooth_speed_array = apply_butterworth_filter(raw_spatiotemporal_data, 'speed')
        
        # Cutoff post peak speed on filtered array
        idx_peak_speed = find_speed_plateau(smooth_speed_array)
        raw_spatiotemporal_data['Smooth 1080 Speed (m/s)'] = smooth_speed_array
             
        spatiotemporal_data = raw_spatiotemporal_data.iloc[:idx_peak_speed+1].copy()

        # Rename to match Morin's Excel sheet
        rename = {
            'time': 'Time (s)',
            'speed': '1080 Speed (m/s)',
            'force': 'F external load (N)'
        }
        spatiotemporal_data = spatiotemporal_data.rename(columns=rename)
        
        v_data = spatiotemporal_data['1080 Speed (m/s)'].to_numpy()
        t_data = spatiotemporal_data['Time (s)'].to_numpy()

        # Model fitting
        model_speed, model_acc, params = self.fit_morin(v_data, t_data)

        # Physics calculations
        spatiotemporal_data['Model speed (m/s)'] = model_speed
        spatiotemporal_data['Acceleration (m/s2)'] = model_acc
        dt = spatiotemporal_data['Time (s)'].diff().fillna(0)
        delta_distance = spatiotemporal_data['Model speed (m/s)'] * dt
        spatiotemporal_data['Position (m)'] = delta_distance.cumsum()
        spatiotemporal_data['Square differences'] = (spatiotemporal_data['Smooth 1080 Speed (m/s)'] - spatiotemporal_data['Model speed (m/s)']) ** 2
        spatiotemporal_data['F Hzt (N)'] = spatiotemporal_data['Acceleration (m/s2)'] * self.weight
        spatiotemporal_data['F air (N)'] = calculate_air_resistance_force(spatiotemporal_data['Model speed (m/s)'], self.air_density, self.frontal_area, self.wind_speed)
        spatiotemporal_data['F Hzt total (N)'] =  spatiotemporal_data['F Hzt (N)'] + spatiotemporal_data['F air (N)'] + spatiotemporal_data['F external load (N)']
        spatiotemporal_data['F Hzt total (N/kg)'] = spatiotemporal_data['F Hzt total (N)'] / self.weight
        spatiotemporal_data['Power Hzt (W/kg)'] = spatiotemporal_data['Model speed (m/s)'] * spatiotemporal_data['F Hzt total (N/kg)']
        spatiotemporal_data['F Resultant (N)'] = np.sqrt(spatiotemporal_data['F Hzt total (N)']**2 + (self.weight * 9.80665)**2)      
        spatiotemporal_data['RF (%)'] = (spatiotemporal_data['F Hzt total (N)'] / spatiotemporal_data['F Resultant (N)']) * 100

        # Mask: Only include lines where athlete is producing force
        mask_force_produced_raw = spatiotemporal_data['F Hzt total (N)'] > spatiotemporal_data['F external load (N)']
        mask_force_produced = mask_force_produced_raw.cumsum().astype(bool)
        
        res_f_v = linregress(x=spatiotemporal_data.loc[mask_force_produced, 'Model speed (m/s)'],
                            y=spatiotemporal_data.loc[mask_force_produced, 'F Hzt total (N)'])
        
        # Model Adherence: Compares filtered real speed with model
        ss_res = spatiotemporal_data['Square differences'].sum()
        mean_speed = spatiotemporal_data['Smooth 1080 Speed (m/s)'].mean()
        ss_tot = ((spatiotemporal_data['Smooth 1080 Speed (m/s)'] - mean_speed) ** 2).sum()

        if ss_tot > 0:
            model_adherence = 1 - (ss_res / ss_tot)
        else:
            model_adherence = 0    
       
        # Final parameters
        slope = -res_f_v.slope
        F0_abs = res_f_v.intercept
        F0 = F0_abs / self.weight
        V0 = F0_abs/slope
        F_V_slope = F0 / V0
        Pmax = (F0 * V0) / 4

        tau = params.get('Tau')
        t0 = params.get('t0')
        
        # Calculate derived metrics via simulation
        Rf_max, DRF = self.calculate_Rfmax_DRF(F0, V0)

        results = {
            'F0': F0,
            'F0_abs': F0_abs,
            'V0': V0,
            'Pmax': Pmax,
            'F_V_slope': F_V_slope,
            'Rf_max': Rf_max,
            'DRF': DRF,
            'Model_Adherence': model_adherence,
            'Tau': tau,
            't0': t0,
            'source_data': spatiotemporal_data,
            'regress_mask': mask_force_produced
        }

        return results