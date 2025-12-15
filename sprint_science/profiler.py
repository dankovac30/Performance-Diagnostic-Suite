from scipy.optimize import curve_fit
import numpy as np
import math
import pandas as pd
from scipy.stats import linregress
from scipy.signal import butter, filtfilt
from sprint_science.physics import calculate_frontal_area, calculate_air_density, calculate_air_resistance_force
from sprint_science.simulator import SprintSimulation


class SprintProfilation:
    
    def __init__(self, raw_spatiometric_data, height, weight, data_fit = 'Morin', wind_speed=0.0, temperature_c=20.0, barometric_preassure_hpa=1013.25):
        self.raw_spatiometric_data = raw_spatiometric_data.copy()
        self.height = height
        self.weight = weight
        self.frontal_area = calculate_frontal_area(height, weight)
        self.air_density = calculate_air_density(temperature_c, barometric_preassure_hpa)
        self.wind_speed = wind_speed
        if data_fit not in ['Morin', 'Sigmoid']:
            raise ValueError("Data fit method must be 'Morin' or 'Sigmoid'")
        self.data_fit = data_fit
    

    @staticmethod    
    def morin_velocity_model(t, vmax, tau, t0):
        return np.where(t > t0, vmax * (1 - np.exp(-(t - t0) / tau)), 0)
    
    
    def fit_morin(self, v_data, t_data):
        
        p0 = [np.max(v_data), 1.3, 0]
        bounds = ([0, 0, -3], [15, 5, 3])

        popt, _ = curve_fit(self.morin_velocity_model, t_data, v_data, p0=p0, bounds=bounds)
        vmax, tau, t0 = popt

        model_speed = np.where(t_data > t0, vmax * (1 - np.exp(-(t_data - t0) / tau)), 0)
        raw_acc = (vmax/tau) * np.exp(-(t_data - t0) / tau)
        model_acc = np.where(t_data > t0, raw_acc, 0)

        params = {'Tau': tau, 't0': t0, 'V_max': vmax}

        return model_speed, model_acc, params


    @staticmethod
    def sigmoid_velocity_model(t, L, k, t0, b):
        return (L / (1 + np.exp(-k * (t - t0)))) + b
        

    def fit_sigmoid(self, v_data, t_data):

        p0 = [np.max(v_data), 1, 2, 0]
        bounds = ([0, 0.01, -2, -np.max(v_data)], [15, 5, 10, 2])

        popt, _ = curve_fit(self.sigmoid_velocity_model, t_data, v_data, p0=p0, bounds=bounds)
        L, k, t0, b = popt

        exponent = -k * (t_data - t0)
        sigmoid_denom = (1 + np.exp(exponent))

        model_speed_raw = (L / sigmoid_denom) + b
        model_speed = np.maximum(0, model_speed_raw)

        raw_acc = (L * k * np.exp(exponent)) / (sigmoid_denom ** 2)
        model_acc = np.maximum(0, raw_acc)

        params = {'Sigmoid_L': L, 'Sigmoid_k': k, 'Sigmoid_t0': t0, 'Sigmoid_b': b}
        
        return model_speed, model_acc, params


    def smooth_filter(self, raw_spatiometric_data, data_to_smooth, cutoff_freq=1.3, order=4):
        
        dt = np.mean(np.diff(raw_spatiometric_data['time']))
        sample_rate = 1 / dt

        nyquist = 0.5 * sample_rate
        normal_cutoff = cutoff_freq / nyquist

        raw_data = raw_spatiometric_data[data_to_smooth].values

        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        y = filtfilt(b, a, raw_data)

        return y
     

    def find_speed_plateau(self, smooth_speed_array):
        
        idx_peak_speed = np.argmax(smooth_speed_array)

        return idx_peak_speed
    

    def calculate_Rfmax_DRF(self, F0, V0):

        sprinter = SprintSimulation(F0, V0, self.weight, self.height, running_distance=100, fatigue_toggle='OFF')
        df = sprinter.run_sprint()

        vertical_force = self.weight * 9.80665

        df_cropped = df[(df['time'] >= 0.3) & (df['acceleration'] > 0.05)].copy()
        df_cropped['RF'] = (df_cropped['propulsion_force'] / np.sqrt(df_cropped['propulsion_force']**2 + vertical_force**2)) * 100

        reg_result = linregress(x=df_cropped['speed'], y=df_cropped['RF'])

        DRF = reg_result.slope
        Rf_max = df_cropped.iloc[0]['RF']

        return Rf_max, DRF
        

    def calculate_profile(self):
        
        raw_spatiometric_data = self.raw_spatiometric_data.copy()
        raw_spatiometric_data = raw_spatiometric_data.drop(columns=['position', 'acceleration'])

        smooth_speed_array = self.smooth_filter(raw_spatiometric_data, 'speed')
        idx_peak_speed = self.find_speed_plateau(smooth_speed_array)

        raw_spatiometric_data['force'] = self.smooth_filter(raw_spatiometric_data, 'force')
        
        spatiometric_data = raw_spatiometric_data.iloc[:idx_peak_speed+1].copy()

        rename = {
            'time': 'Time (s)',
            'speed': '1080 Speed (m/s)',
            'force': 'F external load (N)'
        }

        spatiometric_data = spatiometric_data.rename(columns=rename)
        
        v_data = spatiometric_data['1080 Speed (m/s)'].to_numpy()
        t_data = spatiometric_data['Time (s)'].to_numpy()

        if self.data_fit == 'Morin':
            model_speed, model_acc, params = self.fit_morin(v_data, t_data)

        elif self.data_fit == 'Sigmoid':
            model_speed, model_acc, params = self.fit_sigmoid(v_data, t_data)

        spatiometric_data['Model speed (m/s)'] = model_speed
        spatiometric_data['Acceleration (m/s2)'] = model_acc
        dt = spatiometric_data['Time (s)'].diff().fillna(0)
        delta_distance = spatiometric_data['Model speed (m/s)'] * dt
        spatiometric_data['Position (m)'] = delta_distance.cumsum()
        spatiometric_data['Square differences'] = (spatiometric_data['1080 Speed (m/s)'] - spatiometric_data['Model speed (m/s)']) ** 2
        spatiometric_data['F_Hzt (N)'] = spatiometric_data['Acceleration (m/s2)'] * self.weight
        spatiometric_data['F air (N)'] = calculate_air_resistance_force(spatiometric_data['Model speed (m/s)'], self.air_density, self.frontal_area, self.wind_speed)
        spatiometric_data['F Hzt total (N)'] =  spatiometric_data['F_Hzt (N)'] + spatiometric_data['F air (N)'] + spatiometric_data['F external load (N)']
        spatiometric_data['F Hzt total (N/kg)'] = spatiometric_data['F Hzt total (N)'] / self.weight
        spatiometric_data['Power Hzt (W/kg)'] = spatiometric_data['Model speed (m/s)'] * spatiometric_data['F Hzt total (N/kg)']
        spatiometric_data['F Resultant (N)'] = np.sqrt(spatiometric_data['F Hzt total (N)']**2 + (self.weight * 9.80665)**2)      
        spatiometric_data['RF (%)'] = (spatiometric_data['F Hzt total (N)'] / spatiometric_data['F Resultant (N)']) * 100

        mask_force_produced_raw = spatiometric_data['F Hzt total (N)'] > spatiometric_data['F external load (N)']
        mask_force_produced = mask_force_produced_raw.cumsum().astype(bool)
        idx_peak_acc = spatiometric_data['Acceleration (m/s2)'].idxmax()
        mask_post_peak = spatiometric_data.index >= idx_peak_acc
        final_mask = mask_force_produced & mask_post_peak
        
        res_f_v = linregress(x=spatiometric_data.loc[final_mask, 'Model speed (m/s)'],
                            y=spatiometric_data.loc[final_mask, 'F Hzt total (N)'])
        
        ss_res = spatiometric_data['Square differences'].sum()
        mean_speed = spatiometric_data['1080 Speed (m/s)'].mean()
        ss_tot = ((spatiometric_data['1080 Speed (m/s)'] - mean_speed) ** 2).sum()

        if ss_tot > 0:
            curve_confidence = 1 - (ss_res / ss_tot)
        else:
            curve_confidence = 0    
       
        slope = -res_f_v.slope
        F0_abs = res_f_v.intercept
        F0 = F0_abs / self.weight
        V0 = F0_abs/slope
        F_V_slope = F0 / V0
        Pmax = (F0 * V0) / 4
        
        Rf_max, DRF = self.calculate_Rfmax_DRF(F0, V0)

        b = params.get('Sigmoid_b')
        t = params.get('Sigmoid_t0')
        k = params.get('Sigmoid_k')

        dictionary = {
            'F0': F0,
            'V0': V0,
            'Pmax': Pmax,
            'F_V_slope': F_V_slope,
            'Rf_max': Rf_max,
            'DRF': DRF,
            'Confidence': curve_confidence,
            'Sigmoid_b': b,
            'Sigmoid_t0': t,
            'Sigmoid_k': k
        }

        #return dictionary

        # ==========================================
        # 🕵️‍♂️ DEBUG PLOT START
        # ==========================================
        import matplotlib.pyplot as plt

        plt.figure(figsize=(16, 6))

        # GRAF 1: Rychlost v čase (Model vs Realita)
        plt.subplot(1, 2, 1)
        # Reálná data (modré tečky)
        plt.scatter(spatiometric_data['Time (s)'], spatiometric_data['1080 Speed (m/s)'], 
                    color='blue', s=10, alpha=0.5, label='1080 Raw Data')
        # Modelová data (červená čára)
        plt.plot(spatiometric_data['Time (s)'], spatiometric_data['Model speed (m/s)'], 
                 color='red', linewidth=2, label='Model')     
       

        plt.title(f'Velocity Model Fit')
        plt.xlabel('Time (s)')
        plt.ylabel('Speed (m/s)')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # GRAF 2: F-V Profil (Linearita)
        plt.subplot(1, 2, 2)
        # Body použité pro regresi (vyfiltrované tvou maskou mask_f_v)
        subset_regress = spatiometric_data[final_mask]
        plt.scatter(subset_regress['Model speed (m/s)'], subset_regress['F Hzt total (N)'], 
                    color='black', s=15, label='Regression Data')
        
        # Vykreslení výsledné přímky F-V
        # Vytvoříme si pomocné body od 0 do V0
        x_line = np.linspace(0, V0, 10)
        # Rovnice přímky: F = F0_abs - (F0_abs/V0) * v
        y_line = F0_abs - (F0_abs / V0) * x_line
        
        plt.plot(x_line, y_line, color='red', linestyle='--', linewidth=2, label=f'Linear Fit')
        
        # Zvýraznění F0 (na ose Y)
        plt.plot(0, F0_abs, 'ro', markersize=8, label=f'F0 = {F0_abs:.0f} N')
        
        plt.title(f'F-V Profile (R2={res_f_v.rvalue**2:.3f})')
        plt.xlabel('Model Speed (m/s)')
        plt.ylabel('Total Force (N)')
        plt.xlim(left=0)
        plt.ylim(bottom=0)
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()
        # ==========================================
        # 🕵️‍♂️ DEBUG PLOT END    )