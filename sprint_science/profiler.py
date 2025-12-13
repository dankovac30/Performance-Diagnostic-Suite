from scipy.optimize import curve_fit
import numpy as np
import math
from scipy.stats import linregress
from scipy.signal import butter, filtfilt
from sprint_science.physics import calculate_frontal_area, calculate_air_density, calculate_air_resistance_force
from sprint_science.simulator import SprintSimulation


class SprintProfilation:
    
    def __init__(self, raw_spatiometric_data, height, weight, wind_speed=0.0, temperature_c=20.0, barometric_preassure_hpa=1013.25):
        self.raw_spatiometric_data = raw_spatiometric_data.copy()
        self.height = height
        self.weight = weight
        self.frontal_area = calculate_frontal_area(height, weight)
        self.air_density = calculate_air_density(temperature_c, barometric_preassure_hpa)
        self.wind_speed = wind_speed
    
    @staticmethod    
    def morin_velocity_model(t, vmax, tau, delay):
        return np.where(t > delay, vmax * (1 - np.exp(-(t - delay) / tau)), 0)
    

    def smooth_filter(self, raw_spatiometric_data, cutoff_freq=1.3, order=4):
        
        dt = np.mean(np.diff(raw_spatiometric_data['time']))
        sample_rate = 1 / dt

        nyquist = 0.5 * sample_rate
        normal_cutoff = cutoff_freq / nyquist

        raw_speed = raw_spatiometric_data['speed'].values

        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        y = filtfilt(b, a, raw_speed)

        return y
     

    def find_speed_plateau(self, smooth_speed_array):
        
        idx_peak_speed = np.argmax(smooth_speed_array)

        return idx_peak_speed
    

    def calculate_Rfmax_DRF(self, F0, V0):

        sprinter = SprintSimulation(F0, V0, self.weight, self.height, running_distance=100, fatigue_toggle='OFF')
        df = sprinter.run_sprint()

        vertical_force = self.weight * 9.80665

        df_from_03 = df[df['time'] >= 0.3].copy()
        df_from_03['RF'] = (df_from_03['propulsion_force'] / np.sqrt(df_from_03['propulsion_force']**2 + vertical_force**2)) * 100

        reg_result = linregress(x=df_from_03['speed'], y=df_from_03['RF'])

        DRF = reg_result.slope
        Rf_max = df_from_03.iloc[0]['RF']

        return Rf_max, DRF
        

    def calculate_profile(self):
        
        raw_spatiometric_data = self.raw_spatiometric_data.copy()
        raw_spatiometric_data = raw_spatiometric_data.drop(columns=['position', 'acceleration'])

        smooth_speed_array = self.smooth_filter(self.raw_spatiometric_data)
        idx_peak_speed = self.find_speed_plateau(smooth_speed_array)

        spatiometric_data = raw_spatiometric_data.iloc[:idx_peak_speed+1].copy()

        rename = {
            'time': 'Time (s)',
            'speed': '1080 Speed (m/s)',
            'force': 'F external load (N)'
        }

        spatiometric_data = spatiometric_data.rename(columns=rename)

        v_data = spatiometric_data['1080 Speed (m/s)'].to_numpy()
        t_data = spatiometric_data['Time (s)'].to_numpy()

        try:
            
            p0 = [np.max(v_data), 1.3, 0.0]
            bounds = ([0, 0.1, -3], [15, 5.0, 3])

            popt, _ = curve_fit(self.morin_velocity_model, t_data, v_data, p0=p0, bounds=bounds)
            vmax, tau, delay = popt

        except RuntimeError:
            return {}
        
        spatiometric_data['Model speed (m/s)'] = np.where(spatiometric_data['Time (s)'] > delay, vmax * (1 - np.exp(-(spatiometric_data['Time (s)'] - delay) / tau)), 0)
        dt = spatiometric_data['Time (s)'].diff().fillna(0)
        delta_distance = spatiometric_data['Model speed (m/s)'] * dt
        spatiometric_data['Position (m)'] = delta_distance.cumsum()
        spatiometric_data['Square differences'] = (spatiometric_data['1080 Speed (m/s)'] - spatiometric_data['Model speed (m/s)']) ** 2
        raw_acc = (vmax/tau) * np.exp(-(spatiometric_data['Time (s)'] - delay) / tau)
        spatiometric_data['Acceleration (m/s2)'] = np.where(spatiometric_data['Time (s)'] > delay, raw_acc, 0)
        spatiometric_data['F_Hzt (N)'] = spatiometric_data['Acceleration (m/s2)'] * self.weight
        spatiometric_data['F air (N)'] = calculate_air_resistance_force(spatiometric_data['Model speed (m/s)'], self.air_density, self.frontal_area, self.wind_speed)
        spatiometric_data['F Hzt total (N)'] =  spatiometric_data['F_Hzt (N)'] + spatiometric_data['F air (N)'] + spatiometric_data['F external load (N)']
        spatiometric_data['F Hzt total (N/kg)'] = spatiometric_data['F Hzt total (N)'] / self.weight
        spatiometric_data['Power Hzt (W/kg)'] = spatiometric_data['Model speed (m/s)'] * spatiometric_data['F Hzt total (N/kg)']
        spatiometric_data['F Resultant (N)'] = np.sqrt(spatiometric_data['F Hzt total (N)']**2 + (self.weight * 9.80665)**2)      
        spatiometric_data['RF (%)'] = (spatiometric_data['F Hzt total (N)'] / spatiometric_data['F Resultant (N)']) * 100

        mask_f_v_raw = spatiometric_data['F Hzt total (N)'] > spatiometric_data['F external load (N)']
        mask_f_v = mask_f_v_raw.cumsum().astype(bool)
        res_f_v = linregress(x=spatiometric_data.loc[mask_f_v, 'Model speed (m/s)'],
                            y=spatiometric_data.loc[mask_f_v, 'F Hzt total (N)'])
        
        ss_res = spatiometric_data['Square differences'].sum()
        mean_speed = spatiometric_data['1080 Speed (m/s)'].mean()
        ss_tot = ((spatiometric_data['1080 Speed (m/s)'] - mean_speed) ** 2).sum()

        if ss_tot > 0:
            curve_confidence = 1 - (ss_res / ss_tot)
        else:
            curve_confidence = 0    

        #spatiometric_data.to_excel('pd.xlsx', index=False)
        
        slope = -res_f_v.slope
        F0_abs = res_f_v.intercept
        F0 = F0_abs / self.weight
        V0 = F0_abs/slope
        F_V_slope = F0 / V0
        Pmax = (F0 * V0) / 4
        
        Rf_max, DRF = self.calculate_Rfmax_DRF(F0, V0)

        dictionary = {
            'F0': F0,
            'V0': V0,
            'Pmax': Pmax,
            'F_V_slope': F_V_slope,
            'Rf_max': Rf_max,
            'DRF': DRF,
            'Tau': tau,
            'Delay': delay,
            'Confidence': curve_confidence
        }

        return dictionary
    
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
                 color='red', linewidth=2, label='Morin Model')
        # Čára startu (Delay)
        plt.axvline(x=delay, color='green', linestyle='--', label=f'Calculated Start ({delay:.3f}s)')
        
        plt.title(f'Velocity Model Fit (Tau={tau:.3f})')
        plt.xlabel('Time (s)')
        plt.ylabel('Speed (m/s)')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # GRAF 2: F-V Profil (Linearita)
        plt.subplot(1, 2, 2)
        # Body použité pro regresi (vyfiltrované tvou maskou mask_f_v)
        subset_regress = spatiometric_data[mask_f_v]
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