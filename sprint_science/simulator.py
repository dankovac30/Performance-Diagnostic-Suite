import math
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

class SprintSimulation:
    
    def __init__(self, F0, V0, weight, height, running_distance, wind_speed=0.0, temperature_c=15.0, barometric_preassure_hpa=1013.25, external_force_N=0, unloaded_speed=None, fly_length=30, sex='M', fatigue_threshold=None, fatigue_strength=None):
        self.F0 = F0
        self.V0 = V0
        self.weight = weight
        self.height = height
        self.running_distance = running_distance
        self.wind_speed = wind_speed
        self.temperature_c = temperature_c
        self.barometric_preassure_hpa = barometric_preassure_hpa
        self.external_force_N = external_force_N
        self.unloaded_speed = unloaded_speed
        self.fly_length = fly_length
  
        # minimum time increment  
        self.dt = 0.001

        # body surface area (Du Bois 1916), converted to frontal area *0.266 (Pugh 1971)
        self.A = 0.2025 * (self.height ** 0.725) * (self.weight ** 0.425) * 0.266

        # fatigue setup
        men_fatigue_settings = (5.16, 39.5)
        women_fatigue_settings = (5.96, 33.1)

        if sex == 'W' and fatigue_threshold is None:
            
            self.fatigue_threshold = women_fatigue_settings[0]
            self.fatigue_strength = women_fatigue_settings[1]
        
        elif fatigue_threshold is None:
            self.fatigue_threshold = men_fatigue_settings[0]
            self.fatigue_strength = men_fatigue_settings[1]

        else:
            self.fatigue_threshold = fatigue_threshold
            self.fatigue_strength = fatigue_strength            

        # others
        lane = 6
        self.bend_diameter = 35.28 + 1.22 * lane
        self.f_v_inclination = self.F0 / self.V0

        # final report
        self.results_df = None


    def get_results(self):

        if self.results_df is None:
            self.results_df = self.run_sprint()
        
        return self.results_df
    

    def calculate_air_density(self):
        rho0 = 1.293
        p_std_torr = 760.0
        t_std_kelvin = 273.0

        temperature_kelvin = t_std_kelvin + self.temperature_c
        preassure_torr = self.barometric_preassure_hpa * (p_std_torr / 1013.25)

        rho = rho0 * (preassure_torr / p_std_torr) * (t_std_kelvin / temperature_kelvin)

        return rho
    

    def run_sprint(self):
        
        # variables
        F0 = self.F0
        V0 = self.V0
        original_V0 = self.V0

        # air resistance constants
        Cd = 0.9     # (van IngenSchenauetal. 1991)
        rho = self.calculate_air_density()

        # initial state
        time = 0
        speed = 0
        covered_distance = 0
        fatigie_active = False 

        # return
        time_list = []
        distance_list = []
        speed_list = []
        acceleration_list = []        

        while covered_distance < self.running_distance:

            # propulsive force
            f_propulsion = (F0 - (self.f_v_inclination) * speed) * self.weight
            f_propulsion = max(0, f_propulsion)

            # bend resistance
            if self.running_distance > 100 and covered_distance < (self.running_distance - 84.39):
                f_bend = 0.027 * (self.weight * speed**2.5) / self.bend_diameter

            else:
                f_bend = 0

            # air resistance
            relative_speed = speed - self.wind_speed
            f_resistance = 0.5 * rho * self.A * Cd * (relative_speed * abs(relative_speed))

            # resultant propulsive force
            f_resultant = f_propulsion - f_resistance - f_bend - self.external_force_N
            
            # acceleration
            acceleration = f_resultant / self.weight

            if acceleration < 0.05 and not fatigie_active:
                fatigie_active = True

            if fatigie_active:
                V0 -= ((original_V0 - self.fatigue_threshold) / self.fatigue_strength) * self.dt
                F0 = V0 * self.f_v_inclination

            # return
            time_list.append(time)
            distance_list.append(covered_distance)
            speed_list.append(speed)
            acceleration_list.append(acceleration)
            
            # update
            covered_distance += (speed * self.dt)
            speed += (acceleration * self.dt)
            speed = max(0.0001, speed)
            time += self.dt
            
            # debug
            # print(f"Time: {time:.2f}s | Distance: {covered_distance:.2f}m | Speed: {speed:.2f}m/s | Acceleration: {acceleration:.2f}m/s²")

        results = {
            'time': time_list,
            'distance': distance_list,
            'speed': speed_list,
            'acceleration': acceleration_list
        }

        self.results_df = pd.DataFrame(results)

        return self.results_df


    def top_speed(self):
        results_df = self.get_results()

        results_df['speed_rounded'] = results_df['speed'].round(1)

        index_top_speed = results_df['speed'].idxmax()
        top_speed_row = results_df.loc[index_top_speed]
        top_speed = top_speed_row['speed']
        distance_top_speed = top_speed_row['distance']

        index_top_speed_rounded = results_df['speed_rounded'].idxmax()
        top_speed_rounded_row = results_df.loc[index_top_speed_rounded]
        top_speed_rounded = top_speed_rounded_row['speed_rounded']
        distance_top_speed_rounded = top_speed_rounded_row['distance']
        
        report = {
            'top_speed': top_speed,
            'distance_top_speed': distance_top_speed,
            'top_speed_rounded': top_speed_rounded,
            'distance_top_speed_rounded': distance_top_speed_rounded
        }

        return report


    def segments(self):
        data = self.get_results()
        
        boundary_list = list(range(10, int(self.running_distance + 1), 10))
        
        segment_list = []
        total_time_list = []
        segment_time_list = []
        previous_time = 0

        
        for record in boundary_list:
            
            beyond_boundary_df = data[data['distance'] >= record]
            if beyond_boundary_df.empty:
                continue
            
            first_beyond_boundary_df = beyond_boundary_df.iloc[0]

            
            total_time = first_beyond_boundary_df['time']
            total_time_list.append(total_time)
            segment_time = first_beyond_boundary_df['time'] - previous_time
            segment_time_list.append(segment_time)

            previous_time = first_beyond_boundary_df['time']

            segment = {
                'distance': record,
                'total_time': total_time,
                'segment_time': segment_time
            }

            segment_list.append(segment)
        
        
        last_row = data.iloc[-1]

        segment = {
                'distance': last_row['distance'],
                'total_time': last_row['time'],
                'segment_time': last_row['time'] - previous_time
        }

        segment_list.append(segment)

        segment_df = pd.DataFrame(segment_list)

        return segment_df


    def flying_sections(self):
        
        if self.fly_length > 60:
            
            return "Nejdelší podporovaný letmý úsek je 60m"
        
        elif self.running_distance < 100:

            return "Pro výpočet letmého úseku zadejte délku běhu alespoň 100m"

        else:

            data = self.get_results()

            fastest_time = float('inf')
            fastest_start_m = 0
            fastest_finish_m = 0

            fastest_time_rounded = float('inf')
            fastest_start_m_rounded = 0
            fastest_finish_m_rounded = 0


            time_list = data['time']
            distance_list = data['distance']
            number_of_records = len(time_list)

            loop_marker = 0

            for start_index in range(number_of_records):
                
                start_m = distance_list[start_index]
                finish_m = start_m + self.fly_length

                for finish_index in range(loop_marker, number_of_records):

                    if distance_list[finish_index] >= finish_m:
                        
                        finish_m = distance_list[finish_index]
                        real_segment_distance = finish_m - start_m

                        start_time = time_list[start_index]
                        finish_time = time_list[finish_index]
                        segment_time = ((finish_time - start_time) / real_segment_distance) * self.fly_length
                        segment_time_rounded = round(segment_time, 2)

                        if segment_time < fastest_time:
                            fastest_time = segment_time
                            fastest_start_m = start_m
                            fastest_finish_m = start_m + self.fly_length

                        if segment_time_rounded < fastest_time_rounded:
                            fastest_time_rounded = segment_time_rounded
                            fastest_start_m_rounded = round(start_m, 0)
                            fastest_finish_m_rounded = fastest_start_m_rounded + self.fly_length

                        loop_marker = finish_index
                        break
                        
            fly_report = {
                'fastest': {
                    'time': fastest_time,
                    'start': fastest_start_m,
                    'finish': fastest_finish_m
                },
                'first_fast': {
                    'time': fastest_time_rounded,
                    'start': fastest_start_m_rounded,
                    'finish': fastest_finish_m_rounded
                }
            }

        return fly_report


    def f_v_profile_comparison(self):
        max_power = (self.F0 * self.V0) / 4
        
        f_v_slopes_resuls = []

        min_value = 0.1
        max_value = 2.00
        f_v_slope_increment = 0.01

        f_v_slopes_range = np.arange(min_value, max_value, f_v_slope_increment)


        for value in f_v_slopes_range:
            V0_new = math.sqrt((4*max_power)/value)
            F0_new = V0_new * value

            temp_simulation = SprintSimulation(F0=F0_new, V0=V0_new, weight=self.weight, height=self.height, running_distance=self.running_distance, external_force_N=self.external_force_N)

            data = temp_simulation.run_sprint()
            
            current_f_v = {
                'f_v_slope': value,
                'time': data['time'].iloc[-1],
                'F0': F0_new,
                'V0': V0_new
            }

            f_v_slopes_resuls.append(current_f_v)

        f_v_slopes_resuls_df = pd.DataFrame(f_v_slopes_resuls)

        return f_v_slopes_resuls_df
        

    def drag_coeficient_calibration(self):
        
        # last calibration
        # values = [0.8789, 0.877, 0.8553, 0.8624, 0.8857, 0.9045, 0.8768, 0.8732, 0.8707, 0.8752, 0.8808, 0.8753, 0.8928, 0.8761, 0.8789, 0.8807, 0.8836, 0.8954, 0.8921, 0.8799, 0.874, 0.8738, 0.8773]

        drag_coeficient = None
        smallest_error = float('inf')
        
        for i in range(8500, 9500):
            
            drag_coeficient_loop = i / 10000
            
            temp_simulation = SprintSimulation(F0=self.F0, V0=self.V0, weight=self.weight, height=self.height, running_distance=100, external_force_N=0, drag_coeficient=drag_coeficient_loop)

            speed_loop = temp_simulation.top_speed()

            current_error = abs(self.unloaded_speed - speed_loop['top_speed'])
            
            if current_error < smallest_error:
                smallest_error = current_error
                drag_coeficient = drag_coeficient_loop
                
        return drag_coeficient


    def overspeed_zones(self):
                
        overspeed_zones = []
        baseline_sim = SprintSimulation(F0=self.F0, V0=self.V0, weight=self.weight, height=self.height, running_distance=100, external_force_N=0)
        unloaded_speed = baseline_sim.top_speed()['top_speed']

        for i in range(1, 200):
            
            external_force_N = -i

            temp_simulation = SprintSimulation(F0=self.F0, V0=self.V0, weight=self.weight, height=self.height, running_distance=100, external_force_N=external_force_N)

            top_speed = temp_simulation.top_speed()['top_speed']
            speed_gain = top_speed / unloaded_speed

            overspeed_zones.append({
                'external_force_N': external_force_N,
                'top_speed': top_speed,
                'speed_gain': speed_gain
            })

            if speed_gain > 1.1:
                break

        overspeed_zones_df = pd.DataFrame(overspeed_zones)

        return overspeed_zones_df
