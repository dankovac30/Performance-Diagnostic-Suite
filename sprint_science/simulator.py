from typing import List, Dict, Optional, Tuple, Union
import math
import pandas as pd
import numpy as np
from sprint_science.physics import calculate_frontal_area, calculate_air_density, calculate_air_resistance_force

class SprintSimulation:
    """
    A physics-based engine to simulate 100m sprint performance.
    
    This class models the athlete as a point mass subject to:
    1. Propulsive forces (derived from F0-V0 profile).
    2. Aerodynamic drag.
    3. Bend resistance (centrifugal effects on curve).
    4. Fatigue (decay of V0 profile after acceleration phase).
    """
    def __init__(self,
                 F0: float,
                 V0: float,
                 weight: float,
                 height: float,
                 running_distance: float,
                 wind_speed: float = 0.0,
                 temperature_c: float = 20.0,
                 barometric_pressure_hpa: float = 1013.25,
                 external_force_N: float = 0,
                 fly_length: float = 30,
                 sex: str = 'M',
                 fatigue_toggle: str ='ON', 
                 fatigue_threshold: Optional[float] = None,
                 fatigue_strength: Optional[float] = None):
        """
        Initialize the simulation environment and athlete parameters.

        Args:
            F0 (float): Theoretical maximum horizontal force per unit mass (N/kg).
            V0 (float): Theoretical maximum velocity (m/s).
            weight (float): Athlete's body mass (kg).
            height (float): Athlete's height (cm or m).
            running_distance (float): Total distance to simulate (m).
            wind_speed (float, optional): Headwind (-) or tailwind (+) in m/s.
            temperature_c (float, optional): For air density calculation.
            barometric_pressure_hpa (float, optional): For air density calculation.
            external_force_N (float, optional): Constant external load (1080 resistance). Positive = Resistance.
            fly_length (float, optional): Distance for flying split calculation (e.g., 30m fly).
            sex (str, optional): 'M' or 'W' to determine default fatigue profiles.
            fatigue_toggle (str, optional): 'ON'/'OFF' to enable physiological fatigue modeling.
            fatigue_threshold (float, optional): For brute force calibration on real life data.
            fatigue_strength (float, optional): For brute force calibration on real life data.
        """
        self.F0 = F0
        self.V0 = V0
        self.weight = weight
        self.height = height
        self.running_distance = running_distance
        self.wind_speed = wind_speed
        self.temperature_c = temperature_c
        self.barometric_pressure_hpa = barometric_pressure_hpa
        self.external_force_N = external_force_N
        self.fly_length = fly_length
        self.fatigue_toggle = fatigue_toggle
  
        # Time-step for discrete integration
        self.dt = 0.001

        # Calculate aerodynamic properties
        self.A = calculate_frontal_area(self.height, self.weight)
        self.rho = calculate_air_density(self.temperature_c, self.barometric_pressure_hpa)

        # Fatigue profile configuration based on 100-200m scoring tables conversions
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

        # Track geometry: Don't change lines (will probably break model)
        lane = 6
        self.bend_diameter = 35.28 + 1.22 * lane
        
        # Linear F-V Profile Slope
        self.f_v_inclination = self.F0 / self.V0

        # Cache for simulation results
        self.results_df = None


    def get_results(self) -> pd.DataFrame:
        """
        Runs the simulation if not already run, or returns cached results.

        Returns:
            pd.DataFrame: High-resolution time-series data of the sprint.
        """
        if self.results_df is None:
            self.results_df = self.run_sprint()
        
        return self.results_df
     

    def run_sprint(self) -> pd.DataFrame:
        """
        Core physics engine. Simulates the sprint.
        
        Methodology:
        At each time step `dt`:
        1. Calculate Potential Propulsive Force (F0 - slope * v).
        2. Subtract Resistances (Air, Bend, External).
        3. Apply Newton's Second Law (F=ma) to update Speed and Position.
        4. Apply Fatigue if acceleration phase is over (a < 0.05).
        
        Returns:
            pd.DataFrame: High-resolution time-series data of the sprint.
        """
        # Local variables for simulation
        F0 = self.F0
        V0 = self.V0
        original_V0 = self.V0

        # Initial state
        time = 0
        speed = 0
        covered_distance = 0
        fatigue_active = False 

        # Data containers
        time_list = []
        distance_list = []
        speed_list = []
        acceleration_list = []
        propulsion_force_list = []
                
        # SIMULATION LOOP
        while covered_distance < self.running_distance:

            # Propulsive force generation based on actual speed (F-V Model)
            f_propulsion = (F0 - (self.f_v_inclination * speed)) * self.weight
            f_propulsion = max(0, f_propulsion)

            # Bend resistance: EMPIRICAL APPROXIMATION
            # Applies a time penalty (approx. 0.2s - 0.3s per 100m) for runs > 100m
            if self.running_distance > 100 and covered_distance < (self.running_distance - 84.39):
                f_bend = 0.027 * (self.weight * speed**2.5) / self.bend_diameter
            else:
                f_bend = 0

            # Aerodynamic drag
            f_air = calculate_air_resistance_force(speed, self.rho, self.A, self.wind_speed)

            # Net force calculation
            f_resultant = f_propulsion - f_air - f_bend - self.external_force_N
            
            # Kinematics (Newton's 2nd Law)
            acceleration = f_resultant / self.weight

            # Fatigue activation at the end of the acceleration
            if acceleration < 0.05 and not fatigue_active and self.fatigue_toggle == 'ON':
                fatigue_active = True

            if fatigue_active:
                V0 -= ((original_V0 - self.fatigue_threshold) / self.fatigue_strength) * self.dt
                F0 = V0 * self.f_v_inclination

            # Store data
            time_list.append(time)
            distance_list.append(covered_distance)
            speed_list.append(speed)
            acceleration_list.append(acceleration)
            propulsion_force_list.append(f_propulsion)
            
            
            # Update state
            covered_distance += (speed * self.dt)
            speed += (acceleration * self.dt)
            speed = max(0.0001, speed)
            time += self.dt
            
            # Debug print
            # print(f"Time: {time:.2f}s | Distance: {covered_distance:.2f}m | Speed: {speed:.2f}m/s | Acceleration: {acceleration:.2f}m/s²")
            
            # END LOOP
        
        results = {
            'time': time_list,
            'distance': distance_list,
            'speed': speed_list,
            'acceleration': acceleration_list,
            'propulsion_force': propulsion_force_list
        }

        self.results_df = pd.DataFrame(results)

        return self.results_df


    def top_speed(self) -> Dict[str, float]:
        """
        Identifies the maximum speed reached during the simulation.
        
        Returns:
            Dict: Raw top speed and 'Rounded' top speed to one decimal place.
        """
        results_df = self.get_results()

        # Create rounded speed column
        results_df['speed_rounded'] = results_df['speed'].round(1)

        # Precise peak
        index_top_speed = results_df['speed'].idxmax()
        top_speed_row = results_df.loc[index_top_speed]
        top_speed = top_speed_row['speed']
        distance_top_speed = top_speed_row['distance']

        # Rounded peak
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


    def segments(self) -> pd.DataFrame:
        """
        Calculates split times for every 10 meters (e.g., 0-10m, 10-20m).

        Returns:
            pd.DataFrame: A Dataframe containing calculated times of each segment and cumulative time.
        """
        data = self.get_results()
        
        # List of 10m segments
        boundary_list = list(range(10, int(self.running_distance + 1), 10))
        
        segment_list = []
        total_time_list = []
        segment_time_list = []
        previous_time = 0

        for record in boundary_list:
            # Find the first frame beyond the distance marker
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
        
        # Handle the final segment
        last_row = data.iloc[-1]

        segment = {
                'distance': last_row['distance'],
                'total_time': last_row['time'],
                'segment_time': last_row['time'] - previous_time
        }

        segment_list.append(segment)
        segment_df = pd.DataFrame(segment_list)

        return segment_df


    def flying_sections(self) -> Union[Dict[str, Dict[str, float]], str]:
        """
        Calculates the fastest 'flying' section (e.g., fastest 30m) using a sliding window.
        
        Returns:
            Dict: A dictionary containing the time, start, and finish position of the fastest section.
        """
        if self.fly_length > 60:
            
            raise ValueError(f"Maximum supported fly length is 60m.")
        
        elif self.running_distance < 100:

            raise ValueError(f"To perform the flying section calculation, please enter a run distance of at least 100m.")

        else:

            data = self.get_results()

            # Initialize trackers
            fastest_time = float('inf')
            fastest_start_m = 0
            fastest_finish_m = 0

            # Rounded trackers
            fastest_time_rounded = float('inf')
            fastest_start_m_rounded = 0
            fastest_finish_m_rounded = 0

            time_list = data['time']
            distance_list = data['distance']
            number_of_records = len(time_list)
            loop_marker = 0

            # Sliding window algorithm
            for start_index in range(number_of_records):
                
                start_m = distance_list[start_index]
                finish_m = start_m + self.fly_length
                
                # Start searching for finish from the last known position
                for finish_index in range(loop_marker, number_of_records):

                    if distance_list[finish_index] >= finish_m:
                        
                        finish_m = distance_list[finish_index]
                        real_segment_distance = finish_m - start_m

                        start_time = time_list[start_index]
                        finish_time = time_list[finish_index]
                        
                        # Interpolate exact time for the fly_length
                        segment_time = ((finish_time - start_time) / real_segment_distance) * self.fly_length
                        segment_time_rounded = round(segment_time, 2)
                        
                        # Update fastest
                        if segment_time < fastest_time:
                            fastest_time = segment_time
                            fastest_start_m = start_m
                            fastest_finish_m = start_m + self.fly_length

                        # Update rounded fastest
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


    def f_v_profile_comparison(self) -> pd.DataFrame:
        """
        Optimization Analysis:
        Simulates sprint performance for varying F0/V0 slopes while keeping Pmax constant.

        Returns:
            pd.DataFrame: A DataFrame containing calculated time of the run and force-velocity parameters for each F-V slope.
        """
        max_power = (self.F0 * self.V0) / 4
        
        f_v_slopes_resuls = []

        # Simulation settings
        min_value = 0.1
        max_value = 2.00
        f_v_slope_increment = 0.01
        f_v_slopes_range = np.arange(min_value, max_value, f_v_slope_increment)


        for value in f_v_slopes_range:
            # Recalculate V0 and F0 for the current slope keeping Pmax constant
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

        return pd.DataFrame(f_v_slopes_resuls)
        

    def overspeed_zones(self) -> pd.DataFrame:
        """
        Overspeed/Assisted Speed Simulation.
        Calculates the effect of towing (negative external force) on top speed gain.
        Stops when speed gain exceeds 10%.

        Returns:
            pd.DataFrame: A DataFrame detailing the effect of incremental external assisting force on the athlete's top speed.
        """    
        overspeed_zones = []
        
        # Baseline (unloaded)
        baseline_sim = SprintSimulation(F0=self.F0, V0=self.V0, weight=self.weight, height=self.height, running_distance=100, external_force_N=0)
        unloaded_top_speed = baseline_sim.top_speed()['top_speed']

        # Iterative increasing assistance by 1N
        for i in range(1, 200):
            
            external_force_N = -i

            temp_simulation = SprintSimulation(F0=self.F0, V0=self.V0, weight=self.weight, height=self.height, running_distance=100, external_force_N=external_force_N)

            top_speed = temp_simulation.top_speed()['top_speed']
            speed_gain = top_speed / unloaded_top_speed

            overspeed_zones.append({
                'external_force_N': external_force_N,
                'top_speed': top_speed,
                'speed_gain': speed_gain
            })
            
            if speed_gain > 1.1:
                break

        return pd.DataFrame(overspeed_zones)
