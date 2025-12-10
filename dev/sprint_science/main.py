import matplotlib.pyplot as plt
from sprint_science.simulator import SprintSimulation
from sprint_science.utilities import find_profile_for_time
from .validate_fatigue import validate_fatigue, get_100_200_times
from . import reporting


F0 = 8
V0 = 11.5
weight = 83
height = 1.85
running_distance = 100
external_force = 0
unloaded_speed = 10.5
fly_length = 30

profile = {'F0': F0, 'V0': V0, 'weight': weight, 'height': height, 'running_distance': running_distance, 'external_force_N': external_force, 'unloaded_speed': unloaded_speed, 'fly_length': fly_length}

# saved profiles
jara1 = {'F0': 7.68, 'V0': 10.51, 'weight': 74, 'height': 1.84, 'running_distance': 100, 'external_force_N': 0, 'fly_length': 30}
jara2 = {'F0': 7.65, 'V0': 10.36, 'weight': 74, 'height': 1.84, 'running_distance': 100, 'external_force_N': 0, 'fly_length': 30}
strasky1 = {'F0': 7.65, 'V0': 10.22, 'weight': 84, 'height': 1.84, 'running_distance': 100, 'external_force_N': 0, 'fly_length': 30}
strasky2 = {'F0': 7.85, 'V0': 10.36, 'weight': 84, 'height': 1.84, 'running_distance': 100, 'external_force_N': 0, 'fly_length': 30}
salcmanova1 = {'F0': 6.69, 'V0': 9.33, 'weight': 66, 'height': 1.84, 'running_distance': 100, 'external_force_N': 0, 'fly_length': 30}
salcmanova2 = {'F0': 7.07, 'V0': 9.05, 'weight': 66, 'height': 1.84, 'running_distance': 100, 'external_force_N': 0, 'fly_length': 30}
salcmanova3 = {'F0': 6.78, 'V0': 8.69, 'weight': 66, 'height': 1.84, 'running_distance': 100, 'external_force_N': 0, 'fly_length': 30}
salcmanova4 = {'F0': 6.79, 'V0': 8.79, 'weight': 66, 'height': 1.84, 'running_distance': 100, 'external_force_N': 0, 'fly_length': 30}
splechtnova1 = {'F0': 6.57, 'V0': 9.45, 'weight': 71, 'height': 1.78, 'running_distance': 100, 'external_force_N': 0, 'fly_length': 30}
splechtnova2 = {'F0': 6.56, 'V0': 9.43, 'weight': 71, 'height': 1.78, 'running_distance': 100, 'external_force_N': 0, 'fly_length': 30}
vanek1 = {'F0': 7.23, 'V0': 10.63, 'weight': 85, 'height': 1.95, 'running_distance': 100, 'external_force_N': 0, 'fly_length': 30}
vanek2 = {'F0': 7.15, 'V0': 10.68, 'weight': 85, 'height': 1.95, 'running_distance': 200, 'external_force_N': 0, 'fly_length': 30}
tlaskal1 = {'F0': 7.64, 'V0': 10.15, 'weight': 68.7, 'height': 1.68, 'running_distance': 100, 'external_force_N': 0, 'fly_length': 30}
tlaskal2 = {'F0': 7.75, 'V0': 10.41, 'weight': 68.7, 'height': 1.68, 'running_distance': 100, 'external_force_N': 0, 'fly_length': 30}
tlaskal3 = {'F0': 7.68, 'V0': 10.51, 'weight': 74, 'height': 1.84, 'running_distance': 100, 'external_force_N': 0, 'fly_length': 30}
bartonek1 = {'F0': 7.29, 'V0': 10.43, 'weight': 80.0, 'height': 1.82, 'running_distance': 100, 'external_force_N': 0, 'fly_length': 30}
bartonek2 = {'F0': 7.12, 'V0': 10.48, 'weight': 80.0, 'height': 1.82, 'running_distance': 100, 'external_force_N': 0, 'fly_length': 30}
safar1 = {'F0': 7.89, 'V0': 10.18, 'weight': 74.3, 'height': 1.78, 'running_distance': 200, 'external_force_N': 0, 'fly_length': 30}
safar2 = {'F0': 7.53, 'V0': 10.19, 'weight': 74.3, 'height': 1.78, 'running_distance': 100, 'external_force_N': 0, 'fly_length': 30}
sedlacek1 = {'F0': 7.61, 'V0': 9.81, 'weight': 74.0, 'height': 1.83, 'running_distance': 100, 'external_force_N': 0, 'fly_length': 30}
sedlacek2 = {'F0': 7.81, 'V0': 9.75, 'weight': 74.0, 'height': 1.83, 'running_distance': 100, 'external_force_N': 0, 'fly_length': 30}
dvorakova1 = {'F0': 5.82, 'V0': 8.38, 'weight': 69.0, 'height': 1.81, 'running_distance': 100, 'external_force_N': 0, 'fly_length': 30}
dvorakova2 = {'F0': 5.98, 'V0': 8.36, 'weight': 69.0, 'height': 1.81, 'running_distance': 100, 'external_force_N': 0, 'fly_length': 30}


bolt = {'F0': 9, 'V0': 13, 'weight': 94, 'height': 1.95, 'running_distance': 100, 'external_force_N': 0, 'fly_length': 30}


#profile picker
profile = bolt

analyza = SprintSimulation(**profile)

#reporting.complete_report(analyza)

reporting.segment_report(analyza)

#reporting.top_speed_report(analyza)

#reporting.flying_sections_report(analyza)

#reporting.overspeed_zones_report(analyza)

#reporting.plot_add_trial_to_v_distance(analyza)
#reporting.plot_trial_v_distance(analyza)

#reporting.fastest_f_v_report(analyza)
#reporting.plot_fastest_f_v(analyza)

#reporting.calibration(analyza)

#get_100_200_times(0.75, 'W')

#print(SprintSimulation.calculate_air_density(analyza))




#print(validate_fatigue(sex='W'))



# python -m dev.sprint_simulator.main