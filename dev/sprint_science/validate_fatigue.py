from sprint_science.simulator import SprintSimulation
import numpy as np
import json
import math


def get_100_200_times(sfv, sex='M'):

    men_antropometric = (1.80, 78.8)
    women_antropometric = (1.68, 58.1)

    if sex == 'M':
        antropometric = men_antropometric
    else:
        antropometric = women_antropometric  

    for Pmax in range(10, 31):
        V0 = math.sqrt((4*Pmax)/sfv)
        F0 = V0 * sfv

        simulation_100 = SprintSimulation(F0=F0, V0=V0, weight=antropometric[1], height=antropometric[0], running_distance=100)
        simulation_200 = SprintSimulation(F0=F0, V0=V0, weight=antropometric[1], height=antropometric[0], running_distance=200)

        report_100 = simulation_100.run_sprint()
        report_200 = simulation_200.run_sprint()

        running_time_100 = report_100['time'].iloc[-1]
        running_time_200 = report_200['time'].iloc[-1]
        running_time_100_in_200 = report_200[report_200['distance'] >= 100].iloc[0]['time']

        print(f'Pmax: {Pmax}{running_time_100:>8.2f}{running_time_200:>8.2f}{(running_time_100_in_200 - running_time_100):>8.2f}{(running_time_200/running_time_100):>8.3f}')


def find_profile_for_time_FATIGUE(target_time, sfv, height, weight, fatigue_threshold, fatigue_strength, running_distance=100):

    min_V0 = 1.0
    max_V0 = 20
    tolerance = 0.001

    for _ in range(20):

        V0 = (min_V0 + max_V0) / 2
        F0 = V0 * sfv

        simulation = SprintSimulation(F0 = F0, V0 = V0, weight=weight, height=height, running_distance=running_distance, fatigue_threshold=fatigue_threshold, fatigue_strength=fatigue_strength)
        simulation_run = simulation.run_sprint()
        simulation_time = simulation_run['time'].iloc[-1]

        error = simulation_time - target_time

        if abs(error) < tolerance:
            return F0, V0
        
        elif error > 0:
            min_V0 = V0

        else:
            max_V0 = V0

    return F0, V0


def fatigue_calibration(fatigue_threshold_grid, fatigue_strength_grid, sex = 'M'):

    with open('dev/sprint_simulator/tables.json', 'r', encoding='utf-8') as f:
        sprint_data = json.load(f)

    target_sfv = 0.75
    men_antropometric = (1.80, 78.8)
    women_antropometric = (1.68, 58.1)
     
    if sex == 'M':
        sprint_data = sprint_data['men']
        antropometric = men_antropometric
    else:
        sprint_data = sprint_data['women']
        antropometric = women_antropometric        

    cum_error = float('inf')
    

    for fatigue_threshold in np.arange(*fatigue_threshold_grid):

        for fatigue_strength in np.arange(*fatigue_strength_grid):
            
            local_error = 0

            for i, (val100, val200) in enumerate(zip(sprint_data['100'], sprint_data['200'])):
                height = antropometric[0]
                weight = antropometric[1]

                F0, V0 = find_profile_for_time_FATIGUE(target_time=val100, sfv=target_sfv, height=height, weight=weight, running_distance=100, fatigue_threshold=fatigue_threshold, fatigue_strength=fatigue_strength)

                simulation_200 = SprintSimulation(F0 = F0, V0 = V0, height=height, weight=weight, running_distance=200, fatigue_threshold=fatigue_threshold, fatigue_strength=fatigue_strength)
                simulation_200_run = simulation_200.run_sprint()
                simulation_200_time = simulation_200_run['time'].iloc[-1]

                error = abs(val200 - simulation_200_time)
                local_error += error

                if local_error > cum_error:
                    break


            if local_error < cum_error:

                cum_error = local_error
                result = (fatigue_threshold, fatigue_strength, cum_error)
    
    print(result)
    return result 
                

def validate_fatigue(sex="M"):

    fatigue_threshold_grid = [2, 10, 1]
    fatigue_strength_grid = [10, 100, 10]
    precision = 1
    loop = 1

    while True:
        print(f'\nLoop n.{loop}')

        calculation = fatigue_calibration(fatigue_threshold_grid, fatigue_strength_grid, sex)

        fatigue_threshold_grid = [calculation[0] - precision*2, calculation[0] + precision*2.1, precision/10]
        fatigue_strength_grid = [calculation[1] - precision*20, calculation[1] + precision*21, precision]
        precision /= 10
        loop += 1

        
