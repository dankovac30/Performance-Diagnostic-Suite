from sprint_simulator_core.simulator import SprintSimulation


def find_profile_for_time(target_time, sfv, height, weight, running_distance=100):

    min_V0 = 1.0
    max_V0 = 20
    tolerance = 0.001

    for _ in range(20):

        V0 = (min_V0 + max_V0) / 2
        F0 = V0 * sfv

        simulation = SprintSimulation(F0 = F0, V0 = V0, weight=weight, height=height, running_distance=running_distance)
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