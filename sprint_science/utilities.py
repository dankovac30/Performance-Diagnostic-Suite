from sprint_science.simulator import SprintSimulation


def find_profile_for_time(
    target_time: float, sfv: float, height: float, weight: float, running_distance: float = 100.0
) -> tuple[float, float]:
    """
    Reverse-engineers the mechanical capabilities (F0, V0) required to achieve a specific
    sprint time over a given distance based on F0/V0 ratio.

    Returns:
        Tuple[float, float]: The calculated (F0, V0) pair that results in the target_time.
    """
    # Search bounds for V0
    min_V0 = 1.0
    max_V0 = 20
    tolerance = 0.001

    # Binary search loop
    for _ in range(20):
        V0 = (min_V0 + max_V0) / 2
        F0 = V0 * sfv

        simulation = SprintSimulation(F0=F0, V0=V0, weight=weight, height=height, running_distance=running_distance)
        simulation_run = simulation.run_sprint()
        simulation_time = simulation_run["time"].iloc[-1]

        error = simulation_time - target_time

        if abs(error) < tolerance:
            return F0, V0

        elif error > 0:
            min_V0 = V0

        else:
            max_V0 = V0

    return F0, V0
