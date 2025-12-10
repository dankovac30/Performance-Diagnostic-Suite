# ⚙️ Sprint Simulator Core

### About

This directory contains the core computational engine for the `Performance-Diagnostic-Suite`.

It is **not a runnable script** but a Python library intended to be imported by other applications (like the `Sprint Simulator GUI` or analysis scripts in the `dev/` folder).

The primary component is the `SprintSimulation` class, which is built on a biomechanical model to simulate sprint kinematics from an athlete's Force-Velocity (F-V) profile.

---
### How it Works

The design is built around the central `SprintSimulation` class.

1.  **Initialization:** The class is first initialized with an athlete's physical parameters (F0, V0, weight, height) and the desired simulation settings such running distance or external force.

2.  **Core Simulation:** The main method, `run_sprint()`, executes an iterative physics simulation (using a `dt = 0.001s` time step) to model the entire sprint. This method calculates air resistance, propulsive forces, bend resistance and fatigue to generate a complete time-series of the sprint.

3.  **Helper Methods:** All other methods (`top_speed()`, `segments()`, `flying_sections()`, etc.) are "helper" functions that use cached DataFrame from `run_sprint()` to quickly calculate specific, derived performance metrics without needing to re-run the entire simulation.

---
### Key Methods

* **`run_sprint()`:** The main physics model. Returns a full `pandas` DataFrame of the sprint, including `time`, `distance`, `speed`, and `acceleration` for each time step.
* **`get_results()`:** A safe helper that calls `run_sprint()` only if results haven't been cached yet, then returns the results.
* **`top_speed()`:** Analyzes the results to find the absolute maximum velocity and the distance at which it was achieved.
* **`segments()`:** Calculates cumulative and segment split times for every 10-meter block of the sprint.
* **`flying_sections()`:** Finds the fastest "flying" segment of a given length (default `30m`) within the entire sprint.
* **`f_v_profile_comparison()`:** A powerful analysis tool that simulates multiple sprints (assuming constant Pmax) to find the optimal F-V slope (Sfv) for the given distance.

---

### Dependencies

This core library requires:
* `pandas`
* `numpy`

---

### Installation & Usage

This is a library, not a standalone script. The intended way to use this class is to import it into another script (like the `Sprint Simulator GUI` or your own analysis scripts in the `dev/` folder).

#### Example (from a `dev/` script):

```python
import pandas as pd
from sprint_simulator_core.simulator import SprintSimulation

# 1. Initialize the Simulation with athlete parameters
sim = SprintSimulation(
    F0=8.0, 
    V0=10.0, 
    weight=83.0, 
    height=1.85,
    running_distance=100
)

# 2. Get specific metrics using the helper methods
top_speed_data = sim.top_speed()
flying_30m = sim.flying_sections()

# 4. Print the results
print("--- Basic Results ---")
print(f"Full 100m time: {results_df['time'].iloc[-1]:.2f}s")
print(f"Max Speed: {top_speed_data['top_speed']:.2f} m/s")
print(f"Fastest Flying 30m: {flying_30m['first_fast']['time']:.2f}s")

# 5. Run advanced analysis
print("\n--- Optimal F-V Profile Analysis ---")
optimal_profile_df = sim.f_v_profile_comparison()
fastest = optimal_profile_df.loc[optimal_profile_df['time'].idxmin()]

print(f"Optimal Sfv for 100m: {fastest['f_v_slope']:.2f}")
print(f" (at F0: {fastest['F0']:.1f} and V0: {fastest['V0']:.1f})")