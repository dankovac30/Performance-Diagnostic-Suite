# ⚙️ Sprint Science

### About

This directory contains the core computational engine for the `Performance-Diagnostic-Suite`.

It is **not a runnable script** but a Python library intended to be imported by other applications. It consists of three interconnected modules that handle the biomechanics of sprinting:

1.  **`simulator.py` (Forward):** Predicts sprint performance from an athlete's F-V profile.
2.  **`profilation.py` (Inverse):** Calculates the F-V profile from raw 1080 Sprint data.
3.  **`physics.py`:** A shared physics engine handling air resistance, environmental conditions, and unit conversions.
4.  **`utilities.py`:** Auxiliary functions and supplementary routines.

---
### 1. Sprint Simulation (`simulator.py`)

The `SprintSimulation` class models sprint kinematics based on an athlete's **Force-Velocity (F-V) profile**.

#### How it Works
The model executes an iterative physics simulation (`dt = 0.001s`) to generate a complete time-series of the sprint. It accounts for:
* **Propulsive Forces:** Derived from maximal Force ($F_0$) and Velocity ($V_0$).
* **Resistive Forces:** Air drag (calculated via `physics.py`) and bend resistance (centripetal force).
* **Fatigue:** Modeling the decay of acceleration over distance.

#### Key Methods
* **`run_sprint()`:** Returns a full `pandas` DataFrame (time, distance, velocity, acceleration).
* **`top_speed()`:** Identifies maximum velocity and the distance reached.
* **`f_v_profile_comparison()`:** Simulates multiple scenarios to find the optimal F-V slope ($S_{fv}$) for a specific distance (e.g., 100m).

---
### 2. Sprint Profiling (`profilation.py`)

The `SprintProfilation` class solves the inverse problem: converting **raw time-speed data** into an athlete's physiological parameters ($F_0$, $V_0$, $P_{max}$).

#### How it Works
It uses **Least Squares Optimization** to fit modeled curves against observed field data.
1.  **Input:** Continuous 1080 Sprint data.
2.  **Optimization:** The algorithm adjusts $F_0$, $V_0$, and $\tau$ (tau) to minimize the error (RMSE) between the model and the real-world measurements.
3.  **Output:** A high-precision F-V profile.

#### Key Methods
* **`calculate_profile()`:** The main trigger. Fits the model to the provided splits and returns the optimal athlete parameters.


---
### 3. Physics Engine (`physics.py`)

A stateless utility module that centralizes all physical constants and environmental calculations to ensure consistency across the suite.

#### Features
* **Frontal area:** Calculates athlate's frontal area based on his height and weight.
* **Air Density:** Calculates air density based on temperature and barometric pressure.
* **Air Resistance:** Calculates drag based on barometric pressure, temperature, and athlete body frontal area.

---
### Dependencies

This core library requires:
* `pandas`
* `numpy`
* `scipy` (for optimization algorithms in Profilation)

---
### Installation & Usage

Import the classes directly into your scripts.

#### Example 1: Simulation (Forward)
```python
from sprint_simulator_core.simulator import SprintSimulation

# Initialize with known profile
sim = SprintSimulation(F0=8.0, V0=10.0, weight=83.0, height=1.85, distance=100)

# Run simulation
results = sim.top_speed()
print(f"Predicted Max Speed: {results['top_speed']:.2f} m/s")
```
#### Example 2: Profiling (Inverse)
```python
from sprint_simulator_core.profiler import SprintProfilation

# The 'splits' argument must be a pandas DataFrame 
# containing spatiotemporal data with columns 'time' (s) and 'speed' (m/s).

# Example structure:
splits = pd.DataFrame({
    'time': [0.0, 0.01, 0.02, 0.03, ...], 
    'speed': [0.0, 0.02, 0.05, 0.09, ...]
})

# Calculate profile
profiler = SprintProfilation(splits, weight=80.0, height=1.80)
profile = profiler.calculate_profile()

print(f"Calculated F0: {profile['F0']:.2f} N/kg")
print(f"Calculated V0: {profile['V0']:.2f} m/s")
```