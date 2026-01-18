# Sprint Science

This directory contains the core computational engine for the `Performance-Diagnostic-Suite`.

It is **not a runnable script** but a Python library intended to be imported by other applications. It consists of three interconnected modules that handle the biomechanics of sprinting:

1.  **`simulator.py` (Forward):** Predicts sprint performance from an athlete's F-V profile.
2.  **`profilation.py` (Inverse):** Calculates the F-V profile from raw 1080 Sprint data.
3.  **`step_analysis.py`:** Performs a gait analysis from raw 1080 Sprint data.
4.  **`physics.py`:** A shared physics engine handling air resistance, environmental conditions, and unit conversions.
5.  **`utilities.py`:** Auxiliary functions and supplementary routines.

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
### 2. Sprint Profiling (`profiler.py`)

The profiling logic solves the inverse problem: converting track performance data—whether continuous or discrete—into an athlete's physiological parameters ($F_0$, $V_0$, $P_{max}$, $D_{RF}$).

#### How it Works
The module implements a dual-mode modeling approach to fit modeled curves against observed field data:

1.  **Time-Speed Mode:** Designed for high-frequency data (e.g., 1080 Sprint, Radar). It uses least squares optimization to fit the velocity curve directly.
2.  **Split-Time Mode:** Designed for timing gates (e.g., 5, 10, 20m splits). It utilizes Morin's Macroscopic Model to reconstruct the acceleration phase from limited distance-time points.

In both cases, the algorithm iteratively adjusts $F_0$, $V_0$, and $\tau$ (tau) to minimize the error (RMSE) between the theoretical model and the real-world measurements.

---
### 3. Step Analysis (`step_analysis.py`)

The `StepAnalyzer` class performs a gait analysis by decomposing high-frequency 1080 Sprint data into individual steps using zero-crossing detection algorithms.

#### How it Works
It distinguishes between **Kinematics** (step frequency, length, duration) calculated from propulsive peaks, and **Kinetics** (forces, impulses) calculated from ground contact phases.

* **Step Detection:** Identifies "Foot-strike" (Negative force) events to segment the data into discrete cycles.
* **Impulse Calculation:** Integrates force over time ($\int F \,dt$) to quantify **Propulsive Impulse** (positive force) vs. **Braking Impulse** (negative force/impact).
* **Technical Efficiency:** Evaluates how much energy is lost to braking during specific phases (Acceleration vs. Max Velocity).

#### Key Methods

* **`analyze_steps()`:**
Returns a detailed DataFrame containing metrics for every individual step, including:
    * Left/Right leg identification.
    * Step Length & Frequency.
    * Net Impulse.
    * Braking-to-Propulsive Ratio.

* **`analyze_acc_technical_efficiency()`:**
Quantifies the propulsive vs. braking balance specifically during the initial acceleration phase (typically steps 2–5).

* **`analyze_maxv_technical_efficiency()`:**
Analyzes technical efficiency and force maintenance in the top-speed phase (defined as $>90\%$ $V_{max}$).

---
### 4. Physics Engine (`physics.py`)

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
from sprint_simulator_core.profiler import SprintSpeedTimeProfiler

# The 'splits' argument must be a pandas DataFrame 
# containing spatiotemporal data with columns 'time' (s) and 'speed' (m/s).

# Example structure:
spatiotemporal = pd.DataFrame({
    'time': [0.0, 0.01, 0.02, 0.03, ...], 
    'speed': [0.0, 0.02, 0.05, 0.09, ...],
    'force': [20.0, 20.5, 20.9, 21.0, ...]
})

# Calculate profile
profiler = SprintSpeedTimeProfiler(splits, weight=80.0, height=1.80)
profile = profiler.calculate_profile()

print(f"Calculated F0: {profile['F0']:.2f} N/kg")
print(f"Calculated V0: {profile['V0']:.2f} m/s")
```

#### Example 3: Step Analysis
```python
from sprint_science.step_analysis import StepAnalyzer

# Initialize with raw data and starting leg
analyzer = StepAnalyzer(
    starting_leg='Left', 
    raw_spatiometric_data=spatiotemporal, 
    height=1.80, 
    weight=80.0
)

# 1. Get full step-by-step report
step_report = analyzer.analyze_steps()
print(step_report[['step_number', 'leg', 'step_length', 'braking_propulsive_ratio']])

# 2. Analyze Technical Efficiency in Acceleration
prop_imp, brake_imp = analyzer.analyze_acc_technical_efficiency()
print(f"Acceleration Propulsive Impulse: {prop_imp:.2f} Ns")
print(f"Acceleration Braking Impulse: {brake_imp:.2f} Ns")
```