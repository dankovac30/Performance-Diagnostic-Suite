# Performance-Diagnostic-Suite

A monorepo of tools for athletic performance analysis and diagnostics. Built in Python, this suite provides tools for simulation, data management, and methodology standardization.

The suite is built around a central physics engine (`sprint_simulator_core`) and provides different applications for coaches, athletes, and researchers.

---

### Available Tools

Here is an overview of the projects included in this suite. More detailed information can be found in the `README.md` file within each project's folder.

#### Core
* **Status:** `Active`
* **About:** The foundational library of the suite. Contains shared low-level logic, signal processing algorithms (smoothing, filtering).

#### Vendors
* **Status:** `Active`
* **About:** The hardware integration layer. Contains device-specific logic for parsing, cleaning, and normalizing data from various diagnostic tools.

#### Sprint Science
* **Status:** `Released`
* **About:** The computational core powering all sprint tools. Python library containing the **SprintProfilation** class for calculating Force-Velocity parameters (F0, V0) from raw timing data, **SprintSimulation** class for physics modeling and **StepAnalyzer** for gait analysis.

#### Testing Standards
* **Status:** `Released`
* **About:** A biomechanical tools that standardize diverse performance tests by using inverse kinematics to generate athlete-specific setups and positions based on anthropometry, ensuring consistent and reliable data.

#### Performance Hub
* **Status:** `Under development`
* **About:** The future data-management center of the suite.

#### Applications
* **Scope:** End-User Tools & Interfaces
* **About:** This directory houses standalone applications derived from the suite. It contains independent executable implementations to provide practical access to the suite's scientific algorithms.

### Roadmap

* `[ ]` Implement the `performance_hub` database and report generation.
* `[*]` Finalize the `testing_standards` IMTP script.

---
### Installation & Usage

1.  Clone the repository:
    ```bash
    git clone
    ```

2.  Install dependencies for the specific app (e.g., the GUI):
    ```bash
    pip install -r sprint_simulator_gui/requirements.txt
    ```

3.  Run the application as a module from the **root** folder:
    ```bash
    # Run the Sprint Simulator GUI
    python -m sprint_simulator_gui.app 
    ```