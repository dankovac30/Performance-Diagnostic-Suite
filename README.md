# Performance-Diagnostic-Suite

### About

A monorepo of tools for athletic performance analysis and diagnostics. Built in Python, this suite provides tools for simulation, data management, and methodology standardization.

The suite is built around a central physics engine (`sprint_simulator_core`) and provides different applications for coaches, athletes, and researchers.

### Available Tools

Here is an overview of the projects included in this suite. More detailed information can be found in the `README.md` file within each project's folder.


#### ⚙️ Sprint Science
* **Status:** `Released`
* **About:** The computational core powering all sprint tools. Python library containing the **SprintProfilation** class for calculating Force-Velocity parameters (F0, V0) from raw timing data and **SprintSimulation** class for physics modeling.

#### 🏁 Sprint Simulator GUI
* **Status:** `Released`
* **About:** The main desktop application of the suite. Built with Tkinter, it enables coaches to perform "what-if" analyses of sprint performance based on an athlete’s Force–Velocity profile (F₀, V₀) and anthropometric parameters.

#### 🔬 Testing Standards
* **Status:** `Released`
* **About:** A biomechanical tools that standardize diverse performance tests by using inverse kinematics to generate athlete-specific setups and positions based on anthropometry, ensuring consistent and reliable data.

#### 🗃️ Performance Hub
* **Status:** `Under development`
* **About:** The future data-management center of the suite.

#### 🧪 `dev/`
* **Status:** `Internal`
* **About:** Development and testing folder.

### Roadmap

* `[ ]` Implement the `performance_hub` database and report generation.
* `[*]` Finalize the `testing_standards` IMTP script.

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