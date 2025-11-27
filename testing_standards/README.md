# 🔬 Testing Standards

### About

This directory contains tools designed to standardize biomechanical testing protocols across different athletes and equipment.

The main component currently is the **IMTP Standardizer**, a specialized software tool for standardizing **Isometric Mid-Thigh Pull (IMTP)** testing.

IMTP performance is highly sensitive to body positioning. Traditional setup methods (visual estimation or goniometers) often lead to inconsistent knee and hip angles, reducing test reliability. This tool solves the problem by using a **mathematical model** to calculate the exact rack setting for every athlete based on their anthropometry.

---
### How it Works

The solution consists of a biomechanical computation engine (`imtp_standard.py`) and a graphical user interface (`app.py`).

1.  **Anthropometric Input:** The user enters specific body measurements (foot, shin, thigh, trunk, arm lengths).

2.  **Inverse Kinematics Solver:** The core class `IMTP_Calculations` uses a nested binary search algorithm to simulate thousands of potential body positions within the specific geometry of the ISO rack. A sketch of the setup is included in the screenshots folder.

3.  **Optimization Logic:** The algorithm searches for a rack height that satisfies multiple constraints:
    * **Knee Angle:** Targets ~135°.
    * **Hip Angle:** Targets ~145°.
    * **Shoulder Position:** Must be vertically aligned above the bar.
    * **Anatomical Limits:** Respects physical reach and muscle insertion points.

4.  **Hardware Mapping:** The tool maps the optimal theoretical position to the nearest available setting on a specific ISO rack (defined in JSON).

---
### Key Components

* **`IMTP_Calculations` Class:** The mathematical core. It contains the `find_best_rack_height()` method which executes the brute-force optimization to find the best biomechanical compromise for a given athlete.
* **`Athlete` Class:** A data structure that holds the anthropometric measurements and calculates derived body segment parameters.
* **`app.py`:** A user-friendly Tkinter application that allows coaches to input data and immediately see the result (e.g., "Hole 8") without interacting with the code.
* **`iso_rack.json`:** A configuration file defining the exact heights of the specific rack used in the lab.

---
### Dependencies

This tool requires:
* `tkinter` (Standard Python library)
* `math`, `json`, `os` (Standard Python libraries)

*No external dependencies are needed.*

---
### Usage

This tool is designed to be used as a standalone desktop application.

#### Running the GUI:

To start the application for daily use in the gym/lab:

```bash
python app.py