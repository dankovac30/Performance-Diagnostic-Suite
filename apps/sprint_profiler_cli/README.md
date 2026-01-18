# F-V Sprint Profiler CLI

## About

This tool is the data-crunching engine of the **Performance-Diagnostic-Suite**. It is a robust Command Line Interface (CLI) application designed to automate the analysis of sprint split times.

Based on **J.B. Morin's macroscopic model**, it transforms raw timing gate data (Excel) into complete Force-Velocity biomechanical profiles. It handles batch processing, allowing coaches to analyze hundreds of athletes in seconds.

---

## Workflow

The application follows a "Template-First" philosophy to ensure data consistency.

1.  **Launch:** Run the app. If no input file is found, it automatically generates a template.
2.  **Input:** Fill the `kinematics_sprint_data.xlsx` with your athletes' split times.
3.  **Process:** Run the app again. It validates data, fits the models, and calculates physics.
4.  **Result:** Detailed biomechanical metrics are exported to `kinetics_results.xlsx`.

---

## Quick Start (Binary)

For end-users using the compiled version:

1.  Download the `Sprint-Profiler.exe`.
2.  Place it in a dedicated folder (e.g., `My_Sprint_Analysis`).
3.  **Run the .exe**.
    * *First run:* It will create `kinematics_sprint_data.xlsx` and close.
4.  Open the Excel file and paste your data.
5.  **Run the .exe again**.
    * It will generate `kinetics_results.xlsx` with all calculated metrics.

---

## Features

* **Smart Template System:** Automatically builds the required Excel structure if missing.
* **Dynamic Split Detection:** The app isn't hardcoded to specific distances. You can use any splits (e.g., 5m, 10m, 20m, 30m) just by naming the Excel columns accordingly.
* **Reaction Time Correction:** Automatically subtracts reaction time (e.g., 0.160s) from raw splits to align $t=0$ with movement onset.
* **Batch Processing:** Capable of processing hundreds of athletes in a single loop with error handling for individual lanes.
* **Scientific Validation:** Enforces a **minimum of 4 split segments** to ensure a valid regression fit, adhering to J.B. Morin's recommendations for data reliability.

---

## Data Structure

The input Excel file (`kinematics_sprint_data.xlsx`) requires specific columns.

### Fixed Columns (Mandatory)
| Column | Description |
| :--- | :--- |
| **Name** | Athlete identifier. |
| **Weight (kg)** | Body mass (essential for Force calc). |
| **Height (cm)** | Body height (for air resistance). |
| **Reaction Time (s)** | Time to subtract (e.g., 0.00 for touch-pad, 0.15 for gun). |
| **Temperature / Pressure** | Environmental factors for air density correction. |

### Dynamic Columns (Splits)
Any column with a **numeric header** is treated as a distance marker.
* `5` -> Time at 5m
* `10` -> Time at 10m
* `20` -> Time at 20m
* ...

---

## Installation & Usage (from Source)

For developers running the raw Python script.

1.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/dankovac30/Performance-Diagnostic-Suite.git](https://github.com/dankovac30/Performance-Diagnostic-Suite.git)
    ```

2.  **Install Dependencies:**
    ```bash
    pip install -r sprint_simulator_gui/requirements.txt
    ```

3.  **Run the Application:**
    Navigate to the project root and run:
    ```bash
    python -m sprint_science.profiler_app
    ```

---

## Tech Stack

* **Python 3**
* **Pandas & OpenPyXL** for Excel I/O and data manipulation.
* **SciPy (Optimize & Stats)** for non-linear curve fitting ($d(t)$ model) and linear regression.
* **Colorama/ANSI** for the colored terminal interface.