# Applications


This directory contains the standalone executables derived from the project's scientific core. Each application serves a specific purpose in the performance diagnostic workflow, from raw data processing to predictive modelling.

---

### Sprint Simulator GUI
* **Status:** `Released`
* **About:** Built with Tkinter, it enables coaches to perform "what-if" analyses of sprint performance based on an athlete’s Force–Velocity profile (F₀, V₀) and anthropometric parameters.
---

### Sprint Profiler CLI
* **Status:** `Released`
* **About:** A robust command-line tool for batch processing of raw timing data. It parses variable sprint segments from Excel (e.g., 5m, 10m, 20m), applies the macroscopic mono-exponential correction (Morin et al.), and automatically calculates the complete Force-Velocity profile (F₀, V₀, τ, Pmax) needed for the simulation phase.

