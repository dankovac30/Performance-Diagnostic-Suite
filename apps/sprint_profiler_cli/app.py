import os
import sys
import time

import numpy as np
import pandas as pd

from sprint_science.profiler import SprintSplitTimeProfiler


class SprintProfilerApp:
    """
    Command Line Interface (CLI) application for processing Sprint Profiles.
    Based on J.B. Morin's macroscopic Force-Velocity model.

    Features:
    - Auto-detection of 'frozen' state (PyInstaller compatibility).
    - Dynamic detection of distance columns (works with any set of distances like 5, 10, 20...).
    - Automatic template generation if input file is missing.
    - Reaction time correction.
    """

    def __init__(self):
        """
        Initializes the application context, paths, and terminal settings.
        Detects if running as a script or a compiled executable.
        """
        # Check if the application is "frozen" (bundled by PyInstaller)
        if getattr(sys, "frozen", False):
            self.script_dir = os.path.dirname(sys.executable)
        else:
            self.script_dir = os.path.dirname(os.path.abspath(__file__))

        # Path Configuration
        self.input_name = "kinematics_sprint_data.xlsx"
        self.output_name = "kinetics_results.xlsx"
        self.input_file = os.path.join(self.script_dir, self.input_name)
        self.output_file = os.path.join(self.script_dir, self.output_name)

        # Attempt to get terminal width for pretty printing
        try:
            self.terminal_width = os.get_terminal_size().columns
        except Exception:
            self.terminal_width = 70

    def fill_na(self, value: any, default: any) -> any:
        """
        Utility to handle NaN or empty string values safely.

        Args:
            value (Any): The value to check.
            default (Any): The fallback value if 'value' is empty/NaN.

        Returns:
            Any: The original value or the default.
        """
        if pd.isna(value) or str(value).strip() == "":
            return default
        return value

    def create_dummy(self) -> pd.DataFrame:
        """
        Creates a template DataFrame with dummy data structure.
        Used to generate the input Excel file for first-time users.

        Returns:
            pd.DataFrame: A dataframe with standard columns and one example row.
        """
        # Create a dictionary structure for the template
        dummy = {
            "Name": ["Dan Kovac"],
            "Weight (kg)": [84],
            "Height (cm)": [186],
            "Temperature (°C)": [20],
            "Pressure (hpa)": [1013.25],
            "Reaction Time (s)": [0.160],
            5: [1.40],
            10: [2.23],
            15: [2.98],
            20: [3.64],
            25: [4.29],
            30: [4.99],
        }

        return pd.DataFrame(dummy)

    def print_header(self) -> None:
        """Prints the stylized ASCII header of the application."""
        w = self.terminal_width

        print("\033[96m" + "═" * w + "\033[0m")
        print(f"\033[1m{'F-V SPRINT PROFILER v1.0':^{w}}\033[0m")
        print(f"{'Based on JB Morin Spreadsheet':^{w}}")
        print("\033[96m" + "═" * w + "\033[0m")

    def first_use(self) -> None:
        """
        Handles the first-run scenario where the input file is missing.
        Generates the template and prints instructions.
        """
        w = self.terminal_width

        print("\n[\033[93m!\033[0m] Input file 'kinematics_sprint_data.xlsx' was not found.")
        time.sleep(0.6)
        print("    Building a fresh template for you right now...")

        df_template = self.create_dummy()
        df_template.to_excel(self.input_file, index=False)
        time.sleep(1.2)

        print("\n" + "─" * w)
        time.sleep(0.2)
        print("\033[1mQUICK START GUIDE:\033[0m")
        time.sleep(0.2)
        print("\033[92mDistances:\033[0m Add or modify any numeric columns (e.g., 5, 12, 17.5).")
        time.sleep(0.2)
        print("\033[92mFiltering:\033[0m Script auto-detects the acceleration phase and cuts off the rest.")
        time.sleep(0.2)
        print("\033[92mReaction:\033[0m  Times are auto-corrected via 'Reaction Time (s)'.")
        print("─" * w)
        time.sleep(0.5)

        print("\n\033[90mDeveloped by Dan Kovac\033[0m")
        print("\033[94mEmail:\033[0m  dankovac30@gmail.com")
        print("\033[94mGitHub:\033[0m github.com/dankovac30")

        time.sleep(0.5)
        print(f"\n\033[92m>>> TEMPLATE CREATED AT:\033[0m\n{self.input_file}")
        print("\n" + "=" * w)
        input("Press Enter to exit, edit the file and run again...")

    def save_excel(self, df: pd.DataFrame, file_path: str, file_name: str) -> None:
        """
        Attempts to save the DataFrame to Excel with a retry mechanism for PermissionError.

        Args:
            df (pd.DataFrame): Data to save.
            file_path (str): Full path to the file.
            file_name (str): Name of the file for display purposes.
        """
        while True:
            try:
                df.to_excel(file_path, index=False)
                break
            except PermissionError:
                print(f"\n\033[91mERROR SAVING: File kinetics_{file_name} is opened in Excel.\033[0m")
                input("\033[91mClose it and press Enter to retry saving...\033[0m")

    def handle_calculations(self) -> None:
        """
        Main execution function for the F-V Sprint Profiler CLI application.

        Steps:
        1. Detects the execution environment and sets file paths.
        2. Checks for the input Excel file ('kinematics_sprint_data.xlsx').
            - If missing, generates a template file with dummy data.
        3. Loads the Excel file and detects dynamic numeric columns (distances).
        4. Validates input data (minimum 4 segments required).
        5. Iterates through each athlete (row):
            - Adjusts times by subtracting reaction time.
            - Initializes the SprintSplitTimeProfiler.
            - Calculates the Force-Velocity profile.
        6. Aggregates results and saves them to 'kinetics_results.xlsx'.
        """

        # Data loading
        print(f"\nLoading data from {self.input_name}\n")
        time.sleep(1.2)
        df = pd.read_excel(self.input_file)

        # Check for empty file
        if df.empty:
            print("\nExcel is empty")
            print("Creating new template")
            dummy = self.create_dummy()
            self.save_excel(dummy, self.input_file, self.input_name)
            print("=" * self.terminal_width)
            input("Press Enter to exit, edit the file and run again...")
            return

        # Column validation
        fixed_columns = [
            "Name",
            "Weight (kg)",
            "Height (cm)",
            "Temperature (°C)",
            "Pressure (hpa)",
            "Reaction Time (s)",
        ]
        missing_cols = [col for col in fixed_columns if col not in df.columns]

        if missing_cols:
            print(f"\033[91mError: Missing excel columns: {missing_cols}\033[0m")
            print("=" * self.terminal_width)
            input("Press Enter for exit...")
            return

        # Dynamic distance detection
        distance_cols = []
        for c in df.columns:
            col_str = str(c)
            # Remove one decimal point to check if the rest are digits
            col_str_no_dot = col_str.replace(".", "", 1)

            if col_str_no_dot.isdigit():
                distance_cols.append(c)

        # Sort distances to ensure correct cumulative order
        distance_cols = sorted(distance_cols, key=float)
        distances = np.array(distance_cols, dtype=float)

        # Constraint Check: Morin recommends at least 4 points for validity
        if len(distances) < 4:
            print(f"\033[91mError: Include at least 4 segments, currently {len(distances)}\033[0m")
            print("=" * self.terminal_width)
            input("Press Enter for exit...")
            return

        print(f"Processing {len(df)} athletes...\n")
        time.sleep(0.8)

        # Main processing loop
        results_list = []
        errors = 0

        print("─" * self.terminal_width)

        for index, row in df.iterrows():
            try:
                # Extract split times
                row_times = row[distance_cols].to_numpy(dtype=float)

                # Subtract reaction time to align t=0 with movement onset
                rt = self.fill_na(row["Reaction Time (s)"], 0)
                adjusted_times = row_times - rt

                # Basic data integrity check
                if pd.isna(row["Weight (kg)"]) or pd.isna(row["Height (cm)"]):
                    print(f"\033[91mError in lane {index + 2}: Missing Weight or Height\033[0m")
                    errors += 1
                    continue

                # Calculation core
                profiler = SprintSplitTimeProfiler(
                    distances=distances,
                    times=adjusted_times,
                    weight=row["Weight (kg)"],
                    height=row["Height (cm)"],
                    temperature_c=self.fill_na(row["Temperature (°C)"], 20),
                    barometric_pressure_hpa=self.fill_na(row["Pressure (hpa)"], 1013.25),
                )

                # Compute F-V profile
                report = profiler.calculate_profile()

                name = self.fill_na(row["Name"], f"Lane {index + 2}")

                # Map results to output dictionary
                output_row = {
                    "Name": name,
                    "F0 (N/kg)": report["F0"],
                    "V0 (m/s)": report["V0"],
                    "P max (W/kg)": report["Pmax"],
                    "Rf max (%)": report["Rf_max"],
                    "DRF": report["DRF"],
                    "F-V slope": -report["F_V_slope"],
                    "Sum Square diff": report["Sum Square diff"],
                    "Tau": report["Tau"],
                    "V_max": report["V_max"],
                }

                results_list.append(output_row)

                # Inform user how many segments were used
                print(f"Lane {index + 2} analyzed segments: {len(report['distances'])} of {len(adjusted_times)}")

            except Exception as e:
                print(f"\033[91mError in lane {index + 2}: {e}\033[0m")
                errors += 1
                continue

        # Export results
        print("─" * self.terminal_width)
        print("\nSaving files to: kinetics_results.xlsx")
        time.sleep(1.6)
        self.save_excel(pd.DataFrame(results_list), self.output_file, self.output_name)

        # Final summary
        if errors == 0:
            print("\nCOMPLETED! Processing finished successfully.")
        else:
            print(f"\nProcessing finished with {errors} errors.")

        print("=" * self.terminal_width)
        input("Press Enter for exit...")
        return

    def run_app(self):
        """
        Bootstrap function to start the application.
        Checks for file existence and routes to first-use or calculation logic.
        """
        self.print_header()
        time.sleep(0.5)

        # File existence check & template generation
        if not os.path.exists(self.input_file):
            self.first_use()
            return

        self.handle_calculations()


def main():
    """Entry point of the script."""
    app = SprintProfilerApp()
    app.run_app()


if __name__ == "__main__":
    main()
