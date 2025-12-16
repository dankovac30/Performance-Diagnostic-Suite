from typing import Dict, Tuple, Any
import tkinter as tk
from tkinter import ttk, messagebox
from testing_standards.imtp_standard import IMTP_Calculations
from testing_standards.athlete import Athlete


class IMTP_App:
    """
    Main GUI Application for the IMTP Standardizer.
    
    This class handles the GUI (Tkinter), accepts user input,
    validates data, calls the backend solver, and displays the results.
    """
    def __init__(self, root: tk.Tk):
        """
        Initialize the main window and UI components.
        """
        self.root = root
        self.root.title("IMTP Standardizer")
        self.root.geometry("500x600")
        self.root.resizable(False, False)

        # Styling configuration
        style = ttk.Style()
        style.theme_use('clam')
        style.configure("TLabel", font=("Helvetica", 10))
        style.configure("TButton", font=("Helvetica", 11, "bold"))
        style.configure("Header.TLabel", font=("Helvetica", 16, "bold"), foreground="#333")
        style.configure("Result.TLabel", font=("Helvetica", 14), foreground="#000000")

        # Main layout container
        main_frame = ttk.Frame(root, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Input section
        input_frame = ttk.LabelFrame(main_frame, text="Athlete anthropometry (cm)", padding="15")
        input_frame.pack(fill=tk.X, pady=(0, 10))

        self.entries = {}
        
        # Define fields
        fields = [
            ("Foot length", "foot_length", "27"),
            ("Heel - Ankle", "heel_ankle_length", "5"),
            ("Ankle height", "ankle_height", "7"),
            ("Shin length", "shin_length", "45"),
            ("Thigh length", "thigh_length", "43"),
            ("Trunk length", "trunk_length", "58"),
            ("Arm length", "arm_length", "68")
        ]

        # Generate input rows dynamically
        for i, (label_text, key, default) in enumerate(fields):
            row_frame = ttk.Frame(input_frame)
            row_frame.pack(fill=tk.X, pady=4)
            
            lbl = ttk.Label(row_frame, text=label_text, width=20)
            lbl.pack(side=tk.LEFT)
            
            entry = ttk.Entry(row_frame)
            entry.insert(0, default)
            entry.pack(side=tk.RIGHT, expand=True, fill=tk.X)
            
            self.entries[key] = entry

        # Action button
        calc_btn = ttk.Button(main_frame, text="Calculate", command=self.calculate)
        calc_btn.pack(fill=tk.X, pady=(5, 15), ipady=8)
        
        # Output section
        self.result_frame = ttk.LabelFrame(main_frame, text="Result", padding="15")
        self.result_frame.pack(fill=tk.BOTH, expand=True)

        # Main result
        self.result_label_main = ttk.Label(self.result_frame, text="Enter data...", style="Result.TLabel", anchor="center")
        self.result_label_main.pack(pady=(10, 5))
        
        # Detailed metricks
        self.result_details = ttk.Label(self.result_frame, text="", justify=tk.CENTER)
        self.result_details.pack()


    def calculate(self) -> None:
        """
        Callback function triggered by the 'Calculate' button.
        Collects data, validates input, calls the solver, and updates UI.
        """
        try:
            # Data collection and validation
            data = {}
            for key, entry in self.entries.items():
                val_str = entry.get().replace(',', '.')
                
                if not val_str:
                    raise ValueError(f"Missing value for {key}")
                
                try:
                    val = float(val_str)
                
                except ValueError:
                     raise ValueError(f"Value '{val_str}' for {key} is not a number.")
                
                data[key] = float(val)

            # Create athlete model
            athlete = Athlete(
                foot_length=data['foot_length'],
                heel_ankle_length=data['heel_ankle_length'],
                ankle_height=data['ankle_height'],
                shin_length=data['shin_length'],
                thigh_length=data['thigh_length'],
                trunk_length=data['trunk_length'],
                arm_length=data['arm_length']
            )

            # Run solver
            imtp = IMTP_Calculations(athlete)
            best_result = imtp.find_best_rack_height()

            # Handle result
            if best_result:
                self.display_results(best_result)
            else:
                messagebox.showwarning("Result", "No valid solution was found for the given parameters.")

        except ValueError as e:
            messagebox.showerror("Input error", str(e))
        except Exception as e:
            messagebox.showerror("Error", f"Unexpected error:\n{e}")


    def display_results(self, result: Tuple[str, Dict[str, Any]]) -> None:
        """
        Updates the UI with the calculation results.
        """
        rack_id = result[0]
        data = result[1]

        # Determine display logic (usage of board to split iso rack holes)
        if "D" in rack_id:
            hole_num = rack_id.replace("D", "")
            main_text = f"Hole n. {hole_num}\n+ BOARD"
        else:
            main_text = f"Hole n. {rack_id}\n(Without board)"

        # Update main label
        self.result_label_main.config(text=main_text)

        details_text = (
            f"----------------------------\n"
            f"Knee: {data['knee_angle']:.1f}° (Target 135°)\n"
            f"Hip: {data['hip_angle']:.1f}° (Target 145°)\n"
            f"Thigh placement: {(data['l1']/float(self.entries['thigh_length'].get())*100):.0f} %\n"
        )
        self.result_details.config(text=details_text)

if __name__ == "__main__":
    root = tk.Tk()
    app = IMTP_App(root)
    root.mainloop()