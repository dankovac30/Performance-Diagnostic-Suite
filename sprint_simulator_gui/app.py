import tkinter as tk
from tkinter import ttk, messagebox, font
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sprint_simulator_gui import app_logic


class SprintSimulatorApp:
    """
    The main GUI application class for the Sprint F-V Simulator.
    
    This class handles the user interface creation using Tkinter, manages user inputs,
    visualizes results using Matplotlib, and drives the interaction between
    the user and the simulation logic (app_logic).
    """    
    def __init__(self, root: tk.Tk):
        """
        Initialize the main application window and UI components.
        """
        # UI colours
        background_colour = '#FAFAFA'
        border_colour = "#C0C0C0"

        # Root window configuration
        self.root = root
        self.root.title('Sprint F-V Simulator')
        self.root.configure(bg=background_colour)
        
        # Styling configuration
        self.style = ttk.Style()
        self.style.theme_use('clam')

        # Detect system font
        default_font = font.nametofont("TkDefaultFont")
        self.font_family = default_font.actual()["family"]
        self.base_font = (self.font_family, 10)
        self.heading_font = (self.font_family, 11, "bold")

        # Define custom styles for frames, labels, buttons, and entries
        self.style.configure('.', background=background_colour, foreground='black', font=self.base_font)
        self.style.configure('TFrame', background=background_colour)
        self.style.configure('TLabel', background=background_colour)
        self.style.configure('TRadiobutton', background=background_colour)
        self.style.configure('TLabelframe', background=background_colour, bordercolor=border_colour, relief='solid', borderwidth=1)
        self.style.configure('TLabelframe.Label', background=background_colour, font=self.heading_font)
        
        # Interactive hover button
        self.style.configure('TButton', font=self.base_font, background=background_colour, foreground='black', bordercolor=border_colour, relief='solid', borderwidth=0.5, padding=(10, 5), focuscolor=background_colour)
        self.style.map('TButton', background=[('active', '#E0E0E0'), ('hover', '#F0F0F0')], relief=[('pressed', 'solid'), ('hover', 'solid')])
        self.style.configure('TEntry', font=self.base_font, fieldbackground='#FFFFFF', bordercolor=border_colour, relief='solid', borderwidth=1, padding=(5, 5))
        self.style.map('TEntry', bordercolor=[('focus', '#0078D7')], relief=[('focus', 'solid')])
        self.style.configure('TRadiobutton', focuscolor=background_colour)
        
        # Main layout container
        main_frame = ttk.Frame(self.root, padding="10 10 10 10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Configure grid weight for responsive resizing
        self.root.columnconfigure(0, weight = 1)
        self.root.rowconfigure(0, weight = 1)

        # List to store multiple simulation runs for comparison on the graph
        self.simulation_reports = []
        
        # Variable to toggle between Speed/Distance and Time/Distance views
        self.graph_type_var = tk.StringVar(value="speed")

        self.graph_bg_color = background_colour

        # Widget creation
        self._create_input_widgets(main_frame)
        self._create_output_widgets(main_frame)
        self._create_control_widgets(main_frame)
        self._create_graph_widget(main_frame)

        # Grid configuration for the main area
        main_frame.columnconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(2, weight=1)


    def _create_input_widgets(self, parent_frame: ttk.Frame):
        """Creates the input form for athlete parameters (F0, V0, Height, Weight)."""
        input_frame = ttk.LabelFrame(parent_frame, text='Input parameters', padding=10)
        input_frame.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")

        # F0 input
        ttk.Label(input_frame, text="F0 (N/kg):").grid(row=0, column=0, sticky="w", pady=3, padx=5)
        self.entry_f0 = ttk.Entry(input_frame, width=12)
        self.entry_f0.grid(row=0, column=1, sticky="w", pady=3, padx=5)
        self.entry_f0.insert(0, "8.0")

        # V0 input
        ttk.Label(input_frame, text="V0 (m/s):").grid(row=1, column=0, sticky="w", pady=3, padx=5)
        self.entry_v0 = ttk.Entry(input_frame, width=12)
        self.entry_v0.grid(row=1, column=1, sticky="w", pady=3, padx=5)
        self.entry_v0.insert(0, "10.0")

        # Height Input
        ttk.Label(input_frame, text="Height (m):").grid(row=2, column=0, sticky="w", pady=3, padx=5)
        self.entry_height = ttk.Entry(input_frame, width=12)
        self.entry_height.grid(row=2, column=1, sticky="w", pady=3, padx=5)
        self.entry_height.insert(0, "1.85")

        # Weight Input
        ttk.Label(input_frame, text="Weight (kg):").grid(row=3, column=0, sticky="w", pady=3, padx=5)
        self.entry_weight = ttk.Entry(input_frame, width=12)
        self.entry_weight.grid(row=3, column=1, sticky="w", pady=3, padx=5)
        self.entry_weight.insert(0, "83.0")


    def _create_output_widgets(self, parent_frame: ttk.Frame):
        """Creates the dashboard for displaying simulation results (splits, max speed, etc.)."""
        output_frame = ttk.LabelFrame(parent_frame, text="Simulation results", padding="10")
        output_frame.grid(row=0, column=1, padx=5, pady=5, sticky="nsew")

        self.output_labels = {}
        
        # Metrics to display
        output_definitions = {
            'running_time_100m': "100m Time:",
            'top_speed': "Max. Speed:",
            'top_speed_distance': "Dist. of Max. Speed:",            
            'time_30m': "30m Time:",
            'time_30m_fly': "Flying 30m time:",
            'fly_start': "Flying 30m (start):",
            'fly_finish': "Flying 30m (finish):"

        }

        # Dynamically generate labels
        row = 0
        for key, text in output_definitions.items():
            ttk.Label(output_frame, text=text).grid(row=row, column=0, sticky="w", pady=3, padx=5)
            value_label = ttk.Label(output_frame, text="--", width=12, font=(self.base_font[0], self.base_font[1], "bold"))
            value_label.grid(row=row, column=1, sticky="w", pady=3, padx=5)
            self.output_labels[key] = value_label
            row += 1


    def _create_control_widgets(self, parent_frame: ttk.Frame):
        """Creates action buttons (Calculate, Clear) and graph view controls."""
        # Action buttons
        action_frame = ttk.LabelFrame(parent_frame, text="Actions", padding="10")
        action_frame.grid(row=1, column=0, padx=5, pady=5, sticky="nsew")

        self.btn_calculate = ttk.Button(action_frame, text="Calculate & Add to Graph", command=self._handle_calculate)
        self.btn_calculate.grid(row=0, column=0, sticky="ew", padx=5, pady=2)

        self.btn_clear = ttk.Button(action_frame, text="Clear Graph", command=self._handle_clear_graph)
        self.btn_clear.grid(row=1, column=0, sticky="ew", padx=5, pady=2)

        action_frame.columnconfigure(0, weight=1)

        # Graph settings
        graph_options_frame = ttk.LabelFrame(parent_frame, text="Graph View", padding="10")
        graph_options_frame.grid(row=1, column=1, padx=5, pady=5, sticky="nsew")

        rb_speed = ttk.Radiobutton(graph_options_frame, text="Speed / Distance", variable=self.graph_type_var, value="speed", command=self._draw_graph)
        rb_speed.grid(row=0, column=0, sticky="w", padx=5)

        rb_time = ttk.Radiobutton(graph_options_frame, text="Time / Distance", variable=self.graph_type_var, value="time", command=self._draw_graph)
        rb_time.grid(row=1, column=0, sticky="w", padx=5)


    def _create_graph_widget(self, parent_frame: ttk.Frame):
        """Embeds a Matplotlib figure into the Tkinter window using FigureCanvasTkAgg."""
        graph_frame = ttk.LabelFrame(parent_frame, text="Graph Analysis", padding="5")
        graph_frame.grid(row=2, column=0, columnspan=2, padx=5, pady=5, sticky="nsew")

        graph_frame.columnconfigure(0, weight=1)
        graph_frame.rowconfigure(0, weight=1)

        # Create Figure with custom styling to match the GUI
        self.fig = Figure(figsize=(8, 4), dpi=100, facecolor=self.graph_bg_color)
        self.fig.subplots_adjust(left=0.08, bottom=0.13, right=0.95, top=0.9)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_facecolor(self.graph_bg_color)

        # Embed plot
        self.canvas = FigureCanvasTkAgg(self.fig, master=graph_frame)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

        self._draw_graph()


    def _handle_calculate(self):
        """
        Triggered when 'Calculate' is clicked.
        1. Retrieves data from inputs.
        2. Calls the business logic (app_logic) to run the simulation.
        3. Updates the results dashboard.
        4. Appends the simulation report to the list for plotting.
        """
        try:
            # Data retrieval and validation
            F0 = self.entry_f0.get()
            V0 = self.entry_v0.get()
            weight = self.entry_weight.get()
            height = self.entry_height.get()

            # Get calculations from logic module
            results, report = app_logic.run_simulation_logic(F0, V0, weight, height)

            # Update UI labels
            if results:
                self.output_labels['time_30m'].config(text=f"{results['time_30m']:.2f} s")
                self.output_labels['running_time_100m'].config(text=f"{results['running_time_100m']:.2f} s")
                self.output_labels['top_speed'].config(text=f"{results['top_speed']:.2f} m/s")
                self.output_labels['top_speed_distance'].config(text=f"{results['top_speed_distance']:.0f} m")
                self.output_labels['time_30m_fly'].config(text=results['time_30m_fly'])
                self.output_labels['fly_start'].config(text=results['fly_start'])
                self.output_labels['fly_finish'].config(text=results['fly_finish'])

            # Create a label for the legend
            label = f"F0: {float(F0.replace(',', '.')):.2f}, V0: {float(V0.replace(',', '.')):.2f}, H: {float(height.replace(',', '.')):.2f}, W: {float(weight.replace(',', '.')):.1f} "
            report.name = label

            # Store report for multi-plot comparison
            self.simulation_reports.append(report)

            # Update graph
            self._draw_graph()

        except ValueError as e:
            messagebox.showerror("Input Error", str(e))
        except Exception as e:
            messagebox.showerror("Simulation Error", str(e))

    
    def _draw_graph(self):
        """Redraws the graph based on the stored simulation reports and selected view mode."""
        self.ax.clear()
        self.ax.set_facecolor(self.graph_bg_color)

        graph_type = self.graph_type_var.get()

        if self.simulation_reports:
            # Plot all stored reports to allow comparison
            for report in self.simulation_reports:
                
                if graph_type == "speed":
                    self.ax.plot(report['distance'], report['speed'], label=report.name)
                
                elif graph_type == "time":
                    self.ax.plot(report['distance'], report['time'], label=report.name)
            
            self.ax.legend(prop={'family': self.font_family, 'size': 9}, loc='lower right')

        # Configure Axes based on view type
        if graph_type == "speed":
            self.ax.set_title("Speed vs. Distance", fontfamily=self.font_family, fontsize=11)
            self.ax.set_ylabel("Speed (m/s)", fontfamily=self.font_family, fontsize=10)
            self.ax.set_ylim(0, 14)
        
        elif graph_type == "time":
            self.ax.set_title("Time vs. Distance", fontfamily=self.font_family, fontsize=11)
            self.ax.set_ylabel("Time (s)", fontfamily=self.font_family, fontsize=10)
            self.ax.set_ylim(0, 14)

        self.ax.set_xlabel("Distance (m)", fontfamily=self.font_family, fontsize=10)
        self.ax.set_xlim(0, 100)
        self.ax.grid(True)

        for label in (self.ax.get_xticklabels() + self.ax.get_yticklabels()):
            label.set_fontfamily(self.font_family)
            label.set_fontsize(10)

        self.canvas.draw()


    def _handle_clear_graph(self):
        """Clears all previous simulation runs and resets the plot."""
        self.simulation_reports.clear()
        self._draw_graph()


def main():
    """Entry point for the application."""
    try:
        root = tk.Tk()
        app = SprintSimulatorApp(root)
        root.mainloop()
    
    except ImportError as e:
        print("Import Error. Make sure you are running this as a module from the root project folder.")
        print(f"Error detail: {e}")
        print("SRun using: python -m sprint_simulator_gui.app")            


if __name__ == "__main__":
    main()
