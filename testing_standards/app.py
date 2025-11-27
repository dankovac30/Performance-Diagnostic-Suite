import tkinter as tk
from tkinter import ttk, messagebox
from testing_standards.imtp_standard import IMTP_Calculations
from testing_standards.athlete import Athlete


class IMTP_App:

    def __init__(self, root):
        self.root = root
        self.root.title("IMTP Standardizer")
        self.root.geometry("500x600")
        self.root.resizable(False, False)

        style = ttk.Style()
        style.theme_use('clam')
        style.configure("TLabel", font=("Helvetica", 10))
        style.configure("TButton", font=("Helvetica", 11, "bold"))
        style.configure("Header.TLabel", font=("Helvetica", 16, "bold"), foreground="#333")
        style.configure("Result.TLabel", font=("Helvetica", 14), foreground="#000000")

        main_frame = ttk.Frame(root, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)

        input_frame = ttk.LabelFrame(main_frame, text="Antropometrie atleta (cm)", padding="15")
        input_frame.pack(fill=tk.X, pady=(0, 10))

        self.entries = {}
        fields = [
            ("Délka chodidla", "foot_length", "27"),
            ("Pata - Kotník", "heel_ankle_length", "5"),
            ("Výška kotníku", "ankle_height", "7"),
            ("Délka bérce", "shin_length", "45"),
            ("Délka stehna", "thigh_length", "43"),
            ("Délka trupu", "trunk_length", "58"),
            ("Délka paže", "arm_length", "68")
        ]

        for i, (label_text, key, default) in enumerate(fields):
            row_frame = ttk.Frame(input_frame)
            row_frame.pack(fill=tk.X, pady=4)
            
            lbl = ttk.Label(row_frame, text=label_text, width=20)
            lbl.pack(side=tk.LEFT)
            
            entry = ttk.Entry(row_frame)
            entry.insert(0, default)
            entry.pack(side=tk.RIGHT, expand=True, fill=tk.X)
            
            self.entries[key] = entry

        calc_btn = ttk.Button(main_frame, text="Vypočítat Nastavení", command=self.calculate)
        calc_btn.pack(fill=tk.X, pady=(5, 15), ipady=8)
        
        self.result_frame = ttk.LabelFrame(main_frame, text="Výsledek", padding="15")
        self.result_frame.pack(fill=tk.BOTH, expand=True)

        self.result_label_main = ttk.Label(self.result_frame, text="Zadejte data...", style="Result.TLabel", anchor="center")
        self.result_label_main.pack(pady=(10, 5))
        
        self.result_details = ttk.Label(self.result_frame, text="", justify=tk.CENTER)
        self.result_details.pack()


    def calculate(self):
        try:
            data = {}
            for key, entry in self.entries.items():
                val = entry.get().replace(',', '.')
                if not val:
                    raise ValueError(f"Chybí hodnota pro {key}")
                data[key] = float(val)

            athlete = Athlete(
                foot_length=data['foot_length'],
                heel_ankle_length=data['heel_ankle_length'],
                ankle_height=data['ankle_height'],
                shin_length=data['shin_length'],
                thigh_length=data['thigh_length'],
                trunk_length=data['trunk_length'],
                arm_length=data['arm_length']
            )

            imtp = IMTP_Calculations(athlete)
            
            best_result = imtp.find_best_rack_height()

            if best_result:
                self.display_results(best_result)
            else:
                messagebox.showwarning("Výsledek", "Pro zadané parametry nebylo nalezeno validní řešení.")

        except ValueError as e:
            messagebox.showerror("Chyba vstupu", str(e))
        except Exception as e:
            messagebox.showerror("Chyba", f"Neočekávaná chyba:\n{e}")


    def display_results(self, result):
        rack_id = result[0]
        data = result[1]

        if "D" in rack_id:
            hole_num = rack_id.replace("D", "")
            main_text = f"Díra č. {hole_num}\n+ DESKA"
        else:
            main_text = f"Díra č. {rack_id}\n(Bez desky)"

        self.result_label_main.config(text=main_text)

        details_text = (
            f"----------------------------\n"
            f"Koleno: {data['knee_angle']:.1f}° (Target 135°)\n"
            f"Kyčel: {data['hip_angle']:.1f}° (Target 145°)\n"
            f"Pozice na stehně: {(data['l1']/float(self.entries['thigh_length'].get())*100):.0f} %\n"
        )
        self.result_details.config(text=details_text)

if __name__ == "__main__":
    root = tk.Tk()
    app = IMTP_App(root)
    root.mainloop()