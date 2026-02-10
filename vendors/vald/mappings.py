VALD_CMJ_METRICS = {
    # Antropometry
    655386: "Body Weight [kg]",
    # Kinematics
    6553607: "Jump Height (Flight Time) [cm]",
    6553611: "Jump Height (Imp-Mom) [cm]",
    6553610: "Jump Height (Imp-Dis) [cm]",
    6553603: "Countermovement Depth [cm]",
    6553655: "Takeoff Velocity (Imp-Mom) [m/s]",
    6553701: "Eccentric Peak Velocity [m/s]",
    6553700: "Velocity at Peak Power [m/s]",
    # Timing
    6553606: "Flight Time [s]",
    6553643: "Contraction Time [s]",
    6553644: "Eccentric Duration [s]",
    6553664: "Eccentric Braking Duration [s]",
    6553657: "Concentric Duration [s]",
    6553646: "Time to Peak Power [s]",
    # Kinetics
    6553685: "Concentric Peak Force [N]",  # LR
    6553687: "Eccentric Peak Force [N]",  # LR
    6553713: "Force at Zero Velocity [N]",  # LR
    6553663: "Min Eccentric Force [N]",
    6553673: "Force at Peak Power [N]",  # LR
    # Mean forces
    6553619: "Concentric Mean Force [N]",  # LR
    6553620: "Eccentric Mean Force [N]",  # LR
    6553671: "Eccentric Mean Braking Force [N]",
    # Power and Work
    6553633: "Peak Power [W]",
    6553604: "Peak Power [W/kg]",
    6553702: "Eccentric Peak Power [W]",
    6553623: "Concentric Mean Power [W]",
    6553621: "Eccentric Mean Power [W]",
    6553699: "Total Work [J]",
    # Impulse
    6553712: "Concentric Impulse [N·s]",  # LR
    6553716: "Eccentric Unloading Impulse [N·s]",  # LR
    6553704: "Eccentric Braking Impulse [N·s]",  # LR
    # Others
    6553682: "Eccentric Deceleration RFD [N/s]",
    6553692: "Concentric RPD [W/s]",
    6553691: "Lower-Limb Stiffness [N/m]",
    # Landing
    6553628: "Peak Landing Force [N]",  # LR
}

VALD_DJ_METRICS = {
    # Antropometry
    655386: "Body Weight [kg]",
    # Kinematics
    6553652: "Drop Height [cm]",
    6553607: "Jump Height (Flight Time) [cm]",
    6553611: "Jump Height (Imp-Mom) [cm]",
    6553610: "Jump Height (Imp-Dis) [cm]",
    # Timing
    6553605: "Contact Time [s]",
    6553606: "Flight Time [s]",
    6553650: "Peak Force [N]",  # LR
    6553719: "Drop Landing RFD [N/s]",  # LR
    6553715: "Eccentric Net Impulse [N·s]",  # Calculate total impulse and delete
    6553712: "Concentric Net Impulse [N·s]",  # Calculate total impulse and delete
    # Landing
    6553628: "Peak Landing Force [N]",  # LR
}

VALD_IMTP_METRICS = {
    # Forces
    13107203: "Peak Force [N]",  # LR
    13107223: "Peak Force [N/kg]",
    13107245: "Peak Net Force [N]",  # LR
    # Dopočítat na Kg a hmotnost
    13107202: "Time to Peak Force [s]",  #
    # RFD
    13107209: "RFD 0-50ms [N/s]",  # LR
    13107211: "RFD 0-100ms [N/s]",  # LR
    13107212: "RFD 0-150ms [N/s]",  # LR
    13107213: "RFD 0-200ms [N/s]",  # LR
    # Force in time
    13107204: "Force at 50ms [N]",  # LR
    13107205: "Force at 100ms [N]",  # LR
    13107206: "Force at 150ms [N]",  # LR
    13107207: "Force at 200ms [N]",  # LR
    # Impulse in time
    13107228: "Impulse 0-50ms [N·s]",  # LR
    13107230: "Impulse 0-100ms [N·s]",  # LR
    13107231: "Impulse 0-150ms [N·s]",  # LR
    13107232: "Impulse 0-200ms [N·s]",  # LR
}

VALD_ALL_METRICS = VALD_CMJ_METRICS | VALD_DJ_METRICS | VALD_IMTP_METRICS
