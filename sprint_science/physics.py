"""
Physics module for Sprint Science.

This module contains aerodynamic constants and functions to calculate 
air resistance forces acting on a sprinter. It relies on standard 
anthropometric and aerodynamic formulas used in biomechanics literature 
(Morin, Samozino, et al.).
"""

# Constants

# Standard reference values for air at 0°C and 1 atm (STP)
RHO_STD = 1.293
P_STD_TORR = 760.0
T_STD_KELVIN = 273.0

# Aerodynamic Drag Coefficient (van IngenSchenauetal. 1991)
CD_SPRINT = 0.9


# Calculations

def calculate_frontal_area(height: float, weight: float) -> float:
    """
    Estimates the athlete's frontal Area (Af) based on height and weight.
    
    Methodology:
    1. Calculates Body Surface Area (BSA) using the Du Bois & Du Bois (1916) formula.
    2. Converts BSA to Frontal Area using the correction factor 0.266 (Pugh, 1971).
        
    Returns:
        float: Estimated frontal area (Af) in m^2.
    """
    # Sanity check for units: If height > 3, assume cm and convert to meters.
    if height > 3:
        height /= 100

    A = 0.2025 * (height ** 0.725) * (weight ** 0.425) * 0.266

    return A

def calculate_air_density(temperature_c: float, barometric_pressure_hpa: float) -> float:
    """
    Calculates the actual air density (rho) based on current environmental conditions.

    Returns:
        float: Air density (rho) in kg/m^3.
    """    
    temperature_kelvin = T_STD_KELVIN + temperature_c
    pressure_torr = barometric_pressure_hpa * (P_STD_TORR / 1013.25)

    rho = RHO_STD * (pressure_torr / P_STD_TORR) * (T_STD_KELVIN / temperature_kelvin)

    return rho

def calculate_air_resistance_force(speed: float, rho: float, A: float, wind_speed: float = 0) -> float:
    """
    Calculates the aerodynamic drag force (F_air) acting on the sprinter.
    
    Returns:
        float: Aerodynamic drag force in Newtons (N).
    """
    relative_speed = speed - wind_speed
    # Formula: F_air = 0.5 * rho * A * Cd * (v_rel)^2
    f_air = 0.5 * rho * A * CD_SPRINT * (relative_speed * abs(relative_speed))

    return f_air


