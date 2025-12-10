
# Constants
# Standard reference values for air at 0°C and 1 atm (STP)
rho_std = 1.293
p_std_torr = 760.0
t_std_kelvin = 273.0

# Standard aerodynamic constant for human body in sprint position (van IngenSchenauetal. 1991)
Cd = 0.9


def calculate_frontal_area(height, weight):
   
    # Body surface area (Du Bois 1916), converted to frontal area *0.266 (Pugh 1971)
    if height > 3:
        height /= 100

    A = 0.2025 * (height ** 0.725) * (weight ** 0.425) * 0.266

    return A

def calculate_air_density(temperature_c, barometric_pressure_hpa):
        
        temperature_kelvin = t_std_kelvin + temperature_c
        pressure_torr = barometric_pressure_hpa * (p_std_torr / 1013.25)

        rho = rho_std * (pressure_torr / p_std_torr) * (t_std_kelvin / temperature_kelvin)

        return rho

def calculate_air_resistance_force(speed, rho, A, wind_speed=0):

        relative_speed = speed - wind_speed
        f_resistance = 0.5 * rho * A * Cd * (relative_speed * abs(relative_speed))

        return f_resistance


