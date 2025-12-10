from sprint_science.simulator import SprintSimulation

def run_simulation_logic(f0, v0, weight, height):

    #fixed variables
    running_distance = 100.0
    external_force_N = 0.0
    fly_length = 30.0

    f0_clean = f0.replace(",", ".")
    v0_clean = v0.replace(",", ".")
    weight_clean = weight.replace(",", ".")
    height_clean = height.replace(",", ".")

    try:
        f0_f = float(f0_clean)
        v0_f = float(v0_clean)
        weight_f = float(weight_clean)
        height_f = float(height_clean)

    except ValueError:
        raise ValueError("Invalid input. Please enter numbers only.")

    if f0_f <= 0:
        raise ValueError("F0 must be a positive number (greater than 0).")
        
    if v0_f <= 0:
        raise ValueError("V0 mmust be a positive number (greater than 0).")
   
    if height_f <= 0:
        raise ValueError("Height must be a positive number (greater than 0).")
        
    if weight_f <= 0:
        raise ValueError("Weight must be a positive number (greater than 0).")
        
    try:
        data = SprintSimulation(
            F0 = f0_f,
            V0 = v0_f,
            weight = weight_f,
            height = height_f,
            running_distance = running_distance,
            external_force_N = external_force_N,
            fly_length = fly_length
    )

        report = data.run_sprint()
        running_time = report['time'].iloc[-1]
        top_speed = data.top_speed()
        fly_segment = data.flying_sections()
        fly_time = str(f'{fly_segment['first_fast']['time']:.2f} s')
        fly_start = str(f'{fly_segment['first_fast']['start']:.0f} m')
        fly_finish = str(f'{fly_segment['first_fast']['finish']:.0f} m')
        time_30_m = report[report['distance'] > 30]['time'].iloc[0]


        results = {
            'running_time_100m' : running_time,
            'time_30m' : time_30_m,
            'time_30m_fly' : fly_time,
            'fly_start' : fly_start,
            'fly_finish' : fly_finish,
            'top_speed' : top_speed['top_speed'],
            'top_speed_distance' : top_speed['distance_top_speed']
        }

        return results, report
    
    except Exception as e:
        raise Exception(f'An error occurred during simulation: {e}')

