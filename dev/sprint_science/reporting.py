import matplotlib.pyplot as plt
from sprint_science.simulator import SprintSimulation
import time
import pandas as pd
import math


def complete_report(data):
    
    original_fly_length = data.fly_length

    data.fly_length = 30

    report = data.run_sprint()

    running_time = report['time'].iloc[-1]

    top_speed = data.top_speed()

    fly_segment = data.flying_sections()

    data.fly_length = original_fly_length

    fly_time = str(f'{fly_segment['fastest']['time']:.2f} s')
    fly_start = str(f'{fly_segment['fastest']['start']:.0f}')
    fly_finish = str(f'{fly_segment['fastest']['finish']:.0f} m')
        
    time_30_m = report[report['distance'] > 30]['time'].iloc[0]

    p_max = data.F0 * data.V0 / 4
    sfv = data.F0 / data.V0


    def slow_print(report_lines, delay=0.07):

        for line in report_lines:
            print(line)
            time.sleep(delay)
    
    report_text = [
        '\n',
        '==================================================',
        ' ==      REPORT VÝKONNOSTNÍ DIAGNOSTIKY        ==',
        '==================================================',
        '\n',
        '--- VSTUPNÍ PARAMETRY SIMULACE ---',
        f'- F0: {data.F0} N/kg, V0: {data.V0}',
        f'- Hmotnost: {data.weight} kg, Výška: {data.height} cm',
        f'- Běžená vzdálenost: {data.running_distance} m',
        f'- Externí odpor: {data.external_force_N} N',
        '\n',
        '--- KLÍČOVÉ UKAZATELE VÝKONU ---',
        f'Celkový čas na {data.running_distance}m: {running_time:.2f}',
        '\n',
        '--- ČASOVÁ ANALÝZA VÝKONU ---',
        f'Čas na 30m (akcelerace): {time_30_m:.2f} s',
        f'Čas ma 30m (letmý úsek): {fly_time}',
        f'Ve vzdálenosti {fly_start} - {fly_finish}',        
        f'Maximální rychlost: {top_speed['top_speed']:.2f} m/s',
        f'Dosaženo ve vzdálenosti: {top_speed['distance_top_speed']:.1f} m',
        '\n',
        '--- BIOMECHANICKÝ PROFIL ---',
        f'Maximální výkon (Pmax): {p_max:.2f} W/kg ({(p_max * data.weight):.0f} W)',
        f'F-V Sklon (Sfv): {sfv:.2f}',
        '\n',
        '==================================================',
        'Nyní bude zobrazen graf průběhu rychlosti...',
        '\n',
    ]

    slow_print(report_text)
    

    plt.figure(figsize=(10, 6))
    plt.plot(report['distance'], report['speed'], label='Graf rychlosti a vzdálenosti')
    plt.xlabel('Vzdálenost')
    plt.ylabel('Rychlost')
    plt.grid(True)
    plt.show() 


def segment_report(data):
    print("\n--- MEZIČASY NA SEGMENTECH ---")

    segments_df = data.segments()

    for index, row in segments_df.iterrows():

        print(f"Čas na {row['distance']:.0f} m: {row['total_time']:.2f} | Čas segmentu: {row['segment_time']:.2f}")



def top_speed_report(data):
    print("\n--- MAXIMÁLNÍ RYCHLOST ---")


    top_speed = data.top_speed()

    print(f'Maximální rychlost (bez zaokrouhl.): {top_speed['top_speed']:.2f} m/s')
    print(f'Vzdálenost (bez zaokrouhl.): {top_speed['distance_top_speed']:.1f} m')
    print('\n')    
    print(f'Maximální rychlost (1 d. m.): {top_speed['top_speed_rounded']:.1f} m/s')
    print(f'Vzdálenost (1 d. m.): {top_speed['distance_top_speed_rounded']:.1f} m')


def fastest_f_v_report(data):
    print("\n--- NEJRYCHLEJŠÍ F-V PROFIL ---")

    report = data.f_v_profile_comparison()

    fastest_f_v = report.loc[report['time'].idxmin()]

    print(f'Vzdálenost: {data.running_distance} m')
    print(f'Optimální sklon: {fastest_f_v['f_v_slope']:.2f}')
    print(f'Čas: {fastest_f_v['time']:.2f}')
    print(f'F0: {fastest_f_v['F0']:.2f}')
    print(f'V0: {fastest_f_v['V0']:.2f}')


def calibration(data):
    print("\n--- KALIBRACE ---")

    report = data.run_sprint()

    boudaries = [2.5, 5, 10, 20, 30]
    boundary_index = 0

    for i, time in enumerate(report['time']):

        if boundary_index >= len(boudaries):
            break

        current_boundary = boudaries[boundary_index]
        
        if report['distance'][i] >= current_boundary:

            print(f'{time:.2f}')

            boundary_index += 1
    

def plot_trial_v_distance(data):

    report = data.run_sprint()

    plt.figure(figsize=(10, 6))
    plt.plot(report['distance'], report['speed'])
    plt.xlabel('Distance')
    plt.ylabel('Speed')
    plt.grid(True)
    plt.show() 


def plot_fastest_f_v(data):

    report = data.f_v_profile_comparison()

    f_v_slope_list = report['f_v_slope'].tolist()
    time_list = report['time'].tolist()

    plt.figure(figsize=(10, 6))
    plt.plot(f_v_slope_list, time_list)
    plt.xlabel('F/V Slope')
    plt.ylabel('Time')
    plt.grid(True)
    plt.show() 


def plot_add_trial_to_v_distance(data):
    
    report = data.run_sprint()

    plt.plot(report['distance'], report['speed'])
    plt.xlabel('distance')
    plt.ylabel('speed')
    plt.grid(True)


def flying_sections_report(data):
    print("\n--- ANALÝZA LETMÝCH ÚSEKŮ ---")

    report = data.flying_sections()

    if type(report) == str:
        print(report)

    else:
        print(f'Absolutně nejrychlejší úsek (na tisíciny): {data.fly_length} m za {report['fastest']['time']:.3f} s mezi {report['fastest']['start']:.1f} - {report['fastest']['finish']:.1f} m')
        print(f'První úsek s nejlepším časem (na setiny): {data.fly_length} m za {report['first_fast']['time']} s mezi {report['first_fast']['start']} - {report['first_fast']['finish']} m')


def overspeed_zones_report(data):
    print("\n--- ANALÝZA URYCHLOVAČ ---")

    report = data.overspeed_zones()

    speed_zones = [1.01, 1.02, 1.03, 1.04, 1.05, 1.06, 1.07, 1.08, 1.09, 1.10]


    for record in speed_zones:

        beyond_zone = report[report['speed_gain'] >= record]

        first_beyond_zone = beyond_zone.iloc[0]

        print(f'Relativní rychlost: {record * 100:.0f} %    Absolutní rychlost: {first_beyond_zone['top_speed']:.2f} m/s    Externí síla: {first_beyond_zone['external_force_N']:.0f} N')
