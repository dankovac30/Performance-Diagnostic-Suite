from testing_standards.imtp_standard import IMTP_Calculations
from testing_standards.athlete import Athlete


foot_length = 27
heel_ankle_length = 5
ankle_height = 7
shin_length = 45
thigh_length = 43
trunk_length = 58
arm_length = 68

test = (foot_length, heel_ankle_length, ankle_height, shin_length, thigh_length, trunk_length, arm_length)
me = (27, 5, 7, 45, 43, 58, 68)
kaja = (27, 5, 6, 42, 42, 62, 68)


profile = me

athlete = Athlete(*profile)

imtp = IMTP_Calculations(athlete)


rack_bar = imtp.rank_rack_heights()

for key, value_dict in rack_bar.items():

    print(f'Rack: {key:<6} Knee: {value_dict['knee_angle']:<10.1f} Hip: {value_dict['hip_angle']:<10.1f} Thigh: {value_dict['thigh_uncovered']:<10} Score: {value_dict['score']:<10.4f}  Knee overhang: {value_dict['knee_overhang']:.1f}')


