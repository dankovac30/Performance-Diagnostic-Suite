from .athlete import Athlete
import math
import json
import os

class IMTP_Calculations:

    def __init__(self, athlete):
        
        self.athlete = athlete
        self.ankle_offset = (athlete.foot_length/2) - athlete.heel_ankle_length
        
    
    def update_geometry(self, knee_angle_rad, l1):

        self.l1 = l1
        self.alfa_angle_rad = knee_angle_rad        

        self.beta_angle_rad = math.asin(self.athlete.quadriceps_offset/self.l1)
        self.omega_angle_rad = knee_angle_rad + self.beta_angle_rad

        self.l2 = math.sqrt(self.l1**2 + self.athlete.thigh_length**2 - (2 * self.l1 * self.athlete.thigh_length * math.cos(self.beta_angle_rad)))
        self.psi_angle_rad = math.asin(self.athlete.quadriceps_offset/self.l2)

        self.l3 = math.sqrt(self.l1**2 + self.athlete.shin_length**2 - (2 * self.l1 * self.athlete.shin_length * math.cos(self.omega_angle_rad)))
        self.h = math.sqrt(self.l3**2 - self.ankle_offset**2)

        self.gamma_angle_rad = math.acos((self.l2**2 + self.athlete.trunk_length**2 - self.athlete.arm_length**2) / (2 * self.l2 * self.athlete.trunk_length))
        self.delta_angle_rad = self.gamma_angle_rad + self.psi_angle_rad

        self.theta_angle_rad = math.asin(self.h / self.l3)
        self.lambda_angle_rad = math.acos((self.athlete.shin_length**2 + self.l3**2 - self.l1**2) / (2 * self.athlete.shin_length * self.l3))  
        self.epsilon_angle_rad = self.theta_angle_rad - self.lambda_angle_rad


    def bar_height(self):

        bar_height = self.h + self.athlete.ankle_height

        return bar_height
    

    def segment_angles(self):

        ankle_angle = math.degrees(self.epsilon_angle_rad)
        knee_angle = math.degrees(self.alfa_angle_rad)
        hip_angle = math.degrees(self.delta_angle_rad)

        angles = {
            'ankle': ankle_angle,
            'knee': knee_angle,
            'hip': hip_angle
        }

        return angles
    

    def segment_inclination(self):
     
        shin_incline = self.epsilon_angle_rad
        thigh_incline = shin_incline + (math.radians(180) - self.alfa_angle_rad)
        trunk_incline = thigh_incline - (math.radians(180) - self.delta_angle_rad)

        return shin_incline, thigh_incline, trunk_incline


    def acromion_position(self):

        shin_incline, thigh_incline, trunk_incline = self.segment_inclination()
        
        final_position = - self.ankle_offset

        knee_projection = math.cos(shin_incline) * self.athlete.shin_length
        final_position += knee_projection

        hip_projection = math.cos(thigh_incline) * self.athlete.thigh_length
        final_position += hip_projection

        shoulder_projection = math.cos(trunk_incline) * self.athlete.trunk_length
        final_position += shoulder_projection

        return final_position
    
    
    def asign_possible_rack_numbers(self):

        current_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(current_dir, 'iso_rack_w_board.json')

        with open(file_path) as f:
            rack_data = json.load(f)
        
        possible_rack_numbers = {}

        lower_height_bound = self.athlete.ankle_height + math.cos(math.radians(30)) * self.athlete.shin_length
        upper_height_bound = self.athlete.ankle_height + self.athlete.shin_length + self.athlete.thigh_length

        for key, value in rack_data.items():

            if lower_height_bound < value < upper_height_bound:

                possible_rack_numbers[key] = value
        
        return possible_rack_numbers


    def rank_rack_heights(self):

        rack_positions = self.asign_possible_rack_numbers()
        result_dict = {}

        vertical_arm_reach = math.sqrt(self.athlete.arm_length**2 - self.athlete.quadriceps_offset**2)
        bar_distance_below_hip = max(vertical_arm_reach - self.athlete.trunk_length, 0)
        bar_distance_above_knee = self.athlete.thigh_length - bar_distance_below_hip
        limit_min_l1 = self.athlete.thigh_length / 3
        limit_max_l1 = math.sqrt(bar_distance_above_knee**2 + self.athlete.quadriceps_offset**2)

        for rack_position, rack_bar_height in rack_positions.items():

            min_knee_angle = math.radians(100)
            max_knee_angle = math.radians(170)

            for _ in range(20):

                current_knee_angle = (min_knee_angle + max_knee_angle) / 2
                min_l1 = limit_min_l1
                max_l1 = limit_max_l1
                valid_solution_found = False

                for _ in range(20):
                    
                    current_l1 = (max_l1 + min_l1) / 2

                    try:
                        
                        self.update_geometry(current_knee_angle, current_l1)

                        height_at_top = self.h + self.athlete.ankle_height

                        diff = height_at_top - rack_bar_height

                        if abs(diff) < 0.1:
                            valid_solution_found = True
                            break
                        
                        elif diff > 0:
                            max_l1 = current_l1

                        elif diff < 0:
                            min_l1 = current_l1
                           
                    except ValueError:
                        break
                
                if valid_solution_found:

                    acromion_position = self.acromion_position()
                    target_acromion_position = 0

                    diff = acromion_position - target_acromion_position

                    
                    if abs(diff) < 1:
                        
                        ideal_knee_angle = math.radians(135)
                        ideal_hip_angle = math.radians(145)
                        
                        diff_knee_angle = abs(current_knee_angle - ideal_knee_angle)
                        diff_hip_angle = abs(self.delta_angle_rad - ideal_hip_angle)

                        error_knee = diff_knee_angle / ideal_knee_angle
                        error_hip = diff_hip_angle / ideal_hip_angle

                        thigh_uncovered_distance = math.sqrt(current_l1**2 - self.athlete.quadriceps_offset**2)
                        thigh_uncovered_ratio = thigh_uncovered_distance / self.athlete.thigh_length

                        score = (1.5 * error_knee)**2 + error_hip**2


                        knee_overhang = (math.cos(self.epsilon_angle_rad) * self.athlete.shin_length) - self.ankle_offset


                        current_rack_result = {
                            'l1': current_l1,
                            'thigh_uncovered': str(f'{thigh_uncovered_ratio*100:.0f}%'),
                            'knee_angle': math.degrees(current_knee_angle),
                            'hip_angle': math.degrees(self.delta_angle_rad),
                            'knee_overhang': knee_overhang,
                            'score': score
                        }
                        
                        result_dict[rack_position] = current_rack_result

                        if abs(diff) < 0.1:
                            break
                    
                    if diff > 0:
                        max_knee_angle = current_knee_angle
                    
                    elif diff < 0:
                        min_knee_angle = current_knee_angle
                
                else:
                    try:
                        self.update_geometry(current_knee_angle, limit_max_l1)
                        height_at_top = self.h + self.athlete.ankle_height
                
                    except ValueError:
                        height_at_top = - float('inf')

                    try:
                        self.update_geometry(current_knee_angle, limit_min_l1)
                        height_at_bottom = self.h + self.athlete.ankle_height

                    except ValueError:
                        height_at_bottom = float('inf') 

                    if height_at_top < rack_bar_height:
                        min_knee_angle = current_knee_angle

                    elif height_at_bottom > rack_bar_height:
                        max_knee_angle = current_knee_angle

                    else:
                        break        

        if not result_dict:
            return None
        
        return result_dict
    

    def find_best_rack_height(self):

        result_dict = self.rank_rack_heights()

        sorted_results = sorted(result_dict.items(), key=lambda x: x[1]['score'])
        best_rack_height = sorted_results[0]

        return best_rack_height
        
