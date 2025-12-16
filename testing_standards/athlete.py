class Athlete:
    """
    Data structure holding anthropometric measurements of an athlete.
    Acts as a configuration object for the bio-mechanical models.
    """
    def __init__(self,
                 foot_length: float,
                 heel_ankle_length: float,
                 ankle_height: float,
                 shin_length: float,
                 thigh_length: float,
                 trunk_length: float,
                 arm_length: float,
                 quadriceps_offset: float = 7):
        """
        Initialize athlete parameters. All measurements are in centimeters (cm).

        Args:
            foot_length (float): Total length of the foot (Heel to Toe).
            heel_ankle_length (float): Horizontal distance from the heel to the lateral malleolus (ankle joint).
            ankle_height (float): Vertical height of the ankle joint center from the ground.
            shin_length (float): Length of the shin. Measured from the Lateral Malleolus to the midpoint of the line connecting the Fibular Head and the Femoral Lateral Epicondyle (approximating the knee joint center).
            thigh_length (float): Length of the thigh. Measured from the midpoint of the knee joint (as defined above) to the Greater Trochanter (approximating the hip joint center).
            trunk_length (float): Length of the torso. Measured from the Greater Trochanter to the Acromion (approximating the shoulder joint center).
            arm_length (float): Total arm length. Measured from the Acromion to the Middle Finger Knuckle (when the hand is open).
            quadriceps_offset (float, optional): Anterior thickness of the thigh muscle. Based on studies defaults to 7.0 cm.
        """
        self.foot_length = foot_length
        self.heel_ankle_length = heel_ankle_length
        self.ankle_height = ankle_height
        self.shin_length = shin_length
        self.thigh_length = thigh_length
        self.trunk_length = trunk_length
        self.arm_length = arm_length
        self.quadriceps_offset = quadriceps_offset
        