import sys
sys.path.append('..')

from libs import calculate_line_angle

import mediapipe as mp
mp_pose = mp.solutions.pose

class BackPostureAnalyzer:
    def __init__(self, keypoints):
        self.body_line_angle = calculate_line_angle(
            keypoints[-2], 
            keypoints[-1]
        )
        self.shoulder_line_angle = calculate_line_angle(
            keypoints[mp_pose.PoseLandmark.LEFT_SHOULDER], 
            keypoints[mp_pose.PoseLandmark.RIGHT_SHOULDER],
            ignore_depth=True
        )
        self.offset_1 = 60
        self.offset_2 = 20

    def is_upright(self):
        return self.body_line_angle >= self.offset_1

    def is_leaning_forward(self):
        return self.body_line_angle < self.offset_1
    
    def is_flexous(self):
        return self.body_line_angle >= self.offset_1 and self.shoulder_line_angle > self.offset_2
    
    def is_leaning_forward_and_flexous(self):
        return self.body_line_angle < self.offset_1 and self.shoulder_line_angle > self.offset_2
    
    def predict(self):
        if self.is_flexous():
            return 3
        if self.is_upright():
            return 1
        if self.is_leaning_forward_and_flexous():
            return 4
        if self.is_leaning_forward():
            return 2
        return 1