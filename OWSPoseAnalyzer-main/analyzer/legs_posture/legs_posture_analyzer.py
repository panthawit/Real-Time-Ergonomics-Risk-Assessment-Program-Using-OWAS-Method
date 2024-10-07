import sys
sys.path.append('../..')

from libs import calculate_line_angle

import mediapipe as mp
mp_pose = mp.solutions.pose

class LegsPostureAnalyzer:
    def __init__(self, keypoints):
        self.keypoints = keypoints
        self.left_knee_ankle_line_angle = calculate_line_angle(
            keypoints[mp_pose.PoseLandmark.LEFT_KNEE],
            keypoints[mp_pose.PoseLandmark.LEFT_ANKLE],
        )
        self.right_knee_ankle_line_angle = calculate_line_angle(
            keypoints[mp_pose.PoseLandmark.RIGHT_KNEE],
            keypoints[mp_pose.PoseLandmark.RIGHT_ANKLE],
        )
        self.left_knee_hip_line_angle = calculate_line_angle(
            keypoints[mp_pose.PoseLandmark.LEFT_KNEE],
            keypoints[mp_pose.PoseLandmark.LEFT_HIP],
        )
        self.right_knee_hip_line_angle = calculate_line_angle(
            keypoints[mp_pose.PoseLandmark.RIGHT_KNEE],
            keypoints[mp_pose.PoseLandmark.RIGHT_HIP],
        )
        self.offset_1 = 90
        self.offset_2 = 75
        self.offset_3 = 115
        
        self.distance_threshold = 0.005

    def is_sitting(self):
        return (
            (self.left_knee_hip_line_angle < self.offset_2 and self.right_knee_hip_line_angle < self.offset_2) or
            (self.keypoints[mp_pose.PoseLandmark.LEFT_KNEE][1] < self.keypoints[mp_pose.PoseLandmark.LEFT_HIP][1]) or
            (self.keypoints[mp_pose.PoseLandmark.RIGHT_KNEE][1] < self.keypoints[mp_pose.PoseLandmark.RIGHT_HIP][1])
        )
    
    def is_standing_normally(self):
        return  self.offset_2 >= self.left_knee_hip_line_angle >= self.offset_1 and self.offset_1 >= self.right_knee_hip_line_angle >= self.offset_3
    
    def is_standing_with_one_leg(self):
        return (
            (
                (self.left_knee_hip_line_angle < self.offset_1) and
                (self.keypoints[mp_pose.PoseLandmark.LEFT_ANKLE][1] - self.keypoints[mp_pose.PoseLandmark.RIGHT_ANKLE][1] < -self.distance_threshold)
            ) or 
            (
                (self.right_knee_hip_line_angle < self.offset_1) and
                (self.keypoints[mp_pose.PoseLandmark.LEFT_ANKLE][1] - self.keypoints[mp_pose.PoseLandmark.RIGHT_ANKLE][1] > self.distance_threshold)
            )
        )

    def is_standing_with_legs_bent(self):#4
        return (
            (self.offset_1 > self.left_knee_hip_line_angle >= self.offset_2) and 
            (self.offset_1 > self.right_knee_hip_line_angle >= self.offset_2)
        )

    def is_standing_with_one_leg_bent(self):#5
        return (
            (
                (self.left_knee_hip_line_angle < self.offset_1) and 
                (self.offset_2 <= self.right_knee_hip_line_angle < self.offset_1) and
                (self.keypoints[mp_pose.PoseLandmark.LEFT_ANKLE][1] - self.keypoints[mp_pose.PoseLandmark.RIGHT_ANKLE][1] < -self.distance_threshold)
            ) or  
            (
                (self.right_knee_hip_line_angle < self.offset_1) and 
                (self.offset_2 <= self.left_knee_hip_line_angle < self.offset_1) and
                (self.keypoints[mp_pose.PoseLandmark.LEFT_ANKLE][1] - self.keypoints[mp_pose.PoseLandmark.RIGHT_ANKLE][1] > self.distance_threshold)
            )
        )
    
    def is_kneeling(self):
        return self.left_knee_ankle_line_angle < self.offset_2 and self.right_knee_ankle_line_angle < self.offset_2
    
    def predict(self):
        if self.is_kneeling():
            return 6
        if self.is_sitting():
            return 1
        if self.is_standing_normally():
            return 2
        if self.is_standing_with_legs_bent():
            return 4
        if self.is_standing_with_one_leg_bent():
            return 5
        if self.is_standing_with_one_leg():
            return 3
        return 7 # Walking