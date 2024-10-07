import sys
sys.path.append('..')

from libs import calculate_angle

import mediapipe as mp
mp_pose = mp.solutions.pose

class ForearmsPostureAnalyzer:
    def __init__(self, keypoints):
        self.left_shoulder_outside_angle = calculate_angle(
            keypoints[mp_pose.PoseLandmark.LEFT_HIP], 
            keypoints[mp_pose.PoseLandmark.LEFT_SHOULDER], 
            keypoints[mp_pose.PoseLandmark.LEFT_ELBOW]
        )
        self.right_shoulder_outside_angle = calculate_angle(
            keypoints[mp_pose.PoseLandmark.RIGHT_HIP], 
            keypoints[mp_pose.PoseLandmark.RIGHT_SHOULDER], 
            keypoints[mp_pose.PoseLandmark.RIGHT_ELBOW]
        )
        self.is_left_elbow_higher_than_shoulder = keypoints[mp_pose.PoseLandmark.LEFT_ELBOW][1] < keypoints[mp_pose.PoseLandmark.LEFT_SHOULDER][1]
        self.is_right_elbow_higher_than_shoulder = keypoints[mp_pose.PoseLandmark.RIGHT_ELBOW][1] < keypoints[mp_pose.PoseLandmark.RIGHT_SHOULDER][1]
    
    def is_left_elbow_above(self):
        return self.left_shoulder_outside_angle >= 90 and self.is_left_elbow_higher_than_shoulder
    
    def is_right_elbow_above(self):
        return self.right_shoulder_outside_angle >= 90 and self.is_right_elbow_higher_than_shoulder

    def is_both_elbow_below(self):
        return (not self.is_left_elbow_above()) and (not self.is_right_elbow_above())
    
    def is_one_elbow_below(self):
        return self.is_left_elbow_above() != self.is_right_elbow_above()
    
    def is_both_elbow_above(self):
        return self.is_left_elbow_above() and self.is_right_elbow_above()
    
    def predict(self):
        if self.is_both_elbow_below():
            return 1
        if self.is_one_elbow_below():
            return 2
        if self.is_both_elbow_above():
            return 3
        return 1