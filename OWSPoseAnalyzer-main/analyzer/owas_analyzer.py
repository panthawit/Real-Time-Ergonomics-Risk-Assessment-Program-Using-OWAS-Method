import sys
sys.path.append('..')

from .back_posture.back_posture_analyzer import BackPostureAnalyzer
from .forearms_posture.forearms_posture_analyzer import ForearmsPostureAnalyzer
from .legs_posture.legs_posture_analyzer import LegsPostureAnalyzer
import numpy as np

import mediapipe as mp
mp_pose = mp.solutions.pose

def OWAS_analysis(keypoint):
    # additional keypoint
    
    keypoint = list(keypoint)
    # center of shoulder
    keypoint.append(
        np.mean([
            keypoint[mp_pose.PoseLandmark.LEFT_SHOULDER],
            keypoint[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        ], axis=0)
    )
    # center of hip
    keypoint.append(
        np.mean([
            keypoint[mp_pose.PoseLandmark.LEFT_HIP],
            keypoint[mp_pose.PoseLandmark.RIGHT_HIP]
        ], axis=0)
    )
    keypoint = np.array(keypoint)
    
    back_score = BackPostureAnalyzer(keypoint).predict()
    forearm_score = ForearmsPostureAnalyzer(keypoint).predict()
    leg_score = LegsPostureAnalyzer(keypoint).predict()
    return back_score, forearm_score, leg_score