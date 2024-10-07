import numpy as np
import cv2
from constant import (
    SCORE_CRITERIA
)
from analyzer.owas_analyzer import OWAS_analysis
import mediapipe as mp

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5, model_complexity=1)

if __name__ == '__main__':
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1024)

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = pose.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if results.pose_landmarks:
            connections = mp_pose.POSE_CONNECTIONS

            h, w, _ = image.shape
            for connection in connections:
                start_point = tuple(map(int, (results.pose_landmarks.landmark[connection[0]].x * w, results.pose_landmarks.landmark[connection[0]].y * h)))
                end_point = tuple(map(int, (results.pose_landmarks.landmark[connection[1]].x * w, results.pose_landmarks.landmark[connection[1]].y * h)))
                cv2.line(image, start_point, end_point, (0, 255, 0), 2)

            keypoints = np.array([[landmark.x, landmark.y, landmark.z] for landmark in results.pose_landmarks.landmark])
            back_score, forearm_score, leg_score = OWAS_analysis(keypoints)

            cv2.putText(image, f'Back Score : {back_score}', (0, 50), cv2.FONT_HERSHEY_DUPLEX, 2.0, (255, 0, 0), 3)
            cv2.putText(image, f'Forearm Score : {forearm_score}', (0, 100), cv2.FONT_HERSHEY_DUPLEX, 2.0, (255, 0, 0), 3)
            cv2.putText(image, f'Leg Score : {leg_score}', (0, 150), cv2.FONT_HERSHEY_DUPLEX, 2.0, (255, 0, 0), 3)
            
            if back_score and forearm_score and leg_score:
                score = SCORE_CRITERIA[back_score][forearm_score][leg_score][1]
            else:
                score = 0
            cv2.putText(image, f'Overall Score : {score}', (0, 200), cv2.FONT_HERSHEY_DUPLEX, 2.0, (255, 255, 0), 3)

            cv2.imshow('webcam', image)
            if cv2.waitKey(1) == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()