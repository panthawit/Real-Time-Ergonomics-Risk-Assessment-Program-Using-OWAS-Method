import numpy as np

def calculate_angle(a, b, c):
    ba = np.array(a) - np.array(b)
    bc = np.array(c) - np.array(b)
    cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cos_angle)
    return np.degrees(angle)

def calculate_line_angle(point1, point2, ignore_depth=False):
    if ignore_depth:
        point1 = np.array(point1)[:-1]
        point2 = np.array(point2)[:-1]
    
    point_norm = np.abs(point1 - point2)
    y = point_norm[1]
    distance = np.linalg.norm(point_norm)
    
    # Calculate the angle between the direction vector and the x-axis
    angle_radians = np.arcsin(y / distance)
    angle_degrees = np.degrees(angle_radians)
    return angle_degrees