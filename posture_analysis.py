import numpy as np

def calculate_angle(a, b, c):
    """
    Calculate the angle between three points using the law of cosines.
    a, b, c are (x, y) coordinates of joints.
    """
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    ab = a - b
    cb = c - b

    cosine_angle = np.dot(ab, cb) / (np.linalg.norm(ab) * np.linalg.norm(cb))
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))

    return np.degrees(angle)

def check_posture(angle):
    """
    Checks posture based on calculated joint angle.
    """
    if 160 <= angle <= 180:
        return "Good"
    elif 130 <= angle < 160:
        return "Leaning"
    else:
        return "Bad Posture"
