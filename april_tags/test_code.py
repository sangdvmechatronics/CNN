import numpy as np
import math

def extract_rotation_info(m):
    trace = m[0, 0] + m[1, 1] + m[2, 2]
    angle = math.acos((trace - 1) / 2) if -1 <= (trace - 1) / 2 <= 1 else 0

    if angle != 0:
        x = (m[2, 1] - m[1, 2]) / (2 * math.sin(angle))
        y = (m[0, 2] - m[2, 0]) / (2 * math.sin(angle))
        z = (m[1, 0] - m[0, 1]) / (2 * math.sin(angle))
    else:
        x = y = z = 0

    angle_degrees = math.degrees(angle)

    print(f"Góc xoay: {angle_degrees} độ")
    print(f"Trục xoay: {x}, {y}, {z}")

    return angle_degrees, x, y, z

# Sử dụng hàm với một ma trận xoay ví dụ
R_example = np.array([[0.866, -0.5, 0],
                     [0.5, 0.866, 0],
                     [0, 0, 1]])

extract_rotation_info(R_example)
