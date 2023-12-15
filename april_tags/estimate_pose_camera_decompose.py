import numpy as np
import cv2
from scipy.spatial.transform import Rotation
import math


def estimate_pose_camera():

    # Ma trận camera
    K_camera_matrix  = np.load('camera_matrix.npy')
    D_camera_matrix_coeffs = np.load('dist_coeffs.npy')

    # Ma trận homography
    H = np.loadtxt('homography_matrix.txt')

    print(D_camera_matrix_coeffs)

    print(f"checking homography: \n {H}" +"\n"+ f"mtx: {K_camera_matrix}" +"\n")
    # Tính ma trận quay R và vector dịch chuyển t từ ma trận homography H
    num, Rs, Ts, Ns = cv2.decomposeHomographyMat(H, K_camera_matrix)
    ### Trong kết quả đã đưa ra, Rs là một tuple chứa 4 ma trận quay, và Ts là một tuple chứa 4 vector tịnh tiến. 
    # Mỗi cặp ma trận quay và vector tịnh tiến đại diện cho một giải pháp tư thế của camera tương ứng với ma trận homography.
    #có thể chọn bất kỳ cặp nào trong số chúng để sử dụng tư thế của camera trong ứng dụng của mình. Đối với mỗi cặp,
    #  ma trận quay đại diện cho hướng nhìn của camera và vector tịnh tiến đại diện cho vị trí của camera trong không gian toàn cầu.

    #print(f"ma trận quay: \n {Rs}")
    print(f"ma trận đầu tiên: \n {Rs[0]}")
    #print(f"\n ma trận tịnh tiến \n {Ts}")
    print(f"ma trận tịnh tiến đầu tiên \n {Ts[0]}")


    matrix_to_axis_angle(Rs[0])

## Code tính góc
def matrix_to_axis_angle(m):
    angle = math.acos((m[0, 0] + m[1, 1] + m[2, 2] - 1) / 2)
    
    x = (m[2, 1] - m[1, 2]) / np.sqrt((m[2, 1] - m[1, 2]) ** 2 + (m[0, 2] - m[2, 0]) ** 2 + (m[1, 0] - m[0, 1]) ** 2)
    y = (m[0, 2] - m[2, 0]) / np.sqrt((m[2, 1] - m[1, 2]) ** 2 + (m[0, 2] - m[2, 0]) ** 2 + (m[1, 0] - m[0, 1]) ** 2)
    z = (m[1, 0] - m[0, 1]) / np.sqrt((m[2, 1] - m[1, 2]) ** 2 + (m[0, 2] - m[2, 0]) ** 2 + (m[1, 0] - m[0, 1]) ** 2)

    print(f"goc xoay cac truc: {angle} \n {x}, {y}, {z}")
    
    return angle, x, y, z

## medium https://medium.com/check-visit-computer-vision/convert-camera-poses-in-python-35debb8460ec


if __name__ == "__main__":
    estimate_pose_camera()