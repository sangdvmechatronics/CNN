import numpy as np
import cv2
from scipy.spatial.transform import Rotation


def estimate_pose_camera():

    # Ma trận camera
    K = loaded_mtx = np.load('camera_matrix.npy')


    # Ma trận homography
    H = np.loadtxt('homography_matrix.txt')


    print(f"checking homography: {H}" +"\n"+ f"mtx: {K}" +"\n")
    # Tính ma trận quay R và vector dịch chuyển t từ ma trận homography H
    num, Rs, Ts, Ns = cv2.decomposeHomographyMat(H, K)
    ### Trong kết quả đã đưa ra, Rs là một tuple chứa 4 ma trận quay, và Ts là một tuple chứa 4 vector tịnh tiến. 
    # Mỗi cặp ma trận quay và vector tịnh tiến đại diện cho một giải pháp tư thế của camera tương ứng với ma trận homography.
    #có thể chọn bất kỳ cặp nào trong số chúng để sử dụng tư thế của camera trong ứng dụng của mình. Đối với mỗi cặp,
    #  ma trận quay đại diện cho hướng nhìn của camera và vector tịnh tiến đại diện cho vị trí của camera trong không gian toàn cầu.

    print(f"ma trận quay: \n {Rs}")
    print(f"\n ma trận tịnh tiến \n {Ts}")

    # Xác định ma trận chiếu P
    #P = np.dot(K, np.hstack((R, t)))


    #retval, rvec, tvec, _, _, _, _ = cv2.decomposeProjectionMatrix(P)

    """"
    retval: Trả về 1 nếu thành công.
    rvec: Vector hóa của ma trận quay R.
    tvec: Vector dịch chuyển t.
    K: Ma trận nội suy cảm biến.
    D: Ma trận nội suy biến dạng.
    R: Ma trận quay.
    P: Ma trận chiếu.
    """
    # # Hiển thị kết quả
    # print("Rotation Matrix:")
    # print(R)
    # print("\nTranslation Vector:")
    # print(t)

if __name__ == "__main__":
    estimate_pose_camera()