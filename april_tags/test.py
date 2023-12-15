import numpy as np
import cv2
from scipy.spatial.transform import Rotation
import math


import cv2
import pyrealsense2 as rs
import numpy as np
import argparse
import os
from pupil_apriltags import Detector
import time
import copy



def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--width", help='cap width', type=int, default=960)
    parser.add_argument("--height", help='cap height', type=int, default=540)

    parser.add_argument("--families", type=str, default='tag36h11')
    parser.add_argument("--nthreads", type=int, default=1)
    parser.add_argument("--quad_decimate", type=float, default=2.0)
    parser.add_argument("--quad_sigma", type=float, default=0.0)
    parser.add_argument("--refine_edges", type=int, default=1)
    parser.add_argument("--decode_sharpening", type=float, default=0.25)
    parser.add_argument("--debug", type=int, default=0)

    args = parser.parse_args()

    return args


def main_all():
    args = get_args()

    families = args.families
    nthreads = args.nthreads
    quad_decimate = args.quad_decimate
    quad_sigma = args.quad_sigma
    refine_edges = args.refine_edges
    decode_sharpening = args.decode_sharpening
    debug = args.debug
    
    # os.add_dll_directory("C:/Users/username/Miniconda3/envs/my_env/lib/site-packages/pupil_apriltags.libs")
    os.add_dll_directory("E:\FRAMEWORK\LANGUAGE\Anaconda_install\envs\do_an\Lib\site-packages\pupil_apriltags.libs")
    # os.add_dll_directory("E:\FRAMEWORK\LANGUAGE\Anaconda_install\envs\do_an\Lib\site-packages\pupil_apriltags\lib\apriltag.dll"
    # Detector準備 #############################################################
    at_detector = Detector(
        families=families,
        nthreads=nthreads,
        quad_decimate=quad_decimate,
        quad_sigma=quad_sigma,
        refine_edges=refine_edges,
        decode_sharpening=decode_sharpening,
        debug=debug,
    )

    elapsed_time = 0
    cap = cv2.VideoCapture('17_april_tag.mp4')

    try:
        while True:
            start_time = time.time()
            # Kiểm tra xem video có được mở thành công hay không
            if not cap.isOpened():
                print("Không thể mở video.")
                return

            # Đọc frame từ video
            ret, frame = cap.read()
            frame = cv2.resize(frame, (640,480))
            # Kiểm tra xem frame đã được đọc thành công hay không
            if not ret:
                print("Không thể đọc frame.")
                break
            debug_image = copy.deepcopy(frame)

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            tags = at_detector.detect(
                image,
                estimate_tag_pose=False,
                camera_params=None,
                tag_size=None,
            )
            src_points = np.array([[0, 0], [0, 0], [0, 0], [0, 0]])
            
            debug_image, src_points = draw_tags(debug_image, tags, elapsed_time, src_points)

            elapsed_time = time.time() - start_time


            dst_points = np.array([[0,0,0],[0.075,0,0],[0.075,0.075,0],[0,0.075,0]])
            src_points = src_points.astype('float32')
            dst_points = dst_points.astype('float32')

            K_camera_matrix  = np.load('camera_matrix.npy')
            D_camera_matrix_coeffs = np.load('dist_coeffs.npy')
            print(f"kich thuoc mang dst points \n {dst_points} \nmang src points {src_points}\n ma tran K {K_camera_matrix.shape} \n ma tran D {D_camera_matrix_coeffs.shape}")
            rvec, tvec = estimate_pose_camera(src_points, dst_points, K_camera_matrix, D_camera_matrix_coeffs)
            print(f"Estimated :ma trận xoay: {rvec} \n ma trận tịnh tiến {tvec}" )
            
            key = cv2.waitKey(1)
            if key == 27:  # ESC
                break

            # 画面反映 #############################################################
            cv2.imshow('AprilTag Detect Demo', debug_image)

            # Hiển thị ảnh RGB
            #cv2.imshow("RGB Image", color_image)
                
                # Thoát khỏi vòng lặp khi nhấn phím 'ESC'
            if cv2.waitKey(1) & 0xFF == 27:
                break

    finally:
        # Dừng streaming và đóng các kết nối
    
        cv2.destroyAllWindows()

def draw_tags(
    image,
    tags,
    elapsed_time,
    src_points
):
    for tag in tags:
        tag_family = tag.tag_family
        tag_id = tag.tag_id
        center = tag.center
        corners = tag.corners

        center = (int(center[0]), int(center[1]))
        corner_01 = (int(corners[0][0]), int(corners[0][1]))
        corner_02 = (int(corners[1][0]), int(corners[1][1]))
        corner_03 = (int(corners[2][0]), int(corners[2][1]))
        corner_04 = (int(corners[3][0]), int(corners[3][1]))

        src_points = np.array([[corner_01[0], corner_01[1]], [corner_02[0], corner_02[1]],
                       [corner_03[0], corner_03[1]], [corner_04[0], corner_04[1]]])
        #print("src_points", src_points)
        #print(f'Show_conner: {corner_01}: {corner_02}: {corner_03}: {corner_04}')
        cv2.circle(image, (corner_01[0], corner_01[1]), 5, (0, 0, 255), 2)
        cv2.circle(image, (corner_02[0], corner_02[1]), 5, (0, 0, 0), 2)
        cv2.circle(image, (corner_03[0], corner_03[1]), 5, (0, 0, 0), 2)
        cv2.circle(image, (corner_04[0], corner_04[1]), 5, (0, 0, 0), 2)

        # 中心
        cv2.circle(image, (center[0], center[1]), 5, (0, 0, 255), 2)

        # 各辺
        cv2.line(image, (corner_01[0], corner_01[1]),
                (corner_02[0], corner_02[1]), (255, 0, 0), 2)
        cv2.line(image, (corner_02[0], corner_02[1]),
                (corner_03[0], corner_03[1]), (255, 0, 0), 2)
        cv2.line(image, (corner_03[0], corner_03[1]),
                (corner_04[0], corner_04[1]), (0, 255, 0), 2)
        cv2.line(image, (corner_04[0], corner_04[1]),
                (corner_01[0], corner_01[1]), (0, 255, 0), 2)

        # タグファミリー、タグID
        # cv2.putText(image,
        #            str(tag_family) + ':' + str(tag_id),
        #            (corner_01[0], corner_01[1] - 10), cv2.FONT_HERSHEY_SIMPLEX,
        #            0.6, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(image, str(tag_id), (center[0] - 10, center[1] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2, cv2.LINE_AA)

    # 処理時間
    cv2.putText(image,
               "Elapsed Time:" + '{:.1f}'.format(elapsed_time * 1000) + "ms",
               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2,
               cv2.LINE_AA)

    return image, src_points
def estimate_pose_camera(src_points, dst_points, K_camera_matrix, D_camera_matrix_coeffs):

    # # Ma trận camera
    # K_camera_matrix  = np.load('camera_matrix.npy')
    # D_camera_matrix_coeffs = np.load('dist_coeffs.npy')
    retval, rvec, tvec = cv2.solvePnP(dst_points, src_points, K_camera_matrix, D_camera_matrix_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)

    #retval, rvec, tvec = cv2.solvePnP(dst_points, src_points,  K_camera_matrix,  D_camera_matrix_coeffs)
    return rvec, tvec

def calculate_homography(src_points, dst_points):
    # Chuyển đổi điểm thành numpy array
    src_points = np.array(src_points, dtype=np.float32)
    dst_points = np.array(dst_points, dtype=np.float32)

    # Tính ma trận homography
    homography_matrix, _ = cv2.findHomography(src_points, dst_points, cv2.RANSAC, 5.0)

    return homography_matrix



if __name__ == "__main__":
    main_all()

