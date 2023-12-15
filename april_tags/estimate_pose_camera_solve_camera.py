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
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


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

    # カメラ準備 ###############################################################
   

    
    # os.add_dll_directory("C:/Users/username/Miniconda3/envs/my_env/lib/site-packages/pupil_apriltags.libs")
    os.add_dll_directory("E:\FRAMEWORK\LANGUAGE\Anaconda_install\envs\do_an\Lib\site-packages\pupil_apriltags.libs")
    # os.add_dll_directory("E:\FRAMEWORK\LANGUAGE\Anaconda_install\envs\do_an\Lib\site-packages\pupil_apriltags\lib\apriltag.dll")


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
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')


    # Khởi tạo pipeline cho camera RealSense
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    # Bắt đầu streaming
    pipeline.start(config)
    K_camera_matrix  = np.load('camera_matrix.npy')
    D_camera_matrix_coeffs = np.load('dist_coeffs.npy')
    file_path = "data_angle.txt"
    global previous_tvec
    time_run = 0

    try:
  
        while True:
            start_time = time.time()
            time.sleep(0.1)

            # Đợi và lấy frame từ camera
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue
            # Chuyển đổi frame sang mảng numpy
            color_image = np.asanyarray(color_frame.get_data())
            # color_image_calib = cv2.undistort(color_image, K_camera_matrix, D_camera_matrix_coeffs)

            # debug_image = copy.deepcopy(color_image_calib)
            debug_image = copy.deepcopy(color_image)

            image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
            tags = at_detector.detect(
                image,
                estimate_tag_pose=False,
                camera_params=None,
                tag_size=None,
            )
            src_points = np.array([[0, 0], [0, 0], [0, 0], [0, 0]])

            debug_image, src_points = draw_tags(debug_image, tags, elapsed_time, src_points)
            print(debug_image.shape)

            elapsed_time = time.time() - start_time
            time_run += elapsed_time

            print("Time: ", time_run)

            dst_points = np.array([[0,0,0],[7.5,0,0],[7.5,-7.5,0],[0,-7.5,0]])

            src_points = src_points.astype('float32')
            dst_points = dst_points.astype('float32')
            
            #print(f"kich thuoc mang dst points \n {dst_points} \nmang src points {src_points}\n ma tran K {K_camera_matrix.shape} \n ma tran D {D_camera_matrix_coeffs.shape}")
            R, tvec = estimate_pose_camera(src_points, dst_points, K_camera_matrix, D_camera_matrix_coeffs)
            angle_degrees_X, angle_degrees_Y, angle_degrees_Z = matrix_to_axis_angle(R)
            # Chuyển đổi rvec thành ma trận quay R
            previous_tvec = tvec
            save_data(time_run, file_path, angle_degrees_X, angle_degrees_Y, angle_degrees_Z)

            # Hiển thị đồ thị
            
            # update_plot(ax, rvec, tvec, previous_tvec)
            # plt.pause(0.01)  # Tạm dừng 0.01 giây để đồ thị có thể được cập nhật

            key = cv2.waitKey(1)
            cv2.circle(debug_image, (int(debug_image.shape[1]/2), int(debug_image.shape[0]/2)), 2, (0, 0, 255), 2)
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
        pipeline.stop()
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
    retval, rvec, tvec = cv2.solvePnP(dst_points, src_points, K_camera_matrix, D_camera_matrix_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
    R, _ = cv2.Rodrigues(rvec)
    # Chuyển đổi R và tvec để có ma trận T
    R_inv = R.T
    tvec_inv = -R_inv.dot(tvec)

    # Tạo ma trận T
    T = np.eye(4, dtype=R.dtype)
    T[:3, :3] = R_inv
    T[:3, 3:] = tvec_inv
    # Tạo ma trận tư thế của camera
    pose = np.eye(4, dtype=R.dtype)
    pose[:3, :3] = R
    pose[:3, 3:] = tvec

    # Bổ xung vào ma trận T
    T[:3, :3] = pose[:3, :3]
    T[:3, 3:] = pose[:3, 3:]

    # print(f"Estimated : \n {R} \n ----------\n {tvec}" )
    print(f"pose : \n {pose}")
    print("--------------------------------")
    
    return R, tvec

def calculate_homography(src_points, dst_points):
    # Chuyển đổi điểm thành numpy array
    src_points = np.array(src_points, dtype=np.float32)
    dst_points = np.array(dst_points, dtype=np.float32)

    # Tính ma trận homography
    homography_matrix, _ = cv2.findHomography(src_points, dst_points, cv2.RANSAC, 5.0)

    return homography_matrix

def update_plot(ax, rvec, tvec, previous_tvec):
    
    # Xóa đồ thị trước khi cập nhật
    ax.cla()
    # Thiết lập giới hạn cho các trục
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_zlim([0, 1])

    # Thiết lập độ chia cho các trục
    ax.set_xticks(np.arange(0, 1.1, 0.1))
    ax.set_yticks(np.arange(0, 1.1, 0.1))
    ax.set_zticks(np.arange(0, 1.1, 0.1))
    # Hiển thị hệ toạ độ mới
    ax.quiver(0, 0, 0, 1, 0, 0, color='r', label='X-axis')
    ax.quiver(0, 0, 0, 0, 1, 0, color='g', label='Y-axis')
    ax.quiver(0, 0, 0, 0, 0, 1, color='b', label='Z-axis')

    # Hiển thị vị trí camera mới
    ax.scatter(tvec[0], tvec[1], tvec[2], c='k', marker='o', label='Camera Position')

    # Hiển thị vị trí camera trước đó với màu khác
    ax.scatter(previous_tvec[0], previous_tvec[1], previous_tvec[2], c='r', marker='x', label='Previous Camera Position')

    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')

    plt.legend()
def matrix_to_axis_angle(m):
    angle = math.acos((m[0, 0] + m[1, 1] + m[2, 2] - 1) / 2)
    
    x = (m[2, 1] - m[1, 2]) / np.sqrt((m[2, 1] - m[1, 2]) ** 2 + (m[0, 2] - m[2, 0]) ** 2 + (m[1, 0] - m[0, 1]) ** 2)
    y = (m[0, 2] - m[2, 0]) / np.sqrt((m[2, 1] - m[1, 2]) ** 2 + (m[0, 2] - m[2, 0]) ** 2 + (m[1, 0] - m[0, 1]) ** 2)
    z = (m[1, 0] - m[0, 1]) / np.sqrt((m[2, 1] - m[1, 2]) ** 2 + (m[0, 2] - m[2, 0]) ** 2 + (m[1, 0] - m[0, 1]) ** 2)
    ## đổi sang độ
    angle_degrees = math.degrees(angle)
    angle_degrees_X = math.degrees(x)
    angle_degrees_Y = math.degrees(y)
    angle_degrees_Z = math.degrees(z)

    print(f"goc xoay cac truc: {angle_degrees} \n {angle_degrees_X}, {angle_degrees_Y}, {angle_degrees_Z}")
    return angle_degrees_X, angle_degrees_Y, angle_degrees_Z

def save_data(time_run, file_path, angle_degrees_X, angle_degrees_Y, angle_degrees_Z):
    # Mở tệp để ghi (chế độ thêm dữ liệu vào cuối tệp)
    with open(file_path, "a") as file:
        # Ghi dòng mới vào tệp
        line = f"{time_run}\t{angle_degrees_X}\t{angle_degrees_Y}\t{angle_degrees_Z}\n"
        file.write(line)

    print("------Đã lưu--------")

if __name__ == "__main__":
    main_all()

