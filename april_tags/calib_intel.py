import cv2
import numpy as np
import glob

import cv2
import pyrealsense2 as rs
import numpy as np
import argparse
import os
from pupil_apriltags import Detector
import time
import copy


# Kích thước bàn cờ (số ô trên hàng và cột)
board_size = (7, 10)

# Kích thước ô trên bàn cờ (đơn vị: mét)
square_size = 0.021

# Tạo các điểm của bàn cờ
objp = np.zeros((np.prod(board_size), 3), dtype=np.float32)
objp[:, :2] = np.mgrid[0:board_size[0], 0:board_size[1]].T.reshape(-1, 2) * square_size
#print(objp)

# Danh sách lưu trữ các điểm của bàn cờ và các điểm ảnh tương ứng
objpoints = []  # Điểm trên bàn cờ trong không gian 3D
imgpoints = []  # Điểm trên ảnh trong không gian 2D

# Đường dẫn đến thư mục chứa ảnh
image_folder = 'images_calib/848x480/*.png'  # Thay đổi đường dẫn và định dạng ảnh tương ứng

# Danh sách các đường dẫn của ảnh
images = glob.glob(image_folder)

# for index ,img_path in enumerate (images):
#     # Đọc ảnh từ file
#     img = cv2.imread(img_path)

#     # Chuyển ảnh sang ảnh xám
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


#     # Tìm các góc của bàn cờ
#     ret, corners = cv2.findChessboardCorners(gray, board_size, None)

#     if ret:
#         # Thêm điểm vào danh sách nếu bàn cờ được nhận diện
#         objpoints.append(objp)
#         imgpoints.append(corners)
#     if ret:
#             ####Hiển thị các góc trên ảnh
#             cv2.drawChessboardCorners(img, board_size, corners, ret)
#             # cv2.imshow(f"Chessboard Corners {index}", img)
#             # cv2.waitKey(0)
# # Calibrate camera
# ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
# # ###Lưu ma trận camera matrix (mtx) vào file
# np.save('848_480_camera_matrix_18_12.npy', mtx)

# # # Lưu distortion coefficients (dist) vào file
# np.save('848_480_dist_coeffs_18_12.npy', dist)
# # ###Hiển thị thông số calibrate
# print("Camera matrix:")
# print(mtx)
# print("\nDistortion coefficients:")
# print(dist)


## Load calibration
###Đọc ma trận camera matrix từ file
loaded_mtx = np.load('848_480_camera_matrix_18_12.npy')

# Đọc distortion coefficients từ file
loaded_dist = np.load('848_480_dist_coeffs_18_12.npy')

print (f"ma trận {loaded_mtx}"
       +"\n" + f"dist{loaded_dist}")




img = cv2.imread('images_calib/848x480/image_10.png')
h,  w = img.shape[:2]
print(h, w)
# Hiệu chỉnh ảnh =  0 là loại bỏ các pixel không mong muốn , =1 là giữ lại và bổ xung nên đen
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(loaded_mtx, loaded_dist, img.shape[1::-1], 0, img.shape[1::-1])
undistorted_img = cv2.undistort(img, loaded_mtx, loaded_dist, None, newcameramtx)

#np.save('848_480_new_camera_matrix_18_12.npy', newcameramtx)

print("New Camera Matrix:")
print(newcameramtx)

x, y, w, h = roi
print("roi:", x, y, w, h)

dst = img[y:y+h, x:x+w]
print(" kich thuoc sau hieu chinh", dst.shape[:2])
cv2.imshow("img", dst)
cv2.waitKey(0)
cv2.imshow("img_truoc", img)
cv2.waitKey(0)



# undistort
# dst = cv2.undistort(img, loaded_mtx, loaded_dist, None, newcameramtx)
# # crop the image

# x, y, w, h = roi
# print("roi:", x, y, w, h)

# dst = dst[y:y+h, x:x+w]
# print(" kich thuoc sau hieu chinh", dst.shape[:2])
# cv2.imshow("img", dst)
# cv2.waitKey(0)

# # Khởi tạo pipeline cho camera RealSense
# pipeline = rs.pipeline()
# config = rs.config()
# config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# # Bắt đầu streaming
# pipeline.start(config)

# try:
#     while True:
#         start_time = time.time()

#         # Đợi và lấy frame từ camera
#         frames = pipeline.wait_for_frames()
#         color_frame = frames.get_color_frame()

#         if not color_frame:
#             continue

#         # Chuyển đổi frame sang mảng numpy
#         color_image = np.asanyarray(color_frame.get_data())
# # キー処理(ESC：終了) #################################################
#         key = cv2.waitKey(1)
#         if key == 27:  # ESC
#             break

#         # 画面反映 #############################################################
#         cv2.imshow('AprilTag Detect Demo', debug_image)

#         # Hiển thị ảnh RGB
#         #cv2.imshow("RGB Image", color_image)
            
#             # Thoát khỏi vòng lặp khi nhấn phím 'ESC'
#         if cv2.waitKey(1) & 0xFF == 27:
#             break

# finally:
#     # Dừng streaming và đóng các kết nối
#     pipeline.stop()
#     cv2.destroyAllWindows()