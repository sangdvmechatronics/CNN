#!/usr/bin/python3

import cv2
import numpy as np
import os

def crop_ROI_tags(img, min_cx, min_cy):
    if os.path.exists('img/landmark.jpg'):
        os.remove('img/landmark.jpg')
    x1, y1 = int(min_cx - 20), int(min_cy - 20)
    x2, y2 = int(min_cx + 20), int(min_cy + 20)
    if x1 >= 0 and y1 >= 0 and x2 <= img.shape[1] and y2 <= img.shape[0]:

        cropped_img = img[y1:y2, x1:x2]
        # Lưu ảnh vào thư mục img
        cv2.imwrite('img/landmark.jpg', cropped_img)
        return cropped_img

 
def configure_camera():

    K_camera_matrix = np.load('build_lib/camera_matrix_lib/848_480_new_camera_matrix_18_12.npy')  
    
    #print("K_camera_matrix", K_camera_matrix)
    loaded_mtx = np.load('build_lib/camera_matrix_lib/848_480_camera_matrix_18_12.npy')
    # Đọc distortion coefficients từ file
    loaded_dist = np.load('build_lib/camera_matrix_lib/848_480_dist_coeffs_18_12.npy')
    
    return K_camera_matrix ,loaded_mtx, loaded_dist


def get_images(color_image, K_camera_matrix ,loaded_mtx, loaded_dist):
    # depth_image = np.asanyarray(depth_frame.get_data())
    # color_image = np.asanyarray(color_image.get_data())
    ## thực hiện undistorted_image
    undistorted_img = cv2.undistort(color_image, loaded_mtx, loaded_dist, None, K_camera_matrix)
    x, y, w, h = 0, 0, 847, 479
    color_image = color_image[y:y+h, x:x+w]
    return color_image


##Test camera

# K_camera_matrix ,loaded_mtx, loaded_dist = configure_camera()
# pipeline = rs.pipeline()
# config = rs.config()
# config.enable_stream(rs.stream.color, 848, 480, rs.format.bgr8, 30)
# pipeline.start(config)
# while True:
#     # Đợi và lấy frame từ camera
#     frames = pipeline.wait_for_frames()
#     color_frame = frames.get_color_frame()
#     if not color_frame:
#         continue
#     #color_image = np.asanyarray(color_frame.get_data())
#     color_image = get_images(color_frame, K_camera_matrix ,loaded_mtx, loaded_dist)

#     # Hiển thị ảnh RGB
#     cv2.imshow("RGB Image", color_image)
#     # print(color_image.shape)
#     key = cv2.waitKey(1)
#     if key == 27:  # ESC
#         break
#     # Dừng streaming và đóng các kết nối
# pipeline.stop()
# cv2.destroyAllWindows()

