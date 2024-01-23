#!/usr/bin/python3

import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
import time
from build_lib.info_landmarks import *
from build_lib.find_ROI import *


class calculate_position():
    
    def __init__(self,center_list, point_x, tag_id, dpi_x, dpi_y):

        self.center_list = center_list
        self.point_x = point_x
        self.count = tag_id
        self.info_landmarks = info_landmarks(self.count)
        self.dpi_x = dpi_x
        self.dpi_y = dpi_y

    # Tính toán khoảng cách 
    def calculate_distances(self):
        distances = []
        # đọc các tâm tìm trong toàn bộ ảnh rồi tính toán khoảng cách tới tâm ảnh
        for i , (centers) in enumerate(self.center_list):
            cx, cy = centers
            ## Đnag điều chỉnh để tâm dán về trùng với tâm ảnh
            cx = cx 
            cy = cy 
            #### Cộng thêm quá trình hiệu chỉnh tâm robot về tam ảnh
            vx = cx - (848 / 2)              
            vy = cy - (480 / 2)
            distance = np.sqrt(vx**2 + vy**2)
            distances.append(distance) 
        # Tìm khoảng cách ngắn nhất và lấy thông tin
        distance_robot = min(distances)
        min_index = distances.index(distance_robot)
        min_cx, min_cy = self.center_list[min_index]

        distance_robot *= self.dpi_x ### thực hiện hiểu chỉnh với thực tế
        return distance_robot, min_cx, min_cy

    # Tạo vectors ngắn nhất từ tâm đường tròn tới đường tròn  gần tâm ảnh crop nhất
    ## Dùng để tính theta
    def create_vector_X_crop(self):
        # Tâm ảnh crop thực hiển điều chỉnh cắt
        cx = 20
        cy = 20
        vector_X_cropped = None
        if len(self.point_x_2) > 0:
            center1 = self.point_x_2[0] # lấy tọa độ của điểm đầu tiên trong mảng
            center2 = self.point_x_2[1]# lấy tọa độ của điểm thứ hai trong mảng
            #print(center1, center2)
            vector_X_cropped = np.array([center1-cx, center2-cy])
        else:
            vector_X_cropped = None
        #print("vector", vector_X_cropped)
        return vector_X_cropped
    # Tạo vectors AB ( từ tâm hình tròn tới tâm ảnh) ( rB -rA)
    def create_vector_rAB(self, min_cx, min_cy):
        #### Cộng thêm quá trình hiệu chỉnh tâm robot về tam ảnh
        r_BA = np.array([848/2 - min_cx, 480/2  - min_cy])
        #print("r_BA", r_BA)
        return r_BA

    # Tính góc trục x (ảnh cropped) với phương ngang của ảnh
    def theta(self, vector_X_cropped):
        #print('check vector', vector_X_cropped)
        theta_val_rad = np.arctan2(vector_X_cropped[1], vector_X_cropped[0]) 
        theta_val_degree = np.degrees(theta_val_rad)  # Chuyển đổi sang đơn vị độ
        #print("góc: ", theta_val_rad)


        theta_val_rad = theta_val_rad 
        #print("góc: ", theta_val_rad)
        return theta_val_rad
    
    def pos_tranform_robot_to_landmank(self, theta_val_rad, r_BA):
    ### chuyển từ độ về radian và tôi hỏi chấm hỏi chấm thực hiện quay ngược theta, vì đang quy ước robot đúng như hê tọa độ tuyệt dối
        c = np.cos(theta_val_rad)
        s = np.sin(theta_val_rad)
        #print("góc xoay", theta)
        ## thực hiện tính theta theo chiều dương cùng chiều kim đồng hồ trục z hướng lên. đâm vào màn hình
        R1 = np.array([[c, -s , 0, 0],
                    [s, c, 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]])
        r_BA = np.array([[r_BA[0]],
                    [r_BA[1]],
                    [0],
                    [1]])
        R1_nghich_dao = np.linalg.inv(R1)
        r_B_i =  np.dot(R1_nghich_dao, r_BA) 
        return r_B_i 
    ## Run
    def pos_robot(self, r_B_i,r_BA ,theta_val_rad, vector_X_cropped):
        if self.point_x is not None:
            distance_robot, min_cx, min_cy = self.calculate_distances()
            if vector_X_cropped is not None:
                r_BA = self.create_vector_rAB(min_cx, min_cy)
                theta_val = theta_val_rad

            else:
                r_BA = None
                theta_val= None
        if r_BA is not None and theta_val is not None:
            r_B_i[0] *= self.dpi_x 
            r_B_i[1] *= self.dpi_y 
        else:
            r_B_i = None
        return r_B_i , theta_val
    ## Thực hiện tính toán vị trí robot trong hệ tọa độ thực toàn cầu
    def pos_robot_to_global(self, r_B_i , theta_val):
        tag_id = self.count
        pos_robot_real = None
        if tag_id is not None:
            T, r_O_i, phi = self.info_landmarks
            if r_B_i is not None:
                #print("hệ quy chiếu gắn với là: ", r_O_i)
                #print("ma trận gắn với là ", T)
                r_B_g = r_O_i + np.dot(T, r_B_i)
                theta_val = theta_val ### Do đang bị ngược với quy ước của arctan2 có thẻ thay đổi 
                phi_r = phi - theta_val - np.pi/2
                pos_robot_real = [[r_B_g[0, 0]], [r_B_g[1,0]], [phi_r]]
        print("Pose_theta: ", np.rad2deg(phi_r))
        return pos_robot_real
    def run_position(self, cropped_img):
        try:

            distance_robot, min_cx, min_cy = self.calculate_distances()
            

            pre2_landmarks = detect_landmark(cropped_img, show = False)
            center_list_2, point_x_2, tag_id_2, dpi_x_2, dpi_y_2 = pre2_landmarks.run_detect_tag()
            self.point_x_2 = point_x_2
            vector_X_cropped = self.create_vector_X_crop()
            #print(vector_X_cropped)

            r_BA = self.create_vector_rAB(min_cx, min_cy)
            theta_val_rad = self.theta(vector_X_cropped)

            r_B_i = self.pos_tranform_robot_to_landmank(theta_val_rad, r_BA)
            r_B_i , theta_val = self.pos_robot(r_B_i, r_BA, theta_val_rad, vector_X_cropped)

            #print(theta_val_rad)
            result = self.pos_robot_to_global(r_B_i , theta_val)
            ##print(result)
            return result
        except Exception as e:
            print(f"Error: {e}")
            result = None
            return result

# robot_1 = calculate_position([(130,150)],(200,250), 1)

# result = robot_1.run_position()
# print(result)
