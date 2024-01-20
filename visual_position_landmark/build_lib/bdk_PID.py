#!/usr/bin/python3
import numpy as np
import time

class PIDController():
    
    def __init__(self, x_d, y_d, theta_d, x_actual, y_actual, theta_actual, V_d, oz):
        self.e_x_erorr = x_d - x_actual
        self.e_y_erorr = y_d - y_actual
        self.e_theta_erorr = theta_d - theta_actual
        self.V_d  = V_d
        self.oz = oz
        self.theta_actual = theta_actual
        #print(self.theta_actual)

    def transform_erorr(self):
        e = np.array([[self.e_x_erorr],[self.e_y_erorr], [self.e_theta_erorr]])
        o = self.theta_actual
        matrix_R = np.array([[np.cos(o), -np.sin(o), 0], 
                            [np.sin(o), np.cos(o), 0],
                            [0, 0, 1]])
        matrix_R_T = np.transpose(matrix_R)
        self.e_r = np.dot(matrix_R_T, e)
        #print(self.e_r)

    def parameter_PID(self):
        self.e_rx = self.e_r[0][0]
        self.e_ry = self.e_r[1][0]
        self.e_rtheta = self.e_r[2][0]
        #print(self.e_ry)
        e = np.sqrt(self.e_rx*self.e_rx + self.e_ry*self.e_ry)
        a1 = 0.5
        a2 = 0.5
        b1 = 0.003
        b2 = 0.003
        c1 = 0.05
        c2 = 0.05

        self.Kp =  a1 + a2*e  ######  có thể xem xétĐang chọn dấu ngược so với tài liệu, biểu diễn các hệ số âm nên đảo chiều dấu
        self.Ki =  b1 - b2*e
        self.Kd =  c1 + c2*e
        self.v_max = 8    ### thực nghiệm nên cho nhỏ
        self.w_max = 8    ### thực nghiệm nên cho nhỏ

    def PID_1(self, later):  ## khai bao later = tai dau chuong trinh chay
        now = time.time()
        ## Create parameter
        dt = now - later if (later- now)  else 1e-16
        #print("time1", dt)
        erorr_x = 0
        interal_erorr_x = 0  # tích phân
        erorr_last_x = 0
        derivative_erorr_x = 0
        e_V_x = 0

        erorr_x = self.e_rx
        interal_erorr_x += erorr_x*dt 
        derivative_erorr_x = (erorr_x - erorr_last_x)/dt
        erorr_last_x = erorr_x

        self.e_V_x = self.Kp * erorr_x + self.Ki *interal_erorr_x + self.Kd * derivative_erorr_x
        
        if self.e_V_x >= self.v_max:
            self.e_V_x = self.v_max ## tinh loi u1
        later = now
        return later

    def PID_2(self, later_1): ## tinh loi omega ## khai bao later1 tai dau ct
        
        now_1 = time.time()
        dt = now_1 - later_1 if (now_1 - later_1) else 1e-16

        #print("time", dt)
        erorr_y = 0
        erorr_theta = 0
        
        interal_erorr_y = 0 
        interal_erorr_theta = 0

        erorr_last_y = 0
        erorr_last_theta = 0

        derivative_erorr_y = 0 ## Khởi tạo đạo hàm ey, e_theta =0
        derivative_erorr_theta = 0

        e_theta_derivative = 0
        ## Compute with e_ry
        erorr_y = self.e_ry
        interal_erorr_y += erorr_y*dt   # tính tích phân tích lũy
        derivative_erorr_y = (erorr_y - erorr_last_y)/dt # tính đạo hàm bằng giá trị trước - giá trị sau
        erorr_last_y = erorr_y   # cập nhật giá trị lỗi qua từ vòng
        ## Compute with e_rtheta
        erorr_theta = self.e_rtheta
        interal_erorr_theta += erorr_theta*dt
        derivative_erorr_theta = (erorr_theta - erorr_last_theta)/dt
        erorr_last_theta = erorr_theta
        ## Compute e_theta_cham
        self.e_theta_derivative = (self.Kp * erorr_y + self.Ki *interal_erorr_y + self.Kd * derivative_erorr_y) + \
            (self.Kp * erorr_theta + self.Ki *interal_erorr_theta + self.Kd * derivative_erorr_theta)

        if self.e_theta_derivative >= self.w_max:
            self.e_theta_derivative = self.w_max ### tinh loi cua u2
        later_1 = now_1
        return later_1
    def compute_VG_theta_send_control(self):
        self.V_G = self.V_d + self.e_V_x   ## van toc dieu khien
        self.omega_controll =  self.oz + self.e_theta_derivative     ## omega dieu khien
        return self.V_G, self.omega_controll

    def run_PID(self, later, later_1):
        self.transform_erorr()
        self.parameter_PID()

        later = self.PID_1(later) 
        ## later_1 , later khai báo đầu chương trình trước while
        later_1  = self.PID_2(later_1)
        self.V_G, self.omega_controll = self.compute_VG_theta_send_control()
        return self.V_G, self.omega_controll



# PID_1 = PIDController(2,2,0.3, 1.9,1.9,0.25, 0.3, 0.3)
# later = later_1 = time.time()
# V_G, omega_controll = PID_1.run_PID(later, later_1)
#print(V_G, omega_controll)