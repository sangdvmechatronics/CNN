#!/usr/bin/python3
print("-----------------------")

from build_lib.bdk_PID import *
from build_lib.algorithm_position import *
from build_lib.find_ROI import *
from build_lib.input_images import *
from build_lib.info_landmarks import *
from build_lib.trajectory import *
from build_lib.run_parameter import *
from build_lib.lib import *

def program_main():

    K_camera_matrix ,loaded_mtx, loaded_dist = configure_camera()
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 848, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)

    pipeline.start(config)
    actual_position = [[0],[0],[0]]
    time_input_start = later = later_1 = time.time()
    t_i= 0
    time_input_start = time.time()
    time.sleep(3)
    #out = save_video()
    cap = cv2.VideoCapture("4.mp4")

    try:
        while not exit_flag:
            time.sleep(0.1)
            start_time = time.time()
            time_input_end = time.time()
            t_i =  t_i + (time_input_end - time_input_start) if (time_input_end - time_input_start) else 1
            print(f"Time: {round(t_i,2)} (s)\n")
            time_input_start = time_input_end
            # Đợi và lấy frame từ camera
            #frames = pipeline.wait_for_frames()
            _, frames = cap.read()
            frames = cv2.resize(frames, (848,480))
            cv2.imshow("video", frames)
            color_image = frames.copy()

            #color_frame = frames.get_color_frame()
            color_image = get_images(color_image, K_camera_matrix ,loaded_mtx, loaded_dist)
            #depth_frame = frames.get_depth_frame()  # Lấy frame độ sâu
            color_frame = color_image.copy()
            #color_frame = cv2.imread("build_lib/april_tag_19.jpg")
            # if not color_frame :
            #     continue
            #color_image, depth_frame = get_images(color_frame, depth_frame, K_camera_matrix ,loaded_mtx, loaded_dist)
            
            #out.write(frames)
            color_image_copy = color_image.copy()

            # cv2.imshow("color_image", color_image_copy)
            # if cv2.waitKey(1) == ord('q'):
            #          break
            pre_landmarks = detect_landmark(color_image_copy, show = True)
            if pre_landmarks is not None:
                center_list, point_x, tag_id, dpi_x, dpi_y = pre_landmarks.run_detect_tag()
                #print(f"center_list: {center_list}\n point_x: {point_x}\tag: {tag_id} \n dpi: {dpi}")
                if len(center_list) > 0:
                    robot = calculate_position(center_list, point_x, tag_id, dpi_x, dpi_y)
                    distance_robot, min_cx, min_cy = robot.calculate_distances()
                    cropped_img = crop_ROI_tags(color_image, min_cx, min_cy)

                    # print(f"toạ độ tâm: {min_cx}, min_cy: {min_cy}")
                    # print(f"distance_robot: {distance_robot}")
                    ### Lấy thêm thông tin để khẳng định độ chắc chắn dự đoán dựa trên chiều cao
                    #depth_value = depth_frame.get_distance(min_cx, min_cy) 
                    #print(f"depth_value: {round(depth_value,2)}")
                    pos_robot_real = robot.run_position(cropped_img)
                    
                    if pos_robot_real is not None:
                        actual_position = pos_robot_real
                        # previous_position = actual_position
                    # else:
                        # actual_position = previous_position

                    # run = check_array(actual_position)
                    # rospy.init_node("publisher_bdk_ho_txt", anonymous= True)
                    # pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
                    # rate = rospy.Rate(10)

                    x_actual = actual_position[0][0]
                    y_actual = actual_position[1][0]
                    theta_actual = actual_position[2][0]      
                    print(f"x_thuc {round(x_actual,2)}, y_thuc {round(y_actual,2)}, theta_thuc {round(theta_actual,2)} ")
                    end_time = time.time()
                    time_xl = end_time- start_time

                    #cv2.imshow("RGB Image", color_image)
                    signal.signal(signal.SIGINT, tat_chuong_trinh)
                    if exit_flag is True:
                        break

                    if cv2.waitKey(1) == ord('q'):
                        break
            else:
                print("Error detect landmarks...")
    except KeyboardInterrupt:
            print("End ................")
    finally:
        # Dừng streaming và đóng các kết nối
            print("Releasing...")
            pipeline.stop()
            #out.release()
            cv2.destroyAllWindows()



if __name__=="__main__":
    global exit_flag
    exit_flag = False 
    program_main()

