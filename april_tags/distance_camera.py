import pyrealsense2 as rs
import numpy as np
import cv2

# Khởi tạo pipeline cho camera RealSense D435
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

# Bắt đầu streaming
pipeline.start(config)

try:
    while True:
        # Đợi và lấy frame từ camera
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()

        if not depth_frame:
            continue

        # Lấy thông tin chiều rộng và chiều cao của ảnh depth
        width = depth_frame.get_width()
        height = depth_frame.get_height()

        # Lấy độ sâu tại tâm ảnh (pixel tại giữa ảnh)
        center_x = int(width / 2)
        center_y = int(height / 2)
        depth_value = depth_frame.get_distance(center_x, center_y)

        # In ra giá trị độ sâu
        print(f"Depth value at center: {depth_value} meters")

        # Hiển thị ảnh depth (optional)
        depth_image = np.asanyarray(depth_frame.get_data())
        cv2.imshow("Depth Image", depth_image)
        key = cv2.waitKey(1)
        if key == 27:  # Phím Escape để thoát
            break

finally:
    # Dừng streaming và đóng cửa sổ OpenCV (nếu có)
    pipeline.stop()
    cv2.destroyAllWindows()
