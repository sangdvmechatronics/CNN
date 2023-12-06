import cv2
import numpy as np

def calculate_homography(src_points, dst_points):
    # Chuyển đổi điểm thành numpy array
    src_points = np.array(src_points, dtype=np.float32)
    dst_points = np.array(dst_points, dtype=np.float32)

    # Tính ma trận homography
    homography_matrix, _ = cv2.findHomography(src_points, dst_points, cv2.RANSAC, 5.0)

    return homography_matrix
def main():
    tag_corners = [
        (372, 225),
        (429, 227),
        (431, 170),
        (374, 168)
    ]

    # Điểm toạ độ tương ứng trên ảnh (ví dụ: có thể được đo từ thực tế)
    image_corners = [
        (226, 226),
        (283, 227),
        (285, 169),
        (226, 168)
    ]

    # Tính ma trận homography
    homography_matrix = calculate_homography(tag_corners, image_corners)

    #np.savetxt('homography_matrix.txt',homography_matrix)

    # In ma trận homography
    print("Homography Matrix:")
    print(homography_matrix)
if __name__ == '__main__':
    main()






