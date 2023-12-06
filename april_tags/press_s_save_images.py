import cv2
import os

# Tạo thư mục để lưu ảnh
output_folder = 'images_calib'
os.makedirs(output_folder, exist_ok=True)

# Đối tượng VideoCapture cho camera Intel
cap = cv2.VideoCapture(2)  # 0 là chỉ số của camera, có thể thay đổi nếu bạn có nhiều camera

# Thiết lập độ phân giải
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
# Biến đếm số thứ tự ảnh
image_count = 1

while True:
    # Đọc frame từ camera
    ret, frame = cap.read()

    # Hiển thị frame
    cv2.imshow("Intel Camera", frame)

    # Kiểm tra xem người dùng có ấn phím "s" không
    key = cv2.waitKey(1)
    if key == ord('s'):
        # Lưu ảnh với số thứ tự tăng dần
        image_path = os.path.join(output_folder, f'image_{image_count}.png')
        cv2.imwrite(image_path, frame)
        print(f"Captured image {image_count} and saved to {image_path}")
        image_count += 1

    # Kiểm tra xem người dùng có ấn phím "ESC" không
    elif key == 27:
        break

# Giải phóng tài nguyên
cap.release()
cv2.destroyAllWindows()
