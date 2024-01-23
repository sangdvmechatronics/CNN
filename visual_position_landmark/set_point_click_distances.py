import cv2

def mouse_click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"Clicked at coordinates (x, y): ({x}, {y})")
        # Thêm mã lệnh in toạ độ lên ảnh
        cv2.putText(image, f"({x}, {y})", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.imshow("Image", image)

def main():
    # Đọc hình ảnh
    image_path = "img/landmark.jpg"
    global image
    image = cv2.imread(image_path)

    if image is None:
        print("Error: Could not read the image.")
        return

    # Tạo cửa sổ để hiển thị hình ảnh và kết nối sự kiện chuột
    cv2.namedWindow("Image")
    cv2.setMouseCallback("Image", mouse_click_event)

    while True:
        # Hiển thị hình ảnh
        cv2.imshow("Image", image)

        # Đợi người dùng nhấn phím ESC để thoát
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break

    # Đóng cửa sổ khi kết thúc
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
