import matplotlib.pyplot as plt
import pandas as pd

# Đọc dữ liệu từ file txt
df = pd.read_csv('2.txt', delimiter='\t')

# Lấy dữ liệu từ các cột
x_actual = df['x_actual']
y_actual = df['y_actual']
theta_actual = df['theta_actual']


# Vẽ điểm cho mỗi dòng
for index, row in df.iterrows():
    x_value = row['x_actual']/100
    y_value = row['y_actual']/100
    plt.scatter(x_value, y_value, color = "red")

# Thiết lập các thông số cho biểu đồ
# plt.title('Vị trí tại mỗi thời điểm')

# plt.xlabel('x_actual')
# plt.ylabel('y_actual')
# plt.grid(True)
# # Hiển thị biểu đồ
# plt.show()



## LỌC ____________--------------------
# Áp dụng bộ lọc trung bình động với cửa sổ 5
df_smoothed = df.rolling(window=10).mean()

# Vẽ biểu đồ
plt.scatter(df['x_actual'], df['y_actual'], label='Dữ liệu ban đầu', marker= "o", s = 5 )
plt.scatter(df_smoothed['x_actual'], df_smoothed['y_actual'], label='Dữ liệu sau khi áp dụng bộ lọc trung bình động', marker= "o", s = 5)
plt.legend()
plt.show()

## KALMAN

from filterpy.kalman import KalmanFilter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Đọc dữ liệu từ file txt

# Xây dựng mô hình Kalman cho robot vi sai
kf = KalmanFilter(dim_x=3, dim_z=2)

# Xác định ma trận chuyển động A
kf.F = np.array([[1, 1, 0],
                 [0, 1, 1],
                 [0, 0, 1]])

# Xác định ma trận đo H
kf.H = np.array([[1, 0, 0],
                 [0, 1, 0]])

# Khởi tạo giá trị ban đầu và ma trận hiệp phương sai
kf.x = np.array([df['x_actual'][0], df['y_actual'][0], df['theta_actual'][0]])
kf.P *= 1e2

# Tạo danh sách để lưu trữ giá trị dự đoán
predictions = []

# Lặp qua từng bước thời gian
for i in range(len(df)):
    # Dự đoán trạng thái tiếp theo
    kf.predict()

    # Cập nhật trạng thái với đo đạc thực tế
    measurement = np.array([df['x_actual'][i], df['y_actual'][i]])
    kf.update(measurement)

    # Lấy giá trị dự đoán
    prediction = kf.x[:2]
    predictions.append(prediction)

# Chuyển đổi danh sách dự đoán thành DataFrame
df_predictions = pd.DataFrame(predictions, columns=['x_pred', 'y_pred'])

# Vẽ biểu đồ so sánh dữ liệu thực và dự đoán
plt.scatter(df['x_actual'], df['y_actual'], label='Dữ liệu thực tế', marker='o', s = 5)
plt.scatter(df_predictions['x_pred'], df_predictions['y_pred'], label='Dữ liệu dự đoán', marker='o', color='red', s = 5)
plt.legend()
plt.show()


#### LỌC kết thúc






# # Vẽ biểu đồ
# plt.figure(figsize=(10, 6))

# # Biểu đồ x_actual
# plt.subplot(3, 1, 1)
# plt.plot(x_actual/100, label='x_actual')
# plt.title('Biểu đồ x_actual')
# plt.xlabel('Index')
# plt.ylabel('x_actual')
# plt.legend()

# # Biểu đồ y_actual
# plt.subplot(3, 1, 2)
# plt.plot(y_actual/100, label='y_actual')
# plt.title('Biểu đồ y_actual')
# plt.xlabel('Index')
# plt.ylabel('y_actual')
# plt.legend()

# # Biểu đồ theta_actual
# plt.subplot(3, 1, 3)
# plt.plot(theta_actual/100, label='theta_actual')
# plt.title('Biểu đồ theta_actual')
# plt.xlabel('Index')
# plt.ylabel('theta_actual')
# plt.legend()

# # Hiển thị biểu đồ
# plt.tight_layout()
# plt.show()
