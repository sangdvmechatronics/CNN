import matplotlib.pyplot as plt
import numpy as np

# Đọc dữ liệu từ file txt
file_path = "data_angle.txt"
data = np.loadtxt(file_path, skiprows=0)  # Bỏ qua dòng tiêu đề

# Tách dữ liệu thành cột tương ứng với thời gian, góc X, Y, Z
time_run = data[:, 0]
angle_x = data[:, 1]
angle_y = data[:, 2]
angle_z = data[:, 3]

# Tạo 3 biểu đồ trong 1 figure
fig, axs = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

# Biểu đồ cho góc X
axs[0].plot(time_run, angle_x, label='Góc X', color='r')
axs[0].set_ylabel('Góc X (độ)')
axs[0].legend()

# Biểu đồ cho góc Y
axs[1].plot(time_run, angle_y, label='Góc Y', color='g')
axs[1].set_ylabel('Góc Y (độ)')
axs[1].legend()

# Biểu đồ cho góc Z
axs[2].plot(time_run, angle_z, label='Góc Z', color='b')
axs[2].set_xlabel('Thời gian (s)')
axs[2].set_ylabel('Góc Z (độ)')
axs[2].legend()

# Đặt tiêu đề chung cho toàn bộ figure
fig.suptitle('Biểu đồ góc quay theo thời gian')

# Hiển thị biểu đồ
plt.show()