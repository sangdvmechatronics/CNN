def moving_average(data, window_size):
    result = []
    for i in range(len(data) - window_size + 1):
        window = data[i:i+window_size]
        avg = sum(window) / window_size
        result.append(avg)
    return result

# Sử dụng hàm moving_average để làm mờ dữ liệu
data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
window_size = 5
smoothed_data = moving_average(data, window_size)

print(smoothed_data)
