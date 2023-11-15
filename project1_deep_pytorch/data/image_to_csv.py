##### chuyển dữ liệu hình ảnh sang csv và loại ảnh
import os
import pandas as pd

# Đường dẫn đến các thư mục ảnh
folder_paths = {'cats':0, 'dogs':1, 'horses':2}

# Tạo một list chứa các DataFrame cho từng loại ảnh
dfs = []

for folder_path, class_images in folder_paths.items():
    # List các tệp ảnh trong thư mục
    image_files = [f for f in os.listdir(folder_path) if f.endswith(('.jpg', '.png', '.jpeg'))]

    # Tạo DataFrame cho loại ảnh này
    df = pd.DataFrame({
        'Image_Name': image_files,
        'Image_Path': [os.path.join(folder_path, img) for img in image_files],
        "Class": class_images
    })

    # Thêm DataFrame vào list
    dfs.append(df)

# Ghép nối các DataFrame
merged_df = pd.concat(dfs, ignore_index=True)

# Lưu DataFrame vào CSV
csv_path = 'animals.csv'
merged_df.to_csv(csv_path, index=False)

