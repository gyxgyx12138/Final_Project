import csv
import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder

def TransformLabel(data, csv_path):
    user = LabelEncoder()
    user.fit(data[:, 0])
    data[:, 0] = user.transform(data[:, 0])
    
    item = LabelEncoder()
    item.fit(data[:, 1])
    data[:, 1] = item.transform(data[:, 1])
    
    selected_fields = ['reviewerID', 'itemID', 'overall']

    # Kiểm tra xem tệp CSV đã tồn tại chưa
    if not os.path.exists(csv_path):
        # Tạo tệp mới và ghi tiêu đề
        with open(csv_path, 'w', newline='') as csv_data:
            csv_writer = csv.writer(csv_data)
            csv_writer.writerow(selected_fields)

    # Ghi dữ liệu vào tệp CSV
    with open(csv_path, 'a', newline='') as csv_data:  # Mở tệp ở chế độ 'a' để thêm dữ liệu vào cuối tệp
        csv_writer = csv.writer(csv_data)
        for item in data:
            csv_writer.writerow(item)
    return data

def TransformLabel_Deep(data, csv_path):
     # Khởi tạo LabelEncoder cho user
    user_encoder = LabelEncoder()
    reviewer_ids = data[:, 0]
    
    user_encoded = user_encoder.fit_transform(reviewer_ids)
    data[:, 0] = user_encoded
    selected_fields = ['ID', 'Array']
    with open(csv_path, 'w', newline='') as csv_data:
        csv_writer = csv.writer(csv_data)
        csv_writer.writerow(selected_fields)
        csv_writer.writerows(data)
        

def merge_csv_columns(csv_file1, id_column1, csv_file2, id_column2, value_column2, new_column):
    # Đọc dữ liệu từ file CSV thứ hai và ánh xạ ID với giá trị
    id_to_value = {}
    with open(csv_file2, 'r', newline='') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        for row in csv_reader:
            id_value = row[id_column2]
            value = row[value_column2]
            id_to_value[id_value] = value

    # Đọc dữ liệu từ file CSV đầu tiên và cập nhật dữ liệu trên đó
    updated_rows = []
    with open(csv_file1, 'r', newline='') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        for row in csv_reader:
            id_value = row[id_column1]
            if id_value in id_to_value:
                row[new_column] = id_to_value[id_value]
            else:
                row[new_column] = ''
            updated_rows.append(row)

    # Ghi dữ liệu đã cập nhật trở lại vào file CSV đầu tiên
    with open(csv_file1, 'w', newline='') as csv_file:
        fieldnames = updated_rows[0].keys()
        csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        csv_writer.writeheader()
        csv_writer.writerows(updated_rows)

