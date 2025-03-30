from scipy.io import arff
import pandas as pd
import re

def remove_duplicates_and_increment(input_file, output_file):
    # Đọc file ARFF
    data, meta = arff.loadarff(input_file)

    # Chuyển dữ liệu thành DataFrame
    df = pd.DataFrame(data)

    # Giải mã các giá trị kiểu byte
    df = df.applymap(lambda x: x.decode('utf-8') if isinstance(x, bytes) else x)

    # Chuyển đổi các cột có thể thành kiểu số
    for column in df.columns:
        try:
            df[column] = pd.to_numeric(df[column], errors='ignore')
        except Exception as e:
            print(f"Không thể chuyển đổi cột {column}: {e}")

    # Loại bỏ các dòng trùng lặp
    df_cleaned = df.drop_duplicates()

    # Tịnh tiến các giá trị số lên 1 đơn vị
    for column in df_cleaned.select_dtypes(include='number').columns:
        df_cleaned[column] += 1

    # Ghi lại dữ liệu vào file ARFF mới
    with open(output_file, "w", encoding="utf-8") as f:
        # Ghi phần header của ARFF
        f.write(f"@RELATION {meta.name}\n\n")

        for attribute in meta.names():
            col_data = df_cleaned[attribute]

            # Nếu kiểu nominal, liệt kê chính xác giá trị từ dữ liệu
            if meta[attribute][0] == 'nominal':
                unique_values = sorted(set(col_data.astype(str)))
                f.write(f"@ATTRIBUTE {attribute} {{{','.join(unique_values)}}}\n")
            else:
                f.write(f"@ATTRIBUTE {attribute} {meta[attribute][0]}\n")

        f.write("\n@DATA\n")

        # Ghi dữ liệu đã xử lý
        for row in df_cleaned.itertuples(index=False):
            f.write(",".join(map(str, row)) + "\n")

# Sử dụng hàm
input_path = "Training Dataset.arff"  # Thay bằng đường dẫn file đầu vào
output_path = "Training_Dataset_Cleaned_Incremented.arff"
remove_duplicates_and_increment(input_path, output_path)

print(f"File đã xử lý được lưu tại: {output_path}")
