import tkinter as tk
from tkinter import messagebox, Toplevel, ttk
from PIL import Image, ImageTk
import joblib
import os

# === Hàm dự đoán với mô hình Random Forest ===
def predict_phishing():
    try:
        missing_features = []
        features = []

        # Lấy giá trị từ các combobox
        for i, combobox in enumerate(comboboxes):
            value = combobox.get()
            if value:
                features.append(int(value.split("(")[1].strip(")").strip()) + 1)
            else:
                # Nếu không có giá trị, mặc định là kết quả xấu
                features.append(0)
                missing_features.append(features_list[i][0])

        # Load mô hình Random Forest
        rf_model = joblib.load("random_forest_model.pkl")

        # Dự đoán với mô hình
        prediction_rf = rf_model.predict([features])[0]

        # Tạo thông báo kết quả
        result_message = "🔍 Kết quả dự đoán:\n\n"
        result_message += f"📌 Random Forest: {'KHÔNG LỪA ĐẢO ✅' if prediction_rf == 2 else 'LỪA ĐẢO ⚠️'}\n"

        if missing_features:
            result_message += "\n⚠️ Các thuộc tính thiếu (mặc định kết quả xấu):\n"

        # Tạo cửa sổ thông báo lớn hơn
        result_window = Toplevel(root)
        result_window.title("Kết quả dự đoán")
        result_window.geometry("500x300")

        label = tk.Label(result_window, text=result_message, font=("Arial", 14), justify="left")
        label.pack(padx=20, pady=20)

        close_button = ttk.Button(result_window, text="Đóng", command=result_window.destroy)
        close_button.pack(pady=10)

    except Exception as e:
        messagebox.showerror("Lỗi", str(e))

# === Giao diện Tkinter ===
root = tk.Tk()
root.title("Dự đoán trang web lừa đảo")

# Tạo kiểu với font lớn
style = ttk.Style()
style.configure("Large.TButton", font=("Arial", 16))
style.configure("Large.TLabel", font=("Arial", 14))
style.configure("Large.TCombobox", font=("Arial", 14))

# Danh sách các câu hỏi và giá trị (đã tăng lên 1 đơn vị)
features_list = [
    ("URL chứa địa chỉ IP?", ["Không (2)", "Có (0)"]),
    ("Độ dài URL?", ["Ngắn (2)", "Trung bình (1)", "Dài (0)"]),
    ("Dùng dịch vụ rút gọn URL?", ["Không (2)", "Có (0)"]),
    ("URL có ký tự @?", ["Không (2)", "Có (0)"]),
    ("Có // bất thường?", ["Không (2)", "Có (0)"]),
    ("Tên miền có dấu -?", ["Không (2)", "Có (0)"]),
    ("Số subdomain?", ["Ít (2)", "Trung bình (1)", "Nhiều (0)"]),
    ("Trạng thái SSL?", ["Hợp lệ (2)", "Không rõ (1)", "Không hợp lệ (0)"]),
    ("Thời gian đăng ký?", ["Dài (2)", "Ngắn (0)"]),
    ("Nguồn favicon?", ["Cùng miền (2)", "Khác (0)"]),
    ("Cổng kết nối?", ["Chuẩn (2)", "Bất thường (0)"]),
    ("Tên miền có 'https'?", ["Không (2)", "Có (0)"]),
    ("Nguồn nội dung?", ["Cùng miền (2)", "Khác (0)"]),
    ("Các liên kết trong trang trỏ đến?", ["Cùng trang (2)", "Một số khác (1)", "Nhiều trang khác (0)"]),
    ("Thẻ script/link?", ["Cùng trang (2)", "Một số khác (1)", "Nhiều khác (0)"]),
    ("Xử lý form?", ["Cùng miền (2)", "Không rõ (1)", "Khác (0)"]),
    ("Gửi thẳng email?", ["Không (2)", "Có (0)"]),
    ("URL khớp tên miền?", ["Có (2)", "Không (0)"]),
    ("Số chuyển hướng?", ["Nhiều (2)", "Ít (1)"]),
    ("Thay đổi khi hover?", ["Không (2)", "Có (0)"]),
    ("Cho phép chuột phải?", ["Có (2)", "Không (0)"]),
    ("Cửa sổ popup?", ["Không (2)", "Có (0)"]),
    ("Dùng iframe ẩn?", ["Không (2)", "Có (0)"]),
    ("Tuổi tên miền?", ["Cũ (2)", "Mới (0)"]),
    ("DNS hợp lệ?", ["Có (2)", "Không (0)"]),
    ("Lưu lượng truy cập?", ["Cao (2)", "Trung bình (1)", "Thấp (0)"]),
    ("PageRank?", ["Cao (2)", "Thấp (0)"]),
    ("Google Index?", ["Có (2)", "Không (0)"]),
    ("Các trang web khác liên kết trỏ đến trang này?", ["Nhiều (2)", "Một số (1)", "Ít (0)"]),
    ("Trong danh sách phishing?", ["Không (2)", "Có (0)"])
]

comboboxes = []

# Tạo frame chứa nội dung chính với thanh cuộn
canvas = tk.Canvas(root)
scrollbar = ttk.Scrollbar(root, orient="vertical", command=canvas.yview)
scrollable_frame = ttk.Frame(canvas)

scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))

canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
canvas.configure(yscrollcommand=scrollbar.set)

for i in range(0, len(features_list), 2):
    row = i // 2
    question_left, options_left = features_list[i]

    ttk.Label(scrollable_frame, text=question_left, style="Large.TLabel").grid(row=row, column=0, padx=20, pady=5, sticky="w")
    combo_left = ttk.Combobox(scrollable_frame, values=options_left, state="readonly", width=25, style="Large.TCombobox")
    combo_left.grid(row=row, column=1, padx=20, pady=5, sticky="w")
    comboboxes.append(combo_left)

    if i + 1 < len(features_list):
        question_right, options_right = features_list[i + 1]

        ttk.Label(scrollable_frame, text=question_right, style="Large.TLabel").grid(row=row, column=3, padx=20, pady=5, sticky="w")
        combo_right = ttk.Combobox(scrollable_frame, values=options_right, state="readonly", width=25, style="Large.TCombobox")
        combo_right.grid(row=row, column=4, padx=20, pady=5, sticky="w")
        comboboxes.append(combo_right)

canvas.pack(side="top", fill="both", expand=True)
scrollbar.pack(side="right", fill="y")

predict_btn = ttk.Button(root, text="🚀 Dự đoán", command=predict_phishing, style="Large.TButton")
predict_btn.pack(pady=20)

root.mainloop()
