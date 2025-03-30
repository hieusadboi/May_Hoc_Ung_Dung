import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from scipy.io import arff
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import CategoricalNB

# === 1. Load dữ liệu từ file ARFF ===
def load_data(filepath):
    data, meta = arff.loadarff(filepath)
    df = pd.DataFrame(data)
    df = df.applymap(lambda x: x.decode() if isinstance(x, bytes) else x)
    df = df.apply(pd.to_numeric, errors='coerce')
    df.dropna(inplace=True)
    return df

data_file = "Training_Dataset_Cleaned_Incremented.arff"
df = load_data(data_file)

# === 2. Tách tập dữ liệu ===
X = df.drop(columns=["Result"])
y = df["Result"]

count_0 = 0
count_2 = 0

for value in y:
    if value == 0:
        count_0 += 1
    elif value == 2:
        count_2 += 1

print(f"📊 Số lượng lừa đảo ( 0 ): {count_0}")
print(f"📊 Số lượng Không lừa đảo ( 2 ): {count_2}")

# Vẽ biểu đồ tròn
data = [count_0, count_2]
labels = ["Lừa đảo (0)", "Không lừa đảo (2)"]
colors = ["red", "green"]

plt.figure(figsize=(7, 7))
wedges, _, autotexts = plt.pie(data, autopct='%1.1f%%', colors=colors, startangle=90)

# Định dạng văn bản trong biểu đồ
for text in autotexts:
    text.set_fontsize(12)
    text.set_color("black")

# Thêm chú thích
plt.legend(wedges, [f"Lừa đảo (0) - {count_0}", f"Không lừa đảo (2) - {count_2}"], title="Chú thích", loc="best")
plt.title("Tỷ lệ lừa đảo và không lừa đảo")
plt.axis("equal")  # Đảm bảo hình tròn không bị méo
plt.savefig("fraud_pie_chart.png", dpi=300)
plt.show()


# === 3. Chia train / test (70% train, 30% test) ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# ============================
#  🔹 Mô hình KNN
# ============================

k_values = range(1, 30)
accuracies = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k, metric='hamming')
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    accuracies.append(acc)

best_k = k_values[np.argmax(accuracies)]
knn_best = KNeighborsClassifier(n_neighbors=best_k, metric='hamming')
knn_best.fit(X_train, y_train)
y_pred_knn = knn_best.predict(X_test)

accuracy_knn = accuracy_score(y_test, y_pred_knn)
cm_knn = confusion_matrix(y_test, y_pred_knn)

print(f"🔹 Độ chính xác KNN (K={best_k}): {accuracy_knn:.5f}")
print("📌 Ma trận nhầm lẫn (KNN):\n", cm_knn)

# joblib.dump(knn_best, "knn_model.pkl")

plt.figure(figsize=(10, 5))
plt.plot(k_values, accuracies, marker='o', linestyle='dashed', color='b', label="Độ chính xác KNN")
plt.axvline(x=best_k, color='r', linestyle='--', label=f"K tối ưu: {best_k}")
plt.text(best_k, max(accuracies), f"{best_k}", verticalalignment='bottom', horizontalalignment='right', fontsize=12, color='black')
plt.xlabel("Số hàng xóm (k)")
plt.ylabel("Độ chính xác")
plt.title("Chọn k tối ưu cho thuật toán KNN")
plt.legend()
plt.grid(True)
plt.savefig("knn_accuracy.png", dpi=300)
# plt.show()

# ============================
#  🔹 Mô hình Naive Bayes
# ============================
nb_model = CategoricalNB()
nb_model.fit(X_train, y_train)
y_pred_nb = nb_model.predict(X_test)

accuracy_nb = accuracy_score(y_test, y_pred_nb)
cm_nb = confusion_matrix(y_test, y_pred_nb)

print(f"🔹 Độ chính xác Naive Bayes: {accuracy_nb:.5f}")
print("📌 Ma trận nhầm lẫn (Naive Bayes):\n", cm_nb)

# joblib.dump(nb_model, "naive_bayes_model.pkl")


# ============================
#  🔹 Mô hình Random Forest
# ============================

n_estimators_range = range(100, 201)
max_depth_range = range(2, 21)
min_samples_split_range = range(2, 21)
min_samples_leaf_range = range(1, 21)

# Lưu kết quả độ chính xác
accuracy_results = {}

# 📌 1. Tối ưu số lượng cây con (n_estimators)
accuracy_results["n_estimators"] = []
for n in n_estimators_range:
    acc = accuracy_score(y_test, RandomForestClassifier(n_estimators=n, random_state=42).fit(X_train, y_train).predict(X_test))
    accuracy_results["n_estimators"].append(acc)

best_n = n_estimators_range[np.argmax(accuracy_results["n_estimators"])]
plt.figure(figsize=(10, 5))
plt.plot(n_estimators_range, accuracy_results["n_estimators"], marker='o', linestyle='dashed', color='blue')
plt.axvline(x=best_n, color='r', linestyle='--', label=f"Tối ưu: {best_n}")
plt.text(best_n, max(accuracy_results["n_estimators"]), f"{best_n}", verticalalignment='bottom', horizontalalignment='right', fontsize=12, color='black')
plt.xlabel("Số cây con (n_estimators)")
plt.ylabel("Độ chính xác")
plt.title("Tối ưu n_estimators")
plt.legend()
plt.grid(True)
plt.savefig("n_estimators.png", dpi=300)

# 📌 2. Tối ưu độ sâu tối đa (max_depth)
accuracy_results["max_depth"] = []
for d in max_depth_range:
    acc = accuracy_score(y_test, RandomForestClassifier(n_estimators=best_n, max_depth=d, random_state=42).fit(X_train, y_train).predict(X_test))
    accuracy_results["max_depth"].append(acc)

best_d = max_depth_range[np.argmax(accuracy_results["max_depth"])]
plt.figure(figsize=(10, 5))
plt.plot(max_depth_range, accuracy_results["max_depth"], marker='o', linestyle='dashed', color='blue')
plt.axvline(x=best_d, color='r', linestyle='--', label=f"Tối ưu: {best_d}")
plt.text(best_d, max(accuracy_results["max_depth"]), f"{best_d}", verticalalignment='bottom', horizontalalignment='right', fontsize=12, color='black')
plt.xlabel("Độ sâu tối đa (max_depth)")
plt.ylabel("Độ chính xác")
plt.title("Tối ưu max_depth")
plt.legend()
plt.grid(True)
plt.savefig("max_depth.png", dpi=300)

# 📌 3. Tối ưu số mẫu tối thiểu để chia nhánh (min_samples_split)
accuracy_results["min_samples_split"] = []
for s in min_samples_split_range:
    acc = accuracy_score(y_test, RandomForestClassifier(n_estimators=best_n, max_depth=best_d, min_samples_split=s, random_state=42).fit(X_train, y_train).predict(X_test))
    accuracy_results["min_samples_split"].append(acc)

best_s = min_samples_split_range[np.argmax(accuracy_results["min_samples_split"])]
plt.figure(figsize=(10, 5))
plt.plot(min_samples_split_range, accuracy_results["min_samples_split"], marker='o', linestyle='dashed', color='blue')
plt.axvline(x=best_s, color='r', linestyle='--', label=f"Tối ưu: {best_s}")
plt.text(best_s, max(accuracy_results["min_samples_split"]), f"{best_s}", verticalalignment='bottom', horizontalalignment='right', fontsize=12, color='black')
plt.xlabel("Số mẫu tối thiểu để chia nhánh (min_samples_split)")
plt.ylabel("Độ chính xác")
plt.title("Tối ưu min_samples_split")
plt.legend()
plt.grid(True)
plt.savefig("min_samples_split.png", dpi=300)

# 📌 4. Tối ưu số mẫu tối thiểu để làm nút lá (min_samples_leaf)
accuracy_results["min_samples_leaf"] = []
for l in min_samples_leaf_range:
    acc = accuracy_score(y_test, RandomForestClassifier(n_estimators=best_n, max_depth=best_d, min_samples_split=best_s, min_samples_leaf=l, random_state=42).fit(X_train, y_train).predict(X_test))
    accuracy_results["min_samples_leaf"].append(acc)

best_l = min_samples_leaf_range[np.argmax(accuracy_results["min_samples_leaf"])]
plt.figure(figsize=(10, 5))
plt.plot(min_samples_leaf_range, accuracy_results["min_samples_leaf"], marker='o', linestyle='dashed', color='blue')
plt.axvline(x=best_l, color='r', linestyle='--', label=f"Tối ưu: {best_l}")
plt.text(best_l, max(accuracy_results["min_samples_leaf"]), f"{best_l}", verticalalignment='bottom', horizontalalignment='right', fontsize=12, color='black')
plt.xlabel("Số mẫu tối thiểu để làm lá (min_samples_leaf)")
plt.ylabel("Độ chính xác")
plt.title("Tối ưu min_samples_leaf")
plt.legend()
plt.grid(True)
plt.savefig("min_samples_leaf.png", dpi=300)


# 📌 Huấn luyện mô hình tối ưu
rf_best = RandomForestClassifier(
    n_estimators=best_n, 
    max_depth=best_d, 
    min_samples_split=best_s, 
    min_samples_leaf=best_l,
    random_state=42
)
rf_best.fit(X_train, y_train)
y_pred_rf = rf_best.predict(X_test)

# Tính độ chính xác cuối cùng
accuracy_rf = accuracy_score(y_test, y_pred_rf)
cm_rf = confusion_matrix(y_test, y_pred_rf)

print(f"🔹 Độ chính xác Random Forest: {accuracy_rf:.5f}")
print("📌 Ma trận nhầm lẫn (Ramdom Forest):\n", cm_rf)

# 🔍 Lấy giá trị độ quan trọng của thuộc tính
feature_importances = rf_best.feature_importances_

# 📊 Sắp xếp độ quan trọng giảm dần
feature_names = X_train.columns if isinstance(X_train, pd.DataFrame) else [f"Feature {i}" for i in range(X_train.shape[1])]
sorted_idx = np.argsort(feature_importances)[::-1]  # Sắp xếp giảm dần
sorted_features = [feature_names[i] for i in sorted_idx]
sorted_importances = feature_importances[sorted_idx]

# 📊 Vẽ biểu đồ độ quan trọng của các thuộc tính (tính theo %)
plt.figure(figsize=(12, 6))
plt.barh(sorted_features, sorted_importances * 100, color="royalblue")  # Nhân với 100
plt.xlabel("Mức độ quan trọng (%)")  # Đổi đơn vị trục x
plt.ylabel("Thuộc tính")
plt.title("Độ quan trọng của các thuộc tính trong Random Forest (%)")
plt.gca().invert_yaxis()  # Đảo ngược trục để thuộc tính quan trọng nhất ở trên cùng
plt.grid(axis="x", linestyle="--", alpha=0.7)

# Lưu hình ảnh
plt.savefig("feature_importance.png", dpi=300)

# Vẽ và lưu ma trận nhầm lẫn cho KNN
plt.figure(figsize=(10, 8))
sns.heatmap(cm_knn, annot=True, fmt="d", cmap="Greens", xticklabels=["Không lừa đảo", "Lừa đảo"], 
            yticklabels=["Không lừa đảo", "Lừa đảo"])
plt.title(f"Ma trận nhầm lẫn - KNN (k={best_k})")
plt.xlabel("Giá trị thực tế")
plt.ylabel("Giá trị dự đoán")
plt.savefig("confusion_matrix_knn.png", dpi=300)

# Vẽ và lưu ma trận nhầm lẫn cho Naive Bayes
plt.figure(figsize=(10, 8))
sns.heatmap(cm_nb, annot=True, fmt="d", cmap="Oranges", xticklabels=["Không lừa đảo", "Lừa đảo"], 
            yticklabels=["Không lừa đảo", "Lừa đảo"])
plt.title("Ma trận nhầm lẫn - Naive Bayes")
plt.xlabel("Giá trị thực tế")
plt.ylabel("Giá trị dự đoán")
plt.savefig("confusion_matrix_naive_bayes.png", dpi=300)

# Vẽ và lưu ma trận nhầm lẫn cho Radom Forest
plt.figure(figsize=(10, 8))
sns.heatmap(cm_rf, annot=True, fmt="d", cmap="YlGnBu", xticklabels=["Không lừa đảo", "Lừa đảo"], 
            yticklabels=["Không lừa đảo", "Lừa đảo"])
plt.title("Ma trận nhầm lẫn - Random Forest")
plt.xlabel("Giá trị thực tế")
plt.ylabel("Giá trị dự đoán")
plt.savefig("confusion_matrix_random_forest.png", dpi=300)

plt.show()

# Tìm độ chính xác cao nhất
best_accuracy = max(accuracy_knn, accuracy_nb, accuracy_rf)

if best_accuracy == accuracy_knn:
    joblib.dump(knn_best, "knn_model.pkl")
    print(f"📌 Mô hình có độ chính xác cao nhất: KNN (k={best_k}, {accuracy_knn:.5f}) đã được lưu!")
elif best_accuracy == accuracy_nb:
    joblib.dump(nb_model, "naive_bayes_model.pkl")
    print(f"📌 Mô hình có độ chính xác cao nhất: Naive Bayes ({accuracy_nb:.5f}) đã được lưu!")
else:
    joblib.dump(rf_best, "random_forest_model.pkl")
    print(f"📌 Mô hình có độ chính xác cao nhất: Random Forest ({accuracy_rf:.5f}) đã được lưu!")
