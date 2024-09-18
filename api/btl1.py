from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import StackingRegressor
from flask import Flask, request, jsonify, render_template
import numpy as np
import os

# Tải dữ liệu
data = fetch_california_housing()
print(data)
X = data.data #lưu trữ ma trận dữ liệu đầu vào
y = data.target #lưu trữ đầu ra dữ liệu

# Chia tập dữ liệu
# test_size:phần trăm đưa vào kiểm tra còn lại là huấn luyện
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Chuẩn hóa dữ liệu
scaler = StandardScaler() #Bộ chuẩn hóa sẽ tính giá trị trung bình và độ lệch chuẩn
X_train = scaler.fit_transform(X_train) #Tính giá trị trung bình và độ lệch chuẩn cho dữ liệu huấn luyện và chuẩn hóa nó.
X_test = scaler.transform(X_test) #Chuẩn hóa dữ liệu kiểm tra dựa trên thông tin từ tập huấn luyện.

# Huấn luyện mô hình hồi quy tuyến tính
linear_model = LinearRegression() #Khởi tạo mô hình hồi quy tuyến tính
linear_model.fit(X_train, y_train) #Huấn luyện mô hình để tìm mối quan hệ giữa các đặc trưng đầu vào (X_train) và giá trị mục tiêu (y_train)

# Dự đoán
y_pred_linear = linear_model.predict(X_test)

# Đánh giá mô hình
mse_linear = mean_squared_error(y_test, y_pred_linear) #đo lường sai số bình phương trung bình giữa giá trị dự đoán và giá trị thực tế
rmse_linear = mse_linear ** 0.5 #thể hiện sai số với cùng đơn vị như biến mục tiêu
r2_linear = r2_score(y_test, y_pred_linear) #đánh giá mức độ mà mô hình giải thích được phương sai của biến mục tiêu, càng gần 1 càng tốt

print(f'Linear Regression - MSE: {mse_linear}, RMSE: {rmse_linear}, R²: {r2_linear}')

# Huấn luyện mô hình Ridge
ridge_model = Ridge(alpha=1.0) 
ridge_model.fit(X_train, y_train)

# Dự đoán
y_pred_ridge = ridge_model.predict(X_test)

# Đánh giá mô hình
mse_ridge = mean_squared_error(y_test, y_pred_ridge)
rmse_ridge = mse_ridge ** 0.5
r2_ridge = r2_score(y_test, y_pred_ridge)

print(f'Ridge Regression - MSE: {mse_ridge}, RMSE: {rmse_ridge}, R²: {r2_ridge}')

# Huấn luyện mô hình MLPRegressor
# hidden_layyer_sizes :Xác định số lượng nơ-ron trong các tầng ẩn. Ở đây, có một tầng ẩn với 100 nơ-ron
# max_iter:Xác định số lần lặp tối đa trong quá trình huấn luyện
# random_state:Giúp tái tạo kết quả bằng cách cố định nguồn gốc của các số ngẫu nhiên
mlp_model = MLPRegressor(hidden_layer_sizes=(100,), max_iter=500, random_state=42)
mlp_model.fit(X_train, y_train)

# Dự đoán
y_pred_mlp = mlp_model.predict(X_test)

# Đánh giá mô hình
mse_mlp = mean_squared_error(y_test, y_pred_mlp)
rmse_mlp = mse_mlp ** 0.5
r2_mlp = r2_score(y_test, y_pred_mlp)

print(f'MLP Regressor - MSE: {mse_mlp}, RMSE: {rmse_mlp}, R²: {r2_mlp}')

# Khởi tạo các mô hình cơ bản
estimators = [
    ('linear', LinearRegression()), # hồi quy tuyến tính
    ('ridge', Ridge(alpha=1.0)),    # hồi quy ridge
    ('mlp', MLPRegressor(hidden_layer_sizes=(100,), max_iter=500, random_state=42)) # mạng nơ-ron
]

# Mô hình meta
stacking_model = StackingRegressor(estimators=estimators, final_estimator=Ridge())

# Huấn luyện mô hình stacking
stacking_model.fit(X_train, y_train)

# Dự đoán
y_pred_stacking = stacking_model.predict(X_test)

# Đánh giá mô hình
mse_stacking = mean_squared_error(y_test, y_pred_stacking)
rmse_stacking = mse_stacking ** 0.5
r2_stacking = r2_score(y_test, y_pred_stacking)

print(f'Stacking - MSE: {mse_stacking}, RMSE: {rmse_stacking}, R²: {r2_stacking}')

app = Flask(__name__)
# Tải mô hình đã huấn luyện (ở đây dùng stacking model)
model = stacking_model
scaler = scaler

# Giao diện người dùng
@app.route('/')
def home():
    return render_template('index.html')

# Dự đoán từ dữ liệu người dùng nhập
@app.route('/predict', methods=['POST'])
def predict():
    # Lấy dữ liệu từ form nhập
    features = [float(x) for x in request.form.values()]
    final_features = np.array(features).reshape(1, -1)
    final_features = scaler.transform(final_features)
    # Dự đoán kết quả
    prediction = model.predict(final_features)
    prediction_linear = linear_model.predict(final_features)[0]
    prediction_ridge = ridge_model.predict(final_features)[0]
    prediction_mlp = mlp_model.predict(final_features)[0]
    prediction_stacking = stacking_model.predict(final_features)[0]
    # Kết quả trả về
    return render_template(
    'index.html', 
    prediction_text=f'Giá nhà dự đoán: ${prediction[0]:.2f}',
    prediction_text_linear=f'Giá nhà dự đoán (Linear Regression): ${prediction_linear:.2f}',
    prediction_text_ridge=f'Giá nhà dự đoán (Ridge Regression): ${prediction_ridge:.2f}',
    prediction_text_mlp=f'Giá nhà dự đoán (MLP Regressor): ${prediction_mlp:.2f}',
    prediction_text_stacking=f'Giá nhà dự đoán (Stacking Regressor): ${prediction_mlp:.2f}'
)

if __name__ == "__main__":
    app.run(debug=True)


