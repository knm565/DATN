import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import numpy as np

# Đọc dữ liệu từ file train.csv và test.csv
train_data = pd.read_csv('data/train.csv')
test_data = pd.read_csv('data/test.csv')

# Tiền xử lý dữ liệu
# Chuyển đổi dữ liệu không phải số sang số bằng cách áp dụng dummies
data = pd.get_dummies(data)

# Tách các features và target từ tập huấn luyện
X_train = train_data.drop(columns=['SalePrice'])
y_train = train_data['SalePrice']

# Các features của tập kiểm tra
X_test = test_data.copy()  # Không có cột 'SalePrice' trong tập kiểm tra

# Khởi tạo Random Forest Regressor
rf = RandomForestRegressor(n_estimators=100, random_state=42)

# Huấn luyện mô hình trên tập dữ liệu train
rf.fit(X_train, y_train)
# Dự đoán giá nhà trên tập dữ liệu test
y_pred = rf.predict(X_test)

# Tính toán sai số bình phương trung bình
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')
