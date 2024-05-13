import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Đọc dữ liệu từ file train.csv
data = pd.read_csv("train.csv")

# Loại bỏ cột Id vì nó không mang lại ý nghĩa trong việc dự đoán
data.drop('Id', axis=1, inplace=True)

# Loại bỏ các dòng có giá trị bị thiếu trong cột SalePrice
data.dropna(subset=['SalePrice'], inplace=True)

# Chuyển đổi các biến phân loại sang biến mã hóa one-hot
data = pd.get_dummies(data)

# Chia dữ liệu thành features (X) và target (y)
X = data.drop('SalePrice', axis=1)
y = data['SalePrice']

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Khởi tạo và huấn luyện mô hình RandomForestRegressor
rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
rf_regressor.fit(X_train, y_train)

# Dự đoán trên tập kiểm tra
predictions = rf_regressor.predict(X_test)

# Đánh giá mô hình bằng mean squared error
mse = mean_squared_error(y_test, predictions)
print("Mean Squared Error:", mse)
