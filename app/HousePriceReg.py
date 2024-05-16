import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Đọc dữ liệu từ file train.csv và test.csv
train_data = pd.read_csv('/content/drive/MyDrive/data/train.csv')
test_data = pd.read_csv('/content/drive/MyDrive/data/test.csv')
submission = pd.read_csv('/content/drive/MyDrive/data/sample_submission.csv')

# Loại bỏ cột 'Id'
train_data = train_data.drop(columns=['Id'])
test_data = test_data.drop(columns=['Id'])
submission = submission.drop(columns=['Id'])
# Xử lý các giá trị bị thiếu
train_data['LotFrontage'] = train_data['LotFrontage'].fillna(train_data['LotFrontage'].mean())
train_data['Alley'] = train_data['Alley'].fillna('None')
test_data['LotFrontage'] = test_data['LotFrontage'].fillna(test_data['LotFrontage'].mean())
test_data['Alley'] = test_data['Alley'].fillna('None')

# Xử lý các giá trị bị thiếu cho các cột khác
for col in train_data.columns:
    if train_data[col].dtype == "object":
        train_data[col] = train_data[col].fillna('None')
    else:
        train_data[col] = train_data[col].fillna(train_data[col].mean())

for col in test_data.columns:
    if test_data[col].dtype == "object":
        test_data[col] = test_data[col].fillna('None')
    else:
        test_data[col] = test_data[col].fillna(test_data[col].mean())

# Kết hợp cả tập huấn luyện và tập kiểm tra để đảm bảo one-hot encoding nhất quán
all_data = pd.concat([train_data.drop(columns=['SalePrice']), test_data], axis=0)

# Chuyển đổi các biến phân loại thành các biến số
all_data = pd.get_dummies(all_data)

# Tách lại dữ liệu sau khi chuyển đổi
X_train = all_data.iloc[:len(train_data), :]
X_test = all_data.iloc[len(train_data):, :]
y_train = train_data['SalePrice']

# Khởi tạo và huấn luyện mô hình RandomForestRegressor
rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42, bootstrap=True)
rf_regressor.fit(X_train, y_train)

# Dự đoán trên tập kiểm tra
predictions = rf_regressor.predict(X_test)

plt.scatter(submission, predictions)
plt.xlabel('true price')
plt.ylabel('pre')
plt.show()

from sklearn.metrics import mean_absolute_error as MAE
err = MAE(submission, predictions)
print('Sai so trung binh: ', err)

# Lấy tầm quan trọng của các feature
importances = rf_regressor.feature_importances_
feature_names = X_train.columns

# Tạo DataFrame cho các feature và tầm quan trọng của chúng
feature_importances = pd.DataFrame({'Feature': feature_names, 'Importance': importances})

# Sắp xếp các feature theo tầm quan trọng giảm dần
feature_importances = feature_importances.sort_values(by='Importance', ascending=False)

# In các feature quan trọng theo thứ tự giảm dần
print(feature_importances)
